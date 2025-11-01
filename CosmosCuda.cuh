/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 [Huseyin Tugrul BUYUKISIK]
 *
 * This file is part of the CosmosSimulationWithCuda project.
 * See LICENSE in the root of the project for details.
 */
#define __CUDACC__
#include <math.h>
#include <vector>
#include <iostream>
#include <random>
#include <mutex>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
using ComplexVar = float2;
namespace Constants {
    // Number of threads per CUDA block. This is for 1536 resident threads per SM. For older GPUs, 512 or 1024 can be chosen.
    constexpr int THREADS = 768;
    // Number of CUDA devices (max 2 tested)
    constexpr int NUM_CUDA_DEVICES = 2;

    // FFT uses these (long-range force calculation)
    // N is width of lattice (N x N) and can be only a power of 2. Higher value increases accuracy at the cost of performance.
    constexpr int N = 2048;
    constexpr double MATH_PI = 3.14159265358979323846;
    // Local convolution (short-range force calculation) to increase accuracy for high-accuracy mode. Only computes closest masses within LOCAL_CONV_WIDTH / 2 range.
    constexpr int LOCAL_CONV_WIDTH = 33;

    // Time-step of simulation. Lower values increase accuracy.
    constexpr float dt = 0.002f;

    // For render buffer output. Asynchronously filled.
    constexpr int MAX_FRAMES_BUFFERED = 40;
    constexpr int BLUR_R = 3;
    constexpr int BLUR_HALF_R = (BLUR_R - 1) / 2;
}
namespace Kernels {
    constexpr int SUB_MATRIX_SIZE = 32;
    constexpr int NUM_SUB_MATRICES_PER_DIMENSION = Constants::N / SUB_MATRIX_SIZE;
    constexpr int PAIR_LIST_SIZE = (NUM_SUB_MATRICES_PER_DIMENSION * (NUM_SUB_MATRICES_PER_DIMENSION - 1)) / 2;
    __constant__ float shortRangeGravKern_c[Constants::LOCAL_CONV_WIDTH * Constants::LOCAL_CONV_WIDTH];
    __constant__ float renderBlurKern_c[Constants::BLUR_R * Constants::BLUR_R];
    __device__ int2 pairIndexList_d[PAIR_LIST_SIZE];
    __device__ float minVar;
    __device__ float maxVar;
    __device__ float smoothMin;
    __device__ float smoothMax;
    __device__ ComplexVar wCoefficients[Constants::N * 2];
    __device__ __host__ __forceinline__ ComplexVar makeComplexVar(float x, float y) {
        ComplexVar result;
        result.x = x;
        result.y = y;
        return result;
    }
    __device__ __forceinline__ ComplexVar d_wForNk(const double N, const double k) {
        const double angle = (-2.0 * Constants::MATH_PI * k) / N;
        return makeComplexVar(cos(angle), -sin(angle));
    }
    template<int N>
    __global__ void k_fillCoefficientArray() {
        const int thread = threadIdx.x;
        const int threads = blockDim.x;
        int offset = 0;
        for (int i = 1; i < N; i *= 2) {
            const int steps = (i * 2 + threads - 1) / threads;
            for (int j = 0; j < steps; j++) {
                const int id = j * threads + thread;
                if (id < i * 2) {
                    wCoefficients[offset + id] = d_wForNk(i * 2, id);
                }

            }
            offset += i * 2;
        }
    }

    __device__ __forceinline__ void d_calcWarpDft(ComplexVar& var, const int warpLane, const float inverseMult, ComplexVar* wCoefficients) {
        int ofs = 0;
        #pragma unroll 1
        for (int level = 1; level <= 16; level <<= 1)
        {
            const int level2 = level * 2;
            const int idx = (warpLane % level2);
            const bool firstHalf = idx < level;
            const ComplexVar gather = makeComplexVar(__shfl_xor_sync(0xFFFFFFFF, var.x, level), __shfl_xor_sync(0xFFFFFFFF, var.y, level));
            const ComplexVar select1 = firstHalf ? var : gather;
            const ComplexVar select2 = firstHalf ? gather : var;
            ComplexVar tmp = __ldg(&wCoefficients[ofs + idx]);
            tmp.y *= inverseMult;
            float tmpX = fmaf(select2.x, tmp.x, select1.x);
            var.x = fmaf(-select2.y, tmp.y, tmpX);
            float tmpY = fmaf(select2.x, tmp.y, select1.y);
            var.y = fmaf(select2.y, tmp.x, tmpY);
            ofs += level * 2;
        }
    }


    template<int N>
    __device__ constexpr unsigned int d_bits() {
        int t = 1;
        int ctr = 0;
        while (t < N) {
            t *= 2;
            ctr++;
        }
        return ctr;
    }
    // This is used for 2D
    template<int N>
    __global__ void k_calcFftBatched1D(ComplexVar* data, const bool inverse = false) {
        const unsigned int thread = threadIdx.x;
        const unsigned int warpLane = thread & 31;
        const unsigned int block = blockIdx.x;
        const unsigned int numBlocks = gridDim.x;
        const unsigned int gridSteps = (N + numBlocks - 1) / numBlocks;
        constexpr unsigned int blockSteps = (N + Constants::THREADS - 1) / Constants::THREADS;
        const float inverseMult = inverse ? -1.0f : 1.0f;
        const float divider = inverse ? N : 1.0f;
        ComplexVar vars[blockSteps];
        extern __shared__ ComplexVar s_coalescing[];
        for (unsigned int grid = 0; grid < gridSteps; grid++) {
            const unsigned int row = grid * numBlocks + block;
            if (row < N) {
                #pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * Constants::THREADS + thread;
                    const int element = col + row * N;
                    if (col < N) {
                        s_coalescing[col] = __ldg(&data[element]);
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * Constants::THREADS + thread;
                    if (col < N) {
                        vars[blc] = s_coalescing[__brev(col) >> (32 - d_bits<Constants::N>())];
                    }
                    d_calcWarpDft(vars[blc], warpLane, inverseMult, wCoefficients);
                }

                // todo: ping-pong buffer = less syncthreads, bigger input support
                __syncthreads();
                #pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * Constants::THREADS + thread;
                    if (col < N) {
                        s_coalescing[col] = vars[blc];
                    }
                }
                __syncthreads();
                int wOfs = 62;
                #pragma unroll 1
                for (unsigned int level = 32; level < N; level <<= 1) {
                    const int level2 = level * 2;
                    #pragma unroll
                    for (int blc = 0; blc < blockSteps; blc++) {
                        const unsigned int col = blc * Constants::THREADS + thread;
                        if (col < N) {
                            const int idx = (col % level2);
                            const bool firstHalf = idx < level;
                            const ComplexVar gather = s_coalescing[col ^ level];
                            const ComplexVar select1 = firstHalf ? vars[blc] : gather;
                            const ComplexVar select2 = firstHalf ? gather : vars[blc];
                            ComplexVar tmp = __ldg(&wCoefficients[wOfs + idx]);
                            tmp.y *= inverseMult;
                            float tmpX = fmaf(select2.x, tmp.x, select1.x);
                            vars[blc].x = fmaf(-select2.y, tmp.y, tmpX);
                            float tmpY = fmaf(select2.x, tmp.y, select1.y);
                            vars[blc].y = fmaf(select2.y, tmp.x, tmpY);
                        }
                    }
                    wOfs += level * 2;
                    __syncthreads();
                    #pragma unroll
                    for (int blc = 0; blc < blockSteps; blc++) {
                        const unsigned int col = blc * Constants::THREADS + thread;
                        if (col < N) {
                            s_coalescing[col] = vars[blc];
                        }
                    }
                    __syncthreads();
                }
                #pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * Constants::THREADS + thread;
                    const int element = col + row * N;
                    if (col < N) {
                        vars[blc].x /= divider;
                        vars[blc].y /= divider;
                        data[element] = vars[blc];
                    }
                }

            }
        }
    }
    // Computes pair indices to be used in transpose operations without requiring double-precision calculation at expense of memory fetch.
    template<int N>
    __global__ void k_fillPairIndexList() {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (PAIR_LIST_SIZE + numTotalThreads - 1) / numTotalThreads;
        for (int ii = 0; ii < steps; ii++) {
            const long long index = ii * numTotalThreads + globalThread;
            if (index < PAIR_LIST_SIZE) {
                const long long i = NUM_SUB_MATRICES_PER_DIMENSION - 2.0 - floor(sqrt(4.0 * PAIR_LIST_SIZE * 2ll - 8.0 * index - 7.0) * 0.5 - 0.5);
                const long long j = index + i + 1ll - PAIR_LIST_SIZE + (((NUM_SUB_MATRICES_PER_DIMENSION - i) * (NUM_SUB_MATRICES_PER_DIMENSION - 1ll - i)) >> 1ll);
                pairIndexList_d[index] = make_int2(i, j);
            }
        }

    }

    template<int N>
    __global__ void k_calcTranspose(ComplexVar* data) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int steps = (PAIR_LIST_SIZE + numBlocks - 1) / numBlocks;
        __shared__ ComplexVar s_tile[SUB_MATRIX_SIZE][SUB_MATRIX_SIZE + 1];
        __shared__ ComplexVar s_tile2[SUB_MATRIX_SIZE][SUB_MATRIX_SIZE + 1];
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numBlocks + block;
            if (index < PAIR_LIST_SIZE) {
                const int2 pair = pairIndexList_d[index];
                const int i = pair.x;
                const int j = pair.y;
                if (i < j) {
                    const int tileX = i * SUB_MATRIX_SIZE;
                    const int tileY = j * SUB_MATRIX_SIZE;
                    const int tileOffset = tileX + tileY * N;
                    const int tileOffset2 = tileY + tileX * N;
                    constexpr int SUB_ELEMENTS = SUB_MATRIX_SIZE * SUB_MATRIX_SIZE;
                    const int steps2 = (SUB_ELEMENTS + numThreads - 1) / numThreads;
                    #pragma unroll
                    for (int k = 0; k < steps2; k++) {
                        const int element = k * numThreads + thread;
                        if (element < SUB_ELEMENTS) {
                            const int col = element % SUB_MATRIX_SIZE;
                            const int row = element / SUB_MATRIX_SIZE;
                            s_tile[row][col] = data[tileOffset + col + row * N];
                            s_tile2[row][col] = data[tileOffset2 + col + row * N];
                        }
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int k = 0; k < steps2; k++) {
                        const int element = k * numThreads + thread;
                        if (element < SUB_ELEMENTS) {
                            const int col = element % SUB_MATRIX_SIZE;
                            const int row = element / SUB_MATRIX_SIZE;
                            data[tileOffset2 + col + row * N] = s_tile[col][row];
                            data[tileOffset + col + row * N] = s_tile2[col][row];
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
    template<int N>
    __global__ void k_calcTransposeDiagonals(ComplexVar* data) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;

        const int steps = (NUM_SUB_MATRICES_PER_DIMENSION + numBlocks - 1) / numBlocks;
        __shared__ ComplexVar s_tile[SUB_MATRIX_SIZE][SUB_MATRIX_SIZE + 1];
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numBlocks + block;
            if (index < NUM_SUB_MATRICES_PER_DIMENSION) {
                // if i == j, transpose in-place. if i < j, swap
                const int i = index;
                const int j = index;
                const int tileX = i * SUB_MATRIX_SIZE;
                const int tileY = j * SUB_MATRIX_SIZE;
                const int tileOffset = tileX + tileY * N;
                constexpr int SUB_ELEMENTS = SUB_MATRIX_SIZE * SUB_MATRIX_SIZE;
                const int steps2 = (SUB_ELEMENTS + numThreads - 1) / numThreads;
                #pragma unroll
                for (int k = 0; k < steps2; k++) {
                    const int element = k * numThreads + thread;
                    if (element < SUB_ELEMENTS) {
                        const int col = element % SUB_MATRIX_SIZE;
                        const int row = element / SUB_MATRIX_SIZE;
                        s_tile[row][col] = data[tileOffset + col + row * N];
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int k = 0; k < steps2; k++) {
                    const int element = k * numThreads + thread;
                    if (element < SUB_ELEMENTS) {
                        const int col = element % SUB_MATRIX_SIZE;
                        const int row = element / SUB_MATRIX_SIZE;
                        data[tileOffset + col + row * N] = s_tile[col][row];
                    }
                }
                __syncthreads();
            }
        }
    }
    template<int N>
    __global__ void k_generateFilterLattice(ComplexVar* data, bool accuracy) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        //constexpr float cutoff = 2.0f;
        // when accuracy mode enabled, a second lattice is computed only for closest neighbors (within 16 lattice cells radius) using direct convolution.
        constexpr float selfForceAvoidance = 2.0f;
        constexpr float shortRangeForceRange = (Constants::LOCAL_CONV_WIDTH - 1) / 2.0f;
        const float cutoff = accuracy ? shortRangeForceRange : selfForceAvoidance;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                const int i = index % N;
                const int j = index / N;
                const int cx = N / 2;
                const int cy = N / 2;
                const float dx = i - cx;
                const float dy = j - cy;
                const float r = sqrtf(dx * dx + dy * dy);
                float mult = (N / 2048.0f) * (N / 2048.0f);
                
                if (r > cutoff) {
                    data[((i + N / 2) % N) + ((j + N / 2) % N) * N].x = mult / r;
                    data[((i + N / 2) % N) + ((j + N / 2) % N) * N].y = 0.0f;
                }
                else {
                    data[((i + N / 2) % N) + ((j + N / 2) % N) * N].x = 0.0f;
                    data[((i + N / 2) % N) + ((j + N / 2) % N) * N].y = 0.0f;
                }
            }
        }
    }
    template<int N>
    __global__ void k_multElementwiseLatticeFilter(ComplexVar* data, ComplexVar* data2) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                const int i = index % N;
                const int j = index / N;
                auto d1 = data[i + j * N];
                const auto d2 = data2[i + j * N];
                const float tmpX = d1.x * d2.x - d1.y * d2.y;
                const float tmpY = d1.x * d2.y + d1.y * d2.x;
                d1.x = tmpX;
                d1.y = tmpY;
                data[i + j * N] = d1;
            }
        }
    }
    template<int N>
    __global__ void k_calcGradientLattice(const float* const __restrict__ lattice_d, float2* const __restrict__ latticeForceXY_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N*N + numTotalThreads - 1) / numTotalThreads;
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                const float centerData = __ldca(&lattice_d[index]);
                float left = centerData;
                float right = centerData;
                float top = centerData;
                float bot = centerData;
                float leftLeft = centerData;
                float rightRight = centerData;
                float topTop = centerData;
                float botBot = centerData;
                const int centerX = index % N;
                const int centerY = index / N;
                if (centerX - 1 >= 0) {
                    left = __ldca(&lattice_d[index - 1]);
                }
                if (centerX + 1 < N) {
                    right = __ldca(&lattice_d[index + 1]);
                }
                if (centerY - 1 >= 0) {
                    top = __ldca(&lattice_d[index - N]);
                }
                if (centerY + 1 < N) {
                    bot = __ldca(&lattice_d[index + N]);
                }
                if (centerX - 2 >= 0) {
                    leftLeft = __ldca(&lattice_d[index - 2]);
                }
                if (centerX + 2 < N) {
                    rightRight = __ldca(&lattice_d[index + 2]);
                }
                if (centerY - 2 >= 0) {
                    topTop = __ldca(&lattice_d[index - 2*N]);
                }
                if (centerY + 2 < N) {
                    botBot = __ldca(&lattice_d[index + 2*N]);
                }
                // Gradient
                const float h = 1024.0f / N;
                const float forceComponentX = (-rightRight + 8.0f * right - 8.0f * left + leftLeft) / (h * 12.0f);
                const float forceComponentY = (-botBot + 8.0f * bot - 8.0f * top + topTop) / (h * 12.0f);
                
                latticeForceXY_d[index] = make_float2(forceComponentX, forceComponentY);
            }
        }
    }

    template<int N>
    __global__ void k_forceMultiSampling(const float2* const __restrict__ latticeForceXY_d, float* const __restrict__ x, float* const __restrict__ y, float* const __restrict__ vx, float* const __restrict__ vy, const int numParticles, const bool accuracy) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (numParticles / 4 + numTotalThreads - 1) / numTotalThreads;
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < numParticles / 4) {
                // Particle data is not expected to fit caches.
                const float4 posXr = __ldcs(reinterpret_cast<float4*>(&x[index * 4]));
                const float4 posYr = __ldcs(reinterpret_cast<float4*>(&y[index * 4]));
                const float4 posVXr = __ldcs(reinterpret_cast<float4*>(&vx[index * 4]));
                const float4 posVYr = __ldcs(reinterpret_cast<float4*>(&vy[index * 4]));
                float posX[4] = { posXr.x,posXr.y,posXr.z,posXr.w };
                float posY[4] = { posYr.x,posYr.y,posYr.z,posYr.w };
                float vxr[4] = { posVXr.x, posVXr.y, posVXr.z, posVXr.w };
                float vyr[4] = { posVYr.x, posVYr.y, posVYr.z, posVYr.w };
                #pragma unroll 4
                for (int m = 0; m < 4; m++) {
                    const int centerX = int(posX[m]);
                    const int centerY = int(posY[m]);
                    const int centerIndex = centerX + centerY * N;
                    if (centerX >= 1 && centerX < N - 1 && centerY >= 1 && centerY < N - 1) {
                        // Getting precalculated gradient. This should benefit from caching when many particles access same point.
                        // Then calculating interpolation for a more accurate behavior.
                        float xComponent;
                        float yComponent;
                        if (accuracy) {
                            const float2 forceComponentsCurrent = __ldca(&latticeForceXY_d[centerIndex]);
                            const float2 forceComponentsRight = __ldca(&latticeForceXY_d[centerIndex + 1]);
                            const float2 forceComponentsBottom = __ldca(&latticeForceXY_d[centerIndex + N]);
                            const float2 forceComponentsBottomRight = __ldca(&latticeForceXY_d[centerIndex + 1 + N]);
                            const float fractionalX = posX[m] - centerX;
                            const float fractionalY = posY[m] - centerY;
                            const float xDiff1 = 1.0f - fractionalX;
                            const float yDiff1 = 1.0f - fractionalY;
                            xComponent = forceComponentsCurrent.x * xDiff1 * yDiff1 +
                                forceComponentsRight.x * fractionalX * yDiff1 +
                                forceComponentsBottom.x * xDiff1 * fractionalY +
                                forceComponentsBottomRight.x * fractionalX * fractionalY;
                            yComponent = forceComponentsCurrent.y * xDiff1 * yDiff1 +
                                forceComponentsRight.y * fractionalX * yDiff1 +
                                forceComponentsBottom.y * xDiff1 * fractionalY +
                                forceComponentsBottomRight.y * fractionalX * fractionalY;
                        }
                        else {
                            const float2 forceComponentsCurrent = __ldca(&latticeForceXY_d[centerIndex]);
                            xComponent = forceComponentsCurrent.x;
                            yComponent = forceComponentsCurrent.y;
                        }
                        float newX = fmaf(vxr[m], Constants::dt, posX[m]);
                        float newY = fmaf(vyr[m], Constants::dt, posY[m]);
                        if (newX < 2.0f) {
                            newX += N - 4.0f;
                        }
                        if (newY < 2.0f) {
                            newY += N - 4.0f;
                        }
                        if (newX > N - 2.0f) {
                            newX -= N - 4.0f;
                        }
                        if (newY > N - 2.0f) {
                            newY -= N - 4.0f;
                        }
                        posX[m] = newX;
                        posY[m] = newY;
                        vxr[m] = fmaf(xComponent, Constants::dt, vxr[m]);
                        vyr[m] = fmaf(yComponent, Constants::dt, vyr[m]);
                    }
                }
                __stcs(reinterpret_cast<float4*>(&x[index * 4]), make_float4(posX[0], posX[1], posX[2], posX[3]));
                __stcs(reinterpret_cast<float4*>(&y[index * 4]), make_float4(posY[0], posY[1], posY[2], posY[3]));
                __stcs(reinterpret_cast<float4*>(&vx[index * 4]), make_float4(vxr[0], vxr[1], vxr[2], vxr[3]));
                __stcs(reinterpret_cast<float4*>(&vy[index * 4]), make_float4(vyr[0], vyr[1], vyr[2], vyr[3]));
            }
        }
    }


    template<int N>
    __global__ void k_clearAccumulator(float* const __restrict__ accumulator_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                accumulator_d[index] = 0.0f;
            }
        }
    }

    // Optionally shifts and adds local values to global.
    template<int N>
    __global__ void k_shiftLattice(ComplexVar* lattice_d, float* latticeLocal_d, float* latticeShifted_d, bool accuracy) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                if (accuracy) {
                    latticeShifted_d[index] = lattice_d[index].x + latticeLocal_d[index];
                }
                else {
                    latticeShifted_d[index] = lattice_d[index].x;
                }
            }
        }
    }

    template<int N>
    __global__ void k_sumAccumulators(float* accumulator_d, float* accumulator2_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                accumulator_d[index] += accumulator2_d[index];
            }
        }
    }

    template<int N>
    __global__ void k_scatterMassOnAccumulator(float* const __restrict__ accumulator_d, const float* x, const float* y, const float* m, const int numParticles, bool accuracy) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (numParticles + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < numParticles) {
                const float xf = __ldcs(&x[index]);
                const float yf = __ldcs(&y[index]);
                const float mass = __ldcs(&m[index]);
                const int xi = xf;
                const int yi = yf;
                if (xi >= 1 && xi < N - 1 && yi >= 1 && yi < N - 1) {
                    const float fractionalX = xf - xi;
                    const float fractionalY = yf - yi;
                    const float xDiff1 = 1.0f - fractionalX;
                    const float yDiff1 = 1.0f - fractionalY;
                    const float weightCurrent = xDiff1 * yDiff1;
                    const float weightRight = fractionalX * yDiff1;
                    const float weightBottom = xDiff1 * fractionalY;
                    const float weightBottomRight = fractionalX * fractionalY;
                    // Optional weighted scattering for more accuracy.
                    if (accuracy) {
                        atomicAdd(&accumulator_d[xi + yi * N], weightCurrent * mass);
                        atomicAdd(&accumulator_d[1 + xi + yi * N], weightRight * mass);
                        atomicAdd(&accumulator_d[xi + (yi + 1) * N], weightBottom * mass);
                        atomicAdd(&accumulator_d[1 + xi + (yi + 1) * N], weightBottomRight * mass);
                    } else {
                        atomicAdd(&accumulator_d[xi + yi * N], mass);
                    }
                }
            }
        }
    }
    template<int N>
    __global__ void k_copyAccumulatorIntoLattice(float* const __restrict__ accumulator_d, ComplexVar* const __restrict__ lattice_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                lattice_d[index] = makeComplexVar(accumulator_d[index], 0.0f);
            }
        }
    }
    template<int N>
    __global__ void k_getRealComponentOfLattice(ComplexVar* lattice_d, float* localForceLattice_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                localForceLattice_d[index] = lattice_d[index].x;
            }
        }
    }

    __global__ void k_resetMinMax() {
        minVar = FLT_MAX;
        maxVar = FLT_MIN;
    }
    __device__ static float atomicMax(float* adr, float val)
    {
        int* addressInt = (int*)adr;
        int old = *addressInt;
        int tested;
        do {
            tested = old;
            old = ::atomicCAS(addressInt, tested, __float_as_int(fmaxf(val, __int_as_float(tested))));
        } while (tested != old);
        return __int_as_float(old);
    }
    __device__ static float atomicMin(float* adr, float val)
    {
        int* addressInt = (int*)adr;
        int old = *addressInt;
        int tested;
        do {
            tested = old;
            old = ::atomicCAS(addressInt, tested, __float_as_int(fminf(val, __int_as_float(tested))));
        } while (tested != old);
        return __int_as_float(old);
    }
    template<int N>
    __global__ void k_calcMinMax(float* array_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        float localMin = FLT_MAX;
        float localMax = FLT_MIN;
        __shared__ float s_min[32];
        __shared__ float s_max[32];
        if (thread < 32) {
            s_min[thread] = FLT_MAX;
            s_max[thread] = FLT_MIN;
        }
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                const float data = array_d[index];
                if (data < localMin) {
                    localMin = data;
                }
                if (data > localMax) {
                    localMax = data;
                }
            }
        }
        const int lane = thread & 31;
        const int warp = thread / 32;
        float gather = __shfl_down_sync(0xFFFFFFFF, localMin, 16);
        localMin = (lane < 16) ? ((gather < localMin) ? gather : localMin) : localMin;
        gather = __shfl_down_sync(0xFFFFFFFF, localMin, 8);
        localMin = (lane < 8) ? ((gather < localMin) ? gather : localMin) : localMin;
        gather = __shfl_down_sync(0xFFFFFFFF, localMin, 4);
        localMin = (lane < 4) ? ((gather < localMin) ? gather : localMin) : localMin;
        gather = __shfl_down_sync(0xFFFFFFFF, localMin, 2);
        localMin = (lane < 2) ? ((gather < localMin) ? gather : localMin) : localMin;
        gather = __shfl_down_sync(0xFFFFFFFF, localMin, 1);
        localMin = (lane < 1) ? ((gather < localMin) ? gather : localMin) : localMin;

        gather = __shfl_down_sync(0xFFFFFFFF, localMax, 16);
        localMax = (lane < 16) ? ((gather > localMax) ? gather : localMax) : localMax;
        gather = __shfl_down_sync(0xFFFFFFFF, localMax, 8);
        localMax = (lane < 8) ? ((gather > localMax) ? gather : localMax) : localMax;
        gather = __shfl_down_sync(0xFFFFFFFF, localMax, 4);
        localMax = (lane < 4) ? ((gather > localMax) ? gather : localMax) : localMax;
        gather = __shfl_down_sync(0xFFFFFFFF, localMax, 2);
        localMax = (lane < 2) ? ((gather > localMax) ? gather : localMax) : localMax;
        gather = __shfl_down_sync(0xFFFFFFFF, localMax, 1);
        localMax = (lane < 1) ? ((gather > localMax) ? gather : localMax) : localMax;

        if (lane == 0) {
            s_min[warp] = localMin;
            s_max[warp] = localMax;
        }
        __syncthreads();
        if (warp == 0) {
            localMin = (lane < (numThreads >> 5)) ? s_min[lane] : 0.0f;
            localMax = (lane < (numThreads >> 5)) ? s_max[lane] : 0.0f;
            float gather = __shfl_down_sync(0xFFFFFFFF, localMin, 16);
            localMin = (lane < 16) ? ((gather < localMin) ? gather : localMin) : localMin;
            gather = __shfl_down_sync(0xFFFFFFFF, localMin, 8);
            localMin = (lane < 8) ? ((gather < localMin) ? gather : localMin) : localMin;
            gather = __shfl_down_sync(0xFFFFFFFF, localMin, 4);
            localMin = (lane < 4) ? ((gather < localMin) ? gather : localMin) : localMin;
            gather = __shfl_down_sync(0xFFFFFFFF, localMin, 2);
            localMin = (lane < 2) ? ((gather < localMin) ? gather : localMin) : localMin;
            gather = __shfl_down_sync(0xFFFFFFFF, localMin, 1);
            localMin = (lane < 1) ? ((gather < localMin) ? gather : localMin) : localMin;

            gather = __shfl_down_sync(0xFFFFFFFF, localMax, 16);
            localMax = (lane < 16) ? ((gather > localMax) ? gather : localMax) : localMax;
            gather = __shfl_down_sync(0xFFFFFFFF, localMax, 8);
            localMax = (lane < 8) ? ((gather > localMax) ? gather : localMax) : localMax;
            gather = __shfl_down_sync(0xFFFFFFFF, localMax, 4);
            localMax = (lane < 4) ? ((gather > localMax) ? gather : localMax) : localMax;
            gather = __shfl_down_sync(0xFFFFFFFF, localMax, 2);
            localMax = (lane < 2) ? ((gather > localMax) ? gather : localMax) : localMax;
            gather = __shfl_down_sync(0xFFFFFFFF, localMax, 1);
            localMax = (lane < 1) ? ((gather > localMax) ? gather : localMax) : localMax;
            if (lane == 0) {
                atomicMax(&maxVar, localMax);
                atomicMin(&minVar, localMin);
            }
        }
    }
    __global__ void k_initSmoothMinMax() {
        smoothMin = 1e35f;
        smoothMax = -1.0f;
    }
    __global__ void k_smoothMinMax() {
        if (smoothMin > minVar) {
            smoothMin = minVar;
        }
        if (smoothMax < maxVar) {
            smoothMax = maxVar;
        }
    }
    template<int N>
    __global__ void k_scaleWithMinMax(float* input_d, float* output_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                const float diff = smoothMax - smoothMin;
                output_d[index] = powf((input_d[index] - smoothMin) / diff, 0.2f);
            }
        }
    }
    // This is to capture short-range details missed by FFT.
    constexpr int TILE_SIZE = 32;
    template<int K, int N>
    __global__ void k_calcLocalMassConvolution(const float* const __restrict__ latticeIn_d, float* const __restrict__ latticeOut_d) {
        const int threadX = threadIdx.x;
        const int threadY = threadIdx.y;
        const int blockX = blockIdx.x;
        const int blockY = blockIdx.y;
        const int numThreadsX = blockDim.x;
        const int numThreadsY = blockDim.y;
        const int numBlocksX = gridDim.x;
        const int numBlocksY = gridDim.y;
        const int numTilesPerX = N / TILE_SIZE;
        const int numTilesPerY = N / TILE_SIZE;
        constexpr int tileElements = (TILE_SIZE + K) * (TILE_SIZE + K);
        const int tileLoadSteps = (tileElements + numThreadsX * numThreadsY - 1) / (numThreadsX * numThreadsY);
        constexpr int halfK = (K - 1) / 2;
        extern __shared__ float s_cache[];
        for (int tileY = 0; tileY < numTilesPerY / numBlocksY; tileY++) {
            for (int tileX = 0; tileX < numTilesPerX / numBlocksX; tileX++) {
                const int tileXX = tileX * numBlocksX + blockX;
                const int tileYY = tileY * numBlocksY + blockY;
                const int tileOffset = tileXX * TILE_SIZE + tileYY * TILE_SIZE * N;
                #pragma unroll
                for (int load = 0; load < tileLoadSteps; load++) {
                    const int loadT = load * numThreadsX * numThreadsY + threadX + threadY * numThreadsX;
                    const int loadedX = loadT % (TILE_SIZE + K);
                    const int loadedY = loadT / (TILE_SIZE + K);

                    const int loaded = loadedX + loadedY * (TILE_SIZE + K);
                    if (loadedY < (TILE_SIZE + K) && loadedX < (TILE_SIZE + K)) {
                        if (loadedX + tileXX * TILE_SIZE - halfK >= 0 && loadedX + tileXX * TILE_SIZE - halfK < N &&
                            loadedY + tileYY * TILE_SIZE - halfK >= 0 && loadedY + tileYY * TILE_SIZE - halfK < N) {
                            s_cache[loaded] = __ldg(&latticeIn_d[tileOffset + loadedX - halfK + (loadedY - halfK) * N]);
                        }
                        else {
                            s_cache[loaded] = 0.0f;
                        }
                    }

                }
                __syncthreads();
                float acc[2] = { 0.0f, 0.0f };
                #pragma unroll K
                for (int iy = -halfK; iy <= halfK; iy++) {
                    #pragma unroll K
                    for (int ix = -halfK; ix <= halfK; ix++) {
                        const int neighborX = ix + threadX + halfK;
                        const int neighborY = iy + threadY + halfK;
                        acc[(ix + halfK)&1] = fmaf(s_cache[neighborX + neighborY * (TILE_SIZE + K)], shortRangeGravKern_c[ix + halfK + (iy + halfK) * K], acc[(ix + halfK) & 1]);
                    }
                }
                __syncthreads();
                latticeOut_d[tileOffset + threadX + threadY * N] = acc[0] + acc[1];
            }
        }
    }
    template<int K, int N>
    __global__ void k_calcBlurConvolution(const float* const __restrict__ latticeIn_d, float* const __restrict__ latticeOut_d) {
        const int threadX = threadIdx.x;
        const int threadY = threadIdx.y;
        const int blockX = blockIdx.x;
        const int blockY = blockIdx.y;
        const int numThreadsX = blockDim.x;
        const int numThreadsY = blockDim.y;
        const int numBlocksX = gridDim.x;
        const int numBlocksY = gridDim.y;
        const int numTilesPerX = N / TILE_SIZE;
        const int numTilesPerY = N / TILE_SIZE;
        constexpr int tileElements = (TILE_SIZE + K) * (TILE_SIZE + K);
        const int tileLoadSteps = (tileElements + numThreadsX * numThreadsY - 1) / (numThreadsX * numThreadsY);
        constexpr int halfK = (K - 1) / 2;
        extern __shared__ float s_cache[];
        for (int tileY = 0; tileY < numTilesPerY / numBlocksY; tileY++) {
            for (int tileX = 0; tileX < numTilesPerX / numBlocksX; tileX++) {
                const int tileXX = tileX * numBlocksX + blockX;
                const int tileYY = tileY * numBlocksY + blockY;
                const int tileOffset = tileXX * TILE_SIZE + tileYY * TILE_SIZE * N;
                #pragma unroll
                for (int load = 0; load < tileLoadSteps; load++) {
                    const int loadT = load * numThreadsX * numThreadsY + threadX + threadY * numThreadsX;
                    const int loadedX = loadT % (TILE_SIZE + K);
                    const int loadedY = loadT / (TILE_SIZE + K);

                    const int loaded = loadedX + loadedY * (TILE_SIZE + K);
                    if (loadedY < (TILE_SIZE + K) && loadedX < (TILE_SIZE + K)) {
                        if (loadedX + tileXX * TILE_SIZE - halfK >= 0 && loadedX + tileXX * TILE_SIZE - halfK < N &&
                            loadedY + tileYY * TILE_SIZE - halfK >= 0 && loadedY + tileYY * TILE_SIZE - halfK < N) {
                            s_cache[loaded] = __ldg(&latticeIn_d[tileOffset + loadedX - halfK + (loadedY - halfK) * N]);
                        }
                        else {
                            s_cache[loaded] = 0.0f;
                        }
                    }

                }
                __syncthreads();
                float acc[2] = { 0.0f, 0.0f };
                #pragma unroll K
                for (int iy = -halfK; iy <= halfK; iy++) {
                    #pragma unroll K
                    for (int ix = -halfK; ix <= halfK; ix++) {
                        const int neighborX = ix + threadX + halfK;
                        const int neighborY = iy + threadY + halfK;
                        acc[(ix + halfK) & 1] = fmaf(s_cache[neighborX + neighborY * (TILE_SIZE + K)], renderBlurKern_c[ix + halfK + (iy + halfK) * K], acc[(ix + halfK) & 1]);
                    }
                }
                __syncthreads();
                latticeOut_d[tileOffset + threadX + threadY * N] = acc[0] + acc[1];
            }
        }
    }
}

struct Universe {
private:
    // For host
    std::vector<float> x, y;
    std::vector<float> vx, vy;
    std::vector<float> m;
    std::vector<float> renderColor;
    float* broadcast_h;
    float* frame_h;
    int particleCounter;
    int nbodyCalcCounter;
    int numNbodyStepsPerRender;
    int numParticles[Constants::NUM_CUDA_DEVICES];
    int particleOffsets[Constants::NUM_CUDA_DEVICES];
    int totalNumParticles;
    std::mt19937 rng;

    // For device
    cudaStream_t computeStream[Constants::NUM_CUDA_DEVICES];
    cudaStream_t broadcastStream[Constants::NUM_CUDA_DEVICES];
    cudaEvent_t latticeBroadcastEvent[Constants::NUM_CUDA_DEVICES];
    cudaEvent_t eventStart[Constants::NUM_CUDA_DEVICES];
    cudaEvent_t eventStop[Constants::NUM_CUDA_DEVICES];
    ComplexVar* lattice_d[Constants::NUM_CUDA_DEVICES];
    // 2: copies for double-buffering
    float* accumulator_d[Constants::NUM_CUDA_DEVICES][2];
    float* accumulator2_d[Constants::NUM_CUDA_DEVICES][2];
    float* latticeShifted_d[Constants::NUM_CUDA_DEVICES];
    float2* latticeShiftedForceXY_d[Constants::NUM_CUDA_DEVICES];
    ComplexVar* filter_d[Constants::NUM_CUDA_DEVICES];
    float* x_d[Constants::NUM_CUDA_DEVICES][2];
    float* y_d[Constants::NUM_CUDA_DEVICES][2];
    float* vx_d[Constants::NUM_CUDA_DEVICES];
    float* vy_d[Constants::NUM_CUDA_DEVICES];
    float* m_d[Constants::NUM_CUDA_DEVICES];
    float* renderOutput_d[Constants::NUM_CUDA_DEVICES];
    float* renderOutput2_d[Constants::NUM_CUDA_DEVICES];
    float* localForceLattice_d[Constants::NUM_CUDA_DEVICES];
    float* localForceLatticeResult_d[Constants::NUM_CUDA_DEVICES];
    int numBlocks[Constants::NUM_CUDA_DEVICES];
    int cudaDeviceIndices[Constants::NUM_CUDA_DEVICES];

    // Accuracy setting: increases accuracy of mass projections and force sampling at cost of 50% performance
    bool accuracy;

    // For frame queue
    std::mutex lock;
    std::vector<std::vector<float>> frames;
    std::thread computeThread;
    int pushCtr;
    int popCtr;
    bool working;


public:
    Universe(int particles, const int (&cudaDevices)[Constants::NUM_CUDA_DEVICES], const float(&devicePerformances)[Constants::NUM_CUDA_DEVICES], bool lowAccuracy, int numStepsPerRender) {
        float totalPerf = 0.0f;
        float perf[Constants::NUM_CUDA_DEVICES];
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            totalPerf += (fabsf(devicePerformances[device]) + 0.0001f);
        }
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            perf[device] = (fabsf(devicePerformances[device]) + 0.0001f) / totalPerf;
        }
        accuracy = !lowAccuracy;
        particleCounter = 0;
        nbodyCalcCounter = 0;
        srand(time(0));
        int total = 0;
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            numParticles[device] = particles * perf[device];
            total += numParticles[device];
        }
        int selector = 0;
        while (total > particles) {
            numParticles[(selector++ % 2)]--;
            total--;
        }
        while (total < particles) {
            numParticles[(selector++ % 2)]++;
            total++;
        }
        totalNumParticles = total;
        int sum = 0;
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            particleOffsets[device] = sum;
            sum += numParticles[device];
        }
        x.resize(particles);
        vx.resize(particles);
        y.resize(particles);
        vy.resize(particles);
        m.resize(particles);
        gpuErrchk(cudaMallocHost(&broadcast_h, sizeof(float) * Constants::N * Constants::N * Constants::NUM_CUDA_DEVICES));
        gpuErrchk(cudaMallocHost(&frame_h, sizeof(float) * (Constants::N * Constants::N + 1)));
        
        for (int i = 0; i < particles; i++) {
            x[i] = (rand() % Constants::N);
            y[i] = (rand() % Constants::N);
            vx[i] = 0;
            vy[i] = 0;
            m[i] = 1.0f;
        }
        // For local convolution in force calculation.
        constexpr int HALF_WIDTH = (Constants::LOCAL_CONV_WIDTH - 1) / 2;
        std::vector<float> localForceFilter(Constants::LOCAL_CONV_WIDTH * Constants::LOCAL_CONV_WIDTH);
        for (int iy = -HALF_WIDTH; iy <= HALF_WIDTH; iy++) {
            for (int ix = -HALF_WIDTH; ix <= HALF_WIDTH; ix++) {
                const int index = ix + HALF_WIDTH + (iy + HALF_WIDTH) * Constants::LOCAL_CONV_WIDTH;
                const double r = sqrt((double)(ix * ix + iy * iy));
                float mult = (Constants::N / 2048.0f) * (Constants::N / 2048.0f);
                constexpr float selfAvoidanceRangeForAccuracy = 2.0f;
                if (r > selfAvoidanceRangeForAccuracy && r < HALF_WIDTH) {
                    localForceFilter[index] = mult / r;
                }
                else {
                    localForceFilter[index] = 0.0f;
                }
            }
        }
        // For rendering.
        std::vector<float> renderFilter(Constants::BLUR_R * Constants::BLUR_R);
        for (int iy = -Constants::BLUR_HALF_R; iy <= Constants::BLUR_HALF_R; iy++) {
            for (int ix = -Constants::BLUR_HALF_R; ix <= Constants::BLUR_HALF_R; ix++) {
                const int index = ix + Constants::BLUR_HALF_R + (iy + Constants::BLUR_HALF_R) * Constants::BLUR_R;
                const double r = sqrt((double)(ix * ix + iy * iy));
                if (r < Constants::BLUR_HALF_R) {
                    renderFilter[index] = 1.0f / (r + 1.0f);
                }
                else {
                    renderFilter[index] = 0.0f;
                }
            }
        }
        particleCounter = particles;
        numNbodyStepsPerRender = numStepsPerRender;
        rng = std::mt19937(static_cast<unsigned int>(std::time(0)));
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            cudaDeviceIndices[device] = cudaDevices[device];

            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaStreamCreate(&computeStream[device]));
            gpuErrchk(cudaStreamCreate(&broadcastStream[device]));
            
            cudaDeviceProp prop;
            gpuErrchk(cudaGetDeviceProperties(&prop, cudaDeviceIndices[device]));
            int blocksPerSM = (prop.maxThreadsPerMultiProcessor + Constants::THREADS - 1) / Constants::THREADS;
            numBlocks[device] = prop.multiProcessorCount * blocksPerSM;

            gpuErrchk(cudaMemcpyToSymbolAsync(Kernels::shortRangeGravKern_c, localForceFilter.data(), sizeof(float) * Constants::LOCAL_CONV_WIDTH * Constants::LOCAL_CONV_WIDTH, 0, cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyToSymbolAsync(Kernels::renderBlurKern_c, renderFilter.data(), sizeof(float) * Constants::BLUR_R * Constants::BLUR_R, 0, cudaMemcpyHostToDevice, computeStream[device]));
            
            gpuErrchk(cudaEventCreate(&eventStart[device]));
            gpuErrchk(cudaEventCreate(&eventStop[device]));
            gpuErrchk(cudaEventCreate(&latticeBroadcastEvent[device]));
            
            gpuErrchk(cudaMallocAsync(&lattice_d[device], sizeof(ComplexVar) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&localForceLattice_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&localForceLatticeResult_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&latticeShifted_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&latticeShiftedForceXY_d[device], sizeof(float2) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&filter_d[device], sizeof(ComplexVar) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&x_d[device][0], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&x_d[device][1], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&y_d[device][0], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&y_d[device][1], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&vx_d[device], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&vy_d[device], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&m_d[device], sizeof(float) * numParticles[device], computeStream[device]));

            gpuErrchk(cudaMallocAsync(&accumulator_d[device][0], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&accumulator_d[device][1], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&accumulator2_d[device][0], sizeof(float)* Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&accumulator2_d[device][1], sizeof(float)* Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemsetAsync(accumulator_d[device][0], 0, sizeof(float)* Constants::N* Constants::N, computeStream[device]));
            gpuErrchk(cudaMemsetAsync(accumulator_d[device][1], 0, sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemsetAsync(accumulator2_d[device][0], 0, sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemsetAsync(accumulator2_d[device][1], 0, sizeof(float)* Constants::N* Constants::N, computeStream[device]));

            gpuErrchk(cudaMallocAsync(&renderOutput_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&renderOutput2_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(x_d[device][0], x.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(y_d[device][0], y.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(x_d[device][1], x.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(y_d[device][1], y.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(vx_d[device], vx.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(vy_d[device], vy.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(m_d[device], m.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            cudaFuncSetAttribute(Kernels::k_calcFftBatched1D<Constants::N>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
            cudaFuncSetAttribute(Kernels::k_calcFftBatched1D<Constants::N>, cudaFuncAttributeMaxDynamicSharedMemorySize, Constants::N * sizeof(ComplexVar));
            Kernels::k_fillCoefficientArray<Constants::N><<<1, 1024, 0, computeStream[device] >>>();
            Kernels::k_fillPairIndexList<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device]>>>();
            Kernels::k_generateFilterLattice<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device]>>>(filter_d[device], accuracy);
            Kernels::k_initSmoothMinMax<<<1, 1, 0, computeStream[device]>>>();
            calcFilterFft2D(device);
        }

        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaStreamSynchronize(computeStream[device]));
        }
    }

    // Moves all particles out of simulation area.
    void clear() {
        for (int i = 0; i < totalNumParticles; i++) {
            x[i] = -10.0f;
            y[i] = -10.0f;
        }
        doubleBufferingCtr = 0;
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaMemcpyAsync(x_d[device][0], x.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(y_d[device][0], y.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(x_d[device][1], x.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(y_d[device][1], y.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
        }
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaStreamSynchronize(computeStream[device]));
        }
        particleCounter = 0;
    }
    // Adds a galaxy with n particles, with center at (centerX, centerY) normalized coordinates.
    void addGalaxy(int n, float normalizedCenterX = 0.5f, float normalizedCenterY = 0.5f, float angularVelocity = 1.0f, float massPerParticle = 1.0f, float normalizedRadius = 0.3f, float centerOfMassVelocityX = 0.0f, float centerOfMassVelocityY = 0.0f, bool blackHole = false) {
        const int maxCount = min(particleCounter + n, totalNumParticles);
        angularVelocity *= Constants::N;
        angularVelocity /= 1024.0f;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        const float radiusSquared = (Constants::N * normalizedRadius) * (Constants::N * normalizedRadius);
        const float totalMass = massPerParticle * (maxCount - particleCounter) + (blackHole ? (maxCount - particleCounter) * massPerParticle / 1000.0f : 0.0f);

        for (int i = particleCounter; i < maxCount; i++) {
            const float r = (0.1f + 0.9f * dist(rng)) * (Constants::N * normalizedRadius);
            const float a = Constants::MATH_PI * 2.0 * dist(rng);
            const float characteristicSpeedScaling = sqrtf(0.0000044f * totalMass * r / radiusSquared);
            // orbit position
            x[i] = r * cos(a) + normalizedCenterX * Constants::N;
            y[i] = r * sin(a) + normalizedCenterY * Constants::N;
            const float vecX = x[i] - normalizedCenterX * Constants::N;
            const float vecY = y[i] - normalizedCenterY * Constants::N;
            // orbit velocity
            vx[i] = (vecY * angularVelocity) * characteristicSpeedScaling  + centerOfMassVelocityX * Constants::N;
            vy[i] = (-vecX * angularVelocity)* characteristicSpeedScaling + centerOfMassVelocityY * Constants::N;
            m[i] = massPerParticle;

            if (i == maxCount - 1) {
                if (blackHole) {
                    x[i] = normalizedCenterX * Constants::N;
                    y[i] = normalizedCenterY * Constants::N;
                    vx[i] = centerOfMassVelocityX * Constants::N;
                    vy[i] = centerOfMassVelocityY * Constants::N;
                    m[i] = (maxCount - particleCounter) * massPerParticle / 1000.0f;
                }
            }
        }
        particleCounter = maxCount;
        doubleBufferingCtr = 0;
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaMemcpyAsync(x_d[device][0], x.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(y_d[device][0], y.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(x_d[device][1], x.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(y_d[device][1], y.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(vx_d[device], vx.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(vy_d[device], vy.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(m_d[device], m.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
        }
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaStreamSynchronize(computeStream[device]));
        }
    }
private:
    int doubleBufferingCtr;
    void scatterMassOnLattice() {
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaStreamSynchronize(computeStream[device]));
            gpuErrchk(cudaStreamSynchronize(broadcastStream[device]));
            Kernels::k_clearAccumulator<Constants::N><<<numBlocks[device], Constants::THREADS, 0, broadcastStream[device] >>>(accumulator_d[device][1 - (doubleBufferingCtr % 2)]);
            Kernels::k_scatterMassOnAccumulator<Constants::N><<<numBlocks[device], Constants::THREADS, 0, broadcastStream[device] >>>(accumulator_d[device][1 - (doubleBufferingCtr % 2)], x_d[device][1 - (doubleBufferingCtr % 2)], y_d[device][1 - (doubleBufferingCtr % 2)], m_d[device], numParticles[device], accuracy);
        }

        // Device - device broadcast start
        if (Constants::NUM_CUDA_DEVICES > 1) {
            for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
                gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));

                gpuErrchk(cudaMemcpyAsync(broadcast_h + (device * Constants::N * Constants::N), accumulator_d[device][1 - (doubleBufferingCtr % 2)], sizeof(float) * Constants::N * Constants::N, cudaMemcpyDeviceToHost, broadcastStream[device]));
                gpuErrchk(cudaEventRecord(latticeBroadcastEvent[device], broadcastStream[device]));
            }

            for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
                gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));

                for (int device2 = 0; device2 < Constants::NUM_CUDA_DEVICES; device2++) {
                    if (device != device2) {
                        // Copy each device's broadcast data, then sum.
                        gpuErrchk(cudaStreamWaitEvent(broadcastStream[device], latticeBroadcastEvent[device2]));
                        gpuErrchk(cudaMemcpyAsync(accumulator2_d[device][1 - (doubleBufferingCtr % 2)], broadcast_h + (device2 * Constants::N * Constants::N), sizeof(float) * Constants::N * Constants::N, cudaMemcpyHostToDevice, broadcastStream[device]));
                        Kernels::k_sumAccumulators<Constants::N> << <numBlocks[device], Constants::THREADS, 0, broadcastStream[device] >> > (accumulator_d[device][1 - (doubleBufferingCtr % 2)], accumulator2_d[device][1 - (doubleBufferingCtr % 2)]);
                    }
                }
            }
        }
        // Device - device broadcast stop

        if (doubleBufferingCtr > 0) {
            for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
                gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
                Kernels::k_copyAccumulatorIntoLattice<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (accumulator_d[device][doubleBufferingCtr % 2], lattice_d[device]);
                if (nbodyCalcCounter == numNbodyStepsPerRender - 1) {
                    Kernels::k_getRealComponentOfLattice<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (lattice_d[device], renderOutput_d[device]);
                    Kernels::k_calcBlurConvolution<Constants::BLUR_R, Constants::N> << <dim3(32, 32, 1), dim3(32, 32, 1), sizeof(float)* (Constants::BLUR_R + Kernels::TILE_SIZE)* (Constants::BLUR_R + Kernels::TILE_SIZE), computeStream[device] >> > (renderOutput_d[device], renderOutput2_d[device]);
                    Kernels::k_resetMinMax << <1, 1, 0, computeStream[device] >> > ();
                    Kernels::k_calcMinMax<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (renderOutput2_d[device]);
                    Kernels::k_smoothMinMax << <1, 1, 0, computeStream[device] >> > ();
                    Kernels::k_scaleWithMinMax<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (renderOutput2_d[device], renderOutput_d[device]);
                }
                // Accuracy mode also adds a short-range force component using normal convolution.
                if (accuracy) {
                    Kernels::k_getRealComponentOfLattice<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (lattice_d[device], localForceLattice_d[device]);
                    Kernels::k_calcLocalMassConvolution<Constants::LOCAL_CONV_WIDTH, Constants::N> << <dim3(32, 32, 1), dim3(32, 32, 1), sizeof(float)* (Constants::LOCAL_CONV_WIDTH + Kernels::TILE_SIZE)* (Constants::LOCAL_CONV_WIDTH + Kernels::TILE_SIZE), computeStream[device] >> > (localForceLattice_d[device], localForceLatticeResult_d[device]);
                }
            }
        }
    }
    void calcLatticeFft2D() {
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            Kernels::k_calcFftBatched1D<Constants::N> << <numBlocks[device], Constants::THREADS, Constants::N * sizeof(ComplexVar), computeStream[device] >> > (lattice_d[device], false);
            Kernels::k_calcTranspose<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (lattice_d[device]);
            Kernels::k_calcTransposeDiagonals<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (lattice_d[device]);
            Kernels::k_calcFftBatched1D<Constants::N> << <numBlocks[device], Constants::THREADS, Constants::N * sizeof(ComplexVar), computeStream[device] >> > (lattice_d[device], false);
            Kernels::k_calcTranspose<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (lattice_d[device]);
            Kernels::k_calcTransposeDiagonals<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (lattice_d[device]);
        }

    }
    void calcFilterFft2D(const int device) {
        Kernels::k_calcFftBatched1D<Constants::N><<<numBlocks[device], Constants::THREADS, Constants::N * sizeof(ComplexVar), computeStream[device] >>>(filter_d[device], false);
        Kernels::k_calcTranspose<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device] >>>(filter_d[device]);
        Kernels::k_calcTransposeDiagonals<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device] >>>(filter_d[device]);
        Kernels::k_calcFftBatched1D<Constants::N><<<numBlocks[device], Constants::THREADS, Constants::N * sizeof(ComplexVar), computeStream[device] >>>(filter_d[device], false);
        Kernels::k_calcTranspose<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device] >>>(filter_d[device]);
        Kernels::k_calcTransposeDiagonals<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device] >>>(filter_d[device]);

    }
    void multiplyLatticeFilterElementwise() {
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            Kernels::k_multElementwiseLatticeFilter<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (lattice_d[device], filter_d[device]);
        }
    }
    void calcLatticeIfft2D() {
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            Kernels::k_calcFftBatched1D<Constants::N><<<numBlocks[device], Constants::THREADS, Constants::N * sizeof(ComplexVar), computeStream[device]>>>(lattice_d[device], true);
            Kernels::k_calcTranspose<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device]>>>(lattice_d[device]);
            Kernels::k_calcTransposeDiagonals<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device]>>>(lattice_d[device]);
            Kernels::k_calcFftBatched1D<Constants::N><<<numBlocks[device], Constants::THREADS, Constants::N * sizeof(ComplexVar), computeStream[device]>>>(lattice_d[device], true);
            Kernels::k_calcTranspose<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device]>>>(lattice_d[device]);
            Kernels::k_calcTransposeDiagonals<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device]>>>(lattice_d[device]);
        }
    }
    void multiSampleForces() {
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            Kernels::k_shiftLattice<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (lattice_d[device], localForceLatticeResult_d[device], latticeShifted_d[device], accuracy);
            Kernels::k_calcGradientLattice<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (latticeShifted_d[device], latticeShiftedForceXY_d[device]);
            Kernels::k_forceMultiSampling<Constants::N><<<numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (latticeShiftedForceXY_d[device], x_d[device][doubleBufferingCtr % 2], y_d[device][doubleBufferingCtr % 2], vx_d[device], vy_d[device], numParticles[device], accuracy);
        }
    }

    void pushFrame(float* frame, bool& workingTmp) {
        std::lock_guard<std::mutex> lg(lock);
        if (pushCtr < popCtr + Constants::MAX_FRAMES_BUFFERED - 1) {
            memcpy(frames[pushCtr % Constants::MAX_FRAMES_BUFFERED].data(), frame, sizeof(float) * (Constants::N * Constants::N + 1));
            pushCtr++;
        }
        workingTmp = working;
    }
    public:
    constexpr int getLatticeSize() {
        return Constants::N;
    }

    void nBodyStartGeneratingFrames() {
        {
            std::lock_guard<std::mutex> lg(lock);
            pushCtr = 0;
            popCtr = 0;
            frames.resize(Constants::MAX_FRAMES_BUFFERED);
            for (int i = 0; i < Constants::MAX_FRAMES_BUFFERED; i++) {
                // +1 for elapsed milliseconds data.
                frames[i].resize(Constants::N * Constants::N + 1);
            }
            working = true;
        }
        
        computeThread = std::thread([&]() {
            bool workingTmp = true;
            doubleBufferingCtr = 0;
            while (workingTmp) {
                if (nbodyCalcCounter == 0) {
                    for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
                        gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
                        gpuErrchk(cudaEventRecord(eventStart[device], computeStream[device]));
                    }
                }
                scatterMassOnLattice();
                if (doubleBufferingCtr > 0) {
                    calcLatticeFft2D();
                    multiplyLatticeFilterElementwise();
                    calcLatticeIfft2D();
                    multiSampleForces();
                }
                doubleBufferingCtr++;
                nbodyCalcCounter++;
                if (nbodyCalcCounter == numNbodyStepsPerRender) {
                    nbodyCalcCounter = 0;
                    float milliseconds = 0.0f;
                    for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
                        gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
                        gpuErrchk(cudaEventRecord(eventStop[device], computeStream[device]));
                        gpuErrchk(cudaStreamSynchronize(computeStream[device]));
                        gpuErrchk(cudaStreamSynchronize(broadcastStream[device]));
                        float ms;
                        gpuErrchk(cudaEventElapsedTime(&ms, eventStart[device], eventStop[device]));
                        // Selecting longest completion time as total work time.
                        if (ms > milliseconds) {
                            milliseconds = ms;
                        }
                    }
                    gpuErrchk(cudaSetDevice(cudaDeviceIndices[0]));
                    gpuErrchk(cudaMemcpyAsync(frame_h, renderOutput_d[0], sizeof(float) * Constants::N * Constants::N, cudaMemcpyDeviceToHost, computeStream[0]));
                    gpuErrchk(cudaStreamSynchronize(computeStream[0]));
                    frame_h[Constants::N * Constants::N] = milliseconds / numNbodyStepsPerRender;
                    pushFrame(frame_h, workingTmp);
                }
            }
        });
    }
    // returns frame pixels + last element as elapsed milliseconds per time-step.
    std::vector<float>& popFrame() {
        static std::vector<float> result(Constants::N * Constants::N + 1);
        {
            std::lock_guard<std::mutex> lg(lock);
            if (popCtr < pushCtr) {
                result.swap(frames[popCtr % Constants::MAX_FRAMES_BUFFERED]);
                popCtr++;
            }
        }
        return result;
    }
    void nBodyStop() {
        {
            std::lock_guard<std::mutex> lg(lock);
            working = false;
        }
        computeThread.join();
    }
    ~Universe() {
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaFreeAsync(lattice_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(accumulator_d[device][0], computeStream[device]));
            gpuErrchk(cudaFreeAsync(accumulator_d[device][1], computeStream[device]));
            gpuErrchk(cudaFreeAsync(accumulator2_d[device][0], computeStream[device]));
            gpuErrchk(cudaFreeAsync(accumulator2_d[device][1], computeStream[device]));
            gpuErrchk(cudaFreeAsync(localForceLattice_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(localForceLatticeResult_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(latticeShifted_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(latticeShiftedForceXY_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(filter_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(x_d[device][0], computeStream[device]));
            gpuErrchk(cudaFreeAsync(y_d[device][0], computeStream[device]));
            gpuErrchk(cudaFreeAsync(x_d[device][1], computeStream[device]));
            gpuErrchk(cudaFreeAsync(y_d[device][1], computeStream[device]));
            gpuErrchk(cudaFreeAsync(vx_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(vy_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(m_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(renderOutput_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(renderOutput2_d[device], computeStream[device]));
            gpuErrchk(cudaEventDestroy(eventStart[device]));
            gpuErrchk(cudaEventDestroy(eventStop[device]));
            gpuErrchk(cudaEventDestroy(latticeBroadcastEvent[device]));
        }
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaStreamSynchronize(computeStream[device]));
            gpuErrchk(cudaStreamSynchronize(broadcastStream[device]));
            gpuErrchk(cudaStreamDestroy(computeStream[device]));
            gpuErrchk(cudaStreamDestroy(broadcastStream[device]));
        }
        gpuErrchk(cudaFreeHost(broadcast_h));
        gpuErrchk(cudaFreeHost(frame_h));
    }
};