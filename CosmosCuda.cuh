/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 [Huseyin Tugrul BUYUKISIK]
 *
 * This file is part of the CosmosSimulationWithCuda project.
 * See LICENSE in the root of the project for details.
 */
#define __CUDACC__
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
#include <iostream>
#include <thread>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#ifndef PARALLEL_FOR
#define PARALLEL_FOR(N,O) \
                        struct LocalClass                                                           \
                        {                                                                           \
                            void operator()(int i) const O                                      \
                        } f;                                                                        \
                        std::thread threads[N];                                                     \
                        for(int loopCounterI=0; loopCounterI<N; loopCounterI++)                     \
                        {                                                                           \
                            threads[loopCounterI]=std::thread(f,loopCounterI);                      \
                        }                                                                           \
                        for(int loopCounterI=0; loopCounterI<N; loopCounterI++)                     \
                        {                                                                           \
                            threads[loopCounterI].join();                                           \
                        }                                                                           \

#endif
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace Constants {
    // CUDA grid and block sizes. (currently tuned for RTX4070)
    constexpr int BLOCKS = 46 * 3;
    constexpr int THREADS = 512;
    // FFT uses these (long-range force calculation)
    // N is width of lattice (N x N) and can be only a power of 2. Higher value increases accuracy. For full accuracy, it needs a truncated filter lattice + closest-neighbor search algorithm (for scientific work, which also requires interpolation between cells of lattice).
    constexpr int N = 2048;
    constexpr double MATH_PI = 3.14159265358979323846;
    using ComplexVar = float2;
    // Local convolution (short-range force calculation)
    constexpr int LOCAL_CONV_WIDTH = 33;
}

namespace Kernels {

    __device__ __host__ __forceinline__ Constants::ComplexVar makeComplexVar(float x, float y) {
        Constants::ComplexVar result;
        result.x = x;
        result.y = y;
        return result;
    }
    __device__ Constants::ComplexVar wCoefficients[Constants::N * 2];
    __device__ __forceinline__ Constants::ComplexVar d_wForNk(const double N, const double k) {
        const double angle = (-2.0 * Constants::MATH_PI * k) / N;
        return makeComplexVar(cos(angle), -sin(angle));
    }
    template<int N>
    __global__ void k_fillCoefficientArray() {
        const int thread = threadIdx.x;
        const int threads = blockDim.x;
        int offset = 0;
        for (int i = 1; i < Constants::N; i *= 2) {
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

    __device__ __forceinline__ void d_calcWarpDft(Constants::ComplexVar& var, const int warpLane, const float inverseMult, Constants::ComplexVar* wCoefficients) {
        // Level 1
        int ofs = 0;
        #pragma unroll 1
        for (int level = 1; level <= 16; level <<= 1)
        {
            const int level2 = level * 2;
            const int idx = (warpLane % level2);
            const bool firstHalf = idx < level;
            const Constants::ComplexVar gather = makeComplexVar(__shfl_xor_sync(0xFFFFFFFF, var.x, level), __shfl_xor_sync(0xFFFFFFFF, var.y, level));
            const Constants::ComplexVar select1 = firstHalf ? var : gather;
            const Constants::ComplexVar select2 = firstHalf ? gather : var;
            Constants::ComplexVar tmp = __ldg(&wCoefficients[ofs + idx]);
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
        while (t < Constants::N) {
            t *= 2;
            ctr++;
        }
        return ctr;
    }
    // This is used for 2D
    template<int N>
    __global__ void k_calcFftBatched1D(Constants::ComplexVar* data, const bool inverse = false) {
        const unsigned int thread = threadIdx.x;
        const unsigned int warpLane = thread & 31;
        const unsigned int block = blockIdx.x;
        constexpr unsigned int gridSteps = (Constants::N + Constants::BLOCKS - 1) / Constants::BLOCKS;
        constexpr unsigned int blockSteps = (Constants::N + Constants::THREADS - 1) / Constants::THREADS;
        const float inverseMult = inverse ? -1.0f : 1.0f;
        const float divider = inverse ? Constants::N : 1.0f;
        Constants::ComplexVar vars[blockSteps];
        extern __shared__ Constants::ComplexVar s_coalescing[];
        for (unsigned int grid = 0; grid < gridSteps; grid++) {
            const unsigned int row = grid * Constants::BLOCKS + block;
            if (row < Constants::N) {
                #pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * Constants::THREADS + thread;
                    const int element = col + row * Constants::N;
                    if (col < Constants::N) {
                        s_coalescing[col] = __ldg(&data[element]);
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * Constants::THREADS + thread;
                    if (col < Constants::N) {
                        vars[blc] = s_coalescing[__brev(col) >> (32 - d_bits<Constants::N>())];
                    }
                    d_calcWarpDft(vars[blc], warpLane, inverseMult, wCoefficients);
                }

                // todo: ping-pong buffer = less syncthreads, bigger input support
                __syncthreads();
                #pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * Constants::THREADS + thread;
                    if (col < Constants::N) {
                        s_coalescing[col] = vars[blc];
                    }
                }
                __syncthreads();
                int wOfs = 62;
                #pragma unroll 1
                for (unsigned int level = 32; level < Constants::N; level <<= 1) {
                    const int level2 = level * 2;
                    #pragma unroll
                    for (int blc = 0; blc < blockSteps; blc++) {
                        const unsigned int col = blc * Constants::THREADS + thread;
                        if (col < Constants::N) {
                            const int idx = (col % level2);
                            const bool firstHalf = idx < level;
                            const Constants::ComplexVar gather = s_coalescing[col ^ level];
                            const Constants::ComplexVar select1 = firstHalf ? vars[blc] : gather;
                            const Constants::ComplexVar select2 = firstHalf ? gather : vars[blc];
                            Constants::ComplexVar tmp = __ldg(&wCoefficients[wOfs + idx]);
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
                        if (col < Constants::N) {
                            s_coalescing[col] = vars[blc];
                        }
                    }
                    __syncthreads();
                }
                #pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * Constants::THREADS + thread;
                    const int element = col + row * Constants::N;
                    if (col < Constants::N) {
                        vars[blc].x /= divider;
                        vars[blc].y /= divider;
                        data[element] = vars[blc];
                    }
                }

            }
        }
    }
    // Computes pair indices to be used in transpose operations without requiring double-precision calculation at expense of memory fetch.
    constexpr int SUB_MATRIX_SIZE = 32;
    constexpr int NUM_SUB_MATRICES_PER_DIMENSION = Constants::N / SUB_MATRIX_SIZE;
    constexpr int PAIR_LIST_SIZE = (NUM_SUB_MATRICES_PER_DIMENSION * (NUM_SUB_MATRICES_PER_DIMENSION - 1)) / 2;
    __device__ int2 pairIndexList_d[PAIR_LIST_SIZE];
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
    __global__ void k_calcTranspose(Constants::ComplexVar* data) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int steps = (PAIR_LIST_SIZE + numBlocks - 1) / numBlocks;
        __shared__ Constants::ComplexVar s_tile[SUB_MATRIX_SIZE][SUB_MATRIX_SIZE + 1];
        __shared__ Constants::ComplexVar s_tile2[SUB_MATRIX_SIZE][SUB_MATRIX_SIZE + 1];
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numBlocks + block;
            if (index < PAIR_LIST_SIZE) {
                const int2 pair = pairIndexList_d[index];
                const int i = pair.x;
                const int j = pair.y;
                if (i < j) {
                    const int tileX = i * SUB_MATRIX_SIZE;
                    const int tileY = j * SUB_MATRIX_SIZE;
                    const int tileOffset = tileX + tileY * Constants::N;
                    const int tileOffset2 = tileY + tileX * Constants::N;
                    constexpr int SUB_ELEMENTS = SUB_MATRIX_SIZE * SUB_MATRIX_SIZE;
                    const int steps2 = (SUB_ELEMENTS + numThreads - 1) / numThreads;
                    #pragma unroll
                    for (int k = 0; k < steps2; k++) {
                        const int element = k * numThreads + thread;
                        if (element < SUB_ELEMENTS) {
                            const int col = element % SUB_MATRIX_SIZE;
                            const int row = element / SUB_MATRIX_SIZE;
                            s_tile[row][col] = data[tileOffset + col + row * Constants::N];
                            s_tile2[row][col] = data[tileOffset2 + col + row * Constants::N];
                        }
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int k = 0; k < steps2; k++) {
                        const int element = k * numThreads + thread;
                        if (element < SUB_ELEMENTS) {
                            const int col = element % SUB_MATRIX_SIZE;
                            const int row = element / SUB_MATRIX_SIZE;
                            data[tileOffset2 + col + row * Constants::N] = s_tile[col][row];
                            data[tileOffset + col + row * Constants::N] = s_tile2[col][row];
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
    template<int N>
    __global__ void k_calcTransposeDiagonals(Constants::ComplexVar* data) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;

        const int steps = (NUM_SUB_MATRICES_PER_DIMENSION + numBlocks - 1) / numBlocks;
        __shared__ Constants::ComplexVar s_tile[SUB_MATRIX_SIZE][SUB_MATRIX_SIZE + 1];
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numBlocks + block;
            if (index < NUM_SUB_MATRICES_PER_DIMENSION) {
                // if i == j, transpose in-place. if i < j, swap
                const int i = index;
                const int j = index;
                const int tileX = i * SUB_MATRIX_SIZE;
                const int tileY = j * SUB_MATRIX_SIZE;
                const int tileOffset = tileX + tileY * Constants::N;
                constexpr int SUB_ELEMENTS = SUB_MATRIX_SIZE * SUB_MATRIX_SIZE;
                const int steps2 = (SUB_ELEMENTS + numThreads - 1) / numThreads;
                #pragma unroll
                for (int k = 0; k < steps2; k++) {
                    const int element = k * numThreads + thread;
                    if (element < SUB_ELEMENTS) {
                        const int col = element % SUB_MATRIX_SIZE;
                        const int row = element / SUB_MATRIX_SIZE;
                        s_tile[row][col] = data[tileOffset + col + row * Constants::N];
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int k = 0; k < steps2; k++) {
                    const int element = k * numThreads + thread;
                    if (element < SUB_ELEMENTS) {
                        const int col = element % SUB_MATRIX_SIZE;
                        const int row = element / SUB_MATRIX_SIZE;
                        data[tileOffset + col + row * Constants::N] = s_tile[col][row];
                    }
                }
                __syncthreads();
            }
        }
    }
    template<int N>
    __global__ void k_generateFilterLattice(Constants::ComplexVar* data) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (Constants::N * Constants::N + numTotalThreads - 1) / numTotalThreads;
        constexpr float cutoff = 2.0f;//(Constants::LOCAL_CONV_WIDTH - 1) / 2;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < Constants::N * Constants::N) {
                const int i = index % Constants::N;
                const int j = index / Constants::N;
                const int cx = Constants::N / 2;
                const int cy = Constants::N / 2;
                const float dx = i - cx;
                const float dy = j - cy;
                const float r = sqrtf(dx * dx + dy * dy);
                float mult = 1.0f;
                
                if (r > cutoff) {
                    data[((i + Constants::N / 2) % Constants::N) + ((j + Constants::N / 2) % Constants::N) * Constants::N].x = mult / r;
                    data[((i + Constants::N / 2) % Constants::N) + ((j + Constants::N / 2) % Constants::N) * Constants::N].y = 0.0f;
                }
                else {
                    data[((i + Constants::N / 2) % Constants::N) + ((j + Constants::N / 2) % Constants::N) * Constants::N].x = 0.0f;
                    data[((i + Constants::N / 2) % Constants::N) + ((j + Constants::N / 2) % Constants::N) * Constants::N].y = 0.0f;
                }
            }
        }
    }
    template<int N>
    __global__ void k_multElementwiseLatticeFilter(Constants::ComplexVar* data, Constants::ComplexVar* data2) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (Constants::N * Constants::N + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < Constants::N * Constants::N) {
                const int i = index % Constants::N;
                const int j = index / Constants::N;
                auto d1 = data[i + j * Constants::N];
                const auto d2 = data2[i + j * Constants::N];
                const float tmpX = d1.x * d2.x - d1.y * d2.y;
                const float tmpY = d1.x * d2.y + d1.y * d2.x;
                d1.x = tmpX;
                d1.y = tmpY;
                data[i + j * Constants::N] = d1;
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
                if (centerX + 1 < Constants::N) {
                    right = __ldca(&lattice_d[index + 1]);
                }
                if (centerY - 1 >= 0) {
                    top = __ldca(&lattice_d[index - Constants::N]);
                }
                if (centerY + 1 < Constants::N) {
                    bot = __ldca(&lattice_d[index + Constants::N]);
                }
                if (centerX - 2 >= 0) {
                    leftLeft = __ldca(&lattice_d[index - 2]);
                }
                if (centerX + 2 < Constants::N) {
                    rightRight = __ldca(&lattice_d[index + 2]);
                }
                if (centerY - 2 >= 0) {
                    topTop = __ldca(&lattice_d[index - 2*Constants::N]);
                }
                if (centerY + 2 < Constants::N) {
                    botBot = __ldca(&lattice_d[index + 2*Constants::N]);
                }
                // Gradient
                const float h = 1024.0f / Constants::N;
                const float forceComponentX = (-rightRight + 8.0f * right - 8.0f * left + leftLeft) / (h * 12.0f);
                const float forceComponentY = (-botBot + 8.0f * bot - 8.0f * top + topTop) / (h * 12.0f);
                
                latticeForceXY_d[index] = make_float2(forceComponentX, forceComponentY);
            }
        }
    }

    template<int N>
    __global__ void k_forceMultiSampling(const float2* const __restrict__ latticeForceXY_d, float* const __restrict__ x, float* const __restrict__ y, float* const __restrict__ vx, float* const __restrict__ vy, float* const __restrict__ m, const int numParticles, const bool accuracy) {
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
                const float4 massData = __ldcs(reinterpret_cast<float4*>(&m[index * 4]));
                float posX[4] = { posXr.x,posXr.y,posXr.z,posXr.w };
                float posY[4] = { posYr.x,posYr.y,posYr.z,posYr.w };
                float vxr[4] = { posVXr.x, posVXr.y, posVXr.z, posVXr.w };
                float vyr[4] = { posVYr.x, posVYr.y, posVYr.z, posVYr.w };
                float mass[4] = { massData.x, massData.y, massData.z, massData.w };
                #pragma unroll 4
                for (int m = 0; m < 4; m++) {
                    const int centerX = int(posX[m]);
                    const int centerY = int(posY[m]);
                    const float inverseMass = 1.0f / mass[m];
                    const int centerIndex = centerX + centerY * Constants::N;
                    if (centerX >= 1 && centerX < Constants::N - 1 && centerY >= 1 && centerY < Constants::N - 1) {
                        // Getting precalculated gradient. This should benefit from caching when many particles access same point.
                        // Then calculating interpolation for a more accurate behavior.
                        if (accuracy) {
                            const float2 forceComponentsCurrent = __ldca(&latticeForceXY_d[centerIndex]);
                            const float2 forceComponentsRight = __ldca(&latticeForceXY_d[centerIndex + 1]);
                            const float2 forceComponentsBottom = __ldca(&latticeForceXY_d[centerIndex + Constants::N]);
                            const float2 forceComponentsBottomRight = __ldca(&latticeForceXY_d[centerIndex + 1 + Constants::N]);
                            const float fractionalX = posX[m] - centerX;
                            const float fractionalY = posY[m] - centerY;
                            const float xDiff1 = 1.0f - fractionalX;
                            const float yDiff1 = 1.0f - fractionalY;
                            const float xComponent = forceComponentsCurrent.x * xDiff1 * yDiff1 +
                                forceComponentsRight.x * fractionalX * yDiff1 +
                                forceComponentsBottom.x * xDiff1 * fractionalY +
                                forceComponentsBottomRight.x * fractionalX * fractionalY;
                            const float yComponent = forceComponentsCurrent.y * xDiff1 * yDiff1 +
                                forceComponentsRight.y * fractionalX * yDiff1 +
                                forceComponentsBottom.y * xDiff1 * fractionalY +
                                forceComponentsBottomRight.y * fractionalX * fractionalY;
                            constexpr float dt = 0.01f;
                            posX[m] = fmaf(vxr[m], dt, posX[m]);
                            posY[m] = fmaf(vyr[m], dt, posY[m]);
                            vxr[m] = fmaf(xComponent * inverseMass, dt, vxr[m]);
                            vyr[m] = fmaf(yComponent * inverseMass, dt, vyr[m]);
                        }
                        else {
                            const float2 forceComponentsCurrent = __ldca(&latticeForceXY_d[centerIndex]);
                            constexpr float dt = 0.01f;
                            posX[m] = fmaf(vxr[m], dt, posX[m]);
                            posY[m] = fmaf(vyr[m], dt, posY[m]);
                            vxr[m] = fmaf(forceComponentsCurrent.x * inverseMass, dt, vxr[m]);
                            vyr[m] = fmaf(forceComponentsCurrent.y * inverseMass, dt, vyr[m]);
                        }
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
    __global__ void k_clearLattice(Constants::ComplexVar* lattice_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (Constants::N * Constants::N + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < Constants::N * Constants::N) {
                lattice_d[index] = Constants::ComplexVar{ 0.0f, 0.0f };
            }
        }
    }
    template<int N>
    __global__ void k_clearLatticeForRender(float* lattice_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (Constants::N * Constants::N + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < Constants::N * Constants::N) {
                lattice_d[index] = 0.0f;
            }
        }
    }
    // Optionally shifts and adds local values to global.
    template<int N>
    __global__ void k_shiftLattice(Constants::ComplexVar* lattice_d, float* latticeLocal_d, float* latticeShifted_d, bool accuracy) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (Constants::N * Constants::N + numTotalThreads - 1) / numTotalThreads;
        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < Constants::N * Constants::N) {
                const int x = index % Constants::N;
                const int y = index / Constants::N;
                const int shiftedX = (x - Constants::N / 2 + Constants::N * 2) % Constants::N;
                const int shiftedY = (y - Constants::N / 2 + Constants::N * 2) % Constants::N;
                //latticeShifted_d[index] = lattice_d[shiftedX + shiftedY * Constants::N].x;
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
    __global__ void k_scatterMassOnLattice(Constants::ComplexVar* lattice_d, const float* x, const float* y, const float* m, const int numParticles, bool accuracy) {
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
                if (xi >= 1 && xi < Constants::N - 1 && yi >= 1 && yi < Constants::N - 1) {
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
                        atomicAdd(&lattice_d[xi + yi * Constants::N].x, weightCurrent * mass);
                        atomicAdd(&lattice_d[1 + xi + yi * Constants::N].x, weightRight * mass);
                        atomicAdd(&lattice_d[xi + (yi + 1) * Constants::N].x, weightBottom * mass);
                        atomicAdd(&lattice_d[1 + xi + (yi + 1) * Constants::N].x, weightBottomRight * mass);
                    } else {
                        atomicAdd(&lattice_d[xi + yi * Constants::N].x, mass);
                    }
                }
            }
        }
    }

    template<int N>
    __global__ void k_scatterMassOnLatticeForRender(float* lattice_d, const float* x, const float* y, const int numParticles, const float* renderColor_d, bool colorOnly = false) {
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
                const int xi = x[index];
                const int yi = y[index];
                if (xi >= 0 && xi < Constants::N && yi >= 0 && yi < Constants::N) {
                    atomicAdd(&lattice_d[xi + yi * Constants::N], renderColor_d[index]);
                }
            }
        }
    }

    template<int N>
    __global__ void k_cloneLatticeForShortRangeCalc(Constants::ComplexVar* lattice_d, float* localForceLattice_d) {
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

    // This is to capture short-range details missed by FFT. (todo: use smooth transition from 0 to 16 when combining)
    // Todo: optimize with smem
    __constant__ float shortRangeGravKern_c[Constants::LOCAL_CONV_WIDTH * Constants::LOCAL_CONV_WIDTH];
    template<int N>
    __global__ void k_calcLocalMassConvolution(float* localForceLattice_d, float* localForceLatticeResult_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
        constexpr int HALF_WIDTH = (Constants::LOCAL_CONV_WIDTH - 1) / 2;
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < N * N) {
                const int indexX = index % N;
                const int indexY = index / N;
                
                float accumulator = 0.0f;
                #pragma unroll
                for (int iy = -HALF_WIDTH; iy <= HALF_WIDTH; iy++) {
                    #pragma unroll
                    for (int ix = -HALF_WIDTH; ix <= HALF_WIDTH; ix++) {
                        const int neighborX = ix + indexX;
                        const int neighborY = iy + indexY;
                        const int neighbor = neighborX + neighborY * N;
                        if (neighborX >= 0 && neighborX < Constants::N && neighborY >= 0 && neighborY < Constants::N) {
                            const float neighborData = localForceLattice_d[neighbor];
                            const float weight = shortRangeGravKern_c[ix + HALF_WIDTH + (iy + HALF_WIDTH) * Constants::LOCAL_CONV_WIDTH];
                            // Todo: add more accumulators to reduce rounding error.
                            accumulator = fmaf(neighborData, weight, accumulator);
                        }
                    }
                }
                localForceLatticeResult_d[index] = accumulator;
                
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
    std::vector<float> lattice;
    std::vector<float> renderColor;
    int particleCounter;

    // For OpenCV
    cv::Mat mat;
    int numParticles;

    // For device
    cudaEvent_t eventStart;
    cudaEvent_t eventStop;
    Constants::ComplexVar* lattice_d;
    float* latticeShifted_d;
    float2* latticeShiftedForceXY_d;
    Constants::ComplexVar* filter_d;
    float* x_d;
    float* y_d;
    float* vx_d;
    float* vy_d;
    float* m_d;
    float* renderColor_d;
    float* localForceLattice_d;
    float* localForceLatticeResult_d;
    // Accuracy setting: increases accuracy of mass projections and force sampling at cost of 50% performance
    bool accuracy;
    // Window stats
    int ww;
    int wh;
public:
    Universe(int particles, int cudaDevice, bool lowAccuracy, int windowWidthPixels, int windowHeightPixels) {
        ww = windowWidthPixels;
        wh = windowHeightPixels;
        accuracy = !lowAccuracy;
        particleCounter = 0;
        cudaSetDevice(cudaDevice);
        srand(time(0));
        numParticles = particles;
        x.resize(particles);
        vx.resize(particles);
        y.resize(particles);
        vy.resize(particles);
        m.resize(particles);
        renderColor.resize(particles);
        lattice.resize(Constants::N * Constants::N);
        mat = cv::Mat(cv::Size2i(Constants::N, Constants::N), CV_32FC1);
        float nSqrt = sqrtf(numParticles);
        int nSqrtI = nSqrt;
        for (int i = 0; i < particles; i++) {
            x[i] = (rand() % Constants::N);
            y[i] = (rand() % Constants::N);
            vx[i] = 0;
            vy[i] = 0;
            // random color
            renderColor[i] = 0.2f + ((rand() % 800) / 1000.0f);
            m[i] = 1.0f;
        }
        constexpr int HALF_WIDTH = (Constants::LOCAL_CONV_WIDTH - 1) / 2;
        std::vector<float> localForceFilter(Constants::LOCAL_CONV_WIDTH * Constants::LOCAL_CONV_WIDTH);
        for (int iy = -HALF_WIDTH; iy <= HALF_WIDTH; iy++) {
            for (int ix = -HALF_WIDTH; ix <= HALF_WIDTH; ix++) {
                const int index = ix + HALF_WIDTH + (iy + HALF_WIDTH) * Constants::LOCAL_CONV_WIDTH;
                const double r = sqrt((double)(ix * ix + iy * iy));
                if (r > 0.0 && r < HALF_WIDTH) {
                    localForceFilter[index] = 1.0f / r;
                }
                else {
                    localForceFilter[index] = 0.0f;
                }
            }
        }
        gpuErrchk(cudaMemcpyToSymbol(Kernels::shortRangeGravKern_c, localForceFilter.data(), sizeof(float) * Constants::LOCAL_CONV_WIDTH * Constants::LOCAL_CONV_WIDTH, 0, cudaMemcpyHostToDevice));
        particleCounter = numParticles;
        gpuErrchk(cudaEventCreate(&eventStart));
        gpuErrchk(cudaEventCreate(&eventStop));
        gpuErrchk(cudaMalloc(&lattice_d, sizeof(Constants::ComplexVar) * Constants::N * Constants::N));
        gpuErrchk(cudaMalloc(&localForceLattice_d, sizeof(float) * Constants::N * Constants::N));
        gpuErrchk(cudaMalloc(&localForceLatticeResult_d, sizeof(float) * Constants::N * Constants::N));
        gpuErrchk(cudaMalloc(&latticeShifted_d, sizeof(float) * Constants::N * Constants::N));
        gpuErrchk(cudaMalloc(&latticeShiftedForceXY_d, sizeof(float2) * Constants::N * Constants::N));
        gpuErrchk(cudaMalloc(&filter_d, sizeof(Constants::ComplexVar) * Constants::N * Constants::N));
        gpuErrchk(cudaMalloc(&x_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMalloc(&y_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMalloc(&vx_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMalloc(&vy_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMalloc(&m_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMalloc(&renderColor_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMemcpy(x_d, x.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(y_d, y.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(vx_d, vx.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(vy_d, vy.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(m_d, m.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(renderColor_d, renderColor.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        cudaFuncSetAttribute(Kernels::k_calcFftBatched1D<Constants::N>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(Kernels::k_calcFftBatched1D<Constants::N>, cudaFuncAttributeMaxDynamicSharedMemorySize, Constants::N * sizeof(Constants::ComplexVar));
        Kernels::k_fillCoefficientArray<Constants::N> << <1, 1024 >> > ();
        Kernels::k_fillPairIndexList<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > ();
        Kernels::k_generateFilterLattice<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (filter_d);
        calcFilterFft2D();
        gpuErrchk(cudaDeviceSynchronize());
        cv::namedWindow("Fast Nbody");
    }
    // Requires same number of particles as the simulator.
    void updateParticleData(std::vector<float> sourceX, std::vector<float> sourceY, std::vector<float> sourceVX, std::vector<float> sourceVY, std::vector<float> sourceMass) {
        x.swap(sourceX);
        y.swap(sourceY);
        vx.swap(sourceVX);
        vy.swap(sourceVY);
        m.swap(sourceMass);
        gpuErrchk(cudaMemcpy(x_d, x.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(y_d, y.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(vx_d, vx.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(vy_d, vy.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(m_d, m.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
    }
    // Moves all particles out of simulation area.
    void clear() {
        for (int i = 0; i < numParticles; i++) {
            x[i] = -10.0f;
            y[i] = -10.0f;
        }
        gpuErrchk(cudaMemcpy(x_d, x.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(y_d, y.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        particleCounter = 0;
    }
    // Adds a galaxy with n particles, with center at (centerX, centerY) normalized coordinates.
    void addGalaxy(int n, float normalizedCenterX = 0.5f, float normalizedCenterY = 0.5f, float angularVelocity = 1.0f, float massPerParticle = 1.0f, float normalizedRadius = 0.3f, float centerOfMassVelocityX = 0.0f, float centerOfMassVelocityY = 0.0f) {
        const int maxCount = min(particleCounter + n, numParticles);
        for (int i = particleCounter; i < maxCount; i++) {
            const float r = rand() % (int)(Constants::N * normalizedRadius);
            const float a = Constants::MATH_PI * 2.0 * (rand() % 1000) / 1000.0f;
            // orbit position
            x[i] = r * cos(a) + normalizedCenterX * Constants::N;
            y[i] = r * sin(a) + normalizedCenterY * Constants::N;
            const float vecX = x[i] - normalizedCenterX * Constants::N;
            const float vecY = y[i] - normalizedCenterY * Constants::N;
            // orbit velocity
            vx[i] = vecY * angularVelocity + centerOfMassVelocityX * Constants::N;
            vy[i] = -vecX * angularVelocity + centerOfMassVelocityY * Constants::N;
            // random color
            renderColor[i] = 0.2f + ((rand() % 800) / 1000.0f);
            m[i] = massPerParticle;
        }
        particleCounter = maxCount;
        gpuErrchk(cudaMemcpy(x_d, x.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(y_d, y.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(vx_d, vx.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(vy_d, vy.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(m_d, m.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
    }
    void startBenchmark() {
        gpuErrchk(cudaEventRecord(eventStart));
    }
private:
    // todo: scatter on 9 cells per mass to improve accuracy more.
    void scatterMassOnLattice(bool renderOnly = false) {
        if (renderOnly) {
            Kernels::k_clearLatticeForRender<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (latticeShifted_d);
            Kernels::k_scatterMassOnLatticeForRender<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (latticeShifted_d, x_d, y_d, numParticles, renderColor_d, renderOnly);
        }
        else {
            Kernels::k_clearLattice<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d);
            Kernels::k_scatterMassOnLattice<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d, x_d, y_d, m_d, numParticles, accuracy);
            // Accuracy mode also adds a short-range force component using normal convolution.
            if (accuracy) {
                Kernels::k_cloneLatticeForShortRangeCalc<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d, localForceLattice_d);
                Kernels::k_calcLocalMassConvolution<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (localForceLattice_d, localForceLatticeResult_d);
            }
        }


    }
    void calcLatticeFft2D() {
        Kernels::k_calcFftBatched1D<Constants::N> << <Constants::BLOCKS, Constants::THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (lattice_d, false);
        Kernels::k_calcTranspose<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d);
        Kernels::k_calcFftBatched1D<Constants::N> << <Constants::BLOCKS, Constants::THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (lattice_d, false);
        Kernels::k_calcTranspose<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d);

    }
    void calcFilterFft2D() {
        Kernels::k_calcFftBatched1D<Constants::N> << <Constants::BLOCKS, Constants::THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (filter_d, false);
        Kernels::k_calcTranspose<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (filter_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (filter_d);
        Kernels::k_calcFftBatched1D<Constants::N> << <Constants::BLOCKS, Constants::THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (filter_d, false);
        Kernels::k_calcTranspose<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (filter_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (filter_d);

    }
    void multiplyLatticeFilterElementwise() {
        Kernels::k_multElementwiseLatticeFilter<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d, filter_d);
    }
    void calcLatticeIfft2D() {
        Kernels::k_calcFftBatched1D<Constants::N> << <Constants::BLOCKS, Constants::THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (lattice_d, true);
        Kernels::k_calcTranspose<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d);
        Kernels::k_calcFftBatched1D<Constants::N> << <Constants::BLOCKS, Constants::THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (lattice_d, true);
        Kernels::k_calcTranspose<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d);
    }
    void multiSampleForces() {
        Kernels::k_shiftLattice<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (lattice_d, localForceLatticeResult_d, latticeShifted_d, accuracy);
        Kernels::k_calcGradientLattice<Constants::N> <<<Constants::BLOCKS, Constants::THREADS >>> (latticeShifted_d, latticeShiftedForceXY_d);
        Kernels::k_forceMultiSampling<Constants::N> << <Constants::BLOCKS, Constants::THREADS >> > (latticeShiftedForceXY_d, x_d, y_d, vx_d, vy_d, m_d, numParticles, accuracy);
    }
    public:
    void nBody() {
        scatterMassOnLattice();
        calcLatticeFft2D();
        multiplyLatticeFilterElementwise();
        calcLatticeIfft2D();
        multiSampleForces();
    }
    void stopBenchmark() {
        gpuErrchk(cudaEventRecord(eventStop));
    }
    void sync(int numStepsComputed) {
        gpuErrchk(cudaDeviceSynchronize());
        float milliseconds;
        gpuErrchk(cudaEventElapsedTime(&milliseconds, eventStart, eventStop));
        std::cout << "CUDA Nbody pipeline: " << milliseconds / numStepsComputed << " milliseconds per step" << std::endl;
    }

    void render() {
        scatterMassOnLattice(true);
        gpuErrchk(cudaDeviceSynchronize());
        mat.setTo(cv::Scalar(0.0f));

        gpuErrchk(cudaMemcpy(lattice.data(), latticeShifted_d, sizeof(float) * Constants::N * Constants::N, cudaMemcpyDeviceToHost));
        std::vector<std::thread> renderThread;
        const int nThr = 4;
        for (int i = 0; i < nThr; i++) {
            const int iClone = i;
            renderThread.emplace_back([&, iClone]() {
                const int chunkSize = (Constants::N * Constants::N) / nThr;
                for (int j = iClone * chunkSize; j < iClone * chunkSize + chunkSize; j++) {
                    const auto data = lattice[j];
                    mat.at<float>(j) = (data > 1.0f ? 1.0f : data);
                }
            });
        }
        for (int i = 0; i < nThr; i++) {
            renderThread[i].join();
        }
        cv::Mat resized;
        cv::resize(mat, resized, cv::Size(ww, wh), 0, 0, cv::INTER_LANCZOS4);
        cv::imshow("Fast Nbody", resized);
    }
    ~Universe() {
        cv::destroyAllWindows();
        gpuErrchk(cudaFree(lattice_d));
        gpuErrchk(cudaFree(localForceLattice_d));
        gpuErrchk(cudaFree(localForceLatticeResult_d));
        gpuErrchk(cudaFree(latticeShifted_d));
        gpuErrchk(cudaFree(latticeShiftedForceXY_d));
        gpuErrchk(cudaFree(filter_d));
        gpuErrchk(cudaFree(x_d));
        gpuErrchk(cudaFree(y_d));
        gpuErrchk(cudaFree(vx_d));
        gpuErrchk(cudaFree(vy_d));
        gpuErrchk(cudaFree(m_d));
        gpuErrchk(cudaFree(renderColor_d));
        gpuErrchk(cudaEventDestroy(eventStart));
        gpuErrchk(cudaEventDestroy(eventStop));
    }
};