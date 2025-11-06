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
#pragma comment(lib, "cufft.lib")
#include <cufft.h>
#include <cufftXt.h>
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
#ifndef __OVERRIDE_CONSTANTS__
namespace Constants {
    // Number of threads per CUDA block. This is for 1536 resident threads per SM. For older GPUs, 512 or 1024 can be chosen.
    constexpr int THREADS = 768;
    // Number of CUDA devices (max 2 tested)
    constexpr int NUM_CUDA_DEVICES = 1;

    // FFT uses these (long-range force calculation)
    // N is width of lattice (N x N) and can be only a power of 2. Higher value increases accuracy at the cost of performance.
    constexpr int N = 2048;
    constexpr double MATH_PI = 3.14159265358979323846;


    // Time-step of simulation. Lower values increase accuracy.
    constexpr float dt = 50.0f;
    // Force-multiplier for particles.
    constexpr float gravityMultiplier = 1.0f;

    // For render buffer output. Asynchronously filled.
    constexpr int MAX_FRAMES_BUFFERED = 40;
    constexpr int BLUR_R = 3;
    constexpr int BLUR_HALF_R = (BLUR_R - 1) / 2;
}
#else
namespace Constants {
    // Number of threads per CUDA block. This is for 1536 resident threads per SM. For older GPUs, 512 or 1024 can be chosen.
    constexpr int THREADS = OVERRIDE_CONSTANTS::THREADS;
    // Number of CUDA devices (max 2 tested)
    constexpr int NUM_CUDA_DEVICES = OVERRIDE_CONSTANTS::NUM_CUDA_DEVICES;

    // FFT uses these (long-range force calculation)
    // N is width of lattice (N x N) and can be only a power of 2. Higher value increases accuracy at the cost of performance.
    constexpr int N = OVERRIDE_CONSTANTS::N;
    constexpr double MATH_PI = 3.14159265358979323846;

    // Time-step of simulation. Lower values increase accuracy.
    constexpr float dt = OVERRIDE_CONSTANTS::dt;
    // Force-multiplier for particles.
    constexpr float gravityMultiplier = OVERRIDE_CONSTANTS::gravityMultiplier;

    // For render buffer output. Asynchronously filled.
    constexpr int MAX_FRAMES_BUFFERED = OVERRIDE_CONSTANTS::MAX_FRAMES_BUFFERED;
    constexpr int BLUR_R = OVERRIDE_CONSTANTS::BLUR_R;
    constexpr int BLUR_HALF_R = (BLUR_R - 1) / 2;
}
#endif
namespace Kernels {
    __constant__ float renderBlurKern_c[Constants::BLUR_R * Constants::BLUR_R];
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

    template<int N>
    __global__ void k_generateFilterLattice(ComplexVar* data, bool accuracy) {
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
                const int cx = i <= N / 2 ? 0 : N;
                const int cy = j <= N / 2 ? 0 : N;
                const float dx = i  - cx;
                const float dy = j  - cy;
                const float r = sqrtf(dx * dx + dy * dy);
                if (r > 0.5f) {
                    data[index].x = 1.0f / r;
                    data[index].y = 0.0f;
                }
                else {
                    data[index].x = 3.0f;
                    data[index].y = 0.0f;
                }
            }
        }
    }
    template<int N>
    __global__ void k_divElementwiseLatticeFilter(ComplexVar* data, ComplexVar* data2) {
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

                float denom = d2.x * d2.x + d2.y * d2.y + 0.001f;
                const float tmpX = (d1.x * d2.x + d1.y * d2.y) / denom;
                const float tmpY = (d1.y * d2.x - d1.x * d2.y) / denom;

                d1.x = tmpX;
                d1.y = tmpY;
                data[i + j * N] = d1;
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
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;
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
                    topTop = __ldca(&lattice_d[index - 2 * N]);
                }
                if (centerY + 2 < N) {
                    botBot = __ldca(&lattice_d[index + 2 * N]);
                }
                // Gradient
                const float h = 1.0f;
                const float forceComponentX = (-rightRight + 8.0f * right - 8.0f * left + leftLeft) / (h * 12.0f);
                const float forceComponentY = (-botBot + 8.0f * bot - 8.0f * top + topTop) / (h * 12.0f);

                latticeForceXY_d[index] = make_float2(forceComponentX * Constants::gravityMultiplier, forceComponentY * Constants::gravityMultiplier);
            }
        }
    }

    template<int N>
    __global__ void k_forceMultiSampling(const float2* const __restrict__ latticeForceXY_d, float* const __restrict__ x, float* const __restrict__ y, float* const __restrict__ xOld, float* const __restrict__ yOld, const int numParticles, const bool accuracy) {
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
                const float4 vecOldX = __ldcs(reinterpret_cast<float4*>(&xOld[index * 4]));
                const float4 vecOldY = __ldcs(reinterpret_cast<float4*>(&yOld[index * 4]));
                float posX[4] = { posXr.x,posXr.y,posXr.z,posXr.w };
                float posY[4] = { posYr.x,posYr.y,posYr.z,posYr.w };
                float oldDataX[4] = { vecOldX.x, vecOldX.y, vecOldX.z, vecOldX.w };
                float oldDataY[4] = { vecOldY.x, vecOldY.y, vecOldY.z, vecOldY.w };
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

                        float newX = fmaf(2.0f, posX[m], -oldDataX[m]) + Constants::dt * xComponent * Constants::dt;
                        float newY = fmaf(2.0f, posY[m], -oldDataY[m]) + Constants::dt * yComponent * Constants::dt;
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
                        oldDataX[m] = posX[m];
                        oldDataY[m] = posY[m];
                        posX[m] = newX;
                        posY[m] = newY;
                    }
                }
                __stcs(reinterpret_cast<float4*>(&x[index * 4]), make_float4(posX[0], posX[1], posX[2], posX[3]));
                __stcs(reinterpret_cast<float4*>(&y[index * 4]), make_float4(posY[0], posY[1], posY[2], posY[3]));
                __stcs(reinterpret_cast<float4*>(&xOld[index * 4]), make_float4(oldDataX[0], oldDataX[1], oldDataX[2], oldDataX[3]));
                __stcs(reinterpret_cast<float4*>(&yOld[index * 4]), make_float4(oldDataY[0], oldDataY[1], oldDataY[2], oldDataY[3]));
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
    __global__ void k_shiftLattice(ComplexVar* lattice_d, float* latticeShifted_d, bool accuracy) {
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
            const int x = index % N;
            const int y = index / N;
            const int sX = (x + (N / 2)) % N;
            const int sY = (y + (N / 2)) % N;
            const int s = sX + sY * N;
            if (index < N * N) {
                latticeShifted_d[s] = lattice_d[index].x / (Constants::N * Constants::N);
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
    __global__ void k_generateMassAssignmentKernel(ComplexVar* data, bool accuracy) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = gridDim.x * numThreads;
        const int steps = (N * N + numTotalThreads - 1) / numTotalThreads;

        #pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int index = globalThread + ii * numTotalThreads;
            if (index < N * N) {
                const int i = index % N;
                const int j = index / N;

                const int cx = N / 2;
                const int cy = N / 2;
                const int dx = i - cx;
                const int dy = j - cy;

                if (!accuracy) {

                    data[index].x = (dx == 0 && dy == 0) ? 1.0f : 0.0f;
                    data[index].y = 0.0f;
                }
                else {
                    float fx = 0.5f;
                    float fy = 0.5f;
                    float w = 0.0f;
                    if ((dx == 0 || dx == 1) && (dy == 0 || dy == 1)) {
                        float wx = (dx == 0) ? (1.0f - fx) : fx;
                        float wy = (dy == 0) ? (1.0f - fy) : fy;
                        w = wx * wy;
                    }

                    data[index].x = w;
                    data[index].y = 0.0f;
                }
            }
        }
    }
    template<int N>
    __global__ void k_applyPhaseShift(ComplexVar* data, float fx, float fy) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numThreads = blockDim.x;
        const int numBlocks = gridDim.x;
        const int globalThread = thread + block * numThreads;
        const int totalThreads = numThreads * numBlocks;
        const int steps = (N * N + totalThreads - 1) / totalThreads;

#pragma unroll
        for (int ii = 0; ii < steps; ii++) {
            const int idx = ii * totalThreads + globalThread;
            if (idx < N * N) {
                const int i = idx % N;
                const int j = idx / N;
                const int kx = (i <= N / 2) ? i : i - N;
                const int ky = (j <= N / 2) ? j : j - N;

                const float phase = -2.0f * Constants::MATH_PI * (kx * fx / N + ky * fy / N);

                const ComplexVar c = data[idx];
                const float cosA = cosf(phase);
                const float sinA = sinf(phase);

                data[idx].x = c.x * cosA - c.y * sinA;
                data[idx].y = c.x * sinA + c.y * cosA;
            }
        }
    }
    template<int N>
    __global__ void k_scatterMassOnAccumulator(float* const __restrict__ accumulator_d, const float4* const __restrict__ x, const float4* const __restrict__ y, const float4* const __restrict__ m, const int numParticles, bool accuracy) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int numChunks = numParticles / 4;
        const int steps = (numChunks + numTotalThreads - 1) / numTotalThreads;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
        __shared__ alignas(16) float4 s_xAsync[Constants::THREADS];
        __shared__ alignas(16) float4 s_yAsync[Constants::THREADS];
        __shared__ alignas(16) float4 s_mAsync[Constants::THREADS];
        if (globalThread < numChunks) {
            __pipeline_memcpy_async(&s_xAsync[thread], &x[globalThread], sizeof(float4));
            __pipeline_memcpy_async(&s_yAsync[thread], &y[globalThread], sizeof(float4));
            __pipeline_memcpy_async(&s_mAsync[thread], &m[globalThread], sizeof(float4));
            __pipeline_commit();
        }
#endif

        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < numChunks) {
                float4 xf;
                float4 yf;
                float4 mass;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                __pipeline_wait_prior(0);
                xf = s_xAsync[thread];
                yf = s_yAsync[thread];
                mass = s_mAsync[thread];
                const int nextItem = index + numTotalThreads;
                if (nextItem < numChunks) {
                    __pipeline_memcpy_async(&s_xAsync[thread], &x[nextItem], sizeof(float4));
                    __pipeline_memcpy_async(&s_yAsync[thread], &y[nextItem], sizeof(float4));
                    __pipeline_memcpy_async(&s_mAsync[thread], &m[nextItem], sizeof(float4));
                    __pipeline_commit();
                }
#else
                xf = __ldcs(&x[index]);
                yf = __ldcs(&y[index]);
                mass = __ldcs(&m[index]);
#endif

                // particle 1
                {
                    const int xi = xf.x;
                    const int yi = yf.x;
                    if (xi >= 1 && xi < N - 1 && yi >= 1 && yi < N - 1) {
                        const float fractionalX = xf.x - xi;
                        const float fractionalY = yf.x - yi;
                        const float xDiff1 = 1.0f - fractionalX;
                        const float yDiff1 = 1.0f - fractionalY;
                        const float weightCurrent = xDiff1 * yDiff1;
                        const float weightRight = fractionalX * yDiff1;
                        const float weightBottom = xDiff1 * fractionalY;
                        const float weightBottomRight = fractionalX * fractionalY;
                        // Optional weighted scattering for more accuracy.
                        if (accuracy) {
                            atomicAdd(&accumulator_d[xi + yi * N], weightCurrent * mass.x);
                            atomicAdd(&accumulator_d[1 + xi + yi * N], weightRight * mass.x);
                            atomicAdd(&accumulator_d[xi + (yi + 1) * N], weightBottom * mass.x);
                            atomicAdd(&accumulator_d[1 + xi + (yi + 1) * N], weightBottomRight * mass.x);
                        }
                        else {
                            atomicAdd(&accumulator_d[xi + yi * N], mass.x);
                        }
                    }
                }
                // particle 2
                {
                    const int xi = xf.y;
                    const int yi = yf.y;
                    if (xi >= 1 && xi < N - 1 && yi >= 1 && yi < N - 1) {
                        const float fractionalX = xf.y - xi;
                        const float fractionalY = yf.y - yi;
                        const float xDiff1 = 1.0f - fractionalX;
                        const float yDiff1 = 1.0f - fractionalY;
                        const float weightCurrent = xDiff1 * yDiff1;
                        const float weightRight = fractionalX * yDiff1;
                        const float weightBottom = xDiff1 * fractionalY;
                        const float weightBottomRight = fractionalX * fractionalY;
                        // Optional weighted scattering for more accuracy.
                        if (accuracy) {
                            atomicAdd(&accumulator_d[xi + yi * N], weightCurrent * mass.y);
                            atomicAdd(&accumulator_d[1 + xi + yi * N], weightRight * mass.y);
                            atomicAdd(&accumulator_d[xi + (yi + 1) * N], weightBottom * mass.y);
                            atomicAdd(&accumulator_d[1 + xi + (yi + 1) * N], weightBottomRight * mass.y);
                        }
                        else {
                            atomicAdd(&accumulator_d[xi + yi * N], mass.y);
                        }
                    }
                }
                // particle 3
                {
                    const int xi = xf.z;
                    const int yi = yf.z;
                    if (xi >= 1 && xi < N - 1 && yi >= 1 && yi < N - 1) {
                        const float fractionalX = xf.z - xi;
                        const float fractionalY = yf.z - yi;
                        const float xDiff1 = 1.0f - fractionalX;
                        const float yDiff1 = 1.0f - fractionalY;
                        const float weightCurrent = xDiff1 * yDiff1;
                        const float weightRight = fractionalX * yDiff1;
                        const float weightBottom = xDiff1 * fractionalY;
                        const float weightBottomRight = fractionalX * fractionalY;
                        // Optional weighted scattering for more accuracy.
                        if (accuracy) {
                            atomicAdd(&accumulator_d[xi + yi * N], weightCurrent * mass.z);
                            atomicAdd(&accumulator_d[1 + xi + yi * N], weightRight * mass.z);
                            atomicAdd(&accumulator_d[xi + (yi + 1) * N], weightBottom * mass.z);
                            atomicAdd(&accumulator_d[1 + xi + (yi + 1) * N], weightBottomRight * mass.z);
                        }
                        else {
                            atomicAdd(&accumulator_d[xi + yi * N], mass.z);
                        }
                    }
                }
                // particle 4
                {
                    const int xi = xf.w;
                    const int yi = yf.w;
                    if (xi >= 1 && xi < N - 1 && yi >= 1 && yi < N - 1) {
                        const float fractionalX = xf.w - xi;
                        const float fractionalY = yf.w - yi;
                        const float xDiff1 = 1.0f - fractionalX;
                        const float yDiff1 = 1.0f - fractionalY;
                        const float weightCurrent = xDiff1 * yDiff1;
                        const float weightRight = fractionalX * yDiff1;
                        const float weightBottom = xDiff1 * fractionalY;
                        const float weightBottomRight = fractionalX * fractionalY;
                        // Optional weighted scattering for more accuracy.
                        if (accuracy) {
                            atomicAdd(&accumulator_d[xi + yi * N], weightCurrent * mass.w);
                            atomicAdd(&accumulator_d[1 + xi + yi * N], weightRight * mass.w);
                            atomicAdd(&accumulator_d[xi + (yi + 1) * N], weightBottom * mass.w);
                            atomicAdd(&accumulator_d[1 + xi + (yi + 1) * N], weightBottomRight * mass.w);
                        }
                        else {
                            atomicAdd(&accumulator_d[xi + yi * N], mass.w);
                        }
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
                output_d[index] = 1.8f * powf((input_d[index] - smoothMin) / diff, 0.25f);
            }
        }
    }
    constexpr int TILE_SIZE = 32;
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
template<int NUM_TIME_STEPS_PER_RENDER>
struct Universe {
private:
    // For host
    std::vector<float> x, y;
    std::vector<float> xOld, yOld;
    std::vector<float> m;
    std::vector<float> renderColor;
    float* broadcast_h;
    float* frame_h;
    int particleCounter;
    int numParticles[Constants::NUM_CUDA_DEVICES];
    int particleOffsets[Constants::NUM_CUDA_DEVICES];
    int totalNumParticles;

    // For device
    cufftHandle fftPlan[Constants::NUM_CUDA_DEVICES];
    cudaStream_t computeStream[Constants::NUM_CUDA_DEVICES];
    cudaEvent_t latticeBroadcastEvent[Constants::NUM_CUDA_DEVICES];
    ComplexVar* lattice_d[Constants::NUM_CUDA_DEVICES];
    // 2: copies for double-buffering
    float* accumulator_d[Constants::NUM_CUDA_DEVICES];
    float* accumulator2_d[Constants::NUM_CUDA_DEVICES];
    float* latticeShifted_d[Constants::NUM_CUDA_DEVICES];
    float2* latticeShiftedForceXY_d[Constants::NUM_CUDA_DEVICES];
    ComplexVar* filter_d[Constants::NUM_CUDA_DEVICES];
    ComplexVar* deconvFilter_d[Constants::NUM_CUDA_DEVICES];
    float* x_d[Constants::NUM_CUDA_DEVICES];
    float* y_d[Constants::NUM_CUDA_DEVICES];
    float* xOld_d[Constants::NUM_CUDA_DEVICES];
    float* yOld_d[Constants::NUM_CUDA_DEVICES];
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
    std::vector<std::thread> dedicatedThreads;
    int pushCtr;
    int popCtr;
    bool working;

public:
    Universe(int particles, const int(&cudaDevices)[Constants::NUM_CUDA_DEVICES], const float(&devicePerformances)[Constants::NUM_CUDA_DEVICES], bool lowAccuracy) {
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
        srand(time(0));
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            numParticles[device] = particles * perf[device];
        }
        // For float4 alignment.
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            if ((numParticles[device] % 4) != 0) {
                numParticles[device] += (4 - (numParticles[device] % 4));
            }
        }
        int total = 0;
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            total += numParticles[device];
        }

        while (total > particles) {
            numParticles[Constants::NUM_CUDA_DEVICES - 1]--;
            total--;
        }
        while (total < particles) {
            numParticles[Constants::NUM_CUDA_DEVICES - 1]++;
            total++;
        }

        totalNumParticles = total;
        int sum = 0;
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            particleOffsets[device] = sum;
            sum += numParticles[device];
        }

        x.resize(particles);
        xOld.resize(particles);
        y.resize(particles);
        yOld.resize(particles);
        m.resize(particles);
        gpuErrchk(cudaMallocHost(&broadcast_h, sizeof(float) * Constants::N * Constants::N * Constants::NUM_CUDA_DEVICES));
        gpuErrchk(cudaMallocHost(&frame_h, sizeof(float) * (Constants::N * Constants::N)));
        
        std::vector<std::thread> initThreads;
        const int numThreads = std::thread::hardware_concurrency();
        for (int th = 0; th < numThreads; th++) {
            initThreads.emplace_back([&, th, numThreads]() {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                std::random_device rd;
                std::mt19937 rng = std::mt19937(rd());
                // Cpu-stride loop.
                const int workSize = particles;
                const int chunkSize = (workSize + numThreads - 1) / numThreads;
                for (int i = th * chunkSize; i < (th + 1) * chunkSize; i++) {
                    if (i < particles) {
                        x[i] = 0.01f * Constants::N + (0.98f * dist(rng) * Constants::N);
                        y[i] = 0.01f * Constants::N + (0.98f * dist(rng) * Constants::N);
                        xOld[i] = x[i];
                        yOld[i] = y[i];
                        m[i] = 1.0f / sum;
                    }
                }
                });
        }
        for (int th = 0; th < numThreads; th++) { initThreads[th].join(); }
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
        
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            cudaDeviceIndices[device] = cudaDevices[device];

            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaStreamCreate(&computeStream[device]));

            cudaDeviceProp prop;
            gpuErrchk(cudaGetDeviceProperties(&prop, cudaDeviceIndices[device]));
            int blocksPerSM = (prop.maxThreadsPerMultiProcessor + Constants::THREADS - 1) / Constants::THREADS;
            numBlocks[device] = prop.multiProcessorCount * blocksPerSM;

            
            gpuErrchk(cudaMemcpyToSymbolAsync(Kernels::renderBlurKern_c, renderFilter.data(), sizeof(float) * Constants::BLUR_R * Constants::BLUR_R, 0, cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaEventCreateWithFlags(&latticeBroadcastEvent[device], cudaEventDisableTiming));

            gpuErrchk(cudaMallocAsync(&lattice_d[device], sizeof(ComplexVar) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&localForceLattice_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&localForceLatticeResult_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&latticeShifted_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&latticeShiftedForceXY_d[device], sizeof(float2) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&filter_d[device], sizeof(ComplexVar) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&deconvFilter_d[device], sizeof(ComplexVar)* Constants::N* Constants::N, computeStream[device]));
            
            gpuErrchk(cudaMallocAsync(&x_d[device], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&y_d[device], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&xOld_d[device], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&yOld_d[device], sizeof(float) * numParticles[device], computeStream[device]));
            gpuErrchk(cudaMallocAsync(&m_d[device], sizeof(float) * numParticles[device], computeStream[device]));

            gpuErrchk(cudaMallocAsync(&accumulator_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&accumulator2_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));

            gpuErrchk(cudaMemsetAsync(accumulator_d[device], 0, sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemsetAsync(accumulator2_d[device], 0, sizeof(float) * Constants::N * Constants::N, computeStream[device]));

            gpuErrchk(cudaMemsetAsync(lattice_d[device], 0, sizeof(ComplexVar) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemsetAsync(localForceLattice_d[device], 0, sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemsetAsync(localForceLatticeResult_d[device], 0, sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemsetAsync(latticeShifted_d[device], 0, sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemsetAsync(latticeShiftedForceXY_d[device], 0, sizeof(float2) * Constants::N * Constants::N, computeStream[device]));

            gpuErrchk(cudaMallocAsync(&renderOutput_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMallocAsync(&renderOutput2_d[device], sizeof(float) * Constants::N * Constants::N, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(x_d[device], x.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(y_d[device], y.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(xOld_d[device], x.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(yOld_d[device], y.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(m_d[device], m.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));

            Kernels::k_generateFilterLattice<Constants::N> << <numBlocks[device], Constants::THREADS, 0, computeStream[device] >> > (filter_d[device], accuracy);
            Kernels::k_initSmoothMinMax << <1, 1, 0, computeStream[device] >> > ();
            Kernels::k_generateMassAssignmentKernel<Constants::N><<<numBlocks[device], Constants::THREADS>>>(deconvFilter_d[device], accuracy);

            cufftPlan2d(&fftPlan[device], Constants::N, Constants::N, CUFFT_C2C);
            cufftSetStream(fftPlan[device], computeStream[device]);
            calcFilterFft2D(device);
            calcFilter2Fft2D(device);
            if (accuracy) {
                Kernels::k_applyPhaseShift<Constants::N> << <numBlocks[device], Constants::THREADS >> > (deconvFilter_d[device], -0.5f, -0.5f);
            }
        }

        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaStreamSynchronize(computeStream[device]));
        }
    }

    // Moves all particles out of simulation area.
    void clear() {
        std::vector<std::thread> initThreads;
        const int numThreads = std::thread::hardware_concurrency();
        for (int th = 0; th < numThreads; th++) {
            initThreads.emplace_back([&, th, numThreads]() {
                // Cpu-stride loop.
                const int workSize = totalNumParticles;
                const int chunkSize = (workSize + numThreads - 1) / numThreads;
                for (int i = th * chunkSize; i < (th + 1) * chunkSize; i++) {
                    if (i < totalNumParticles) {
                        x[i] = -10.0f;
                        y[i] = -10.0f;
                        xOld[i] = -10.0f;
                        yOld[i] = -10.0f;
                    }
                }
                });
        }
        for (int th = 0; th < numThreads; th++) { initThreads[th].join(); }
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaMemcpyAsync(x_d[device], x.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(y_d[device], y.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(xOld_d[device], xOld.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
            gpuErrchk(cudaMemcpyAsync(yOld_d[device], yOld.data() + (particleOffsets[device]), sizeof(float) * numParticles[device], cudaMemcpyHostToDevice, computeStream[device]));
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
        angularVelocity /= Constants::dt;
        const float radiusSquared = (Constants::N * normalizedRadius) * (Constants::N * normalizedRadius);
        const float totalMass = massPerParticle * (maxCount - particleCounter) + (blackHole ? (maxCount - particleCounter) * massPerParticle / 1000.0f : 0.0f);
        centerOfMassVelocityX /= Constants::dt;
        centerOfMassVelocityY /= Constants::dt;
        const int numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> initThreads;
        for (int th = 0; th < numThreads; th++) {
            initThreads.emplace_back([&, th, numThreads]() {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                std::random_device rd;
                std::mt19937 rng = std::mt19937(rd());
                // Cpu-stride loop.
                const int workSize = (maxCount - particleCounter);
                const int chunkSize = (workSize + numThreads - 1) / numThreads;
                for (int i = particleCounter + th * chunkSize; i < particleCounter + (th + 1) * chunkSize; i++) {
                    if (i < maxCount) {
                        const float r = (0.1f + 0.9f * dist(rng)) * (Constants::N * normalizedRadius);
                        const float a = Constants::MATH_PI * 2.0 * dist(rng);
                        const float characteristicSpeedScaling = sqrtf(0.0000044f * totalMass * r / radiusSquared);
                        // orbit position
                        x[i] = r * cos(a) + normalizedCenterX * Constants::N;
                        y[i] = r * sin(a) + normalizedCenterY * Constants::N;
                        const float vecX = x[i] - normalizedCenterX * Constants::N;
                        const float vecY = y[i] - normalizedCenterY * Constants::N;
                        // orbit velocity
                        xOld[i] = -Constants::dt * ((vecY * angularVelocity) * characteristicSpeedScaling + centerOfMassVelocityX * Constants::N) + x[i];
                        yOld[i] = -Constants::dt * ((-vecX * angularVelocity) * characteristicSpeedScaling + centerOfMassVelocityY * Constants::N) + y[i];
                        m[i] = massPerParticle / totalNumParticles;

                        if (i == maxCount - 1) {
                            if (blackHole) {
                                x[i] = normalizedCenterX * Constants::N;
                                y[i] = normalizedCenterY * Constants::N;
                                xOld[i] = x[i] - centerOfMassVelocityX * Constants::N * Constants::dt;
                                yOld[i] = y[i] - centerOfMassVelocityY * Constants::N * Constants::dt;
                                m[i] = ((maxCount - particleCounter) * massPerParticle / 1000.0f) / totalNumParticles;
                            }
                        }
                    }
                }
            });
        }
        for (int th = 0; th < numThreads; th++) { initThreads[th].join(); }
        const int start = particleCounter;
        const int stop = maxCount;
        particleCounter = maxCount;
        bool cache[Constants::NUM_CUDA_DEVICES];
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            bool copy = false;
            int copyLength = 0;
            int copyStartDestination = 0;
            int copyStartSource = 0;

            if (start < particleOffsets[device] && stop > particleOffsets[device] + numParticles[device]) {
                copy = true;
                copyLength = numParticles[device];
                copyStartDestination = 0;
                copyStartSource = particleOffsets[device];
            } else if(start < particleOffsets[device] && stop > particleOffsets[device] && stop <= particleOffsets[device] + numParticles[device]){
                copy = true;
                copyLength = stop - particleOffsets[device];
                copyStartDestination = 0;
                copyStartSource = particleOffsets[device];
            } else if (start >= particleOffsets[device] && stop <= particleOffsets[device] + numParticles[device]) {
                copy = true;
                copyLength = stop - start;
                copyStartDestination = start - particleOffsets[device];
                copyStartSource = start;
            } else if (start >= particleOffsets[device] && start < particleOffsets[device] + numParticles[device] && stop > particleOffsets[device] + numParticles[device]) {
                copy = true;
                copyLength = numParticles[device] - (start - particleOffsets[device]);
                copyStartDestination = start - particleOffsets[device];
                copyStartSource = start;
            }
            cache[device] = (copy && copyLength > 0);
            if(cache[device]){
                gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
                gpuErrchk(cudaMemcpyAsync(x_d[device] + copyStartDestination, x.data() + copyStartSource, sizeof(float) * copyLength, cudaMemcpyHostToDevice, computeStream[device]));
                gpuErrchk(cudaMemcpyAsync(y_d[device] + copyStartDestination, y.data() + copyStartSource, sizeof(float) * copyLength, cudaMemcpyHostToDevice, computeStream[device]));
                gpuErrchk(cudaMemcpyAsync(xOld_d[device] + copyStartDestination, xOld.data() + copyStartSource, sizeof(float) * copyLength, cudaMemcpyHostToDevice, computeStream[device]));
                gpuErrchk(cudaMemcpyAsync(yOld_d[device] + copyStartDestination, yOld.data() + copyStartSource, sizeof(float) * copyLength, cudaMemcpyHostToDevice, computeStream[device]));
                gpuErrchk(cudaMemcpyAsync(m_d[device] + copyStartDestination, m.data() + copyStartSource, sizeof(float) * copyLength, cudaMemcpyHostToDevice, computeStream[device]));
            }
        }
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            if (cache[device]) {
                gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
                gpuErrchk(cudaStreamSynchronize(computeStream[device]));
            }
        }

        if (particleCounter >= totalNumParticles) {
            particleCounter = 0;
        }
    }
private:
    struct Barrier { 
        Barrier() { 
            std::unique_lock<std::mutex> lock(lockMutex); 
            count = 0; 
            waitCount = 0; 
        } 
        void wait(bool stopAll = false) { 
            std::unique_lock<std::mutex> lock(lockMutex); 
            if (stopAll) { 
                waitCount++;
                cv.notify_all(); 
                return; 
            } 
            unsigned int currentCounter = count; 
            waitCount++; 
            if (Constants::NUM_CUDA_DEVICES == waitCount) { 
                waitCount = 0; 
                count++; 
                cv.notify_all(); 
            } else { 
                cv.wait(lock, [this, currentCounter] { return currentCounter != count; }); 
            } 
        } 
        private: 
        std::mutex lockMutex; 
        std::condition_variable cv; 
        int count; 
        int waitCount; 
    };
    Barrier timeStepBarriers[NUM_TIME_STEPS_PER_RENDER];

    void scatterMassOnLattice(int gpu, int nbodyCalcCounter) {

        if (Constants::NUM_CUDA_DEVICES > 1) {;
            // Device - device broadcast: send
            Kernels::k_clearAccumulator<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (accumulator_d[gpu]);
            Kernels::k_scatterMassOnAccumulator<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (accumulator_d[gpu], reinterpret_cast<float4*>(x_d[gpu]), reinterpret_cast<float4*>(y_d[gpu]), reinterpret_cast<float4*>(m_d[gpu]), numParticles[gpu], accuracy);
            gpuErrchk(cudaMemcpyAsync(broadcast_h + (gpu * Constants::N * Constants::N),
                accumulator_d[gpu],
                sizeof(float) * Constants::N * Constants::N,
                cudaMemcpyDeviceToHost, computeStream[gpu]));
            gpuErrchk(cudaEventRecord(latticeBroadcastEvent[gpu], computeStream[gpu]));
            timeStepBarriers[nbodyCalcCounter].wait();
            // broadcast: receive
            for (int device2 = 0; device2 < Constants::NUM_CUDA_DEVICES; device2++) {
                if (gpu != device2) {
                    // Copy each device's broadcast data, then sum.
                    gpuErrchk(cudaStreamWaitEvent(computeStream[gpu], latticeBroadcastEvent[device2]));
                    gpuErrchk(cudaMemcpyAsync(accumulator2_d[gpu],
                        broadcast_h + (device2 * Constants::N * Constants::N),
                        sizeof(float) * Constants::N * Constants::N,
                        cudaMemcpyHostToDevice, computeStream[gpu]));
                    Kernels::k_sumAccumulators<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (
                        accumulator_d[gpu],
                        accumulator2_d[gpu]);
                }
            }
            
        }
        else {
            Kernels::k_clearAccumulator<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (accumulator_d[gpu]);
            Kernels::k_scatterMassOnAccumulator<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (accumulator_d[gpu], reinterpret_cast<float4*>(x_d[gpu]), reinterpret_cast<float4*>(y_d[gpu]), reinterpret_cast<float4*>(m_d[gpu]), numParticles[gpu], accuracy);
        }
        
        Kernels::k_copyAccumulatorIntoLattice<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (
            accumulator_d[gpu],
            lattice_d[gpu]
            );
        if (nbodyCalcCounter == NUM_TIME_STEPS_PER_RENDER - 1) {
            Kernels::k_getRealComponentOfLattice<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (lattice_d[gpu], renderOutput_d[gpu]);
            Kernels::k_calcBlurConvolution<Constants::BLUR_R, Constants::N> << <dim3(32, 32, 1), dim3(32, 32, 1), sizeof(float)* (Constants::BLUR_R + Kernels::TILE_SIZE)* (Constants::BLUR_R + Kernels::TILE_SIZE), computeStream[gpu] >> > (renderOutput_d[gpu], renderOutput2_d[gpu]);
            Kernels::k_resetMinMax << <1, 1, 0, computeStream[gpu] >> > ();
            Kernels::k_calcMinMax<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (renderOutput2_d[gpu]);
            Kernels::k_smoothMinMax << <1, 1, 0, computeStream[gpu] >> > ();
            Kernels::k_scaleWithMinMax<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (renderOutput2_d[gpu], renderOutput_d[gpu]);
        }
        // Accuracy mode also adds a short-range force component using normal convolution.
        if (accuracy) {
            Kernels::k_getRealComponentOfLattice<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (
                lattice_d[gpu],
                localForceLattice_d[gpu]
                );
        }
    }
    void calcLatticeFft2D(int gpu) {
        cufftExecC2C(fftPlan[gpu],
            reinterpret_cast<cufftComplex*>(lattice_d[gpu]),
            reinterpret_cast<cufftComplex*>(lattice_d[gpu]),
            CUFFT_FORWARD);
    }
    void calcFilterFft2D(int gpu) {
        cufftSetStream(fftPlan[gpu], computeStream[gpu]);
        cufftExecC2C(fftPlan[gpu],
            reinterpret_cast<cufftComplex*>(filter_d[gpu]),
            reinterpret_cast<cufftComplex*>(filter_d[gpu]),
            CUFFT_FORWARD);
    }
    void calcFilter2Fft2D(int gpu) {
        cufftSetStream(fftPlan[gpu], computeStream[gpu]);
        cufftExecC2C(fftPlan[gpu],
            reinterpret_cast<cufftComplex*>(deconvFilter_d[gpu]),
            reinterpret_cast<cufftComplex*>(deconvFilter_d[gpu]),
            CUFFT_FORWARD);
    }
    void multiplyLatticeFilterElementwise(int gpu) {
        Kernels::k_multElementwiseLatticeFilter<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (lattice_d[gpu], filter_d[gpu]);
        Kernels::k_divElementwiseLatticeFilter<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (lattice_d[gpu], deconvFilter_d[gpu]);
    }
    void calcLatticeIfft2D(int gpu) {
        cufftSetStream(fftPlan[gpu], computeStream[gpu]);
        cufftExecC2C(fftPlan[gpu],
            reinterpret_cast<cufftComplex*>(lattice_d[gpu]),
            reinterpret_cast<cufftComplex*>(lattice_d[gpu]),
            CUFFT_INVERSE);
    }
    void multiSampleForces(int gpu) {
        Kernels::k_shiftLattice<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (lattice_d[gpu], latticeShifted_d[gpu], accuracy);
        Kernels::k_calcGradientLattice<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (latticeShifted_d[gpu], latticeShiftedForceXY_d[gpu]);
        Kernels::k_forceMultiSampling<Constants::N> << <numBlocks[gpu], Constants::THREADS, 0, computeStream[gpu] >> > (latticeShiftedForceXY_d[gpu], x_d[gpu], y_d[gpu], xOld_d[gpu], yOld_d[gpu], numParticles[gpu], accuracy);
    }

    void pushFrame(float* frame) {
        std::lock_guard<std::mutex> lg(lock);
        if (pushCtr < popCtr + Constants::MAX_FRAMES_BUFFERED - 1) {
            memcpy(frames[pushCtr % Constants::MAX_FRAMES_BUFFERED].data(), frame, sizeof(float) * (Constants::N * Constants::N));
            pushCtr++;
        }
    }
    void checkWorkStatus(bool& workingTmp) {
        std::lock_guard<std::mutex> lg(lock);
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
                frames[i].resize(Constants::N * Constants::N);
            }
            working = true;
        }
        for (int i = 0; i < Constants::NUM_CUDA_DEVICES; i++) {
            int gpu = i;
            dedicatedThreads.emplace_back(std::thread([&, gpu]() {
                int nbodyCalcCounter = 0;
                bool workingTmp = true;
                gpuErrchk(cudaSetDevice(cudaDeviceIndices[gpu]));
                cufftSetStream(fftPlan[gpu], computeStream[gpu]);
                while (workingTmp) {
                    scatterMassOnLattice(gpu, nbodyCalcCounter);
                    calcLatticeFft2D(gpu);
                    multiplyLatticeFilterElementwise(gpu);
                    calcLatticeIfft2D(gpu);
                    multiSampleForces(gpu);

                    nbodyCalcCounter++;
                    if (nbodyCalcCounter == NUM_TIME_STEPS_PER_RENDER) {
                        nbodyCalcCounter = 0;
                        gpuErrchk(cudaStreamSynchronize(computeStream[gpu]));
                        
                        if (gpu == 0) {
                            gpuErrchk(cudaSetDevice(cudaDeviceIndices[0]));
                            gpuErrchk(cudaMemcpyAsync(frame_h, renderOutput_d[0], sizeof(float) * Constants::N * Constants::N, cudaMemcpyDeviceToHost, computeStream[0]));
                            gpuErrchk(cudaStreamSynchronize(computeStream[0]));
                            pushFrame(frame_h);
                        }
                    }
                    checkWorkStatus(workingTmp);
                }
                for (int i = 0; i < NUM_TIME_STEPS_PER_RENDER; i++) {
                    timeStepBarriers[i].wait(true);
                }
            }));
        }
    }
    // returns frame pixels.
    std::vector<float>& popFrame(bool& ready) {
        static std::vector<float> result(Constants::N * Constants::N);
        {
            std::lock_guard<std::mutex> lg(lock);
            if (popCtr < pushCtr) {
                result.swap(frames[popCtr % Constants::MAX_FRAMES_BUFFERED]);
                popCtr++;
                ready = true;
            }
            else {
                ready = false;
            }
        }
        return result;
    }
    void nBodyStop() {
        {
            std::lock_guard<std::mutex> lg(lock);
            working = false;
        }
        for (int i = 0; i < Constants::NUM_CUDA_DEVICES; i++) {
            if (dedicatedThreads[i].joinable()) {
                dedicatedThreads[i].join();
            }
        }
    }
    ~Universe() {
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            gpuErrchk(cudaFreeAsync(lattice_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(accumulator_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(accumulator2_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(localForceLattice_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(localForceLatticeResult_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(latticeShifted_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(latticeShiftedForceXY_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(filter_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(deconvFilter_d[device], computeStream[device]));
            
            gpuErrchk(cudaFreeAsync(x_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(y_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(xOld_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(yOld_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(m_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(renderOutput_d[device], computeStream[device]));
            gpuErrchk(cudaFreeAsync(renderOutput2_d[device], computeStream[device]));
            gpuErrchk(cudaEventDestroy(latticeBroadcastEvent[device]));
        }
        for (int device = 0; device < Constants::NUM_CUDA_DEVICES; device++) {
            gpuErrchk(cudaSetDevice(cudaDeviceIndices[device]));
            cufftSetStream(fftPlan[device], computeStream[device]);
            cufftDestroy(fftPlan[device]);
            gpuErrchk(cudaStreamSynchronize(computeStream[device]));
            gpuErrchk(cudaStreamDestroy(computeStream[device]));
        }
        gpuErrchk(cudaFreeHost(broadcast_h));
        gpuErrchk(cudaFreeHost(frame_h));
    }
};