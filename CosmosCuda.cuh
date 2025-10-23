#define __CUDACC__
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
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
    // Constants::N can be only power of 2. The higher Constants::N the better approximation.
    constexpr int Constants::N = 2048;
    // RTX4070
    const int BLOCKS = 46 * 3;
    const int THREADS = 512;
    // FFT uses these
    constexpr double Constants::MATH_PI = 3.14159265358979323846;
    using ComplexVar = float2;
}

namespace Kernels {

    __device__ __host__ __forceinline__ Constants::ComplexVar makeComplexVar(float x, float y) {
        Constants::ComplexVar result;
        result.x = x;
        result.y = y;
        return result;
    }
    __device__ Constants::ComplexVar wCoefficients[Constants::N * 2];
    __device__ __forceinline__ Constants::ComplexVar d_wForNk(const double Constants::N, const double k) {
        const double angle = (-2.0 * Constants::MATH_PI * k) / Constants::N;
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
            const int leveld2 = level / 2;
            const int idx = (warpLane % level2);
            const bool firstHalf = idx < level;
            const float index = idx;
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
        constexpr unsigned int gridSteps = (Constants::N + BLOCKS - 1) / BLOCKS;
        constexpr unsigned int blockSteps = (Constants::N + THREADS - 1) / THREADS;
        const float inverseMult = inverse ? -1.0f : 1.0f;
        const float divider = inverse ? Constants::N : 1.0f;
        Constants::ComplexVar vars[blockSteps];
        extern __shared__ Constants::ComplexVar s_coalescing[];


        for (unsigned int grid = 0; grid < gridSteps; grid++) {
            const unsigned int row = grid * BLOCKS + block;
            if (row < Constants::N) {
#pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * THREADS + thread;
                    const int element = col + row * Constants::N;
                    if (col < Constants::N) {
                        s_coalescing[col] = __ldg(&data[element]);
                    }
                }
                __syncthreads();
#pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * THREADS + thread;
                    Constants::ComplexVar var;
                    if (col < Constants::N) {
                        vars[blc] = s_coalescing[__brev(col) >> (32 - d_bits<Constants::N>())];
                    }
                    d_calcWarpDft(vars[blc], warpLane, inverseMult, wCoefficients);
                }

                // todo: ping-pong buffer = less syncthreads, bigger input support
                __syncthreads();
#pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * THREADS + thread;
                    const int element = col + row * Constants::N;
                    if (col < Constants::N) {
                        s_coalescing[col] = vars[blc];
                    }
                }
                __syncthreads();
                int x = 0;
                int wOfs = 62;
#pragma unroll 1
                for (unsigned int level = 32; level < Constants::N; level <<= 1) {
                    const int level2 = level * 2;
                    const int leveld2 = level / 2;
#pragma unroll
                    for (int blc = 0; blc < blockSteps; blc++) {
                        const unsigned int col = blc * THREADS + thread;
                        if (col < Constants::N) {
                            const int idx = (col % level2);
                            const bool firstHalf = idx < level;
                            const float index = idx;
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
                        const unsigned int col = blc * THREADS + thread;
                        if (col < Constants::N) {
                            s_coalescing[col] = vars[blc];
                        }
                    }
                    __syncthreads();
                }
#pragma unroll
                for (int blc = 0; blc < blockSteps; blc++) {
                    const unsigned int col = blc * THREADS + thread;
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
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
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
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
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
                for (int k = 0; k < steps2; k++) {
                    const int element = k * numThreads + thread;
                    if (element < SUB_ELEMENTS) {
                        const int col = element % SUB_MATRIX_SIZE;
                        const int row = element / SUB_MATRIX_SIZE;
                        s_tile[row][col] = data[tileOffset + col + row * Constants::N];
                    }
                }
                __syncthreads();
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
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < Constants::N * Constants::N) {
                const int i = index % Constants::N;
                const int j = index / Constants::N;
                const int cx = Constants::N / 2;
                const int cy = Constants::N / 2;
                const float dx = i - cx;
                const float dy = j - cy;
                const float r2 = dx * dx + dy * dy;
                if (r2 > 0.0f) {
                    data[i + j * Constants::N].x = 1.0f / sqrtf(r2);
                    data[i + j * Constants::N].y = 0.0f;
                }
                else {
                    data[i + j * Constants::N].x = 0.0f;
                    data[i + j * Constants::N].y = 0.0f;
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
    __global__ void k_forceMultiSampling(Constants::ComplexVar* lattice_d, float* x, float* y, float* vx, float* vy, const int numParticles) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (numParticles + numTotalThreads - 1) / numTotalThreads;
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < numParticles) {
                const float posX = x[index];
                const float posY = y[index];

                // Sampling
                const int centerX = int(posX);
                const int centerY = int(posY);
                const int centerIndex = centerX + centerY * Constants::N;
                if (centerX >= 0 && centerX < Constants::N && centerY >= 0 && centerY < Constants::N) {
                    const float centerData = lattice_d[centerIndex].x;
                    float left = centerData;
                    float right = centerData;
                    float top = centerData;
                    float bot = centerData;
                    if (centerX - 1 >= 0) {
                        left = lattice_d[centerIndex - 1].x;
                    }
                    if (centerX + 1 < Constants::N) {
                        right = lattice_d[centerIndex + 1].x;
                    }
                    if (centerY - 1 >= 0) {
                        top = lattice_d[centerIndex - Constants::N].x;
                    }
                    if (centerY + 1 < Constants::N) {
                        bot = lattice_d[centerIndex + Constants::N].x;
                    }
                    const float forceComponentX = (right - left) * 0.5f;
                    const float forceComponentY = (bot - top) * 0.5f;
                    constexpr float dt = 0.002f;
                    x[index] += vx[index] * dt;
                    y[index] += vy[index] * dt;
                    vx[index] += forceComponentX * dt;
                    vy[index] += forceComponentY * dt;
                }
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
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < Constants::N * Constants::N) {
                lattice_d[index] = Constants::ComplexVar{ 0.0f, 0.0f };
            }
        }
    }
    template<int N>
    __global__ void k_shiftLattice(Constants::ComplexVar* lattice_d, Constants::ComplexVar* latticeShifted_d) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (Constants::N * Constants::N + numTotalThreads - 1) / numTotalThreads;
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < Constants::N * Constants::N) {
                const int x = index % Constants::N;
                const int y = index / Constants::N;
                const int shiftedX = (x - Constants::N / 2 + Constants::N * 2) % Constants::N;
                const int shiftedY = (y - Constants::N / 2 + Constants::N * 2) % Constants::N;
                latticeShifted_d[index] = lattice_d[shiftedX + shiftedY * Constants::N];
            }
        }
    }
    template<int N>
    __global__ void k_scatterMassOnLattice(Constants::ComplexVar* lattice_d, const float* x, const float* y, const int numParticles) {
        const int thread = threadIdx.x;
        const int block = blockIdx.x;
        const int numBlocks = gridDim.x;
        const int numThreads = blockDim.x;
        const int globalThread = thread + block * numThreads;
        const int numTotalThreads = numThreads * numBlocks;
        const int steps = (numParticles + numTotalThreads - 1) / numTotalThreads;
        for (int ii = 0; ii < steps; ii++) {
            const int index = ii * numTotalThreads + globalThread;
            if (index < numParticles) {
                const int xi = x[index];
                const int yi = y[index];
                if (xi >= 0 && xi < Constants::N && yi >= 0 && yi < Constants::N) {
                    atomicAdd(&lattice_d[xi + yi * Constants::N].x, 1.0f);
                }
            }
        }
    }
}

struct Universe {
    // For host
    std::vector<float> x, y;
    std::vector<float> vx, vy;
    std::vector<Constants::ComplexVar> lattice;
    std::vector<int> renderIndices;


    // For OpenCV
    cv::Mat mat;
    int numParticles;
    int tileSize;

    // For device
    cudaEvent_t eventStart;
    cudaEvent_t eventStop;
    Constants::ComplexVar* lattice_d;
    Constants::ComplexVar* latticeShifted_d;
    Constants::ComplexVar* filter_d;
    float* x_d;
    float* y_d;
    float* vx_d;
    float* vy_d;
    Universe(int particles = 0, int tileSizeParameter = 32) {
        tileSize = tileSizeParameter;
        numParticles = particles;
        x.resize(particles);
        vx.resize(particles);
        y.resize(particles);
        vy.resize(particles);

        lattice.resize(Constants::N * Constants::N);
        mat = cv::Mat(cv::Size2i(Constants::N, Constants::N), CV_32FC1);
        const int centerX = Constants::N / 2;
        const int centerY = Constants::N / 2;
        const float speed = 1.50f;
        for (int i = 0; i < particles; i++) {
            const float r = rand() % (Constants::N / 3);
            const float a = Constants::MATH_PI * 2.0 * (rand() % 1000) / 1000.0f;
            x[i] = r * cos(a) + Constants::N / 2;
            y[i] = r * sin(a) + Constants::N / 2;
            const float vecX = x[i] - centerX;
            const float vecY = y[i] - centerY;

            vx[i] = vecY * speed / (sqrt(sqrt(r)) + 1.0f);
            vy[i] = -vecX * speed / (sqrt(sqrt(r)) + 1.0f);
        }
        gpuErrchk(cudaEventCreate(&eventStart));
        gpuErrchk(cudaEventCreate(&eventStop));
        gpuErrchk(cudaMalloc(&lattice_d, sizeof(Constants::ComplexVar) * Constants::N * Constants::N));
        gpuErrchk(cudaMalloc(&latticeShifted_d, sizeof(Constants::ComplexVar) * Constants::N * Constants::N));
        gpuErrchk(cudaMalloc(&filter_d, sizeof(Constants::ComplexVar) * Constants::N * Constants::N));
        gpuErrchk(cudaMalloc(&x_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMalloc(&y_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMalloc(&vx_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMalloc(&vy_d, sizeof(float) * numParticles));
        gpuErrchk(cudaMemcpy(x_d, x.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(y_d, y.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(vx_d, vx.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(vy_d, vy.data(), sizeof(float) * numParticles, cudaMemcpyHostToDevice));
        cudaFuncSetAttribute(Kernels::k_calcFftBatched1D<Constants::N>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(Kernels::k_calcFftBatched1D<Constants::N>, cudaFuncAttributeMaxDynamicSharedMemorySize, Constants::N * sizeof(Constants::ComplexVar));
        Kernels::k_fillCoefficientArray<Constants::N> << <1, 1024 >> > ();
        Kernels::k_fillPairIndexList<Constants::N> << <BLOCKS, THREADS >> > ();
        Kernels::k_generateFilterLattice<Constants::N> << <BLOCKS, THREADS >> > (filter_d);
        gpuErrchk(cudaDeviceSynchronize());
        cv::namedWindow("Fast Nbody");
    }
    void startBenchmark() {
        gpuErrchk(cudaEventRecord(eventStart));
    }
    // todo: scatter on 9 cells per mass
    void scatterMassOnLattice() {
        Kernels::k_clearLattice<Constants::N> << <BLOCKS, THREADS >> > (lattice_d);
        Kernels::k_scatterMassOnLattice<Constants::N> << <BLOCKS, THREADS >> > (lattice_d, x_d, y_d, numParticles);

    }
    void calcLatticeFft2D() {
        Kernels::k_calcFftBatched1D<Constants::N> << <BLOCKS, THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (lattice_d, false);
        Kernels::k_calcTranspose<Constants::N> << <BLOCKS, THREADS >> > (lattice_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <BLOCKS, THREADS >> > (lattice_d);
        Kernels::k_calcFftBatched1D<Constants::N> << <BLOCKS, THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (lattice_d, false);
        Kernels::k_calcTranspose<Constants::N> << <BLOCKS, THREADS >> > (lattice_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <BLOCKS, THREADS >> > (lattice_d);

    }
    void calcFilterFft2D() {
        Kernels::k_calcFftBatched1D<Constants::N> << <BLOCKS, THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (filter_d, false);
        Kernels::k_calcTranspose<Constants::N> << <BLOCKS, THREADS >> > (filter_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <BLOCKS, THREADS >> > (filter_d);
        Kernels::k_calcFftBatched1D<Constants::N> << <BLOCKS, THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (filter_d, false);
        Kernels::k_calcTranspose<Constants::N> << <BLOCKS, THREADS >> > (filter_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <BLOCKS, THREADS >> > (filter_d);

    }
    void multiplyLatticeFilterElementwise() {
        Kernels::k_multElementwiseLatticeFilter<Constants::N> << <BLOCKS, THREADS >> > (lattice_d, filter_d);
    }
    void calcLatticeIfft2D() {
        Kernels::k_calcFftBatched1D<Constants::N> << <BLOCKS, THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (lattice_d, true);
        Kernels::k_calcTranspose<Constants::N> << <BLOCKS, THREADS >> > (lattice_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <BLOCKS, THREADS >> > (lattice_d);
        Kernels::k_calcFftBatched1D<Constants::N> << <BLOCKS, THREADS, Constants::N * sizeof(Constants::ComplexVar) >> > (lattice_d, true);
        Kernels::k_calcTranspose<Constants::N> << <BLOCKS, THREADS >> > (lattice_d);
        Kernels::k_calcTransposeDiagonals<Constants::N> << <BLOCKS, THREADS >> > (lattice_d);
    }
    void multiSampleForces() {
        Kernels::k_shiftLattice<Constants::N> << <BLOCKS, THREADS >> > (lattice_d, latticeShifted_d);
        Kernels::k_forceMultiSampling<Constants::N> << <BLOCKS, THREADS >> > (latticeShifted_d, x_d, y_d, vx_d, vy_d, numParticles);
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
        scatterMassOnLattice();
        gpuErrchk(cudaDeviceSynchronize());
        mat.setTo(cv::Scalar(0.0f));


        gpuErrchk(cudaMemcpy(lattice.data(), lattice_d, sizeof(Constants::ComplexVar) * Constants::N * Constants::N, cudaMemcpyDeviceToHost));
        for (int i = 0; i < Constants::N * Constants::N; i++) {
            mat.at<float>(i) = lattice[i].x;
        }
        cv::Mat resized;
        cv::resize(mat, resized, cv::Size(1280, 1280), 0, 0, cv::INTER_LINEAR);

        cv::imshow("Fast Nbody", resized);
        cv::waitKey(1);
    }
    ~Universe() {
        cv::destroyAllWindows();
        gpuErrchk(cudaFree(lattice_d));
        gpuErrchk(cudaFree(latticeShifted_d));
        gpuErrchk(cudaFree(filter_d));
        gpuErrchk(cudaFree(x_d));
        gpuErrchk(cudaFree(y_d));
        gpuErrchk(cudaFree(vx_d));
        gpuErrchk(cudaFree(vy_d));
        gpuErrchk(cudaEventDestroy(eventStart));
        gpuErrchk(cudaEventDestroy(eventStop));
    }
};