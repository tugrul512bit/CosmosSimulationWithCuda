# CosmosSimulationWithCuda

Real-time N-body algorithm for a billion particles (3.7GB memory per 100M particles required), accelerated with CUDA.

- Dependency: Single header-only project with OpenCV(for demo) and CUDA(for compute) APIs. Uses Vcpkg to load OpenCV.
- Multi-GPU work distribution: particles are computed only in their own GPUs.
- Double-buffering: to overlap I/O with computations as an asynchronous pipeline.
- Render: frames are generated asynchronously and buffered to the user's thread, further hiding latency.
- Real-time performance for 500 million particles, with 2 main-stream CUDA GPUs.
![time 1](/cuda-streams.png)
![time 1](/t1.png)
![time 2](/t2.png)
![time 3](/t3.png)
![test](/testc.png)

Algorithm: 
- Mass values of particles are projected onto a lattice of 2048x2048 cells (this Constants::N value can be changed from header)
- The lattice is sent to two convolution operations.
- - FFT for infinite ranged forces (filter weights = 1 / r)
- - 33x33 direct convolution for short-ranged forces (filter weights = 1 / r)
- Each convolution has weights close to center to have zero value to avoid particles pulling themselves
- Then both results are summed elementwise to have a total potential
- Gradient of the potential is sampled by each particle and used as force acting on them
- Euler integration is used for velocity and position updates (when other parts are optimized for higher accuracy, this will become Verlet Integration)
- Complexity: O(N Log(N)) with a low constant cost so that it can run 120 FPS for 20 million particles using a mainstream gpu.

FFT Convolution:
- Particle mass lattice is 2D
- First, FFT of all rows are computed.
- Then FFT of all columns are computed.
- Repeated for both mass-lattice and gravity-lattice
- Element-wise (complex-value) multiplication of both results --> this is FFT of convolution
- Inverse-FFT of the output is computed --> convolution complete

Gravity Lattice:
- This is another 2D array with its center at (N/2,N/2) both in indexing of array and 1/r calculations of cells.
- 1/r is used as the gravitational potential.

Particle lattice:
- Particle masses are projected onto another 2D array
- Each cell contains all or partial mass values of the particles contributing
- High-accuracy mode uses 4 cells, low-accuracy mode uses 1 cell per particle

Gradient:
- Similar to the mass projection, but opposite
- Gradient at each point is computed
- Each particle samples or multisamples (depends on accuracy mode) gradients and computes the force acting on it
- Force is divided by the mass of particle to compute movement during that time step (Euler Integration), together with velocity updates.

10 Million Stars, 1024 x 1024 map (RTX4070):
```
CUDA Nbody pipeline: 1.71934 milliseconds per step
CUDA Nbody pipeline: 1.71413 milliseconds per step
CUDA Nbody pipeline: 1.72573 milliseconds per step
CUDA Nbody pipeline: 1.73992 milliseconds per step
CUDA Nbody pipeline: 1.72532 milliseconds per step
CUDA Nbody pipeline: 1.71991 milliseconds per step
CUDA Nbody pipeline: 1.72161 milliseconds per step
CUDA Nbody pipeline: 1.72772 milliseconds per step
CUDA Nbody pipeline: 1.72846 milliseconds per step
```
50 Million Stars, 2048 x 2048 map (RTX4070):
```
CUDA Nbody pipeline: 9.55624 milliseconds per step
CUDA Nbody pipeline: 9.56117 milliseconds per step
CUDA Nbody pipeline: 9.58674 milliseconds per step
CUDA Nbody pipeline: 9.58286 milliseconds per step
CUDA Nbody pipeline: 9.5803 milliseconds per step
CUDA Nbody pipeline: 9.62326 milliseconds per step
CUDA Nbody pipeline: 9.6405 milliseconds per step
CUDA Nbody pipeline: 9.63198 milliseconds per step
CUDA Nbody pipeline: 9.60792 milliseconds per step
CUDA Nbody pipeline: 9.61549 milliseconds per step
CUDA Nbody pipeline: 9.64701 milliseconds per step
CUDA Nbody pipeline: 9.6755 milliseconds per step
```
![test](/testa.png)
![test](/testb.png)

