# CosmosSimulationWithCuda

Real-time N-body algorithm for a billion particles (2.3GB memory per 100M particles required), accelerated with CUDA.

- Dependency: Single header-only project with OpenCV(for demo) and CUDA(for compute) APIs. Uses Vcpkg to load OpenCV.
- Multi-GPU work distribution: particles are computed only in their own GPUs.
- GPU-GPU communication: partially overlaps with computations to hide latency.
- Render: frames are generated asynchronously and buffered to the user's thread, further hiding latency.
- Real-time performance for 500 million particles, with 2 main-stream CUDA GPUs.
![time 1](/timeline.png)
![time 1](/t1.png)
![time 2](/t2.png)
![time 3](/t3.png)
![test](/testc.png)

Algorithm: 
- Mass values of particles are projected onto a lattice of 2048x2048 cells (this Constants::N value can be changed from header)
- The lattice is sent to two convolution operations.
- - FFT for infinite ranged forces (filter weights = 1 / r)
- - deconvolution of mass-scatter kernel for short-ranged forces (to undo self-pull)
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


![test](/binary_star1.png)
![test](/binary_star2.png)
![test](/binary_star3.png)
![test](/binary_star4.png)

