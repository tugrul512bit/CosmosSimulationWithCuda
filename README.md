# CosmosSimulationWithCuda

Simulate millions of stars using a CUDA GPU, in few miliseconds.

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

---
# Optimizations To Add
- Short-ranged force calculation from nearest-neighbors pair-wise: accuracy improvement
- Partial sorting of particles in each tile: performance improvement
- Rendering: double-buffering for more FPS

