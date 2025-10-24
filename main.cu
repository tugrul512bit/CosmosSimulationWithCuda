#include "CosmosCuda.cuh"
// Change these values to utilize your gpu better (currently they are tuned for RTX4070)
// Constants::BLOCKS
// Constants::THREADS
// Change this to another power-of-2 to tune accuracy of long-ranged forces. Short-ranged forces are not calculated currently (todo).
// Constants::N
int main() {
    const int numNbodySimulationsPerRender = 2;
    // 20 Bytes per particle is allocated.
    const int numParticles = 1000 * 1000 * 10;
    // cuda device index
    const int device = 0;
    // true: more performance
    const bool lowAccuracy = false;
    // Window width/height
    const int w = 1200;
    const int h = 1200;
    Universe cosmos(numParticles, device, lowAccuracy, w, h);
    
    while (true) {
        cosmos.startBenchmark();
        for (int i = 0; i < numNbodySimulationsPerRender; i++) {
            cosmos.nBody();
        }
        cosmos.stopBenchmark();
        cosmos.sync(numNbodySimulationsPerRender);
        cosmos.render();
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    return 0;
}