#include "CosmosCuda.cuh"

int main() {
    const int numNbodySimulationsPerRender = 5;
    srand(time(0));
    Universe uni(1000 * 1000 * 100);
    uni.calcFilterFft2D();
    while (true) {
        uni.startBenchmark();
        for (int i = 0; i < numNbodySimulationsPerRender; i++) {
            uni.scatterMassOnLattice();
            uni.calcLatticeFft2D();
            uni.multiplyLatticeFilterElementwise();
            uni.calcLatticeIfft2D();
            uni.multiSampleForces();
        }
        uni.stopBenchmark();
        uni.sync(numNbodySimulationsPerRender);
        uni.render();
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    return 0;
}