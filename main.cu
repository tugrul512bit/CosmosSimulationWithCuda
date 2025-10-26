#include "CosmosCuda.cuh"
// Change these values to utilize your gpu better (currently they are tuned for RTX4070)
// Constants::BLOCKS
// Constants::THREADS
// Change this to another power-of-2 to tune accuracy of long-ranged forces. Short-ranged forces are not calculated accurately with FFT alone, so a direct-convolution is used to partially reduce error in there.
// Constants::N
int main() {
    const int numNbodySimulationsPerRender = 2;
    // 100M particles require 2.5GB memory
    const int maximumParticles = 1000 * 1000 * 20;
    // cuda device index
    const int device = 0;
    // true = more performance + single force sampling + single mass projection + pure FFT convolution
    const bool lowAccuracy = true;
    // Window width/height
    const int w = 1340;
    const int h = 1340;
    Universe cosmos(maximumParticles, device, lowAccuracy, w, h);
    const bool galaxyCollisionScenario = true;
    if (galaxyCollisionScenario) {
        cosmos.clear();
        // Creating two galaxies in a collision course.
        const int numParticlesPerGalaxy = 10000000;
        const float centerOfGalaxyX = 0.25f;
        const float centerOfGalaxyY = 0.25f;
        const float angularVelocityOfGalaxy = 0.4f;
        const float massPerParticle = 0.01f;
        const float radiusOfGalaxy = 0.2f;
        const float firstGalaxyCenterOfMassVelocityX = 0.01f;
        const float firstGalaxyCenterOfMassVelocityY = 0.01f;
        const float secondGalaxyCenterOfMassVelocityX = -0.01f;
        const float secondGalaxyCenterOfMassVelocityY = -0.01f;
        cosmos.addGalaxy(numParticlesPerGalaxy, centerOfGalaxyX, centerOfGalaxyY, angularVelocityOfGalaxy, massPerParticle, radiusOfGalaxy, firstGalaxyCenterOfMassVelocityX, firstGalaxyCenterOfMassVelocityY);
        cosmos.addGalaxy(numParticlesPerGalaxy, centerOfGalaxyX + 0.50f, centerOfGalaxyY + 0.50f, angularVelocityOfGalaxy, massPerParticle, radiusOfGalaxy, secondGalaxyCenterOfMassVelocityX, secondGalaxyCenterOfMassVelocityY);
    }

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