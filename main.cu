#include "CosmosCuda.cuh"
// Change these values to utilize your gpu better (currently they are tuned for RTX4070)
// Constants::BLOCKS
// Constants::THREADS
// Change this to another power-of-2 to tune accuracy of long-ranged forces. Short-ranged forces are not calculated currently (todo).
// Constants::N
int main() {
    const int numNbodySimulationsPerRender = 2;
    // 20 Bytes per particle is allocated.
    const int maximumParticles = 1000 * 1000 * 10;
    // cuda device index
    const int device = 0;
    // true: more performance
    const bool lowAccuracy = true;
    // Window width/height
    const int w = 1340;
    const int h = 1340;
    Universe cosmos(maximumParticles, device, lowAccuracy, w, h);
    const bool galaxyCollisionScenario = true;
    if (galaxyCollisionScenario) {
        cosmos.clear();
        // Creating two galaxies in a collision course.
        const int numParticlesPerGalaxy = 7000000;
        const float centerOfGalaxyX = 0.25f;
        const float centerOfGalaxyY = 0.25f;
        const float angularVelocityOfGalaxy = 10.8f;
        const float massPerParticle = 1.0f;
        const float radiusOfGalaxy = 0.2f;
        const float firstGalaxyCenterOfMassVelocityX = 0.01f;
        const float firstGalaxyCenterOfMassVelocityY = 0.01f;
        const float secondGalaxyCenterOfMassVelocityX = -0.01f;
        const float secondGalaxyCenterOfMassVelocityY = -0.01f;
        // This galaxy gets 7M particles.
        cosmos.addGalaxy(numParticlesPerGalaxy, centerOfGalaxyX, centerOfGalaxyY, angularVelocityOfGalaxy, massPerParticle, radiusOfGalaxy, firstGalaxyCenterOfMassVelocityX, firstGalaxyCenterOfMassVelocityY);
        // Remaining particles (3M) are left for this galaxy.
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