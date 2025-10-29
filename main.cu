#include "CosmosCuda.cuh"
#include <opencv2/opencv.hpp>
#include <thread>
// Change these values to utilize your gpu better (currently they are tuned for RTX4070)
// Constants::BLOCKS
// Constants::THREADS
// Change this to another power-of-2 to tune accuracy of long-ranged forces. Short-ranged forces are not calculated accurately with FFT alone, so a direct-convolution is used to partially reduce error in there.
// Constants::N
int main() {
    cv::namedWindow("Fast Nbody");

    const int numNbodySimulationsPerRender = 2;
    // 100M particles require 2.5GB memory
    const int maximumParticles = 1000 * 1000 * 20;
    // cuda device index
    const int device = 0;
    // true = more performance + single force sampling + single mass projection + pure FFT convolution
    // false = multi sampled forces per particle + multi-point mass projection per particle + FFT + local convolution
    const bool lowAccuracy = true;
    // Window width/height
    const int w = 1340;
    const int h = 1340;
    Universe cosmos(maximumParticles, device, lowAccuracy, numNbodySimulationsPerRender);
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
    // For rendering output.
    cv::Mat mat = cv::Mat(cv::Size2i(cosmos.getLatticeSize(), cosmos.getLatticeSize()), CV_32FC1);

    // Start nbody thread.
    cosmos.nBodyStartGeneratingFrames();
    // Asynchronously reading the generated frames.
    int escTestCtr = 0;
    while(true){
        std::vector<float> frame = cosmos.popFrame();

        if (frame.size() > 0) {
            std::cout << "CUDA Nbody pipeline: " << frame[frame.size() - 1] << " milliseconds per step" << std::endl;
            // Clear render output.
            mat.setTo(cv::Scalar(0.0f));
            // Copy lattice to opencv mat.
            const int n = frame.size() - 1;
            
            for (int j = 0; j < n; j++) {
                const auto data = frame[j];
                mat.at<float>(j) = (data > 1.0f ? 1.0f : data);
            }
            
            cv::Mat resized;
            cv::resize(mat, resized, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
            cv::imshow("Fast Nbody", resized);
            
            // ESC = exit
            if (escTestCtr++ % 10 == 0) {
                if (cv::waitKey(1) == 27) {
                    break;
                }
            }
        }
    }
    // Stop cpu thread that is generating frames.
    cosmos.nBodyStop();
    cv::destroyAllWindows();
    return 0;
}