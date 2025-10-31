#include "CosmosCuda.cuh"
#include <opencv2/opencv.hpp>
#include <thread>
// Constants that can be changed:
// Constants::N --> lattice size
// Constants::THREADS --> cuda threads per cuda block
// Constants::dt --> time step
// Constants::MAX_FRAMES_BUFFERED --> maximum number of frames stored in queue

int main() {
    cv::namedWindow("Fast Nbody");
    // Multiple time-steps can be computed before each render.
    const int numNbodySimulationsPerRender = 1;
    // 100M particles require 2.5GB memory
    const int maximumParticles = 1000 * 1000 * 24;
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
        const int numParticlesPerGalaxy = 1000 * 1000 * 12;
        const float centerOfGalaxyX = 0.25f;
        const float centerOfGalaxyY = 0.25f;
        const float angularVelocityOfGalaxy = 1.0f;
        const float massPerParticle = 1.0f;
        const float radiusOfGalaxy = 0.2f;
        const float firstGalaxyCenterOfMassVelocityX = 0.01f;
        const float firstGalaxyCenterOfMassVelocityY = 0.01f;
        const float secondGalaxyCenterOfMassVelocityX = -0.02f;
        const float secondGalaxyCenterOfMassVelocityY = -0.02f;
        const bool addBlackHoleToCenter = true;
        cosmos.addGalaxy(numParticlesPerGalaxy, centerOfGalaxyX, centerOfGalaxyY, angularVelocityOfGalaxy, massPerParticle, radiusOfGalaxy, firstGalaxyCenterOfMassVelocityX, firstGalaxyCenterOfMassVelocityY, addBlackHoleToCenter);
        cosmos.addGalaxy(numParticlesPerGalaxy, centerOfGalaxyX + 0.50f, centerOfGalaxyY + 0.50f, angularVelocityOfGalaxy, massPerParticle, radiusOfGalaxy, secondGalaxyCenterOfMassVelocityX, secondGalaxyCenterOfMassVelocityY, addBlackHoleToCenter);
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
            memcpy(mat.data, frame.data(), sizeof(float) * n);
           
            cv::Mat resized;
            cv::Mat resizedColored;
            cv::resize(mat, resized, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
            resized.convertTo(resizedColored, CV_8UC3, 255.0f);
            cv::applyColorMap(resizedColored, resizedColored, cv::COLORMAP_JET);
            cv::imshow("Fast Nbody", resizedColored);
            
            // ESC = exit
            if (cv::waitKey(1) == 27) {
                    break;
            }
        }
    }
    // Stop cpu thread that is generating frames.
    cosmos.nBodyStop();
    cv::destroyAllWindows();
    return 0;
}