#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
// Optionally overriding the constants of simulation.
#define __OVERRIDE_CONSTANTS__
namespace OVERRIDE_CONSTANTS {
    // Number of threads per CUDA block. This is for 1536 resident threads per SM. For older GPUs, 512 or 1024 can be chosen.
    constexpr int THREADS = 768;
    // Number of CUDA devices (max 2 tested). Increase this to use more gpus and choose relevant device index in deviceIndex parameter of Universe constructor. Also use performance hint inside devicePerformance parameter of constructor.
    constexpr int NUM_CUDA_DEVICES = 1;

    // FFT uses these (long-range force calculation)
    // N is width of lattice (N x N) and can be only a power of 2. Higher value increases accuracy at the cost of performance.
    constexpr int N = 2048;

    // Time-step of simulation. Lower values increase accuracy.
    constexpr float dt = 0.004f;
    // Force-multiplier for particles.
    constexpr float gravityMultiplier = 1.0f;

    // For render buffer output. Asynchronously filled.
    constexpr int MAX_FRAMES_BUFFERED = 40;
    // Blur radius in lattice-cell units
    constexpr int BLUR_R = 3;
}
#include "CosmosCuda.cuh"
int main() {
    cv::namedWindow("Fast Nbody");
    // Multiple time-steps can be computed before each render.
    constexpr int NUM_TIME_STEPS_PER_RENDER = 1;
    // 100M particles require 2.3GB memory. Distributed to multiple gpus (in same ratio with devicePerformance[Constants::NUM_CUDA_DEVICES]).
    const int maximumParticles = 1000 * 1000 * 80;
    // Indices of CUDA devices to use. When both are same, single device computes all particles. When different devices selected, load-balancing between two devices is made.
    // The algorithm is only scalable to few GPUs for simplicity.
    // Use device index with fastest PCIE connection as the first value here (in case of multi-gpu)
    // const int deviceIndex[Constants::NUM_CUDA_DEVICES] = { 1, 0 }; if device=1 has faster PCIE -------> Multi GPU
    // const int deviceIndex[Constants::NUM_CUDA_DEVICES] = { 0, 1 }; if device=0 has faster PCIE -------> Multi GPU
    // const int deviceIndex[Constants::NUM_CUDA_DEVICES] = { 0 }; --------------------------------------> Single GPU
    // const int deviceIndex[Constants::NUM_CUDA_DEVICES] = { 0, 0 }; -----------------------------------> Single GPU but has extra i/o that is overlapped with compute
    const int deviceIndex[Constants::NUM_CUDA_DEVICES] = { 0 };
    // Expected relative performance of devices. It's normalized internally. { 0.5f, 0.5f } for dual identical gpus, { 0.33f, 0.33f, 0.33f } for 3 identical gpus.
    const float devicePerformance[Constants::NUM_CUDA_DEVICES] = { 1.0f };
    // true = more performance + single force sampling + single mass projection + pure FFT convolution
    // false = multi sampled forces per particle + multi-point mass projection per particle + FFT + local convolution
    const bool lowAccuracy = true;
    // Window width/height
    const int w = 1340;
    const int h = 1340;
    Universe<NUM_TIME_STEPS_PER_RENDER> cosmos(maximumParticles, deviceIndex, devicePerformance, lowAccuracy);
    const bool galaxyCollisionScenario = true;
    if (galaxyCollisionScenario) {
        cosmos.clear();
        // Creating two galaxies in a collision course.
        const int numParticlesPerGalaxy = 1000 * 1000 * 40;
        const float centerOfGalaxyX = 0.25f;
        const float centerOfGalaxyY = 0.25f;
        const float angularVelocityOfGalaxy = 0.6f;
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
    int frameCount = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    while(true){
        bool ready;
        std::vector<float> frame = cosmos.popFrame(ready);

        if (ready) {
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
            if (frameCount++ % 100 == 99) {
                auto t1 = std::chrono::high_resolution_clock::now();
                auto elapsedSeconds = std::chrono::duration<double>(t1 - t0).count();
                float fps = frameCount / elapsedSeconds;
                std::cout << "Aggregate FPS=" << fps<< " SPS="<< (NUM_TIME_STEPS_PER_RENDER * fps) << std::endl;
                t0 = std::chrono::high_resolution_clock::now();
                frameCount = 0;
            }
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