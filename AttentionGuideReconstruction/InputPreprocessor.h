#ifndef ATTENTIONGUIDERECONSTRUCTION_INPUTPREPROCESSOR_H
#define ATTENTIONGUIDERECONSTRUCTION_INPUTPREPROCESSOR_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace AttentionGuideReconstruction
{

    class InputPreprocessor
    {
    public:
        enum class DepthFormat : std::uint8_t
        {
            kUnknown = 0,
            kUInt16Millimeters,
            kFloat32Meters
        };

        struct DepthImage
        {
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint32_t rowStrideBytes = 0;
            DepthFormat format = DepthFormat::kUnknown;
            float scaleToMeters = 0.001f;
            std::vector<std::uint8_t> data;
        };

        struct Config
        {
            float minValidDepthMeters = 0.05f;
            float maxValidDepthMeters = 5.0f;
            bool enableMedianFilter = true;
            std::uint32_t medianFilterRadius = 1;
            bool enableHoleFill = true;
            std::uint32_t holeFillRadius = 1;
            std::uint32_t minHoleFillNeighbors = 3;
        };

        struct DepthProcessResult
        {
            bool success = false;
            std::string errorMessage;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            DepthFormat sourceFormat = DepthFormat::kUnknown;
            std::vector<float> depthMeters;
            float validDepthRatio = 0.0f;
            float minDepthMeters = 0.0f;
            float maxDepthMeters = 0.0f;
            std::uint32_t filteredPixelCount = 0;
            std::uint32_t holeFilledPixelCount = 0;
        };

        InputPreprocessor();
        explicit InputPreprocessor(Config config);

        DepthProcessResult Process(DepthImage depthImage);

    private:
        std::size_t ToIndex(std::uint32_t x, std::uint32_t y, std::uint32_t width);
        bool IsValidDepth(float depth);
        std::uint32_t GetDepthBytesPerPixel(DepthFormat format);
        bool ValidateDepthImage(DepthImage depthImage, std::string *errorMessage);
        std::vector<float> DecodeDepthImage(DepthImage depthImage);
        void ApplyDepthRange(std::vector<float> *depthMeters);
        std::vector<float> ApplyMedianFilter(std::vector<float> depthMeters, std::uint32_t width, std::uint32_t height, std::uint32_t radius, std::uint32_t *changedPixelCount);
        std::vector<float> FillDepthHoles(std::vector<float> depthMeters, std::uint32_t width, std::uint32_t height, std::uint32_t radius, std::uint32_t minNeighbors, std::uint32_t *filledPixelCount);
        void ComputeStatistics(std::vector<float> depthMeters, float *validDepthRatio, float *minDepthMeters, float *maxDepthMeters);

        Config config_;
    };

} // namespace AttentionGuideReconstruction

#endif
