#ifndef ATTENTIONGUIDERECONSTRUCTION_INPUTPREPROCESSOR_H
#define ATTENTIONGUIDERECONSTRUCTION_INPUTPREPROCESSOR_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace AttentionGuideReconstruction
{

    class InputPreprocessor
    {
    public:
        enum class ColorFormat : std::uint8_t
        {
            kUnknown = 0,
            kGray8,
            kRGB8,
            kBGR8,
            kRGBA8,
            kBGRA8
        };

        enum class DepthFormat : std::uint8_t
        {
            kUnknown = 0,
            kUInt16Millimeters,
            kFloat32Meters
        };

        struct ColorImage
        {
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint32_t channels = 0;
            std::uint32_t rowStrideBytes = 0;
            ColorFormat format = ColorFormat::kUnknown;
            std::vector<std::uint8_t> data;
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

        struct PreprocessConfig
        {
            std::size_t maxCachedFrames = 12;
            std::uint32_t medianFilterRadius = 1;
            std::uint32_t holeFillRadius = 1;
            std::uint32_t minHoleFillNeighbors = 3;
            float minValidDepthMeters = 0.05f;
            float maxValidDepthMeters = 5.0f;
            bool enableMedianFilter = true;
            bool enableHoleFill = true;
            int alignmentOffsetX = 0;
            int alignmentOffsetY = 0;
        };

        struct RawFrame
        {
            std::uint64_t frameId = 0;
            double timestampSec = 0.0;
            ColorImage colorImage;
            DepthImage depthImage;
            std::vector<float> depthMeters;
            ColorImage alignedColorImage;
        };

        struct ProcessedFrame
        {
            std::uint64_t frameId = 0;
            double timestampSec = 0.0;
            ColorImage colorImage;
            std::vector<float> depthMeters;
            float rawValidDepthRatio = 0.0f;
            float processedValidDepthRatio = 0.0f;
            std::uint32_t filteredPixelCount = 0;
            std::uint32_t holeFilledPixelCount = 0;
            double deltaTimeFromPreviousFrameSec = 0.0;
        };

        InputPreprocessor();
        InputPreprocessor(PreprocessConfig config);

        void SetConfig(PreprocessConfig config);
        std::shared_ptr<RawFrame> ReceiveFrame(ColorImage colorImage, DepthImage depthImage, double timestampSec, std::uint64_t frameId, std::string *errorMessage = nullptr);
        bool ProcessDepth(std::shared_ptr<RawFrame> frame, std::string *errorMessage = nullptr);
        bool AlignFrame(std::shared_ptr<RawFrame> frame, std::string *errorMessage = nullptr);
        std::shared_ptr<ProcessedFrame> PackFrame(std::shared_ptr<RawFrame> frame, std::string *errorMessage = nullptr);
        bool CacheFrame(std::shared_ptr<ProcessedFrame> frame, std::string *errorMessage = nullptr);
        std::shared_ptr<ProcessedFrame> RunPipeline(ColorImage colorImage, DepthImage depthImage, double timestampSec, std::uint64_t frameId, std::string *errorMessage = nullptr);
        std::shared_ptr<ProcessedFrame> GetLatestFrame();
        std::vector<std::shared_ptr<ProcessedFrame>> GetFramesByTimeWindow(double beginTimestampSec, double endTimestampSec);
        void ClearCache();

    private:
        std::size_t ToIndex(std::uint32_t x, std::uint32_t y, std::uint32_t width);
        bool IsValidDepth(float depth);
        std::uint32_t ClampCoordinate(int value, std::uint32_t upperBoundExclusive);
        std::uint32_t GetColorChannels(ColorFormat format);
        std::uint32_t GetDepthBytesPerPixel(DepthFormat format);
        bool ValidateColorImage(ColorImage colorImage, std::string *errorMessage);
        bool ValidateDepthImage(DepthImage depthImage, std::string *errorMessage);
        std::vector<float> DecodeDepthImage(DepthImage depthImage);
        std::vector<float> ApplyMedianFilter(std::vector<float> depthMeters, std::uint32_t width, std::uint32_t height, std::uint32_t radius);
        std::vector<float> FillDepthHoles(std::vector<float> depthMeters, std::uint32_t width, std::uint32_t height, std::uint32_t radius, std::uint32_t minNeighbors);
        ColorImage ResizeColorImage(ColorImage colorImage, std::uint32_t targetWidth, std::uint32_t targetHeight, int offsetX, int offsetY);
        float ComputeValidDepthRatio(std::vector<float> depthMeters);
        void ComputeFrameStatistics(std::vector<float> rawDepthMeters, std::vector<float> processedDepthMeters, float &rawValidDepthRatio, float &processedValidDepthRatio, std::uint32_t &filteredPixelCount, std::uint32_t &holeFilledPixelCount);

        PreprocessConfig config_;
        std::deque<std::shared_ptr<ProcessedFrame>> cachedFrames_;
    };

}

#endif
