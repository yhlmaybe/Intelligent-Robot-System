#include "InputPreprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace AttentionGuideReconstruction
{

    InputPreprocessor::InputPreprocessor()
        : config_()
    {
    }

    InputPreprocessor::InputPreprocessor(PreprocessConfig config)
        : config_(config)
    {
    }

    void InputPreprocessor::SetConfig(PreprocessConfig config)
    {
        config_ = config;

        while (cachedFrames_.size() > config_.maxCachedFrames)
        {
            cachedFrames_.pop_front();
        }
    }

    std::shared_ptr<InputPreprocessor::RawFrame> InputPreprocessor::ReceiveFrame(ColorImage colorImage, DepthImage depthImage, double timestampSec, std::uint64_t frameId, std::string *errorMessage)
    {
        if (!std::isfinite(timestampSec) || timestampSec < 0.0)
        {
            if (errorMessage)
            {
                *errorMessage = "timestampSec must be a non-negative finite value.";
            }
            return nullptr;
        }

        if (!ValidateColorImage(colorImage, errorMessage) || !ValidateDepthImage(depthImage, errorMessage))
        {
            return nullptr;
        }

        std::shared_ptr<RawFrame> frame = std::make_shared<RawFrame>();
        frame->frameId = frameId;
        frame->timestampSec = timestampSec;
        frame->colorImage = std::move(colorImage);
        frame->depthImage = std::move(depthImage);
        return frame;
    }

    bool InputPreprocessor::ProcessDepth(std::shared_ptr<RawFrame> frame, std::string *errorMessage)
    {
        if (!frame)
        {
            if (errorMessage)
            {
                *errorMessage = "frame is null.";
            }
            return false;
        }

        if (!ValidateDepthImage(frame->depthImage, errorMessage))
        {
            return false;
        }

        frame->depthMeters = DecodeDepthImage(frame->depthImage);
        if (frame->depthMeters.empty())
        {
            if (errorMessage)
            {
                *errorMessage = "Failed to decode depth image.";
            }
            return false;
        }

        if (config_.enableMedianFilter)
        {
            frame->depthMeters = ApplyMedianFilter(
                std::move(frame->depthMeters),
                frame->depthImage.width,
                frame->depthImage.height,
                config_.medianFilterRadius);
        }

        if (config_.enableHoleFill)
        {
            frame->depthMeters = FillDepthHoles(
                std::move(frame->depthMeters),
                frame->depthImage.width,
                frame->depthImage.height,
                config_.holeFillRadius,
                config_.minHoleFillNeighbors);
        }

        return true;
    }

    bool InputPreprocessor::AlignFrame(std::shared_ptr<RawFrame> frame, std::string *errorMessage)
    {
        if (!frame)
        {
            if (errorMessage)
            {
                *errorMessage = "frame is null.";
            }
            return false;
        }

        if (!ValidateColorImage(frame->colorImage, errorMessage) || !ValidateDepthImage(frame->depthImage, errorMessage))
        {
            return false;
        }

        frame->alignedColorImage = ResizeColorImage(
            frame->colorImage,
            frame->depthImage.width,
            frame->depthImage.height,
            config_.alignmentOffsetX,
            config_.alignmentOffsetY);

        return true;
    }

    std::shared_ptr<InputPreprocessor::ProcessedFrame> InputPreprocessor::PackFrame(std::shared_ptr<RawFrame> frame, std::string *errorMessage)
    {
        if (!frame)
        {
            if (errorMessage)
            {
                *errorMessage = "frame is null.";
            }
            return nullptr;
        }

        if (frame->depthMeters.empty())
        {
            if (errorMessage)
            {
                *errorMessage = "ProcessDepth must be called before PackFrame.";
            }
            return nullptr;
        }

        if (frame->alignedColorImage.data.empty())
        {
            if (!AlignFrame(frame, errorMessage))
            {
                return nullptr;
            }
        }

        std::vector<float> rawDepthMeters = DecodeDepthImage(frame->depthImage);
        if (rawDepthMeters.empty())
        {
            if (errorMessage)
            {
                *errorMessage = "Failed to decode raw depth image during PackFrame.";
            }
            return nullptr;
        }

        std::shared_ptr<ProcessedFrame> processedFrame = std::make_shared<ProcessedFrame>();
        processedFrame->frameId = frame->frameId;
        processedFrame->timestampSec = frame->timestampSec;
        processedFrame->colorImage = frame->alignedColorImage;
        processedFrame->depthMeters = frame->depthMeters;
        processedFrame->deltaTimeFromPreviousFrameSec = cachedFrames_.empty() ? 0.0 : frame->timestampSec - cachedFrames_.back()->timestampSec;

        ComputeFrameStatistics(
            std::move(rawDepthMeters),
            processedFrame->depthMeters,
            processedFrame->rawValidDepthRatio,
            processedFrame->processedValidDepthRatio,
            processedFrame->filteredPixelCount,
            processedFrame->holeFilledPixelCount);

        return processedFrame;
    }

    bool InputPreprocessor::CacheFrame(std::shared_ptr<ProcessedFrame> frame, std::string *errorMessage)
    {
        if (!frame)
        {
            if (errorMessage)
            {
                *errorMessage = "processed frame is null.";
            }
            return false;
        }

        cachedFrames_.push_back(frame);
        while (cachedFrames_.size() > config_.maxCachedFrames)
        {
            cachedFrames_.pop_front();
        }

        return true;
    }

    std::shared_ptr<InputPreprocessor::ProcessedFrame> InputPreprocessor::RunPipeline(ColorImage colorImage, DepthImage depthImage, double timestampSec, std::uint64_t frameId, std::string *errorMessage)
    {
        std::shared_ptr<RawFrame> frame = ReceiveFrame(std::move(colorImage), std::move(depthImage), timestampSec, frameId, errorMessage);
        if (!frame)
        {
            return nullptr;
        }

        if (!ProcessDepth(frame, errorMessage))
        {
            return nullptr;
        }

        if (!AlignFrame(frame, errorMessage))
        {
            return nullptr;
        }

        std::shared_ptr<ProcessedFrame> processedFrame = PackFrame(frame, errorMessage);
        if (!processedFrame)
        {
            return nullptr;
        }

        if (!CacheFrame(processedFrame, errorMessage))
        {
            return nullptr;
        }

        return processedFrame;
    }

    std::shared_ptr<InputPreprocessor::ProcessedFrame> InputPreprocessor::GetLatestFrame()
    {
        if (cachedFrames_.empty())
        {
            return nullptr;
        }

        return cachedFrames_.back();
    }

    std::vector<std::shared_ptr<InputPreprocessor::ProcessedFrame>> InputPreprocessor::GetFramesByTimeWindow(double beginTimestampSec, double endTimestampSec)
    {
        std::vector<std::shared_ptr<ProcessedFrame>> frames;
        if (beginTimestampSec > endTimestampSec)
        {
            return frames;
        }

        for (std::shared_ptr<ProcessedFrame> frame : cachedFrames_)
        {
            if (frame && frame->timestampSec >= beginTimestampSec && frame->timestampSec <= endTimestampSec)
            {
                frames.push_back(frame);
            }
        }

        return frames;
    }

    void InputPreprocessor::ClearCache()
    {
        cachedFrames_.clear();
    }

    std::size_t InputPreprocessor::ToIndex(std::uint32_t x, std::uint32_t y, std::uint32_t width)
    {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
    }

    bool InputPreprocessor::IsValidDepth(float depth)
    {
        return std::isfinite(depth) && depth > 0.0f;
    }

    std::uint32_t InputPreprocessor::ClampCoordinate(int value, std::uint32_t upperBoundExclusive)
    {
        if (upperBoundExclusive == 0)
        {
            return 0;
        }

        if (value <= 0)
        {
            return 0;
        }

        const std::uint32_t maxValue = upperBoundExclusive - 1;
        if (static_cast<std::uint32_t>(value) >= upperBoundExclusive)
        {
            return maxValue;
        }

        return static_cast<std::uint32_t>(value);
    }

    std::uint32_t InputPreprocessor::GetColorChannels(ColorFormat format)
    {
        switch (format)
        {
        case ColorFormat::kGray8:
            return 1;
        case ColorFormat::kRGB8:
        case ColorFormat::kBGR8:
            return 3;
        case ColorFormat::kRGBA8:
        case ColorFormat::kBGRA8:
            return 4;
        default:
            return 0;
        }
    }

    std::uint32_t InputPreprocessor::GetDepthBytesPerPixel(DepthFormat format)
    {
        switch (format)
        {
        case DepthFormat::kUInt16Millimeters:
            return 2;
        case DepthFormat::kFloat32Meters:
            return 4;
        default:
            return 0;
        }
    }

    bool InputPreprocessor::ValidateColorImage(ColorImage colorImage, std::string *errorMessage)
    {
        if (colorImage.width == 0 || colorImage.height == 0)
        {
            if (errorMessage)
            {
                *errorMessage = "Color image width and height must be positive.";
            }
            return false;
        }

        std::uint32_t expectedChannels = GetColorChannels(colorImage.format);
        if (expectedChannels == 0)
        {
            if (errorMessage)
            {
                *errorMessage = "Unsupported color image format.";
            }
            return false;
        }

        if (colorImage.channels != expectedChannels)
        {
            if (errorMessage)
            {
                *errorMessage = "Color image channel count does not match the declared format.";
            }
            return false;
        }

        std::uint64_t minimumStride = static_cast<std::uint64_t>(colorImage.width) * colorImage.channels;
        if (colorImage.rowStrideBytes < minimumStride)
        {
            if (errorMessage)
            {
                *errorMessage = "Color image rowStrideBytes is too small.";
            }
            return false;
        }

        std::uint64_t requiredBytes = static_cast<std::uint64_t>(colorImage.rowStrideBytes) * colorImage.height;
        if (colorImage.data.size() < requiredBytes)
        {
            if (errorMessage)
            {
                *errorMessage = "Color image buffer is too small.";
            }
            return false;
        }

        return true;
    }

    bool InputPreprocessor::ValidateDepthImage(DepthImage depthImage, std::string *errorMessage)
    {
        if (depthImage.width == 0 || depthImage.height == 0)
        {
            if (errorMessage)
            {
                *errorMessage = "Depth image width and height must be positive.";
            }
            return false;
        }

        std::uint32_t bytesPerPixel = GetDepthBytesPerPixel(depthImage.format);
        if (bytesPerPixel == 0)
        {
            if (errorMessage)
            {
                *errorMessage = "Unsupported depth image format.";
            }
            return false;
        }

        std::uint64_t minimumStride = static_cast<std::uint64_t>(depthImage.width) * bytesPerPixel;
        if (depthImage.rowStrideBytes < minimumStride)
        {
            if (errorMessage)
            {
                *errorMessage = "Depth image rowStrideBytes is too small.";
            }
            return false;
        }

        std::uint64_t requiredBytes = static_cast<std::uint64_t>(depthImage.rowStrideBytes) * depthImage.height;
        if (depthImage.data.size() < requiredBytes)
        {
            if (errorMessage)
            {
                *errorMessage = "Depth image buffer is too small.";
            }
            return false;
        }

        if (!std::isfinite(depthImage.scaleToMeters) || depthImage.scaleToMeters <= 0.0f)
        {
            if (errorMessage)
            {
                *errorMessage = "Depth image scaleToMeters must be positive.";
            }
            return false;
        }

        return true;
    }

    std::vector<float> InputPreprocessor::DecodeDepthImage(DepthImage depthImage)
    {
        std::vector<float> depthMeters(static_cast<std::size_t>(depthImage.width) * depthImage.height, 0.0f);
        std::uint32_t bytesPerPixel = GetDepthBytesPerPixel(depthImage.format);
        if (bytesPerPixel == 0)
        {
            return depthMeters;
        }

        for (std::uint32_t y = 0; y < depthImage.height; ++y)
        {
            std::uint8_t *rowPointer = depthImage.data.data() + static_cast<std::size_t>(y) * depthImage.rowStrideBytes;
            for (std::uint32_t x = 0; x < depthImage.width; ++x)
            {
                std::uint8_t *pixelPointer = rowPointer + static_cast<std::size_t>(x) * bytesPerPixel;
                float depthValue = 0.0f;

                if (depthImage.format == DepthFormat::kUInt16Millimeters)
                {
                    std::uint16_t rawDepth = 0;
                    std::memcpy(&rawDepth, pixelPointer, sizeof(rawDepth));
                    depthValue = static_cast<float>(rawDepth) * depthImage.scaleToMeters;
                }
                else if (depthImage.format == DepthFormat::kFloat32Meters)
                {
                    float rawDepth = 0.0f;
                    std::memcpy(&rawDepth, pixelPointer, sizeof(rawDepth));
                    depthValue = rawDepth * depthImage.scaleToMeters;
                }

                if (!std::isfinite(depthValue) || depthValue < config_.minValidDepthMeters || depthValue > config_.maxValidDepthMeters)
                {
                    depthValue = 0.0f;
                }

                depthMeters[ToIndex(x, y, depthImage.width)] = depthValue;
            }
        }

        return depthMeters;
    }

    std::vector<float> InputPreprocessor::ApplyMedianFilter(std::vector<float> depthMeters, std::uint32_t width, std::uint32_t height, std::uint32_t radius)
    {
        if (radius == 0 || depthMeters.empty())
        {
            return depthMeters;
        }

        std::vector<float> filteredDepth = depthMeters;
        std::vector<float> values;
        values.reserve(static_cast<std::size_t>((radius * 2 + 1) * (radius * 2 + 1)));

        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                std::size_t centerIndex = ToIndex(x, y, width);
                if (!IsValidDepth(depthMeters[centerIndex]))
                {
                    continue;
                }

                values.clear();
                int minY = std::max<int>(0, static_cast<int>(y) - static_cast<int>(radius));
                int maxY = std::min<int>(static_cast<int>(height) - 1, static_cast<int>(y) + static_cast<int>(radius));
                int minX = std::max<int>(0, static_cast<int>(x) - static_cast<int>(radius));
                int maxX = std::min<int>(static_cast<int>(width) - 1, static_cast<int>(x) + static_cast<int>(radius));

                for (int sampleY = minY; sampleY <= maxY; ++sampleY)
                {
                    for (int sampleX = minX; sampleX <= maxX; ++sampleX)
                    {
                        float value = depthMeters[ToIndex(static_cast<std::uint32_t>(sampleX), static_cast<std::uint32_t>(sampleY), width)];
                        if (IsValidDepth(value))
                        {
                            values.push_back(value);
                        }
                    }
                }

                if (values.size() < 3)
                {
                    continue;
                }

                std::size_t medianIndex = values.size() / 2;
                std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(medianIndex), values.end());
                filteredDepth[centerIndex] = values[medianIndex];
            }
        }

        return filteredDepth;
    }

    std::vector<float> InputPreprocessor::FillDepthHoles(std::vector<float> depthMeters, std::uint32_t width, std::uint32_t height, std::uint32_t radius, std::uint32_t minNeighbors)
    {
        if (radius == 0 || depthMeters.empty())
        {
            return depthMeters;
        }

        std::vector<float> filledDepth = depthMeters;
        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                std::size_t centerIndex = ToIndex(x, y, width);
                if (IsValidDepth(depthMeters[centerIndex]))
                {
                    continue;
                }

                float sum = 0.0f;
                std::uint32_t validNeighbors = 0;
                int minY = std::max<int>(0, static_cast<int>(y) - static_cast<int>(radius));
                int maxY = std::min<int>(static_cast<int>(height) - 1, static_cast<int>(y) + static_cast<int>(radius));
                int minX = std::max<int>(0, static_cast<int>(x) - static_cast<int>(radius));
                int maxX = std::min<int>(static_cast<int>(width) - 1, static_cast<int>(x) + static_cast<int>(radius));

                for (int sampleY = minY; sampleY <= maxY; ++sampleY)
                {
                    for (int sampleX = minX; sampleX <= maxX; ++sampleX)
                    {
                        float value = depthMeters[ToIndex(static_cast<std::uint32_t>(sampleX), static_cast<std::uint32_t>(sampleY), width)];
                        if (IsValidDepth(value))
                        {
                            sum += value;
                            ++validNeighbors;
                        }
                    }
                }

                if (validNeighbors >= minNeighbors)
                {
                    filledDepth[centerIndex] = sum / static_cast<float>(validNeighbors);
                }
            }
        }

        return filledDepth;
    }

    InputPreprocessor::ColorImage InputPreprocessor::ResizeColorImage(ColorImage colorImage, std::uint32_t targetWidth, std::uint32_t targetHeight, int offsetX, int offsetY)
    {
        ColorImage resizedImage;
        resizedImage.width = targetWidth;
        resizedImage.height = targetHeight;
        resizedImage.channels = colorImage.channels;
        resizedImage.rowStrideBytes = targetWidth * colorImage.channels;
        resizedImage.format = colorImage.format;
        resizedImage.data.resize(static_cast<std::size_t>(resizedImage.rowStrideBytes) * targetHeight, 0U);

        if (targetWidth == 0 || targetHeight == 0 || colorImage.width == 0 || colorImage.height == 0)
        {
            return resizedImage;
        }

        for (std::uint32_t targetY = 0; targetY < targetHeight; ++targetY)
        {
            std::uint32_t sourceY = ClampCoordinate(
                static_cast<int>((static_cast<std::uint64_t>(targetY) * colorImage.height) / targetHeight) + offsetY,
                colorImage.height);
            for (std::uint32_t targetX = 0; targetX < targetWidth; ++targetX)
            {
                std::uint32_t sourceX = ClampCoordinate(
                    static_cast<int>((static_cast<std::uint64_t>(targetX) * colorImage.width) / targetWidth) + offsetX,
                    colorImage.width);

                std::uint8_t *sourcePixel = colorImage.data.data() + static_cast<std::size_t>(sourceY) * colorImage.rowStrideBytes + static_cast<std::size_t>(sourceX) * colorImage.channels;
                std::uint8_t *targetPixel = resizedImage.data.data() + static_cast<std::size_t>(targetY) * resizedImage.rowStrideBytes + static_cast<std::size_t>(targetX) * resizedImage.channels;

                std::memcpy(targetPixel, sourcePixel, colorImage.channels);
            }
        }

        return resizedImage;
    }

    float InputPreprocessor::ComputeValidDepthRatio(std::vector<float> depthMeters)
    {
        if (depthMeters.empty())
        {
            return 0.0f;
        }

        std::size_t validCount = 0;
        for (float depth : depthMeters)
        {
            if (IsValidDepth(depth))
            {
                ++validCount;
            }
        }

        return static_cast<float>(validCount) / static_cast<float>(depthMeters.size());
    }

    void InputPreprocessor::ComputeFrameStatistics(std::vector<float> rawDepthMeters, std::vector<float> processedDepthMeters, float &rawValidDepthRatio, float &processedValidDepthRatio, std::uint32_t &filteredPixelCount, std::uint32_t &holeFilledPixelCount)
    {
        rawValidDepthRatio = ComputeValidDepthRatio(rawDepthMeters);
        processedValidDepthRatio = ComputeValidDepthRatio(processedDepthMeters);
        filteredPixelCount = 0;
        holeFilledPixelCount = 0;

        std::size_t count = std::min(rawDepthMeters.size(), processedDepthMeters.size());
        for (std::size_t index = 0; index < count; ++index)
        {
            bool rawValid = IsValidDepth(rawDepthMeters[index]);
            bool processedValid = IsValidDepth(processedDepthMeters[index]);

            if (!rawValid && processedValid)
            {
                ++holeFilledPixelCount;
            }
            else if (rawValid && processedValid && std::fabs(rawDepthMeters[index] - processedDepthMeters[index]) > 1e-5f)
            {
                ++filteredPixelCount;
            }
        }
    }

}
