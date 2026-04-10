#include "InputPreprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace AttentionGuideReconstruction
{

    InputPreprocessor::InputPreprocessor()
        : config_()
    {
    }

    InputPreprocessor::InputPreprocessor(Config config)
        : config_(config)
    {
    }

    InputPreprocessor::DepthProcessResult InputPreprocessor::Process(DepthImage depthImage)
    {
        DepthProcessResult result;
        result.width = depthImage.width;
        result.height = depthImage.height;
        result.sourceFormat = depthImage.format;

        if (!ValidateDepthImage(depthImage, &result.errorMessage))
        {
            return result;
        }

        result.depthMeters = DecodeDepthImage(depthImage);
        if (result.depthMeters.size() != static_cast<std::size_t>(depthImage.width) * static_cast<std::size_t>(depthImage.height))
        {
            result.errorMessage = "Failed to decode depth image.";
            result.depthMeters.clear();
            return result;
        }

        ApplyDepthRange(&result.depthMeters);

        if (config_.enableMedianFilter && config_.medianFilterRadius > 0)
        {
            result.depthMeters = ApplyMedianFilter(
                std::move(result.depthMeters),
                depthImage.width,
                depthImage.height,
                config_.medianFilterRadius,
                &result.filteredPixelCount);
        }

        if (config_.enableHoleFill && config_.holeFillRadius > 0)
        {
            result.depthMeters = FillDepthHoles(
                std::move(result.depthMeters),
                depthImage.width,
                depthImage.height,
                config_.holeFillRadius,
                config_.minHoleFillNeighbors,
                &result.holeFilledPixelCount);
        }

        ComputeStatistics(
            result.depthMeters,
            &result.validDepthRatio,
            &result.minDepthMeters,
            &result.maxDepthMeters);

        result.success = true;
        return result;
    }

    std::size_t InputPreprocessor::ToIndex(std::uint32_t x, std::uint32_t y, std::uint32_t width)
    {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
    }

    bool InputPreprocessor::IsValidDepth(float depth)
    {
        return std::isfinite(depth) && depth > 0.0f;
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

    bool InputPreprocessor::ValidateDepthImage(DepthImage depthImage, std::string *errorMessage)
    {
        if (depthImage.width == 0 || depthImage.height == 0)
        {
            if (errorMessage)
            {
                *errorMessage = "Depth image width and height must be greater than zero.";
            }
            return false;
        }

        if (depthImage.format == DepthFormat::kUnknown)
        {
            if (errorMessage)
            {
                *errorMessage = "Depth image format is unknown.";
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

        std::uint64_t minStride = static_cast<std::uint64_t>(depthImage.width) * static_cast<std::uint64_t>(bytesPerPixel);
        if (depthImage.rowStrideBytes < minStride)
        {
            if (errorMessage)
            {
                *errorMessage = "Depth image rowStrideBytes is smaller than the minimum required stride.";
            }
            return false;
        }

        std::uint64_t requiredBytes = static_cast<std::uint64_t>(depthImage.rowStrideBytes) * static_cast<std::uint64_t>(depthImage.height);
        if (depthImage.data.size() < requiredBytes)
        {
            if (errorMessage)
            {
                *errorMessage = "Depth image data buffer is smaller than rowStrideBytes * height.";
            }
            return false;
        }

        if (depthImage.format == DepthFormat::kUInt16Millimeters && (!std::isfinite(depthImage.scaleToMeters) || depthImage.scaleToMeters <= 0.0f))
        {
            if (errorMessage)
            {
                *errorMessage = "Depth image scaleToMeters must be a positive finite value for uint16 depth input.";
            }
            return false;
        }

        return true;
    }

    std::vector<float> InputPreprocessor::DecodeDepthImage(DepthImage depthImage)
    {
        std::vector<float> depthMeters(static_cast<std::size_t>(depthImage.width) * static_cast<std::size_t>(depthImage.height), 0.0f);

        for (std::uint32_t y = 0; y < depthImage.height; ++y)
        {
            const std::uint8_t *row = depthImage.data.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(depthImage.rowStrideBytes);
            for (std::uint32_t x = 0; x < depthImage.width; ++x)
            {
                std::size_t index = ToIndex(x, y, depthImage.width);
                if (depthImage.format == DepthFormat::kUInt16Millimeters)
                {
                    std::uint16_t rawDepth = 0;
                    std::memcpy(&rawDepth, row + static_cast<std::size_t>(x) * 2, sizeof(std::uint16_t));
                    depthMeters[index] = rawDepth == 0 ? 0.0f : static_cast<float>(rawDepth) * depthImage.scaleToMeters;
                }
                else if (depthImage.format == DepthFormat::kFloat32Meters)
                {
                    float rawDepth = 0.0f;
                    std::memcpy(&rawDepth, row + static_cast<std::size_t>(x) * 4, sizeof(float));
                    depthMeters[index] = rawDepth;
                }
            }
        }

        return depthMeters;
    }

    void InputPreprocessor::ApplyDepthRange(std::vector<float> *depthMeters)
    {
        if (!depthMeters)
        {
            return;
        }

        for (float &depth : *depthMeters)
        {
            if (!std::isfinite(depth) || depth < config_.minValidDepthMeters || depth > config_.maxValidDepthMeters)
            {
                depth = 0.0f;
            }
        }
    }

    std::vector<float> InputPreprocessor::ApplyMedianFilter(std::vector<float> depthMeters, std::uint32_t width, std::uint32_t height, std::uint32_t radius, std::uint32_t *changedPixelCount)
    {
        if (changedPixelCount)
        {
            *changedPixelCount = 0;
        }

        if (depthMeters.empty() || radius == 0)
        {
            return depthMeters;
        }

        std::vector<float> source = depthMeters;
        std::vector<float> neighbors;

        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                std::size_t index = ToIndex(x, y, width);
                if (!IsValidDepth(source[index]))
                {
                    continue;
                }

                neighbors.clear();
                std::int32_t xBegin = static_cast<std::int32_t>(x) - static_cast<std::int32_t>(radius);
                std::int32_t xEnd = static_cast<std::int32_t>(x) + static_cast<std::int32_t>(radius);
                std::int32_t yBegin = static_cast<std::int32_t>(y) - static_cast<std::int32_t>(radius);
                std::int32_t yEnd = static_cast<std::int32_t>(y) + static_cast<std::int32_t>(radius);

                for (std::int32_t ny = yBegin; ny <= yEnd; ++ny)
                {
                    if (ny < 0 || ny >= static_cast<std::int32_t>(height))
                    {
                        continue;
                    }
                    for (std::int32_t nx = xBegin; nx <= xEnd; ++nx)
                    {
                        if (nx < 0 || nx >= static_cast<std::int32_t>(width))
                        {
                            continue;
                        }

                        float neighborDepth = source[ToIndex(static_cast<std::uint32_t>(nx), static_cast<std::uint32_t>(ny), width)];
                        if (IsValidDepth(neighborDepth))
                        {
                            neighbors.push_back(neighborDepth);
                        }
                    }
                }

                if (neighbors.empty())
                {
                    continue;
                }

                std::size_t medianIndex = neighbors.size() / 2;
                std::nth_element(neighbors.begin(), neighbors.begin() + static_cast<std::ptrdiff_t>(medianIndex), neighbors.end());
                float filteredDepth = neighbors[medianIndex];
                if (std::fabs(filteredDepth - source[index]) > 1e-6f && changedPixelCount)
                {
                    *changedPixelCount += 1;
                }
                depthMeters[index] = filteredDepth;
            }
        }

        return depthMeters;
    }

    std::vector<float> InputPreprocessor::FillDepthHoles(std::vector<float> depthMeters, std::uint32_t width, std::uint32_t height, std::uint32_t radius, std::uint32_t minNeighbors, std::uint32_t *filledPixelCount)
    {
        if (filledPixelCount)
        {
            *filledPixelCount = 0;
        }

        if (depthMeters.empty() || radius == 0)
        {
            return depthMeters;
        }

        std::vector<float> source = depthMeters;
        std::vector<float> neighbors;

        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                std::size_t index = ToIndex(x, y, width);
                if (IsValidDepth(source[index]))
                {
                    continue;
                }

                neighbors.clear();
                std::int32_t xBegin = static_cast<std::int32_t>(x) - static_cast<std::int32_t>(radius);
                std::int32_t xEnd = static_cast<std::int32_t>(x) + static_cast<std::int32_t>(radius);
                std::int32_t yBegin = static_cast<std::int32_t>(y) - static_cast<std::int32_t>(radius);
                std::int32_t yEnd = static_cast<std::int32_t>(y) + static_cast<std::int32_t>(radius);

                for (std::int32_t ny = yBegin; ny <= yEnd; ++ny)
                {
                    if (ny < 0 || ny >= static_cast<std::int32_t>(height))
                    {
                        continue;
                    }
                    for (std::int32_t nx = xBegin; nx <= xEnd; ++nx)
                    {
                        if (nx < 0 || nx >= static_cast<std::int32_t>(width))
                        {
                            continue;
                        }

                        float neighborDepth = source[ToIndex(static_cast<std::uint32_t>(nx), static_cast<std::uint32_t>(ny), width)];
                        if (IsValidDepth(neighborDepth))
                        {
                            neighbors.push_back(neighborDepth);
                        }
                    }
                }

                if (neighbors.size() < static_cast<std::size_t>(minNeighbors))
                {
                    continue;
                }

                float sumDepth = 0.0f;
                for (float neighborDepth : neighbors)
                {
                    sumDepth += neighborDepth;
                }

                depthMeters[index] = sumDepth / static_cast<float>(neighbors.size());
                if (filledPixelCount)
                {
                    *filledPixelCount += 1;
                }
            }
        }

        return depthMeters;
    }

    void InputPreprocessor::ComputeStatistics(std::vector<float> depthMeters, float *validDepthRatio, float *minDepthMeters, float *maxDepthMeters)
    {
        if (validDepthRatio)
        {
            *validDepthRatio = 0.0f;
        }
        if (minDepthMeters)
        {
            *minDepthMeters = 0.0f;
        }
        if (maxDepthMeters)
        {
            *maxDepthMeters = 0.0f;
        }

        if (depthMeters.empty())
        {
            return;
        }

        std::size_t validCount = 0;
        float minDepth = std::numeric_limits<float>::max();
        float maxDepth = 0.0f;

        for (float depth : depthMeters)
        {
            if (!IsValidDepth(depth))
            {
                continue;
            }

            validCount += 1;
            minDepth = std::min(minDepth, depth);
            maxDepth = std::max(maxDepth, depth);
        }

        if (validDepthRatio)
        {
            *validDepthRatio = static_cast<float>(validCount) / static_cast<float>(depthMeters.size());
        }

        if (validCount == 0)
        {
            return;
        }

        if (minDepthMeters)
        {
            *minDepthMeters = minDepth;
        }
        if (maxDepthMeters)
        {
            *maxDepthMeters = maxDepth;
        }
    }

} // namespace AttentionGuideReconstruction
