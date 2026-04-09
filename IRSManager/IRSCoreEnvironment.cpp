#include "IRSCoreEnvironment.h"

namespace IRSCoreEnvironment
{
    EnvironmentalPerception::EnvironmentalPerception(unsigned interval_ms) : interval_ms_(interval_ms)
    {
    }

    EnvironmentalPerception *EnvironmentalPerception::GetInstance()
    {
        static EnvironmentalPerception *instance = new EnvironmentalPerception(100);
        return instance;
    }

    std::vector<Eigen::Vector3d> EnvironmentalPerception::GetPointClouds()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return cloud_points_;
    }

    void EnvironmentalPerception::Reset()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        cloud_points_.clear();
    }
}