#ifndef IRSCOREENVIRONMENT_H
#define IRSCOREENVIRONMENT_H

#include <Eigen/Geometry>
#include <mutex>
#include <vector>

#include "../include/IRSFunction.h"

namespace IRSCoreEnvironment
{

    class EnvironmentalPerception : public IRSThreadTools::IRSThreadBase
    {
    public:
        EnvironmentalPerception(unsigned interval_ms = 100);

        static EnvironmentalPerception *GetInstance();

        std::vector<Eigen::Vector3d> GetPointClouds();

        void Reset();

    private:
        unsigned interval_ms_;
        std::vector<Eigen::Vector3d> cloud_points_;
        mutable std::mutex mtx_;

        EnvironmentalPerception(const EnvironmentalPerception &) = delete;
        EnvironmentalPerception &operator=(const EnvironmentalPerception &) = delete;
    };
} // namespace IRSCoreEnvironment

#endif