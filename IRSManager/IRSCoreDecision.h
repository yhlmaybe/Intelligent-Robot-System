#ifndef IRSCOREDECISION_H
#define IRSCOREDECISION_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "../include/IRSFunction.h"

namespace IRSCoreDecision
{
    class Keyboard104PoseStore
    {
    public:
        struct KeyPoseData
        {
            bool present = false;
            Eigen::Vector3d localPressPoint = Eigen::Vector3d::Zero();
            Eigen::Vector3d worldPressPoint = Eigen::Vector3d::Zero();
        };

        Keyboard104PoseStore();

        bool Load(std::string filePath = "");

        void Reset();

        bool HasKey(std::string keyName);

        bool GetWorldPressPoint(std::string keyName, Eigen::Vector3d &point);

        std::string GetLoadedFilePath();

    private:
        struct KeyboardPoseData
        {
            Eigen::Vector3d position = Eigen::Vector3d::Zero();
            Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
        };

        mutable std::mutex dataMutex_;
        std::string loadedFilePath_;
        KeyboardPoseData keyboardPose_;
        std::map<std::string, KeyPoseData> keyPoseMap_;
    };

    class BrainDecisionNode : public IRSThreadTools::IRSThreadBase
    {
    public:
        BrainDecisionNode();

        static BrainDecisionNode *GetInstance();

        void Reset();

        void PopDecisionAndPushGoalPoints();

        std::vector<Eigen::Vector3d> DecodeDecisionJson(std::string jsonText);

        std::string GetLastDecisionJson();

        std::vector<Eigen::Vector3d> GetLastGoalPoints();

    private:
        mutable std::mutex dataMutex_;
        Keyboard104PoseStore keyboardPoseStore_;
        std::string lastDecisionJson_;
        std::vector<Eigen::Vector3d> lastGoalPoints_;
    };
}

#endif
