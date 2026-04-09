#ifndef IRSCOREMANAGER_H
#define IRSCOREMANAGER_H

#include "IRSCoreModule.h"
#include "IRSCoreDecision.h"

namespace IRSCoreManager
{
    class IRSCoreHandle
    {
    public:
        static IRSThreadTools::ThreadSafeQueue<std::vector<Eigen::Vector3d>> &GetGoalPointsQueue();

        static void Start();

        static void End();

        static void ResetServoState();

        static void SetJointPosition(std::string name, double position);

        static std::shared_ptr<moveit::core::RobotState> GetCurrentRobotState();

        static std::vector<Eigen::Vector3d> GetEnvironmentPoints();

        static std::vector<std::string> GetActiveNodeName();

        static bool IsActiveNode(std::string name);

    private:
        static void UrdfSrdfXMLInitial();

        static bool ReplacePathsInUrdf(std::string &urdfContent, const std::string &oldKey, const std::string &newKey);

        static std::string ROSNodeInitial(std::shared_ptr<moveit::core::RobotState> robotState);
    };

}

#endif
