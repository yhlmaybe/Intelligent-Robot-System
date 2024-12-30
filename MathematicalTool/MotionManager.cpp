#include "MotionManager.h"


MotionPlanning::MotionPlanning(std::string urdf, std::string srdf)
{
    auto urdf_model = urdf::parseURDF(urdf);

    srdf::ModelSharedPtr srdf_model = std::make_shared<srdf::Model>();
    srdf_model->initString(*urdf_model, srdf);
    robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
    
    if(!robot_model)
    {
        IRS_MESSAGE("set robot model error");
        return;
    }
    robot_state = std::make_shared<moveit::core::RobotState>(robot_model);

    std::vector<srdf::Model::EndEffector> endEffectors = srdf_model->getEndEffectors();
    for (srdf::Model::EndEffector& endEffector : endEffectors)
    {
        
    }
    robot_state->setToDefaultValues();
}

sensor_msgs::msg::JointState MotionPlanning::GetCurrentJointStateMsg()
{
    sensor_msgs::msg::JointState jointStateMSgs = sensor_msgs::msg::JointState();
    std::vector<std::string> jointNames = robot_model->getVariableNames();
    double* jointPositions = robot_state->getVariablePositions();
    std::vector<double> jointPositionsVec(jointPositions, jointPositions + jointNames.size());
    jointStateMSgs.name = jointNames;
    jointStateMSgs.position = jointPositionsVec;
}