#include "MotionManager.h"

EndEffector::EndEffector(std::string name, std::string partGroupName, std::string completeGroupName, EndEffectorType type)
{
    name = name;
    partGroupName = partGroupName;
    completeGroupName = completeGroupName;
    type = type;
}

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
        std::string partGroupName = "";
        std::string completeGroupName = "";
        EndEffectorType type = EndEffectorType::None;
        if(endEffector.name_ == "thumb_end")
        {
            partGroupName = "thumb";
            completeGroupName = "thumbWrist";
            type = EndEffectorType::ThumbEnd;
        }
        else if(endEffector.name_ == "index_end")
        {
            partGroupName = "index";
            completeGroupName = "indexWrist";
            type = EndEffectorType::IndexEnd;
        }
        else if(endEffector.name_ == "fourth_end")
        {
            partGroupName = "fourth";
            completeGroupName = "fourthWrist";
            type = EndEffectorType::FourthEnd;
        }
        else if(endEffector.name_ == "mid_end")
        {
            partGroupName = "mid";
            completeGroupName = "midWrist";
            type = EndEffectorType::MidEnd;
        }
        else if(endEffector.name_ == "little_end")
        {
            partGroupName = "little";
            completeGroupName = "littleWrist";
            type = EndEffectorType::LittleEnd;
        }
        else if(endEffector.name_ == "wrist_end")
        {
            partGroupName = "wrist";
            completeGroupName = "wrist";
            type = EndEffectorType::WristEnd;
        }
        else
        {
            continue;
        }
        std::shared_ptr<EndEffector> endEff= std::make_shared<EndEffector>(endEffector.name_, partGroupName, completeGroupName, type);
        endEffectorsMap[type] = endEff;
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
    return jointStateMSgs;
}

Point3D MotionPlanning::ConvertPointFromBaseToEndEffector(EndEffectorType endEffector, Point3D point)
{

}

bool MotionPlanning::EndEffectorPlan(EndEffectorType endEffector, Point3D goalPointRelativeBaseLink)
{

}