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
    }
}