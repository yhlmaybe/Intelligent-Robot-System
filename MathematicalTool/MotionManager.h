#include <string>

#include <urdf/model.h>
#include <urdf_parser/urdf_parser.h>
#include <sensor_msgs/msg/joint_state.hpp>


#include <moveit/planning_interface/planning_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/trajectory_processing/trajectory_tools.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/move_group_interface/move_group_interface.h>

#include "../include/ROSParametersData.h"
#include "../include/IRSFunction.h"
#include "../include/IRSFunction.h"

#include "GeometricManager.h"

class EndEffector
{
public:
    std::string name = "";
    bool IsUsed = false;
    Point3D Coordinate;

    EndEffector(std::string name, double x, double y, double z);
};


class MotionPlanning
{
public:
    std::list<std::shared_ptr<EndEffector>> endEffectors;

    MotionPlanning(std::string urdf, std::string srdf);
    bool EndEffectorPlan(std::shared_ptr<EndEffector> endEffector, Point3D goalPoint);
    sensor_msgs::msg::JointState GetCurrentJointStateMsg();

private:
    std::string overallGroupName = "r_arm";
    std::string overallInitPosdName = "r_arm_home";

    std::shared_ptr<moveit::core::RobotModel> robot_model;
    std::shared_ptr<moveit::core::RobotState> robot_state;
};


