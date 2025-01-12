#ifndef MOTIONMANAGER_H
#define MOTIONMANAGER_H

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

#include "../include/IRSParametersData.h"
#include "../include/IRSFunction.h"
#include "../include/IRSFunction.h"

#include "GeometricManager.h"

enum EndEffectorType 
{
    None,
    WristEnd,
    ThumbEnd,
    IndexEnd,
    FourthEnd,
    MidEnd,
    LittleEnd,
};

class EndEffector
{
public:
    const EndEffectorType type = EndEffectorType::None;
    const std::string name = "";
    const std::string partGroupName = "";
    const std::string completeGroupName = "";
    bool IsUsed = false;
    EndEffector(std::string name, std::string partGroupName, std::string completeGroupName, EndEffectorType type);
};


class MotionPlanning
{
public:
    MotionPlanning(std::string urdf, std::string srdf);
    bool EndEffectorPlan(EndEffectorType endEffector, Point3D goalPointRelativeBaseLink);
    Point3D ConvertPointFromBaseToEndEffector(EndEffectorType endEffector, Point3D point);
    Point3D ConvertPointFromEndEffectorToBase(EndEffectorType endEffector, Point3D point);
    sensor_msgs::msg::JointState GetCurrentJointStateMsg();

    bool PointInEndEffectorRange(EndEffectorType endEffector, Point3D pointRelativeBaseLink);
private:
    std::string overallGroupName = "r_arm";
    std::string overallInitPosdName = "r_arm_home";

    std::shared_ptr<moveit::core::RobotModel> robot_model;
    std::shared_ptr<moveit::core::RobotState> robot_state;

    std::map<EndEffectorType, std::shared_ptr<EndEffector>> endEffectorsMap;
};

#endif


