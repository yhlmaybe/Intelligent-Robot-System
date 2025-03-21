#ifndef MOTIONMANAGER_H
#define MOTIONMANAGER_H

#include <string>

#include <urdf/model.h>
#include <urdf_parser/urdf_parser.h>
#include <sensor_msgs/msg/joint_state.hpp>

#include <pluginlib/pluginlib/class_loader.hpp>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/trajectory_processing/trajectory_tools.h>
#include <moveit/robot_model/robot_model.h>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/moveit_cpp/moveit_cpp.h>
#include <moveit/moveit_cpp/planning_component.h>

#include "../include/IRSParametersData.h"
#include "../include/IRSFunction.h"

#include "Math3D.h"
#include "KinematicTool.h"
#include "PlanningTool.h"

enum EndEffectorType 
{
    NoneType,
    WristEnd,
    ThumbEnd,
    IndexEnd,
    FourthEnd,
    MidEnd,
    LittleEnd,
};

struct TrajectoryPoint 
{
    std::vector<double> positions; 
    double time_from_start;
    double time_interval;        
};

class EndEffector
{
public:
    const EndEffectorType type = EndEffectorType::NoneType;
    const std::string name = "";
    const std::string partGroupName = "";
    const std::string completeGroupName = "";
    std::shared_ptr<IRS_IK::IRS_IK> partGroupIkSolver = nullptr;
    std::shared_ptr<IRS_IK::IRS_IK> completeGroupIkSolver = nullptr;
    bool IsUsed = false;
    EndEffector(std::string URDF, std::string name, std::string partGroupName, std::string completeGroupName, std::string partGroupFirstLinkName, std::string completeGroupFirstLinkName, EndEffectorType type);
};


class MotionManager
{
public:
    MotionManager(std::string urdf, std::string srdf);

    Eigen::Isometry3d ConvertPoseFromRelBaseToRelEnd(EndEffectorType endEffector, Eigen::Isometry3d pose);
    Eigen::Isometry3d ConvertPoseFromRelEndToRelBase(EndEffectorType endEffector, Eigen::Isometry3d pose);
    Eigen::Isometry3d ConvertPoseFromRelEndToRelAny(EndEffectorType endEffector, std::string anyLinkName, Eigen::Isometry3d pose);
    sensor_msgs::msg::JointState GetCurrentJointStateMsg();
    Eigen::Isometry3d GetDefaultTouchPoseFromPoint(EndEffectorType endEffector, Eigen::Vector3d pointRelBase, Eigen::Vector3d pointNormal, double fingerAngle = 60);
    void InitialJointState();

    double CalGoalJointPosition(std::string jointName, double position);

    std::shared_ptr<moveit::core::RobotState> GetCurrentRobotState();

    EndEffectorType GetClosestEndEffector(Eigen::Vector3d point);

    void UpdateState(std::shared_ptr<moveit::core::RobotState> state);

    bool JointIKCal(std::vector<std::string>& jointNameResult, std::vector<double>& jointValueResult, EndEffectorType endEffector, Eigen::Isometry3d pointRelativeEndEff, bool isPart);
    std::vector<TrajectoryPoint> Plan(std::vector<std::string> goalJointNames, std::vector<double> goalJointPositions, std::vector<Eigen::Vector3d> envPointClouds, double totalTime, int interpolateCount = 0);
    sensor_msgs::msg::JointState UpdateRobotStateAndGetMsg(std::vector<std::string> goalJointNames, std::vector<double> goalJointPositions);
    sensor_msgs::msg::JointState UpdateRobotStateAndGetMsg(std::string goalJointName, double goalJointPosition);

private:
    std::string overallGroupName = "r_arm";
    std::string overallInitPosdName = "r_arm_home";

    std::shared_ptr<moveit::core::RobotModel> robotModel;
    std::shared_ptr<moveit::core::RobotState> robotState;

    std::map<EndEffectorType, std::shared_ptr<EndEffector>> endEffectorsMap;

    Eigen::Isometry3d ConvertToIsometry3d(KDL::Frame frame);
    KDL::Frame ConvertToFrame(Eigen::Isometry3d Isometry3d);

    std::shared_ptr<PlanningTool::OMPLTool> omplTool;
};

#endif


