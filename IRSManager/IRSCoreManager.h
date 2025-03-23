#ifndef IRSCOREMANAGER_H
#define IRSCOREMANAGER_H

#include <iostream>
#include <fstream>
#include <sstream>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <list>
#include <map>
#include <memory>

#include "MsgManager.h"

//#include <rclcpp_components/register_node_macro.hpp>
//#include <robot_state_publisher/robot_state_publisher.h>
#include <robot_state_publisher/robot_state_publisher.hpp>

#include <urdf/model.h>
#include <urdf_parser/urdf_parser.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include "../ServoControl/ServoManager.h"
#include "../include/IRSParametersData.h"
#include "../include/IRSFunction.h"
#include "../MathematicalTool/MotionManager.h"

using MimicMap = std::map<std::string, urdf::JointMimicSharedPtr>;

class JointStateListenerNode : public rclcpp::Node
{
public:
     JointStateListenerNode(std::shared_ptr<moveit::core::RobotState> robotState);
    
private:
    std::shared_ptr<moveit::core::RobotState> robot_state;

    std::vector<std::string> end_eff_names;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription;

    void DoListen(const sensor_msgs::msg::JointState msg);

};


class SegmentPair
{
public:
    SegmentPair(const KDL::Segment &p_segment, const std::string &p_root, const std::string &p_tip) : segment(p_segment), root(p_root), tip(p_tip){}

    KDL::Segment segment;
    std::string root;
    std::string tip;
};

class UrdfPublisherNode : public rclcpp::Node
{
public:
    UrdfPublisherNode();

protected:
    KDL::Tree ParseURDF(urdf::Model &model);
    void SetupURDF();
    void AddChildren(const urdf::Model &model, const KDL::SegmentMap::const_iterator segment);
    void PublishTransforms(const std::map<std::string, double> &joint_positions, const builtin_interfaces::msg::Time &time);
    void PublishFixedTransforms();
    void CallbackJointState(const sensor_msgs::msg::JointState::ConstSharedPtr state);
    rcl_interfaces::msg::SetParametersResult parameterUpdate(const std::vector<rclcpp::Parameter> &parameters); 
    void OnParameterEvent(std::shared_ptr<rcl_interfaces::msg::ParameterEvent> event);
    geometry_msgs::msg::TransformStamped KDLToTransform(const KDL::Frame & k);
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr description_pub;

    std::map<std::string, SegmentPair> segment_dynamic;
    std::map<std::string, SegmentPair> segment_fixed;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub;
    
    MimicMap mimic;
};

class EnvironmentalPerception : public IRSThreadBase
{
public:
    EnvironmentalPerception(unsigned interval_ms = 100);

    std::vector<Eigen::Vector3d> GetPointClouds();

    void Reset();

private:
    unsigned interval_ms_;  
    std::vector<Eigen::Vector3d> cloud_points_;
    mutable std::mutex mtx_;

    void ExecuteTask() override;
};

class VisualImageProcessing
{
public:
    VisualImageProcessing();

    void CalGoalPoints();
};

class ServoManagerNode : public rclcpp::Node , public IRSThreadBase
{
public: 
    ServoManagerNode();

    void Reset();

    void ToGoalPoint(); 

    void SetJointPosition(std::string jointName, double position);

    std::shared_ptr<moveit::core::RobotState> GetCurrentRobotState();

private:

    std::map<std::string, std::shared_ptr<ServoManager>> joint_servos;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointState_pub;
    std::shared_ptr<MotionManager> motion_manager;
    mutable std::mutex motion_servo_mtx;

    std::vector<std::shared_ptr<ServoManager>> GetServoManagerFromName(std::vector<std::string> names);
    std::shared_ptr<moveit::core::RobotState> UpdateRobotstateBoundary();

    void ExecuteTask() override;
};


class IRSCoreHandle
{
public:
    static std::shared_ptr<ThreadSafeQueue<std::vector<Eigen::Vector3d>>> goal_points_queue; 

    static void Start();

    static void End();

    static void ResetServoState();

    static void SetJointPosition(std::string name, double position);

    static std::shared_ptr<moveit::core::RobotState> GetCurrentRobotState();

    static std::vector<Eigen::Vector3d> GetEnvironmentPoints();

    static std::vector<std::string> GetActiveNodeName();

    static bool IsActiveNode(std::string name);

private:
    static std::shared_ptr<UrdfPublisherNode> urdf_publisher_node;
    static std::shared_ptr<JointStateListenerNode> joint_state_listen_node;
    static std::shared_ptr<ServoManagerNode> servo_manager_node;
    static std::shared_ptr<EnvironmentalPerception> env_perce;  

    static void UrdfSrdfXMLInitial();

    static bool ReplacePathsInUrdf(std::string& urdfContent, const std::string& oldKey, const std::string& newKey);

    static std::string UrdfNodeInitial();

    static std::string JointStateNodeInitial(std::shared_ptr<moveit::core::RobotState> robotState);
};

#endif

