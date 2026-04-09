#ifndef IRSCOREMODULE_H
#define IRSCOREMODULE_H

#include <iostream>
#include <fstream>
#include <sstream>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <list>
#include <map>
#include <memory>

// #include <rclcpp_components/register_node_macro.hpp>
// #include <robot_state_publisher/robot_state_publisher.h>
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
#include "../BrainDeepLearn/Interface.h"

namespace IRSCoreModule
{
    using MimicMap = std::map<std::string, urdf::JointMimicSharedPtr>;

    class IRSGoalPoints
    {
    public:
        static IRSThreadTools::ThreadSafeQueue<std::vector<Eigen::Vector3d>> &GetGoalPointsQueue();
    };

    class ROSNodeExecutor
    {
    public:
        static ROSNodeExecutor *GetInstance();

        void AddNode(rclcpp::Node::SharedPtr node);

        void Start();

        void Stop();

    private:
        ROSNodeExecutor() : executor_(std::make_shared<rclcpp::executors::MultiThreadedExecutor>()) {}
        ~ROSNodeExecutor() { Stop(); }

        std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
        std::vector<rclcpp::Node::SharedPtr> nodes_;
        std::thread executor_thread_;
        std::atomic<bool> running_{false};
        std::mutex mutex_;

        ROSNodeExecutor(const ROSNodeExecutor &) = delete;
        ROSNodeExecutor &operator=(const ROSNodeExecutor &) = delete;
    };

    class JointStateListenerNode : public rclcpp::Node
    {
    public:
        JointStateListenerNode(std::shared_ptr<moveit::core::RobotState> robotState);

    private:
        std::shared_ptr<moveit::core::RobotState> robot_state;

        std::vector<std::string> end_eff_names;

        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription;

        void DoListen(const sensor_msgs::msg::JointState::SharedPtr msg);
    };

    class SegmentPair
    {
    public:
        SegmentPair(const KDL::Segment &p_segment, const std::string &p_root, const std::string &p_tip) : segment(p_segment), root(p_root), tip(p_tip) {}

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
        // void OnParameterEvent(std::shared_ptr<rcl_interfaces::msg::ParameterEvent> event);
        geometry_msgs::msg::TransformStamped KDLToTransform(const KDL::Frame &k);

        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr description_pub;

        std::map<std::string, SegmentPair> segment_dynamic;
        std::map<std::string, SegmentPair> segment_fixed;

        std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
        std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster;

        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub;

        MimicMap mimic;
    };

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


    class ServoManagerNode : public IRSThreadTools::IRSThreadBase
    {
    public:
        ServoManagerNode();

        static ServoManagerNode *GetInstance();

        void Reset();

        void ClickToGoalPoint();

        void SetJointPosition(std::string jointName, double position);

        std::shared_ptr<moveit::core::RobotState> GetCurrentRobotState();

    private:
        ServoManagerNode(const ServoManagerNode &) = delete;
        ServoManagerNode &operator=(const ServoManagerNode &) = delete;

        std::shared_ptr<rclcpp::Node> node;
        std::map<std::string, std::shared_ptr<ServoManager>> joint_servos;
        rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointState_pub;
        std::shared_ptr<MotionManager> motion_manager;
        mutable std::mutex motion_servo_mtx;

        std::vector<std::shared_ptr<ServoManager>> GetServoManagerFromName(std::vector<std::string> names);
        std::shared_ptr<moveit::core::RobotState> UpdateRobotstateBoundary();
        void PublishJointState();
    };

} 

#endif