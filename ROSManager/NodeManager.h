#include <iostream>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <list>
#include <map>
#include <memory>

#include "MsgManager.h"
#include "../ServoControl/ServoInitiate.h"
#include "../include/ROSParametersData.h"
#include "../include/IRSFunction.h"

#include <rclcpp_components/register_node_macro.hpp>
#include <robot_state_publisher/robot_state_publisher.h>

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

using MimicMap = std::map<std::string, urdf::JointMimicSharedPtr>;

class ServoDriveNodeListenerNode : public rclcpp::Node
{
public:
     ServoDriveNodeListenerNode(std::list<std::shared_ptr<Servo>> servoList);
    

private:

    std::map<int, std::shared_ptr<Servo>> idServoKeyValues;

    std::list<std::shared_ptr<Servo>> servoList;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription;

    void DoListen(const std_msgs::msg::String::SharedPtr msg);

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


class ROSNodeManager
{
public:
    static std::string UrdfInitial();

    static std::vector<std::string> GetActiveNodeName();

    static bool IsActiveNode(std::string name);
};



