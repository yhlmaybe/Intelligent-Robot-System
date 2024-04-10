#include <iostream>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <list>

#include "MsgManager.h"
#include "../ServoControl/ServoInitiate.h"

#include <rclcpp_components/register_node_macro.hpp>
#include <robot_state_publisher/robot_state_publisher.h>

#include <urdf/model.h>
#include <urdf_parser/urdf_parser.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>


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


class UrdfPublisherNode : public rclcpp::Node
{
public:
    UrdfPublisherNode();

private:
    void DoUrdfPublisher();  

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr urdf_pub;  
    std::shared_ptr<urdf::Model> urdf_model;
};


class RvizUrdfManager
{
public:
    static int Initial();
};



