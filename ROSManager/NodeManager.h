#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <list>

#include "MsgManager.h"
#include "../ServoControl/ServoInitiate.h"

class ServoDriveNodeListener : public rclcpp::Node
{
public:
     ServoDriveNodeListener(std::list<std::shared_ptr<Servo>> servoList);
    

private:

    std::map<int, std::shared_ptr<Servo>> idServoKeyValues;

    std::list<std::shared_ptr<Servo>> servoList;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription;

    void DoListen(const std_msgs::msg::String::SharedPtr msg);

};