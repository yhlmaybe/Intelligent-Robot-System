#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <list>

#include "MsgManager.h"
#include "../ServoControl/SCDrive.h"

class ServoDriveNodeListener : public rclcpp::Node
{
public:
     ServoDriveNodeListener(std::map<int, Servo> servoKeyValues);
    

private:

    std::map<int, Servo> servoKeyValues;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription;

    void DoListen(const std_msgs::msg::String::SharedPtr msg);

};