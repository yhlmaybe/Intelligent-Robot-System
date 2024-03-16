#include "NodeManager.h"

ServoDriveNodeListener::ServoDriveNodeListener(std::map<int, Servo> servoKeyValues) 
    : Node("serve_drive_node_listener"), servoKeyValues(std::move(servoKeyValues))
{
    subscription = this->create_subscription<std_msgs::msg::String>(
       "ServeMsg", 10, std::bind(&ServoDriveNodeListener::DoListen, this, std::placeholders::_1));
    //subscription = this->create_subscription<std_msgs::msg::String>(
    //   "ServeMsg", 10, [this](const std_msgs::msg::String::SharedPtr msg){this->DoListen(msg);});
}

void ServoDriveNodeListener::DoListen(const std_msgs::msg::String::SharedPtr msg)
{
    std::stringstream msgString;
    msgString << msg->data.c_str();
    auto ServoMsgs = MsgConvertHandle::StringToServos(msgString.str());
    for(ServoMsg servoMsg : ServoMsgs)
    {
        //Servo servo = 
    }
    
    //DriveHandle::SetServoPosition();
   
}