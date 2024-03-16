#include "NodeManager.h"

ServoDriveNodeListener::ServoDriveNodeListener(std::map<int, Servo> servoKeyValues) : Node("serve_drive_node_listener")
{
    this->servoKeyValues = servoKeyValues;
    subscription = this->create_subscription<std_msgs::msg::String>(
       "ServeMsg", 10, std::bind(&ServoDriveNodeListener::DoListen, this, std::placeholders::_1));
}

void ServoDriveNodeListener::DoListen(std_msgs::msg::String msg)
{
    std::stringstream msgString;
    msgString << msg.data;
    auto ServoMsgs = MsgConvertHandle::StringToServos(msgString.str());
    for(ServoMsg servoMsg : ServoMsgs)
    {
        //Servo servo = 
    }
    
    //DriveHandle::SetServoPosition();
   
}
