#include "NodeManager.h"

ServoDriveNodeListener::ServoDriveNodeListener(std::list<Servo> servos) 
    : Node("serve_drive_node_listener"), servoList(std::move(servos))
{
    for(auto& it : servoList)
    {
        idServoKeyValues.insert(std::make_pair(it.id, it));
    }

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
    std::list<ServoDriveInfo> servoInof;
    for(ServoMsg servoMsg : ServoMsgs)
    {
        std::map<int, Servo>::iterator servoIter = idServoKeyValues.find(servoMsg.id);
        if(servoIter != idServoKeyValues.end())
        {
            Servo* servo = &servoIter->second;
            ServoDriveInfo info = {servo, servoMsg.position, servoMsg.time};
            servoInof.push_back(info);
        }
    }
    
    DriveHandle::SetServoPosition(servoInof); 
}