#include "NodeManager.h"

ServoDriveNodeListenerNode::ServoDriveNodeListenerNode(std::list<std::shared_ptr<Servo> > servos) 
    : Node("serve_drive_node_listener"), servoList(std::move(servos))
{
    for(auto& it : servoList)
    {
        //idServoKeyValues.insert(std::pair<int, std::shared_ptr<Servo>>(it->id, it));
        idServoKeyValues[it->id] = it;
    }

    subscription = this->create_subscription<std_msgs::msg::String>(
       "ServeMsg", 10, std::bind(&ServoDriveNodeListenerNode::DoListen, this, std::placeholders::_1));
    //subscription = this->create_subscription<std_msgs::msg::String>(
    //   "ServeMsg", 10, [this](const std_msgs::msg::String::SharedPtr msg){this->DoListen(msg);});
}

void ServoDriveNodeListenerNode::DoListen(const std_msgs::msg::String::SharedPtr msg)
{
    std::stringstream msgString;
    msgString << msg->data.c_str();
    auto ServoMsgs = MsgConvertHandle::StringToServos(msgString.str());
    std::list<ServoDriveInfo> servoInof;
    for(ServoMsg servoMsg : ServoMsgs)
    {
        std::map<int, std::shared_ptr<Servo>>::iterator servoIter = idServoKeyValues.find(servoMsg.id);
        if(servoIter != idServoKeyValues.end())
        {
            auto servo = servoIter->second;
            ServoDriveInfo info = {servo, servoMsg.position, servoMsg.time};
            servoInof.push_back(info);
        }
    }
    
    DriveHandle::SetServoPosition(servoInof); 
}


UrdfPublisherNode::UrdfPublisherNode() : Node("urdf_publisher")
{
    urdf_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("urdf_markers", 10);
    urdf_model = std::make_shared<urdf::Model>();
    if(!urdf_model->initFile(""))
    {
        RCLCPP_ERROR(this->get_logger(), "failer to load URDF file");
        return;
    }
    DoUrdfPublisher();
}

void UrdfPublisherNode::DoUrdfPublisher()
{
    visualization_msgs::msg::MarkerArray markers;

    for(const auto& link_pair : urdf_model->links_)
    {
        const auto& link = link_pair.second;

        visualization_msgs::msg::Marker marker;
        urdf::JointSharedPtr parent_joint = link->parent_joint;
        std::string frame_id = "base_link";
        if(parent_joint)
        {
            frame_id = parent_joint->name;
        }
        marker.header.frame_id = frame_id;
        marker.header.stamp = this->now();
        marker.ns = "urdf";
        marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = link->visual->origin.position.x;
        marker.pose.position.y = link->visual->origin.position.y;
        marker.pose.position.z = link->visual->origin.position.z;
        marker.pose.orientation.x = link->visual->origin.rotation.x;
        marker.pose.orientation.y = link->visual->origin.rotation.y;
        marker.pose.orientation.z = link->visual->origin.rotation.z;
        marker.pose.orientation.w = link->visual->origin.rotation.w;
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0;
        marker.color.a = 1.0;
        marker.color.r = marker.color.g = marker.color.b = 0.5;
        if (link->visual)
        {
            if (link->visual->geometry)
            {
                link->visual->geometry->type == urdf::Geometry::MESH;
                {
                    marker.mesh_resource = std::static_pointer_cast<urdf::Mesh>(link->visual->geometry)->filename;
                }
            }
        }

        markers.markers.push_back(marker);
    }

    urdf_pub->publish(markers);
}


int RvizUrdfManager::Initial()
{
    auto node = std::make_shared<rclcpp::Node>("robot_state_publisher_node");
    
    char cwd[PATH_MAX];
    if(getcwd(cwd, sizeof(cwd)) == NULL)
    {
        perror("getcwd() error");
        return 1;
    }

    std::string urdf_file = std::string(cwd) + "/ROSManager/urdf/text.uref";
    std::ifstream urdf_ifs(urdf_file);
    std::string urdf_string((std::istreambuf_iterator<char>(urdf_ifs)), std::istreambuf_iterator<char>());



    return 0;
}