#include "NodeManager.h"

std::atomic<bool> IsNodeRunning(true);

std::shared_ptr<ServoDriveNodeListenerNode> ROSNodeManager::servoDriveNodeListenerNode = nullptr;
std::shared_ptr<UrdfPublisherNode> ROSNodeManager::urdfPublisherNode = nullptr;
std::shared_ptr<MotionPlanning> ROSNodeManager::motionPlanning = nullptr;

ServoDriveNodeListenerNode::ServoDriveNodeListenerNode(std::list<std::shared_ptr<Servo>> servos)
    : Node(SERVE_DRIVE_LISTENER), servoList(std::move(servos))
{
    for (auto &it : servoList)
    {
        // idServoKeyValues.insert(std::pair<int, std::shared_ptr<Servo>>(it->id, it));
        idServoKeyValues[it->id] = it;
    }

    subscription = this->create_subscription<std_msgs::msg::String>(
        "ServeMsg", 10, std::bind(&ServoDriveNodeListenerNode::DoListen, this, std::placeholders::_1));
    // subscription = this->create_subscription<std_msgs::msg::String>(
    //    "ServeMsg", 10, [this](const std_msgs::msg::String::SharedPtr msg){this->DoListen(msg);});
}

void ServoDriveNodeListenerNode::DoListen(const std_msgs::msg::String::SharedPtr msg)
{
    std::stringstream msgString;
    msgString << msg->data.c_str();
    auto ServoMsgs = MsgConvertHandle::StringToServos(msgString.str());
    std::list<std::shared_ptr<ServoDriveInfo>> servoInof;
    for (ServoMsg servoMsg : ServoMsgs)
    {
        std::map<int, std::shared_ptr<Servo>>::iterator servoIter = idServoKeyValues.find(servoMsg.id);
        if (servoIter != idServoKeyValues.end())
        {
            auto servo = servoIter->second;
            std::shared_ptr<ServoDriveInfo> info = std::make_shared<ServoDriveInfo>(servo, servoMsg.position, servoMsg.time);
            servoInof.push_back(info);
        }
    }

    DriveHandle::SetServoPosition(servoInof);
}


geometry_msgs::msg::TransformStamped UrdfPublisherNode::KDLToTransform(const KDL::Frame &k)
{
    geometry_msgs::msg::TransformStamped t;
    t.transform.translation.x = k.p.x();
    t.transform.translation.y = k.p.y();
    t.transform.translation.z = k.p.z();
    k.M.GetQuaternion(
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
        t.transform.rotation.w);
    return t;
}

UrdfPublisherNode::UrdfPublisherNode() : Node(URDF_PUBLISHER)
{
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == NULL)
    {
        IRS_MESSAGE("get urdf and srdf cwd() error");
        return;
    }
    std::string urdf_file = std::string(cwd) + "/Configure/Arm_R_SLDASM.urdf";
    std::ifstream inurdf(urdf_file);
    if (inurdf)
    {
        inurdf.seekg(0, std::ios::end);
        URDF_XML.resize(inurdf.tellg());
        inurdf.seekg(0, std::ios::beg);
        inurdf.read(&URDF_XML[0], URDF_XML.size());
        inurdf.close();
    }

    std::string oldMeshPath = "${URDF_MESH_PATH}"; 
    std::string newMeshPath = std::string(cwd) + "/Configure/meshes"; 

    ReplacePathsInUrdf(URDF_XML, oldMeshPath, newMeshPath);

    std::string srdf_file = std::string(cwd) + "/Configure/Arm_R_SLDASM.srdf";
    std::ifstream insrdf(srdf_file);
    if (insrdf)
    {
        insrdf.seekg(0, std::ios::end);
        SRDF_XML.resize(insrdf.tellg());
        insrdf.seekg(0, std::ios::beg);
        insrdf.read(&SRDF_XML[0], SRDF_XML.size());
        insrdf.close();
    }

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    static_tf_broadcaster = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);

    description_pub = this->create_publisher<std_msgs::msg::String>("robot_description", rclcpp::QoS(1).transient_local());
    SetupURDF();

    auto subscriber_options = rclcpp::SubscriptionOptions();
    subscriber_options.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
    // subscribe to joint state
    joint_state_sub = this->create_subscription<sensor_msgs::msg::JointState>(
        "joint_states",
        rclcpp::SensorDataQoS(),
        std::bind(&UrdfPublisherNode::CallbackJointState, this, std::placeholders::_1),
        subscriber_options);

    PublishFixedTransforms();
}

bool UrdfPublisherNode::ReplacePathsInUrdf(std::string &urdfContent, const std::string &oldKey, const std::string &newKey)
{
    size_t pos = 0;
    bool modified = false;
    while ((pos = urdfContent.find(oldKey, pos)) != std::string::npos) 
    {
        urdfContent.replace(pos, oldKey.length(), newKey);
        pos += newKey.length(); 
        modified = true;
    }
    return modified;
}

KDL::Tree UrdfPublisherNode::ParseURDF(urdf::Model &model)
{
    // Initialize the model
    if (!model.initString(URDF_XML))
    {
        IRS_MESSAGE("Unable to initialize urdf::model from robot description");
    }

    // Initialize the KDL tree
    KDL::Tree tree;
    if (!kdl_parser::treeFromUrdfModel(model, tree))
    {
        IRS_MESSAGE("Failed to extract kdl tree from robot description");
    }

    return tree;
}

void UrdfPublisherNode::SetupURDF()
{
    urdf::Model model;
    KDL::Tree tree = ParseURDF(model);

    // Initialize the mimic map
    mimic.clear();
    for (const std::pair<const std::string, urdf::JointSharedPtr> &i : model.joints_)
    {
        if (i.second->mimic)
        {
            // Just taking a reference to the model shared pointers ends up in a crash.
            // Explicitly make a copy of the JointMimic.
            auto jm = std::make_shared<urdf::JointMimic>();
            jm->offset = i.second->mimic->offset;
            jm->multiplier = i.second->mimic->multiplier;
            jm->joint_name = i.second->mimic->joint_name;
            mimic[i.first] = jm;
        }
    }

    KDL::SegmentMap segments_map = tree.getSegments();
    for (const std::pair<const std::string, KDL::TreeElement> &segment : segments_map)
    {
        IRS_MESSAGE("Got segment " + segment.first);
    }

    // walk the tree and add segments to segments_
    segment_dynamic.clear();
    segment_fixed.clear();
    AddChildren(model, tree.getRootSegment());

    auto msg = std::make_unique<std_msgs::msg::String>();
    msg->data = URDF_XML;

    // Publish the robot description
    description_pub->publish(std::move(msg));
}

void UrdfPublisherNode::AddChildren(const urdf::Model &model, const KDL::SegmentMap::const_iterator segment)
{
    const std::string &root = GetTreeElementSegment(segment->second).getName();

    std::vector<KDL::SegmentMap::const_iterator> children = GetTreeElementChildren(segment->second);
    for (unsigned i = 0; i < children.size(); i++)
    {
        const KDL::Segment &child = GetTreeElementSegment(children[i]->second);
        SegmentPair s(GetTreeElementSegment(children[i]->second), root, child.getName());
        if (child.getJoint().getType() == KDL::Joint::None)
        {
            if (model.getJoint(child.getJoint().getName()) &&
                model.getJoint(child.getJoint().getName())->type == urdf::Joint::FLOATING)
            {
                std::string root_str = root.c_str();
                std::string child_str = child.getName().c_str();
                IRS_MESSAGE("floating joint is not supported; skipping segment form " + root_str  + "to" + child_str);
            }
            else
            {
                segment_fixed.insert(make_pair(child.getJoint().getName(), s));
            }
        }
        else
        {
            segment_dynamic.insert(make_pair(child.getJoint().getName(), s));
        }
        AddChildren(model, children[i]);
    }
}

void UrdfPublisherNode::PublishTransforms(const std::map<std::string, double> &joint_positions, const builtin_interfaces::msg::Time &time)
{

    std::vector<geometry_msgs::msg::TransformStamped> tf_transforms;

    for (const std::pair<const std::string, double> &jnt : joint_positions)
    {
        std::map<std::string, SegmentPair>::iterator seg = segment_dynamic.find(jnt.first);
        if (seg != segment_dynamic.end())
        {
            geometry_msgs::msg::TransformStamped tf_transform = KDLToTransform(seg->second.segment.pose(jnt.second));
            tf_transform.header.stamp = time;
            tf_transform.header.frame_id = seg->second.root;
            tf_transform.child_frame_id = seg->second.tip;
            tf_transforms.push_back(tf_transform);
        }
    }
    tf_broadcaster->sendTransform(tf_transforms);
}

void UrdfPublisherNode::PublishFixedTransforms()
{
    std::vector<geometry_msgs::msg::TransformStamped> tf_transforms;

    rclcpp::Time now = this->now();
    for (const std::pair<const std::string, SegmentPair> &seg : segment_fixed)
    {
        geometry_msgs::msg::TransformStamped tf_transform = KDLToTransform(seg.second.segment.pose(0));
        tf_transform.header.stamp = now;

        tf_transform.header.frame_id = seg.second.root;
        tf_transform.child_frame_id = seg.second.tip;
        tf_transforms.push_back(tf_transform);
    }
    static_tf_broadcaster->sendTransform(tf_transforms);
}

void UrdfPublisherNode::CallbackJointState(const sensor_msgs::msg::JointState::ConstSharedPtr state)
{
    if (state->name.size() != state->position.size())
    {
        if (state->position.empty())
        {
            IRS_MESSAGE("position member was empty");
        }
        else
        {
            IRS_MESSAGE("Robot state publisher ignored an invalid JointState message");
        }
        return;
    }

    // get joint positions from state message
    std::map<std::string, double> joint_positions;
    for (size_t i = 0; i < state->name.size(); i++)
    {
        joint_positions.insert(std::make_pair(state->name[i], state->position[i]));
    }

    for (const std::pair<const std::string, urdf::JointMimicSharedPtr> &i : mimic)
    {
        if (joint_positions.find(i.second->joint_name) != joint_positions.end())
        {
            double pos = joint_positions[i.second->joint_name] * i.second->multiplier + i.second->offset;
            joint_positions.insert(std::make_pair(i.first, pos));
        }
    }

    PublishTransforms(joint_positions, state->header.stamp);
}

JointStatePublisherNode::JointStatePublisherNode() : Node(COMPONENT_ROTATE_STATE_PUBLISHER)
{
    jointState_pub = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    PublishJointStates();
    //timer = this->create_wall_timer(std::chrono::milliseconds(TIMEINTERVAL),std::bind(&JointStatePublisherNode::PublishJointStates, this));
}

void JointStatePublisherNode::PublishJointStates()
{
    jointState_pub->publish(ROSNodeManager::motionPlanning->GetCurrentJointStateMsg());
}

void ROSNodeManager::StartNodes(std::list<std::shared_ptr<Servo>> servos)
{
    if(!urdfPublisherNode)
    {
        UrdfInitial();
    }
    if(!servoDriveNodeListenerNode)
    {
        StartROSServoDriveListenerNode(servos);
    }
        if(!motionPlanning)
    {
        motionPlanning = std::make_shared<MotionPlanning>(URDF_XML, SRDF_XML);
    }
    std::shared_ptr<JointStatePublisherNode> pub = std::make_shared<JointStatePublisherNode>();
}

void ROSNodeManager::EndNodes()
{
    IsNodeRunning.store(false, std::memory_order_release);
}

std::string ROSNodeManager::UrdfInitial()
{
    urdfPublisherNode  = std::make_shared<UrdfPublisherNode>();
    std::thread([]()
    {
        while (IsNodeRunning.load())
        {
            rclcpp::spin_some(urdfPublisherNode);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();
    return "urdf publisher start";
}

std::string ROSNodeManager::StartROSServoDriveListenerNode(std::list<std::shared_ptr<Servo>> servos)
{ 
    servoDriveNodeListenerNode = std::make_shared<ServoDriveNodeListenerNode>(servos);
    std::thread([servos]()
    {
        while (IsNodeRunning.load())
        {
            rclcpp::spin_some(servoDriveNodeListenerNode);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();
    return "servoDrive listener start";
}

std::vector<std::string> ROSNodeManager::GetActiveNodeName()
{
    auto node = rclcpp::Node::make_shared("node_names_collector");
    auto node_graph = node->get_node_graph_interface();
    return node_graph->get_node_names();
}

bool ROSNodeManager::IsActiveNode(std::string name)
{
    std::vector<std::string> names = ROSNodeManager::GetActiveNodeName();
    return std::find(names.begin(), names.end(), name) != names.end();
}

