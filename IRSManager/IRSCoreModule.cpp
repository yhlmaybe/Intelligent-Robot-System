#include "IRSCoreModule.h"

namespace IRSCoreModule
{
    IRSThreadTools::ThreadSafeQueue<std::vector<Eigen::Vector3d>> &IRSGoalPoints::GetGoalPointsQueue()
    {
        static IRSThreadTools::ThreadSafeQueue<std::vector<Eigen::Vector3d>> instance;
        return instance;
    }

    ROSNodeExecutor *ROSNodeExecutor::GetInstance()
    {
        static ROSNodeExecutor *instance = new ROSNodeExecutor();
        return instance;
    }

    void ROSNodeExecutor::AddNode(rclcpp::Node::SharedPtr node)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        executor_->add_node(node);
        nodes_.push_back(node);
    }

    void ROSNodeExecutor::Start()
    {
        if (running_)
            return;
        running_ = true;
        executor_thread_ = std::thread([this]()
                                       {
      while (running_ && rclcpp::ok()) 
      {
        executor_->spin_once(std::chrono::milliseconds(100));
      } });
    }

    void ROSNodeExecutor::Stop()
    {
        running_ = false;
        if (executor_thread_.joinable())
        {
            executor_thread_.join();
        }

        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &node : nodes_)
        {
            executor_->remove_node(node);
        }
        nodes_.clear();
        executor_.reset();
    }

    JointStateListenerNode::JointStateListenerNode(std::shared_ptr<moveit::core::RobotState> robotState)
        : Node(SERVE_DRIVE_LISTENER)
    {
        robot_state = std::make_shared<moveit::core::RobotState>(*robotState);

        std::vector<const moveit::core::JointModelGroup *> end_eff = robot_state->getRobotModel()->getEndEffectors();

        for (const moveit::core::JointModelGroup *eff : end_eff)
        {
            std::vector<std::string> names = eff->getLinkModelNames();
            std::string name = names.back();
            end_eff_names.push_back(name);
        }

        subscription = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states",
            rclcpp::QoS(10),
            [this](const sensor_msgs::msg::JointState::SharedPtr msg)
            { this->DoListen(msg); }
            // std::bind(&JointStateListenerNode::DoListen, this, std::placeholders::_1)
        );
        // subscription = this->create_subscription<std_msgs::msg::String>(
        //    "ServeMsg", 10, [this](const std_msgs::msg::String::SharedPtr msg){this->DoListen(msg);});
    }

    void JointStateListenerNode::DoListen(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        for (size_t i = 0; i < msg->name.size(); ++i)
        {
            const std::string &joint_name = msg->name[i];
            double joint_position = msg->position[i];

            robot_state->setVariablePosition(joint_name, joint_position);
        }
        robot_state->update();

        for (size_t i = 0; i < end_eff_names.size(); ++i)
        {
            Eigen::Isometry3d pose = robot_state->getGlobalLinkTransform(end_eff_names[i]);

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);

            oss << "[End Effector: " << end_eff_names[i] << "]\n";

            const Eigen::Vector3d &position = pose.translation();
            oss << "  Position (xyz): ["
                << position.x() << ", "
                << position.y() << ", "
                << position.z() << "]\n";

            Eigen::Quaterniond quat(pose.rotation());
            oss << "  Orientation (xyzw): ["
                << quat.x() << ", "
                << quat.y() << ", "
                << quat.z() << ", "
                << quat.w() << "]\n";

            IRS_FORM_MESSAGE(oss.str(), MessageFunction::JointDatas);
        }
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
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
        static_tf_broadcaster = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);

        description_pub = this->create_publisher<std_msgs::msg::String>("robot_description", rclcpp::QoS(1).transient_local());
        SetupURDF();

        auto subscriber_options = rclcpp::SubscriptionOptions();
        // subscriber_options.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
        //  subscribe to joint state
        joint_state_sub = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states",
            rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::JointState::ConstSharedPtr state)
            { this->CallbackJointState(state); },
            // std::bind(&UrdfPublisherNode::CallbackJointState, this, std::placeholders::_1),
            subscriber_options);

        PublishFixedTransforms();
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
                    IRS_MESSAGE("floating joint is not supported; skipping segment form " + root_str + "to" + child_str);
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

    EnvironmentalPerception::EnvironmentalPerception(unsigned interval_ms) : interval_ms_(interval_ms)
    {
    }

    EnvironmentalPerception *EnvironmentalPerception::GetInstance()
    {
        static EnvironmentalPerception *instance = new EnvironmentalPerception(100);
        return instance;
    }

    std::vector<Eigen::Vector3d> EnvironmentalPerception::GetPointClouds()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return cloud_points_;
    }

    void EnvironmentalPerception::Reset()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        cloud_points_.clear();
    }


    ServoManagerNode::ServoManagerNode()
    {
        node = std::make_shared<rclcpp::Node>(MOTION_SERVO_MANAGER);
        jointState_pub = node->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
        ROSNodeExecutor::GetInstance()->AddNode(node);

        joint_servos = ServoTools::Initiate();

        motion_manager = std::make_shared<MotionManager>(URDF_XML, SRDF_XML);
        motion_manager->UpdateState(UpdateRobotstateBoundary());

        RegisterTask([this]
                     { ClickToGoalPoint(); });
        RegisterTask([this]
                     { PublishJointState(); });
    }

    ServoManagerNode *ServoManagerNode::GetInstance()
    {
        static ServoManagerNode *instance = new ServoManagerNode();
        return instance;
    }

    void ServoManagerNode::Reset()
    {
        std::lock_guard<std::mutex> lock(motion_servo_mtx);
        for (auto it : joint_servos)
        {
            it.second->Reset();
        }
        motion_manager->InitialJointState();
    }

    std::vector<std::shared_ptr<ServoManager>> ServoManagerNode::GetServoManagerFromName(std::vector<std::string> names)
    {
        std::vector<std::shared_ptr<ServoManager>> res;
        for (std::string name : names)
        {
            auto it = joint_servos.find(name);
            if (it != joint_servos.end())
            {
                res.push_back(it->second);
            }
        }
        return res;
    }

    void ServoManagerNode::ClickToGoalPoint()
    {
        std::vector<Eigen::Vector3d> datas = IRSGoalPoints::GetGoalPointsQueue().pop();
        if (datas.size() == 0)
            return;
        std::lock_guard<std::mutex> lock(motion_servo_mtx);
        for (size_t i = 0; i < datas.size(); ++i)
        {
            std::shared_ptr<EndEffector> eff = motion_manager->GetClosestEndEffector(datas[i]);
            // EndEffectorType type = EndEffectorType::LittleEnd;

            // Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

            // pose.translation() = Eigen::Vector3d(0.033, 0.234, -0.079);

            // Eigen::Quaterniond q(0.274, -0.548, -0.095, 0.784);
            // pose.linear() = q.toRotationMatrix();

            // std::shared_ptr<Eigen::Vector3d> pose_nor = std::make_shared<Eigen::Vector3d>(0, 0, -1);
            // Eigen::Isometry3d pose = motion_manager->GetDefaultTouchPoseFromPoint(type, datas[i], *pose_nor);

            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            pose.translation() = datas[i];

            std::vector<std::string> joint_names;
            std::vector<double> joint_values;

            if (!motion_manager->JointIKCal(joint_names, joint_values, eff->type_, pose, true))
            {
                bool isSolution = motion_manager->JointIKCal(joint_names, joint_values, eff->type_, pose, false);
                if (!isSolution)
                {
                    IRS_MESSAGE("point x = %f, y = %f, z = %f is unreachable", datas[i].x(), datas[i].y(), datas[i].z());
                    continue;
                }
            }
            std::vector<Eigen::Vector3d> env_points = EnvironmentalPerception::GetInstance()->GetPointClouds();
            std::vector<TrajectoryPoint> trace = motion_manager->Plan(joint_names, joint_values, env_points, TIMEINTERVAL);
            if (trace.size() == 0)
            {
                IRS_MESSAGE("point x = %f, y = %f, z = %f is planning failure", datas[i].x(), datas[i].y(), datas[i].z());
                continue;
            }
            std::vector<std::shared_ptr<ServoManager>> servos = GetServoManagerFromName(joint_names);
            double servo_time = TIMEINTERVAL / trace.size();
            for (size_t j = 1; j < trace.size(); ++j)
            {
                ServoTools::SetServoPositions(servos, trace[j - 1].positions, trace[j].positions, servo_time);
                sensor_msgs::msg::JointState msg = motion_manager->UpdateRobotStateAndGetMsg(joint_names, trace[j].positions);
                msg.header.stamp = node->get_clock()->now();
                jointState_pub->publish(msg);
            }

            // The two positions of the backtrack simulate clicking the keyboard
            /*size_t k = 0;
            for (size_t j = trace.size() - 1; j > 0; --j)
            {
                if(k > 1) break;
                ServoTools::SetServoPositions(servos, trace[j].positions, trace[j - 1].positions, servo_time);
                sensor_msgs::msg::JointState msg = motion_manager->UpdateRobotStateAndGetMsg(joint_names, trace[j].positions);
                jointState_pub->publish(msg);
                k++;
            }*/
        }
    }

    void ServoManagerNode::SetJointPosition(std::string jointName, double position)
    {
        std::lock_guard<std::mutex> lock(motion_servo_mtx);
        double before_position = motion_manager->GetCurrentRobotState()->getVariablePosition(jointName);
        double after_position = motion_manager->CalGoalJointPosition(jointName, position);

        auto it = joint_servos.find(jointName);
        if (it != joint_servos.end())
        {
            ServoTools::SetServoPosition(it->second, before_position, after_position, TIMEINTERVAL);
            sensor_msgs::msg::JointState msg = motion_manager->UpdateRobotStateAndGetMsg(jointName, after_position);
            msg.header.stamp = node->get_clock()->now();
            jointState_pub->publish(msg);
        }
    }

    std::shared_ptr<moveit::core::RobotState> ServoManagerNode::GetCurrentRobotState()
    {
        std::lock_guard<std::mutex> lock(motion_servo_mtx);
        return motion_manager->GetCurrentRobotState();
    }

    std::shared_ptr<moveit::core::RobotState> ServoManagerNode::UpdateRobotstateBoundary()
    {
        auto urdf_model = urdf::parseURDF(URDF_XML);
        srdf::ModelSharedPtr srdf_model = std::make_shared<srdf::Model>();
        srdf_model->initString(*urdf_model, SRDF_XML);
        std::shared_ptr<moveit::core::RobotModel> robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
        for (auto it : joint_servos)
        {
            moveit::core::JointModel *joint_model = robot_model->getJointModel(it.first);
            std::vector<moveit::core::VariableBounds> var_bounds = joint_model->getVariableBounds();

            if (!var_bounds.empty())
            {
                double min = var_bounds[0].min_position_;
                double max = var_bounds[0].max_position_;

                double update_min = 0, update_max = 0;
                if (it.second->IsUpdateJointPosBound(update_max, update_min, max, min))
                {
                    var_bounds[0].min_position_ = update_min;
                    var_bounds[0].max_position_ = update_max;
                }
            }
            joint_model->setVariableBounds(it.first, var_bounds[0]);
        }
        std::shared_ptr<moveit::core::RobotState> robot_state = std::make_shared<moveit::core::RobotState>(robot_model);
        robot_state->setToDefaultValues();
        return robot_state;
    }

    void ServoManagerNode::PublishJointState()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::lock_guard<std::mutex> lock(motion_servo_mtx);
        sensor_msgs::msg::JointState msg = motion_manager->GetCurrentJointStateMsg();
        msg.header.stamp = node->get_clock()->now();
        /*if (!msg.name.empty())
        {
            IRS_MESSAGE("no message");
        }
        else IRS_MESSAGE("publish message");*/
        jointState_pub->publish(msg);
    }
}
