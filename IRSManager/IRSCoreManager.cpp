#include "IRSCoreManager.h"

namespace IRSCoreManager
{

    std::atomic<bool> IsNodeRunning(false);

    IRSThreadTools::ThreadSafeQueue<std::vector<Eigen::Vector3d>> &IRSCoreHandle::GetGoalPointsQueue()
    {
        return IRSCoreModule::IRSGoalPoints::GetGoalPointsQueue();
    }

    void IRSCoreHandle::Start()
    {
        if (!IsNodeRunning.load())
        {
            IsNodeRunning.store(true, std::memory_order_release);

            UrdfSrdfXMLInitial();

            GetGoalPointsQueue().reset();

            BrainDeepLearnInterface::GetJsonQueue().reset();

            IRSCoreEnvironment::EnvironmentalPerception::GetInstance()->Start();

            IRSCoreModule::ServoManagerNode::GetInstance()->Start();

            IRSCoreDecision::BrainDecisionNode::GetInstance()->Start();

            auto state = IRSCoreModule::ServoManagerNode::GetInstance()->GetCurrentRobotState();

            ROSNodeInitial(state);

            IRSCoreModule::ROSNodeExecutor::GetInstance()->Start();
        }
        else
            IRS_MESSAGE("IRS core already start");
    }

    void IRSCoreHandle::End()
    {
        if (IsNodeRunning.load())
        {
            IsNodeRunning.store(false, std::memory_order_release);
            IRSCoreEnvironment::EnvironmentalPerception::GetInstance()->Stop();
            BrainDeepLearnInterface::GetJsonQueue().stop();
            IRSCoreDecision::BrainDecisionNode::GetInstance()->Stop();
            GetGoalPointsQueue().stop();
            IRSCoreModule::ServoManagerNode::GetInstance()->Stop();
            IRSCoreModule::ROSNodeExecutor::GetInstance()->Stop();
        }
    }

    void IRSCoreHandle::ResetServoState()
    {
        if (IsNodeRunning.load())
        {
            IRSCoreEnvironment::EnvironmentalPerception::GetInstance()->Stop();
            BrainDeepLearnInterface::GetJsonQueue().stop();
            IRSCoreDecision::BrainDecisionNode::GetInstance()->Stop();
            GetGoalPointsQueue().stop();
            IRSCoreModule::ServoManagerNode::GetInstance()->Stop();

            BrainDeepLearnInterface::GetJsonQueue().reset();
            GetGoalPointsQueue().reset();
            IRSCoreEnvironment::EnvironmentalPerception::GetInstance()->Reset();
            IRSCoreModule::ServoManagerNode::GetInstance()->Reset();
            IRSCoreDecision::BrainDecisionNode::GetInstance()->Reset();

            IRSCoreEnvironment::EnvironmentalPerception::GetInstance()->Start();
            IRSCoreModule::ServoManagerNode::GetInstance()->Start();
            IRSCoreDecision::BrainDecisionNode::GetInstance()->Start();
        }
    }

    void IRSCoreHandle::SetJointPosition(std::string name, double position)
    {
        IRSCoreModule::ServoManagerNode::GetInstance()->SetJointPosition(name, position);
    }

    std::shared_ptr<moveit::core::RobotState> IRSCoreHandle::GetCurrentRobotState()
    {
        return IRSCoreModule::ServoManagerNode::GetInstance()->GetCurrentRobotState();
    }

    std::vector<Eigen::Vector3d> IRSCoreHandle::GetEnvironmentPoints()
    {
        return IRSCoreEnvironment::EnvironmentalPerception::GetInstance()->GetPointClouds();
    }

    void IRSCoreHandle::UrdfSrdfXMLInitial()
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

        std::string oldMeshVisualPath = "${URDF_VISUAL_MESH_PATH}";
        std::string newMeshVisualPath = "file://" + std::string(cwd) + "/Configure/meshes/visual";
        std::string oldMeshcollisionPath = "${URDF_COLLISION_MESH_PATH}";
        std::string newMeshcollisionPath = "file://" + std::string(cwd) + "/Configure/meshes/collision";

        ReplacePathsInUrdf(URDF_XML, oldMeshVisualPath, newMeshVisualPath);
        ReplacePathsInUrdf(URDF_XML, oldMeshcollisionPath, newMeshcollisionPath);

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
    }

    bool IRSCoreHandle::ReplacePathsInUrdf(std::string &urdfContent, const std::string &oldKey, const std::string &newKey)
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

    std::string IRSCoreHandle::ROSNodeInitial(std::shared_ptr<moveit::core::RobotState> robotState)
    {
        std::shared_ptr<IRSCoreModule::UrdfPublisherNode> urdf_publisher_node = std::make_shared<IRSCoreModule::UrdfPublisherNode>();
        IRSCoreModule::ROSNodeExecutor::GetInstance()->AddNode(urdf_publisher_node);

        std::shared_ptr<IRSCoreModule::JointStateListenerNode> joint_state_listen_node = std::make_shared<IRSCoreModule::JointStateListenerNode>(robotState);
        IRSCoreModule::ROSNodeExecutor::GetInstance()->AddNode(joint_state_listen_node);

        return "ros node initial";
    }

    std::vector<std::string> IRSCoreHandle::GetActiveNodeName()
    {
        auto node = rclcpp::Node::make_shared("node_names_collector");
        auto node_graph = node->get_node_graph_interface();
        return node_graph->get_node_names();
    }

    bool IRSCoreHandle::IsActiveNode(std::string name)
    {
        std::vector<std::string> names = IRSCoreHandle::GetActiveNodeName();
        return std::find(names.begin(), names.end(), name) != names.end();
    }

}
