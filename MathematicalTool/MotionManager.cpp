#include "MotionManager.h"

EndEffector::EndEffector(std::string URDF, std::string name, std::string partGroupName, std::string completeGroupName, std::string partGroupFirstLinkName, std::string completeGroupFirstLinkName, EndEffectorType type)
{
    name_ = name;
    part_group_name = partGroupName;
    complete_group_name = completeGroupName;
    type_ = type;
    part_group_first_link_name = partGroupFirstLinkName;
    complete_group_first_link_name = completeGroupFirstLinkName;
    part_group_IkSolver = std::make_shared<IRS_IK::IRS_IK>(partGroupFirstLinkName, name, URDF, 0.005, 1e-6);
    complete_group_IkSolver = std::make_shared<IRS_IK::IRS_IK>(completeGroupFirstLinkName, name, URDF, 0.005, 1e-6);
}

MotionManager::MotionManager(std::string urdf, std::string srdf)
{
    auto urdf_model = urdf::parseURDF(urdf);

    srdf::ModelSharedPtr srdf_model = std::make_shared<srdf::Model>();
    srdf_model->initString(*urdf_model, srdf);
    robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
    
    if(!robot_model)
    {
        IRS_MESSAGE("set robot model error");
        return;
    }
    robot_state = std::make_shared<moveit::core::RobotState>(robot_model);

    std::vector<const moveit::core::JointModelGroup*> end_eff = robot_model->getEndEffectors();

    for(const moveit::core::JointModelGroup* eff : end_eff)
    {
        std::vector<std::string> names = eff->getLinkModelNames();
        std::string name = names.back();

        std::string partGroupName = "";
        std::string completeGroupName = "";
        EndEffectorType type = EndEffectorType::NoneType;
        if(name == "ThumbEnd_Link")
        {
            partGroupName = "thumb";
            completeGroupName = "thumbWrist";
            type = EndEffectorType::ThumbEnd;
        }
        else if(name == "IndexEnd_Link")
        {
            partGroupName = "index";
            completeGroupName = "indexWrist";
            type = EndEffectorType::IndexEnd;
        }
        else if(name == "FourthEnd_Link")
        {
            partGroupName = "fourth";
            completeGroupName = "fourthWrist";
            type = EndEffectorType::FourthEnd;
        }
        else if(name == "MidEnd_Link")
        {
            partGroupName = "mid";
            completeGroupName = "midWrist";
            type = EndEffectorType::MidEnd;
        }
        else if(name == "LittleEnd_Link")
        {
            partGroupName = "little";
            completeGroupName = "littleWrist";
            type = EndEffectorType::LittleEnd;
        }
        else
        {
            continue;
        }
        moveit::core::JointModelGroup *partGroup = robot_model->getJointModelGroup(partGroupName);
        std::string partFirstLInkName = "";
        std::vector<const moveit::core::LinkModel *> partLinks = partGroup->getLinkModels();
        if (!partLinks.empty())
        {
            partFirstLInkName = partLinks[0]->getParentLinkModel()->getName();
        }
        moveit::core::JointModelGroup *completeGroup = robot_model->getJointModelGroup(completeGroupName);
        std::string completeFirstLInkName = "";
        std::vector<const moveit::core::LinkModel *> completeLinks = completeGroup->getLinkModels();
        if (!completeLinks.empty())
        {
            completeFirstLInkName = completeLinks[0]->getName();
        }
        std::shared_ptr<EndEffector> endEff = std::make_shared<EndEffector>(URDF_XML, name, partGroupName, completeGroupName, partFirstLInkName, completeFirstLInkName, type);
        end_effectors_map[type] = endEff;
    }
    InitialJointState();

    ompl_tool = std::make_shared<PlanningTool::OMPLTool>(robot_model);
}

sensor_msgs::msg::JointState MotionManager::GetCurrentJointStateMsg()
{
    sensor_msgs::msg::JointState jointStateMsgs = sensor_msgs::msg::JointState();
    std::vector<std::string> jointNames = robot_model->getVariableNames();
    double* jointPositions = robot_state->getVariablePositions();
    std::vector<double> jointPositionsVec(jointPositions, jointPositions + jointNames.size());
    jointStateMsgs.name = jointNames;
    jointStateMsgs.position = jointPositionsVec;
    return jointStateMsgs;
}

Eigen::Isometry3d MotionManager::ConvertPoseFromRelBaseToRelEnd(EndEffectorType endEffector, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d endEffectorTF = robot_state->getGlobalLinkTransform(end_effectors_map[endEffector]->name_);
    return endEffectorTF.inverse() * pose;
}

Eigen::Isometry3d MotionManager::ConvertPoseFromRelEndToRelBase(EndEffectorType endEffector, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d endEffectorTF = robot_state->getGlobalLinkTransform(end_effectors_map[endEffector]->name_);
    return endEffectorTF * pose;
}

Eigen::Isometry3d MotionManager::ConvertPoseFromRelEndToRelAny(EndEffectorType endEffector, std::string anyLinkName, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d endEffectorTF = robot_state->getGlobalLinkTransform(end_effectors_map[endEffector]->name_);
    Eigen::Isometry3d linkTF = robot_state->getGlobalLinkTransform(anyLinkName);
    Eigen::Isometry3d TF = linkTF.inverse() * endEffectorTF;
    return TF * pose;
}

Eigen::Isometry3d MotionManager::ConvertPoseFromRelBaseToRelAny(std::string anyLinkName, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d linkTF = robot_state->getGlobalLinkTransform(anyLinkName);
    return linkTF.inverse() * pose;
}

void MotionManager::InitialJointState()
{
    const moveit::core::JointModelGroup* group = robot_model->getJointModelGroup("r_arm");
    robot_state->setToDefaultValues(group, "r_arm_home");
    robot_state->update();
}

double MotionManager::CalGoalJointPosition(std::string jointName, double position)
{
    moveit::core::JointModel* joint_model = robot_model->getJointModel(jointName);
    const moveit::core::VariableBounds& bounds = joint_model->getVariableBounds(jointName);

    double limited_position = position;

    if (position < bounds.min_position_) 
    {
        IRS_MESSAGE("the %s position is set beyond the boundary and the joint position is set to the upper boundary", jointName);
        limited_position = bounds.min_position_;
    }
    else if (position > bounds.max_position_) 
    {
        IRS_MESSAGE("the %s position is set beyond the boundary and the joint position is set to the lower boundary", jointName);
        limited_position = bounds.max_position_;
    }

    return limited_position;
}

std::shared_ptr<moveit::core::RobotState> MotionManager::GetCurrentRobotState()
{
    return std::make_shared<moveit::core::RobotState>(*robot_state);
}

void MotionManager::UpdateState(std::shared_ptr<moveit::core::RobotState> state)
{
    robot_state = state;
}

std::shared_ptr<EndEffector> MotionManager::GetClosestEndEffector(Eigen::Vector3d pointRelBase)
{
    int i = 0;
    std::shared_ptr<EndEffector> eff;
    double min_distance = 0;
    for (auto endEff : end_effectors_map)
    {
        Eigen::Isometry3d ee_pose = robot_state->getGlobalLinkTransform(endEff.second->name_);
        Eigen::Vector3d position = ee_pose.translation(); // get (x,y,z)
        double distance = (position - pointRelBase).norm();
        if (i == 0)
        {
            min_distance = distance;
            eff = endEff.second;
        }
        else
        {
            if (distance < min_distance)
            {
                min_distance = distance;
                eff = endEff.second;
            }
        }
        i++;
    }
    return eff;
}

Eigen::Isometry3d MotionManager::GetDefaultTouchPoseFromPoint(Eigen::Vector3d pointRelBase, Eigen::Vector3d pointNormal, double fingerAngle)
{
    double eps = 1e-6;
    Eigen::Vector3d op = pointRelBase;
    Eigen::Vector3d x = pointNormal.normalized();
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = pointRelBase;

    Eigen::Vector3d y_proj = op - op.dot(x) * x;
    if (y_proj.norm() < eps)
    {
        y_proj = Eigen::Vector3d::UnitY() - Eigen::Vector3d::UnitY().dot(x) * x;
        if (y_proj.norm() < eps)
        {
            y_proj = Eigen::Vector3d::UnitZ() - Eigen::Vector3d::UnitZ().dot(x) * x;
        }
    }
    Eigen::Vector3d y = y_proj.normalized();
    Eigen::Vector3d z = x.cross(y);
    pose.linear().col(0) = x;
    pose.linear().col(1) = y;
    pose.linear().col(2) = z;

    //check pointNormal and op is parallel
    Eigen::Vector3d cross = op.cross(pointNormal);
    if (cross.norm() < eps) return pose;

    Eigen::Vector3d n = cross.normalized();
    double theta_rad = fingerAngle * M_PI / 180.0;
    Eigen::AngleAxisd rotation(theta_rad, n);
    Eigen::Vector3d x_new = rotation * x;

    Eigen::Vector3d y_new_proj = op - op.dot(x_new) * x_new;
    if (y_new_proj.norm() < eps)
    {
        y_new_proj = Eigen::Vector3d::UnitY() - Eigen::Vector3d::UnitY().dot(x_new) * x_new;
        if (y_new_proj.norm() < eps)
        {
            y_new_proj = Eigen::Vector3d::UnitZ() - Eigen::Vector3d::UnitZ().dot(x_new) * x_new;
        }
    }
    Eigen::Vector3d y_new = y_new_proj.normalized();
    Eigen::Vector3d z_new = x_new.cross(y_new);

    pose.linear().col(0) = x_new;
    pose.linear().col(1) = y_new;
    pose.linear().col(2) = z_new;
    return pose;
}

bool MotionManager::JointIKCal(std::vector<std::string>& jointNameResult, std::vector<double>& jointValueResult, EndEffectorType endEffector, Eigen::Isometry3d poseRelativeBaseLink, bool isPart, bool ignorePoseRotation)
{
    jointNameResult.clear();
    jointValueResult.clear();
    auto eff = end_effectors_map[endEffector];
    moveit::core::JointModelGroup* group;
    KDL::Chain chain;
    bool valid;
    std::shared_ptr<IRS_IK::IRS_IK> ik = nullptr;
    std::string first_link_name = "";
    KDL::Frame endEffectorPose;
    if(isPart) 
    {
        group = robot_model->getJointModelGroup(eff->part_group_name);
        ik = eff->part_group_IkSolver;
        valid = ik->getKDLChain(chain);    
        first_link_name = eff->part_group_first_link_name;
        Eigen::Isometry3d poseRelFirstLink = ConvertPoseFromRelBaseToRelAny(first_link_name, poseRelativeBaseLink);
        endEffectorPose = ConvertToFrame(poseRelFirstLink);  
    }
    else 
    {
        group = robot_model->getJointModelGroup(eff->complete_group_name);
        ik = eff->complete_group_IkSolver;
        valid = ik->getKDLChain(chain);
        first_link_name = eff->complete_group_first_link_name;
        endEffectorPose = ConvertToFrame(poseRelativeBaseLink); 
    }
    const std::vector<std::string>& jointNames = group->getVariableNames();   
    if (!valid)
    {
        IRS_MESSAGE("There was no valid KDL chain found");
        return false;
    }
    KDL::JntArray nominal(chain.getNrOfJoints());

    //KDL::Frame pp;
    //KDL::JntArray q(chain.getNrOfJoints());
    //for (uint j = 0; j < jointNames.size(); j++)
    //{
    //    if(jointNames[j] == "LittleFingerArth_2_Joint") q(j) = 1;
    //    else q(j) = robot_state->getVariablePosition(jointNames[j]);
    //}
    //KDL::ChainFkSolverPos_recursive fk_solver(chain);
    //fk_solver.JntToCart(q, pp);

    for (size_t i = 0; i < jointNames.size(); ++i)
    {
        nominal(i) = robot_state->getVariablePosition(jointNames[i]);
    }

    KDL::JntArray angle(chain.getNrOfJoints());

    KDL::Twist bounds = KDL::Twist::Zero();
    if(ignorePoseRotation)
    {
        bounds.rot = KDL::Vector(M_PI, M_PI, M_PI);
    }

    int rc = ik->CartToJnt(nominal, endEffectorPose, angle, bounds);
    //int rc = ik->CartToJnt(nominal, pp, angle);
    if (rc >= 0)
    {
        for (size_t i = 0; i < jointNames.size(); ++i)
        {
            jointNameResult.push_back(jointNames[i]);
            jointValueResult.push_back(angle(i));
        }
    }
    return rc >= 0;
}

std::vector<TrajectoryPoint> MotionManager::Plan(std::vector<std::string> goalJointNames, std::vector<double> goalJointPositions, std::vector<Eigen::Vector3d> envPointClouds, double totalTime, int interpolateCount)
{
    std::vector<TrajectoryPoint> res;
    ompl::base::PathPtr path = ompl_tool->Plan(robot_state, goalJointNames, goalJointPositions, envPointClouds);
    if (path == nullptr) return res;
    if (ompl::geometric::PathGeometric *pathGeo = dynamic_cast<ompl::geometric::PathGeometric *>(path.get()))
    {
        int jointCount = goalJointNames.size();
        int stateCount = pathGeo->getStateCount();
        if (interpolateCount != 0)
            pathGeo->interpolate(stateCount + interpolateCount);
        for (size_t i = 0; i < pathGeo->getStateCount(); ++i)
        {
            ompl::base::State *state = pathGeo->getState(i);
            ompl::base::RealVectorStateSpace::StateType *joint_angles = state->as<ompl::base::RealVectorStateSpace::StateType>();

            TrajectoryPoint point;
            point.positions.assign(joint_angles->values, joint_angles->values + jointCount);
            point.time_from_start = totalTime * static_cast<double>(i) / (pathGeo->getStateCount() - 1);

            res.push_back(point);
        }
    }
    return res;
}

sensor_msgs::msg::JointState MotionManager::UpdateRobotStateAndGetMsg(std::vector<std::string> goalJointNames, std::vector<double> goalJointPositions)
{
    robot_state->setVariablePositions(goalJointNames, goalJointPositions);
    robot_state->update();
    return GetCurrentJointStateMsg();
}

sensor_msgs::msg::JointState MotionManager::UpdateRobotStateAndGetMsg(std::string goalJointName, double goalJointPosition)
{
    robot_state->setVariablePosition(goalJointName, goalJointPosition);
    robot_state->update();
    return GetCurrentJointStateMsg();
}

Eigen::Isometry3d MotionManager::ConvertToIsometry3d(KDL::Frame frame)
{
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.translation() = Eigen::Vector3d(frame.p.x(), frame.p.y(), frame.p.z());
    double x, y, z, w;
    frame.M.GetQuaternion(x, y, z, w);
    Eigen::Quaterniond quat(w, x, y, z);
    result.linear() = quat.toRotationMatrix();
    return result;
}

KDL::Frame MotionManager::ConvertToFrame(Eigen::Isometry3d Isometry3d)
{
    Eigen::Matrix3d eigen_rot = Isometry3d.linear();
    Eigen::Vector3d eigen_trans = Isometry3d.translation();

    KDL::Rotation kdl_rot(
        eigen_rot(0,0), eigen_rot(0,1), eigen_rot(0,2),
        eigen_rot(1,0), eigen_rot(1,1), eigen_rot(1,2),
        eigen_rot(2,0), eigen_rot(2,1), eigen_rot(2,2)
    );

    KDL::Vector kdl_trans(
        eigen_trans.x(), 
        eigen_trans.y(), 
        eigen_trans.z()
    );

    return KDL::Frame(kdl_rot, kdl_trans);
}