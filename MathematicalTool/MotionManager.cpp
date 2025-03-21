#include "MotionManager.h"

EndEffector::EndEffector(std::string URDF, std::string name, std::string partGroupName, std::string completeGroupName, std::string partGroupFirstLinkName, std::string completeGroupFirstLinkName, EndEffectorType type)
{
    name = name;
    partGroupName = partGroupName;
    completeGroupName = completeGroupName;
    type = type;
    partGroupIkSolver = std::make_shared<IRS_IK::IRS_IK>(partGroupFirstLinkName, name, URDF);
    completeGroupIkSolver = std::make_shared<IRS_IK::IRS_IK>(completeGroupFirstLinkName, name, URDF);
}

MotionManager::MotionManager(std::string urdf, std::string srdf)
{
    auto urdf_model = urdf::parseURDF(urdf);

    srdf::ModelSharedPtr srdf_model = std::make_shared<srdf::Model>();
    srdf_model->initString(*urdf_model, srdf);
    robotModel = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
    
    if(!robotModel)
    {
        IRS_MESSAGE("set robot model error");
        return;
    }
    robotState = std::make_shared<moveit::core::RobotState>(robotModel);

    std::vector<srdf::Model::EndEffector> endEffectors = srdf_model->getEndEffectors();
    for (srdf::Model::EndEffector& endEffector : endEffectors)
    {
        std::string partGroupName = "";
        std::string completeGroupName = "";
        EndEffectorType type = EndEffectorType::NoneType;
        if(endEffector.name_ == "thumb_end")
        {
            partGroupName = "thumb";
            completeGroupName = "thumbWrist";
            type = EndEffectorType::ThumbEnd;
        }
        else if(endEffector.name_ == "index_end")
        {
            partGroupName = "index";
            completeGroupName = "indexWrist";
            type = EndEffectorType::IndexEnd;
        }
        else if(endEffector.name_ == "fourth_end")
        {
            partGroupName = "fourth";
            completeGroupName = "fourthWrist";
            type = EndEffectorType::FourthEnd;
        }
        else if(endEffector.name_ == "mid_end")
        {
            partGroupName = "mid";
            completeGroupName = "midWrist";
            type = EndEffectorType::MidEnd;
        }
        else if(endEffector.name_ == "little_end")
        {
            partGroupName = "little";
            completeGroupName = "littleWrist";
            type = EndEffectorType::LittleEnd;
        }
        else if(endEffector.name_ == "wrist_end")
        {
            partGroupName = "wrist";
            completeGroupName = "wrist";
            type = EndEffectorType::WristEnd;
        }
        else
        {
            continue;
        }
        moveit::core::JointModelGroup *partGroup = robotModel->getJointModelGroup(partGroupName);
        std::string partFirstLInkName = "";
        std::vector<const moveit::core::LinkModel *> partLinks = partGroup->getLinkModels();
        if (!partLinks.empty())
        {
            partFirstLInkName = partLinks[0]->getName();
        }
        moveit::core::JointModelGroup *completeGroup = robotModel->getJointModelGroup(completeGroupName);
        std::string completeFirstLInkName = "";
        std::vector<const moveit::core::LinkModel *> completeLinks = completeGroup->getLinkModels();
        if (!completeLinks.empty())
        {
            completeFirstLInkName = completeLinks[0]->getName();
        }
        std::shared_ptr<EndEffector> endEff = std::make_shared<EndEffector>(URDF_XML, endEffector.name_, partGroupName, completeGroupName, partFirstLInkName, completeFirstLInkName, type);
        endEffectorsMap[type] = endEff;
    }

    InitialJointState();

    omplTool = std::make_shared<PlanningTool::OMPLTool>(robotModel);
}

sensor_msgs::msg::JointState MotionManager::GetCurrentJointStateMsg()
{
    sensor_msgs::msg::JointState jointStateMsgs = sensor_msgs::msg::JointState();
    std::vector<std::string> jointNames = robotModel->getVariableNames();
    double* jointPositions = robotState->getVariablePositions();
    std::vector<double> jointPositionsVec(jointPositions, jointPositions + jointNames.size());
    jointStateMsgs.name = jointNames;
    jointStateMsgs.position = jointPositionsVec;
    return jointStateMsgs;
}

Eigen::Isometry3d MotionManager::ConvertPoseFromRelBaseToRelEnd(EndEffectorType endEffector, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d endEffectorTF = robotState->getGlobalLinkTransform(endEffectorsMap[endEffector]->name);
    return (endEffectorTF.inverse() * pose);
}

Eigen::Isometry3d MotionManager::ConvertPoseFromRelEndToRelBase(EndEffectorType endEffector, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d endEffectorTF = robotState->getGlobalLinkTransform(endEffectorsMap[endEffector]->name);
    return endEffectorTF * pose;
}

Eigen::Isometry3d MotionManager::ConvertPoseFromRelEndToRelAny(EndEffectorType endEffector, std::string anyLinkName, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d endEffectorTF = robotState->getGlobalLinkTransform(endEffectorsMap[endEffector]->name);
    Eigen::Isometry3d linkTF = robotState->getGlobalLinkTransform(anyLinkName);
    Eigen::Isometry3d TF = endEffectorTF * linkTF.inverse();
    return TF * pose;
}

void MotionManager::InitialJointState()
{
    robotState->setToDefaultValues("r_arm", "r_arm_home");
    robotState->update();
}

double MotionManager::CalGoalJointPosition(std::string jointName, double position)
{
    moveit::core::JointModel* joint_model = robotModel->getJointModel(jointName);
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
    return std::make_shared<moveit::core::RobotState>(*robotState);
}

void MotionManager::UpdateState(std::shared_ptr<moveit::core::RobotState> state)
{
    robotState = state;
}

EndEffectorType MotionManager::GetClosestEndEffector(Eigen::Vector3d pointRelBase)
{
    int i = 0;
    EndEffectorType type = EndEffectorType::NoneType;
    double min_distance = 0;
    for (auto endEff : endEffectorsMap)
    {
        Eigen::Isometry3d ee_pose = robotState->getGlobalLinkTransform(endEff.second->name);
        Eigen::Vector3d position = ee_pose.translation(); // get (x,y,z)
        double distance = (position - pointRelBase).norm();
        if (i == 0)
        {
            min_distance = distance;
            type = endEff.first;
        }
        else
        {
            if (distance < min_distance)
            {
                min_distance = distance;
                type = endEff.first;
            }
        }
    }
    return type;
}

Eigen::Isometry3d MotionManager::GetDefaultTouchPoseFromPoint(EndEffectorType endEffector, Eigen::Vector3d pointRelBase, Eigen::Vector3d pointNormal, double fingerAngle)
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

bool MotionManager::JointIKCal(std::vector<std::string>& jointNameResult, std::vector<double>& jointValueResult, EndEffectorType endEffector, Eigen::Isometry3d pointRelativeEndEff, bool isPart)
{
    jointNameResult.clear();
    jointValueResult.clear();
    auto eff = endEffectorsMap[endEffector];
    moveit::core::JointModelGroup* group;
    KDL::Chain chain;
    bool valid;
    std::shared_ptr<IRS_IK::IRS_IK> ik = nullptr;
    if(isPart) 
    {
        group = robotModel->getJointModelGroup(eff->partGroupName);
        ik = eff->partGroupIkSolver;
        valid = ik->getKDLChain(chain);    
    }
    else 
    {
        group = robotModel->getJointModelGroup(eff->completeGroupName);
        ik = eff->completeGroupIkSolver;
        valid = ik->getKDLChain(chain);
    }
    const std::vector<std::string>& jointNames = group->getVariableNames();   
    if (!valid)
    {
        IRS_MESSAGE("There was no valid KDL chain found");
        return false;
    }
    KDL::JntArray nominal(chain.getNrOfJoints());

    for (size_t i = 0; i < jointNames.size(); ++i)
    {
        nominal(i) = robotState->getVariablePosition(jointNames[i]);
    }
    Eigen::Isometry3d poseRelFirstLink = ConvertPoseFromRelEndToRelAny(endEffector, jointNames.front(), pointRelativeEndEff);
    KDL::Frame endEffectorPose = ConvertToFrame(poseRelFirstLink);  
    KDL::JntArray angle(chain.getNrOfJoints());
    int rc = ik->CartToJnt(nominal, endEffectorPose, angle);
    if (rc == 0)
    {
        for (size_t i = 0; i < jointNames.size(); ++i)
        {
            jointNameResult.push_back(jointNames[i]);
            jointValueResult.push_back(angle(i));
        }
    }
    return rc == 0;
}

std::vector<TrajectoryPoint> MotionManager::Plan(std::vector<std::string> goalJointNames, std::vector<double> goalJointPositions, std::vector<Eigen::Vector3d> envPointClouds, double totalTime, int interpolateCount)
{
    std::vector<TrajectoryPoint> res;
    ompl::base::PathPtr path = omplTool->Plan(robotState, goalJointNames, goalJointPositions, envPointClouds);
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
    robotState->setVariablePositions(goalJointNames, goalJointPositions);
    robotState->update();
    return GetCurrentJointStateMsg();
}

sensor_msgs::msg::JointState MotionManager::UpdateRobotStateAndGetMsg(std::string goalJointName, double goalJointPosition)
{
    robotState->setVariablePosition(goalJointName, goalJointPosition);
    robotState->update();
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
    Eigen::Matrix3d rotationMatrix = Isometry3d.linear();
    Eigen::Quaterniond quaternion(rotationMatrix);
    KDL::Rotation rotation = KDL::Rotation::Quaternion(quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
    Eigen::Vector3d translation_vector = Isometry3d.translation();
    KDL::Vector translation(translation_vector.x(), translation_vector.y(), translation_vector.z());
    return KDL::Frame(rotation, translation);
}