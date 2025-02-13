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

MotionPlanning::MotionPlanning(std::string urdf, std::string srdf)
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
        EndEffectorType type = EndEffectorType::None;
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

    robotState->setToDefaultValues();

    sharedSphere = std::make_shared<fcl::Sphered>(0.01);
    envManager = std::make_shared<fcl::DynamicAABBTreeCollisionManagerd>();
    omplTool = std::make_shared<PlanningTool::OMPLTool>(robotModel, envManager);
}

sensor_msgs::msg::JointState MotionPlanning::GetCurrentJointStateMsg()
{
    sensor_msgs::msg::JointState jointStateMSgs = sensor_msgs::msg::JointState();
    std::vector<std::string> jointNames = robotModel->getVariableNames();
    double* jointPositions = robotState->getVariablePositions();
    std::vector<double> jointPositionsVec(jointPositions, jointPositions + jointNames.size());
    jointStateMSgs.name = jointNames;
    jointStateMSgs.position = jointPositionsVec;
    return jointStateMSgs;
}

Eigen::Isometry3d MotionPlanning::ConvertPoseFromRelBaseToRelEnd(EndEffectorType endEffector, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d endEffectorTF = robotState->getGlobalLinkTransform(endEffectorsMap[endEffector]->name);
    return (endEffectorTF.inverse() * pose);
}

Eigen::Isometry3d MotionPlanning::ConvertPoseFromRelEndToRelBase(EndEffectorType endEffector, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d endEffectorTF = robotState->getGlobalLinkTransform(endEffectorsMap[endEffector]->name);
    return endEffectorTF * pose;
}

Eigen::Isometry3d MotionPlanning::ConvertPoseFromRelEndToRelAny(EndEffectorType endEffector, std::string anyLinkName, Eigen::Isometry3d pose)
{
    Eigen::Isometry3d endEffectorTF = robotState->getGlobalLinkTransform(endEffectorsMap[endEffector]->name);
    Eigen::Isometry3d linkTF = robotState->getGlobalLinkTransform(anyLinkName);
    Eigen::Isometry3d TF = endEffectorTF * linkTF.inverse();
    return TF * pose;
}

bool MotionPlanning::JointIKCal(std::map<std::string, double>& result, EndEffectorType endEffector, Eigen::Isometry3d pointRelativeEndEff, bool isPart)
{
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
    for (size_t i = 0; i < jointNames.size(); ++i)
    {
        result[jointNames[i]] = (angle(i));
    }
    return rc == 0;
}

bool MotionPlanning::PlanAndExecute(std::map<std::string, double> goalNameAngles)
{

}

void MotionPlanning::UpDateEnvironment(std::vector<Eigen::Vector3d> pointClouds)
{
    
}

Eigen::Isometry3d MotionPlanning::ConvertToIsometry3d(KDL::Frame frame)
{
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.translation() = Eigen::Vector3d(frame.p.x(), frame.p.y(), frame.p.z());
    double x, y, z, w;
    frame.M.GetQuaternion(x, y, z, w);
    Eigen::Quaterniond quat(w, x, y, z);
    result.linear() = quat.toRotationMatrix();
    return result;
}

KDL::Frame MotionPlanning::ConvertToFrame(Eigen::Isometry3d Isometry3d)
{
    Eigen::Matrix3d rotationMatrix = Isometry3d.linear();
    Eigen::Quaterniond quaternion(rotationMatrix);
    KDL::Rotation rotation = KDL::Rotation::Quaternion(quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
    Eigen::Vector3d translation_vector = Isometry3d.translation();
    KDL::Vector translation(translation_vector.x(), translation_vector.y(), translation_vector.z());
    return KDL::Frame(rotation, translation);
}