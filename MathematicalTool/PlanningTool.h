#ifndef PLANNINGTOOL_H
#define PLANNINGTOOL_H

#include <fcl/fcl.h>
#include <geometric_shapes/geometric_shapes/shapes.h>
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/narrowphase/collision.h>
#include <fcl/narrowphase/collision_object.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/collision_detection/collision_tools.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/base/samplers/ObstacleBasedValidStateSampler.h>

#include "../include/IRSParametersData.h"

namespace PlanningTool
{   
    struct CollisionData 
    {
    fcl::CollisionRequestd request;
    fcl::CollisionResultd result;
    bool done = false;
    };

    struct LinkCollisionObject 
    {
        std::shared_ptr<fcl::CollisionGeometryd> geometry;
        fcl::Transform3d transform; 
    };

    class FCLTool
    {
    public:

        FCLTool(std::shared_ptr<moveit::core::RobotModel> robotModel, std::string completePlanningGroupName);
        ~FCLTool(); 
        std::vector<std::pair<std::vector<LinkCollisionObject>, std::vector<LinkCollisionObject>>> GetSelfAllowLinkColls();
        std::vector<std::pair<std::vector<fcl::CollisionObjectd*>, std::vector<fcl::CollisionObjectd*>>> GetSelfAllowLinkFclColls();
        bool IsSelfCollision();
        bool IsEnvCollision(std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd> envManager);
        void SetCollObjTF(std::string name, Eigen::Isometry3d tf);
        std::vector<fcl::CollisionObjectd*> GetCollObj();
        void UpdateLInkTF(std::vector<std::string> names, std::vector<Eigen::Isometry3d> tfs);
        void UpdateLInkTF(moveit::core::RobotState state);
        void UpdateLInkTF(moveit::core::RobotState state, std::vector<std::string> names);

    private:
        std::shared_ptr<fcl::CollisionGeometryd> CreateFCLGeometry(shapes::ShapeConstPtr& shape); 
        std::vector<fcl::CollisionObjectd*> ConvertToFclColl(std::vector<LinkCollisionObject> linkObjs);

        std::shared_ptr<moveit::core::RobotModel> robotModel;
        std::vector<fcl::CollisionObjectd*> collObjs;
        std::vector<std::pair<std::vector<fcl::CollisionObjectd*>, std::vector<fcl::CollisionObjectd*>>> allowSelfFclCollPairs;
        std::unordered_map<std::string, std::vector<fcl::CollisionObjectd*>> linkFclCollisions;
        std::unordered_map<std::string, std::vector<LinkCollisionObject>> linkCollisions;
    };

    class CustomStateValidator : public ompl::base::StateValidityChecker 
    {
    public:
        CustomStateValidator
        (ompl::base::SpaceInformationPtr& si,
        std::vector<std::string> linkNames,
        std::vector<std::string> jointNames,
        std::shared_ptr<moveit::core::RobotState> robotState,
        std::shared_ptr<FCLTool> fclTool,
        std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd>  envManager);
        
    bool isValid(const ompl::base::State* state) const override ;

private:
    std::shared_ptr<moveit::core::RobotState> robotState;
    std::shared_ptr<FCLTool> fclTool;
    std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd> envManager;
    std::vector<std::string> linkNames;
    std::vector<std::string> jointNames;
    size_t dimsion;
    };

    class OMPLTool
    {
    public:
        OMPLTool(std::shared_ptr<moveit::core::RobotModel> robotModel, std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd>  envManager);

        ompl::base::PathPtr plan(std::shared_ptr<moveit::core::RobotState> robotState, std::vector<std::string> goalJointNames, std::vector<double> goalJointAngles);

    private:
        void UpdateCollisionTF();

        std::shared_ptr<moveit::core::RobotModel> robotModel;
        std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd> envManager;
        std::map<std::string, moveit::core::VariableBounds> jointBounds;
        std::shared_ptr<FCLTool> fclTool;
    };

}

#endif