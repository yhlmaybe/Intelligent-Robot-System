#include "PlanningTool.h"

namespace PlanningTool
{
    FCLTool::FCLTool(std::shared_ptr<moveit::core::RobotModel> robotModel, std::string completePlanningGroupName)
    {   
        robot_model = robotModel;
        moveit::core::JointModelGroup *jointGroup = robot_model->getJointModelGroup(completePlanningGroupName);
        std::vector<std::string> linkNames = jointGroup->getLinkModelNames();
        for (auto &name : linkNames)
        {
            moveit::core::LinkModel *link = robot_model->getLinkModel(name);         
            std::vector<shapes::ShapeConstPtr> shapes = link->getShapes();
            std::vector<LinkCollisionObject> collisionObjects;
            for (size_t i = 0; i < shapes.size(); ++i)
            {
                auto &shape = shapes[i];
                std::shared_ptr<fcl::CollisionGeometryd> fclGeo = CreateFCLGeometry(shape);
                Eigen::Isometry3d originTf = link->getCollisionOriginTransforms()[i];
                fcl::Transform3d fclTf(originTf);
                collisionObjects.push_back({fclGeo, fclTf});
            }
            if (!collisionObjects.empty())
            {
                link_collisions[name] = std::move(collisionObjects);
                std::vector<fcl::CollisionObjectd*> fclColls = ConvertToFclColl(collisionObjects);
                for (fcl::CollisionObjectd* obj : fclColls) { coll_objs.push_back(obj); }
                link_fcl_collisions[name] = fclColls;
            }
        }
        allow_self_fcl_collPairs = GetSelfAllowLinkFclColls();

        robot_manager = std::make_shared<fcl::DynamicAABBTreeCollisionManagerd>();
        robot_manager->registerObjects(coll_objs);
        robot_manager->update();
    }

    FCLTool::~FCLTool()
    {
        for(auto obj : coll_objs) 
        {
            delete obj;
        }
    }

    std::vector<std::pair<std::vector<LinkCollisionObject>, std::vector<LinkCollisionObject>>> FCLTool::GetSelfAllowLinkColls()
    {
        std::vector<std::pair<std::vector<LinkCollisionObject>, std::vector<LinkCollisionObject>>> res;
        std::vector<srdf::Model::CollisionPair> collisionPairs  = robot_model->getSRDF()->getEnabledCollisionPairs(); 
        for(size_t i = 0; i < collisionPairs.size(); ++i)
        {   
            srdf::Model::CollisionPair pair = collisionPairs[i];
            std::vector<LinkCollisionObject> obj1 = link_collisions[pair.link1_];
            std::vector<LinkCollisionObject> obj2 = link_collisions[pair.link2_];
            res.push_back(std::pair<std::vector<LinkCollisionObject>, std::vector<LinkCollisionObject>>(obj1, obj2));
        }
        return res;
    }

    std::vector<std::pair<std::vector<fcl::CollisionObjectd*>, std::vector<fcl::CollisionObjectd*>>> FCLTool::GetSelfAllowLinkFclColls()
    {
        std::vector<std::pair<std::vector<fcl::CollisionObjectd*>, std::vector<fcl::CollisionObjectd*>>> res;
        auto srdf = robot_model->getSRDF();
        std::vector<srdf::Model::CollisionPair> collisionPairs  = srdf->getEnabledCollisionPairs(); 
        for(size_t i = 0; i < collisionPairs.size(); ++i)
        {   
            srdf::Model::CollisionPair pair = collisionPairs[i];
            std::vector<fcl::CollisionObjectd*> obj1 = link_fcl_collisions[pair.link1_];
            std::vector<fcl::CollisionObjectd*> obj2 = link_fcl_collisions[pair.link2_];
            res.push_back(std::pair<std::vector<fcl::CollisionObjectd*>, std::vector<fcl::CollisionObjectd*>>(obj1, obj2));
        }
        return res;
    }

    bool FCLTool::IsSelfCollision()
    {
        fcl::CollisionRequestd req;
        for (size_t i = 0; i < allow_self_fcl_collPairs.size(); ++i)
        {
            for (size_t j = 0; j < allow_self_fcl_collPairs[i].first.size(); ++j)
            {
                fcl::CollisionObjectd* obj1 = allow_self_fcl_collPairs[i].first[j];
                for (size_t k = 0; k < allow_self_fcl_collPairs[i].second.size(); ++k)
                {
                    fcl::CollisionObjectd* obj2 = allow_self_fcl_collPairs[i].second[k];
                    fcl::CollisionResultd res;
                    fcl::collide(obj1, obj2, req, res);
                    if (res.isCollision()) return true;
                }
            }   
        }
        return false;
    }

    bool FCLTool::IsEnvCollision(std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd> envManager)
    {      
        robot_manager->update();
        CollisionData cdata;
        envManager->collide(robot_manager.get() ,&cdata, fcl::DefaultCollisionFunction);
        if(cdata.result.isCollision()) return true;
        return false;
    }

    std::vector<fcl::CollisionObjectd*> FCLTool::ConvertToFclColl(std::vector<LinkCollisionObject> linkObjs)
    {
        std::vector<fcl::CollisionObjectd*> res;
        for (size_t i = 0 ; i < linkObjs.size(); ++i)
        {
            fcl::CollisionObjectd* obj = new fcl::CollisionObjectd(linkObjs[i].geometry);
            res.push_back(obj);
        }
        return res;
    }

    void FCLTool::SetCollObjTF(std::string name, Eigen::Isometry3d tf)
    {
        auto it = link_fcl_collisions.find(name);
        if(it != link_fcl_collisions.end())
        {
            fcl::Transform3d fclTf(tf);
            std::vector<fcl::CollisionObjectd*> objs = it->second;
            for (fcl::CollisionObjectd* obj : objs) 
            { 
                obj->setTransform(fclTf);
            }
        }
    }

    std::vector<fcl::CollisionObjectd*> FCLTool::GetCollObj()
    {
        return coll_objs;
    }

    void FCLTool::UpdateLinkTF(std::vector<std::string> linkNames, std::vector<Eigen::Isometry3d> tfs)
    {
        for (size_t i = 0; i < linkNames.size(); ++i)
        {      
            SetCollObjTF(linkNames[i], tfs[i]);      
        }
    }

    void FCLTool::UpdateLinkTF(moveit::core::RobotState state)
    {
        std::vector<moveit::core::LinkModel *> linkModels = robot_model->getLinkModels();
        for (auto linkModel : linkModels)
        {
            Eigen::Isometry3d tf = state.getGlobalLinkTransform(linkModel);
            SetCollObjTF(linkModel->getName(), tf);
        }
    }

    void FCLTool::UpdateLinkTF(moveit::core::RobotState state, std::vector<std::string> linkNames)
    {
        for (auto name : linkNames)
        {
            Eigen::Isometry3d tf = state.getGlobalLinkTransform(name);
            SetCollObjTF(name, tf);
        }
    }

    std::shared_ptr<fcl::CollisionGeometryd> FCLTool::CreateFCLGeometry(shapes::ShapeConstPtr &shape)
    {
        switch (shape->type)
        {
        case shapes::BOX:
        {
            const auto *box = static_cast<const shapes::Box *>(shape.get());
            return std::make_shared<fcl::Boxd>(box->size[0], box->size[1], box->size[2]);
        }
        case shapes::SPHERE:
        {
            const auto *sphere = static_cast<const shapes::Sphere *>(shape.get());
            return std::make_shared<fcl::Sphered>(sphere->radius);
        }
        case shapes::CYLINDER:
        {
            const auto *cylinder = static_cast<const shapes::Cylinder *>(shape.get());
            return std::make_shared<fcl::Cylinderd>(cylinder->radius, cylinder->length);
        }
        case shapes::MESH:
        {
            const shapes::Mesh *mesh = dynamic_cast<const shapes::Mesh *>(shape.get());
            double *vertices = mesh->vertices;
            unsigned int *triangles = mesh->triangles;

            size_t numVertices = mesh->vertex_count;
            size_t numTriangles = mesh->triangle_count;

            std::vector<fcl::Vector3d> fclVertices;
            std::vector<fcl::Triangle> fclTriangles;

            for (size_t i = 0; i < numVertices; ++i)
            {
                fclVertices.emplace_back(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
            }

            for (size_t i = 0; i < numTriangles; ++i)
            {
                fclTriangles.emplace_back(triangles[i * 3], triangles[i * 3 + 1], triangles[i * 3 + 2]);
            }
            std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd>> collisionGeometry = std::make_shared<fcl::BVHModel<fcl::OBBRSSd>>();
            collisionGeometry->beginModel();
            collisionGeometry->addSubModel(fclVertices, fclTriangles);
            collisionGeometry->endModel();
            return std::dynamic_pointer_cast<fcl::CollisionGeometryd>(collisionGeometry);
        }
        default:
            return nullptr;
        }
    }

    CustomStateValidator::CustomStateValidator(ompl::base::SpaceInformationPtr &si,
                                               std::vector<std::string> linkNames,
                                               std::vector<std::string> jointNames,
                                               std::shared_ptr<moveit::core::RobotState> robotState,
                                               std::shared_ptr<FCLTool> fclTool,
                                               std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd> envManager) : ompl::base::StateValidityChecker(si),
                                                                                                    robot_state(robotState),
                                                                                                    fcl_tool(fclTool),
                                                                                                    env_manager(envManager),
                                                                                                    link_names(linkNames),
                                                                                                    joint_names(jointNames)
    {
        dimsion = joint_names.size();
    }

    bool CustomStateValidator::isValid(const ompl::base::State *state) const
    {
        for (size_t i = 0; i < dimsion; ++i)
        {
            double angle = state->as<ompl::base::RealVectorStateSpace::StateType>()->values[i];
            robot_state->setVariablePosition(joint_names[i], angle);
        }
      
        robot_state->updateLinkTransforms();

        fcl_tool->UpdateLinkTF(*robot_state, link_names);

        bool isSelfColl = fcl_tool->IsSelfCollision();

        if(isSelfColl) return false;

        if (env_manager->size() != 0)
        {
            bool isEnvColl = fcl_tool->IsEnvCollision(env_manager);

            if (isEnvColl) return false;
        }

        return true;
    }

    OMPLTool::OMPLTool(std::shared_ptr<moveit::core::RobotModel> robotModel)
    {   
        robot_model = robotModel;
        fcl_tool = std::make_shared<FCLTool>(robot_model, IRS_GROUP_NAME);

        moveit::core::JointModelGroup* group = robot_model->getJointModelGroup(IRS_GROUP_NAME);

        std::vector<moveit::core::VariableBounds> variableBounds;

        for (auto joint : group->getActiveJointModels())
        {
            const std::vector<std::string> &jointNames = joint->getVariableNames();
            for (const std::string &varName : jointNames)
            {
                const moveit::core::VariableBounds &bounds = joint->getVariableBounds(varName);
                joint_bounds[varName] = bounds;
            }
        }
    }

    ompl::base::PathPtr OMPLTool::Plan(std::shared_ptr<moveit::core::RobotState> robotState, std::vector<std::string> goalJointNames, std::vector<double> goalJointPositions, std::vector<Eigen::Vector3d> envPointClouds)
    {   
        fcl_tool->UpdateLinkTF(*robotState);

        int dimsion = goalJointNames.size();
        size_t count = goalJointNames.size();

        auto space(std::make_shared<ompl::base::RealVectorStateSpace>(dimsion));
        //std::shared_ptr<ompl::base::RealVectorStateSpace> space = std::make_shared<ompl::base::RealVectorStateSpace>(dimsion);

        ompl::base::RealVectorBounds bounds(dimsion);
        std::vector<double> startJointValues;
        for (size_t i = 0; i < count; ++i)
        {   
            std::string name = goalJointNames[i];
            if(joint_bounds.find(name) != joint_bounds.end())
            {
                auto bound = joint_bounds[name];
                bounds.setLow(i, bound.min_position_);
                bounds.setHigh(i, bound.max_position_);
                const double* angle = robotState->getJointPositions(name);
                startJointValues.push_back(*angle);
            }
            else return nullptr;
        }

        space->setBounds(bounds);

        auto ss(std::make_shared<ompl::base::SpaceInformation>(space));
        //std::shared_ptr<ompl::base::SpaceInformation> ss = std::make_shared<ompl::base::SpaceInformation>(space);
        ss->setValidStateSamplerAllocator([](const ompl::base::SpaceInformation *si)
                                          { return std::make_shared<ompl::base::ObstacleBasedValidStateSampler>(si); });

        auto pdef(std::make_shared<ompl::base::ProblemDefinition>(ss));

        ompl::base::ScopedState<> start(space);
        for (size_t i = 0; i < count; ++i)
            start[i] = startJointValues[i];

        ompl::base::ScopedState<> goal(space);
        for (size_t i = 0; i < count; ++i)
            goal[i] = goalJointPositions[i];

        pdef->setStartAndGoalStates(start, goal);

        std::shared_ptr<moveit::core::RobotState> planState = std::make_shared<moveit::core::RobotState>(*robotState);

        std::vector<std::string> linkNames;
        for(size_t i = 0; i < goalJointNames.size(); ++i)
        {
            moveit::core::JointModel* jm = robot_model->getJointModel(goalJointNames[i]);
            linkNames.push_back(jm->getChildLinkModel()->getName());
        }

        std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd> envManager = GetAABBEnvManager(envPointClouds);
        ss->setStateValidityChecker(std::make_shared<CustomStateValidator>(ss, linkNames, goalJointNames, planState, fcl_tool, envManager));
        ss->setup();

        //ompl::geometric::RRTConnect planner(ss);
        //ompl::geometric::RRTConnect *planner = new ompl::geometric::RRTConnect(ss);
        //delete planner;
        //auto planner(std::make_shared<ompl::geometric::RRTConnect>(ss));
        std::shared_ptr<ompl::geometric::RRTConnect> planner = std::make_shared<ompl::geometric::RRTConnect>(ss);
        planner->setProblemDefinition(pdef);
        planner->setup();

        //ss->printSettings(std::cout);
	    //pdef->print(std::cout);

	    ompl::base::PlannerStatus solved = planner->ompl::base::Planner::solve(1.0);

        if (solved)
        {
            ompl::base::PathPtr path = pdef->getSolutionPath();
            return path;
        }
        return nullptr;
    }

    std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd> OMPLTool::GetAABBEnvManager(std::vector<Eigen::Vector3d> envPointClouds)
    {
        std::shared_ptr<fcl::DynamicAABBTreeCollisionManagerd> envManager = std::make_shared<fcl::DynamicAABBTreeCollisionManagerd>();
        if(envPointClouds.size() == 0) return envManager;
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        for (auto point : envPointClouds)
        {
            pcl::PointXYZ pclPoint;
            pclPoint.x = point.x();
            pclPoint.y = point.y();
            pclPoint.z = point.z();
            cloudPoints->points.push_back(pclPoint);
        }

        std::shared_ptr<pcl::PolygonMesh> mesh = PointCloudTool::Date::GenerateMesh(cloudPoints);

        if (mesh->polygons.empty()) 
        {
            throw std::runtime_error("Mesh reconstruction failed");
        }

        pcl::PointCloud<pcl::PointXYZ> meshVertices;
        pcl::fromPCLPointCloud2(mesh->cloud, meshVertices);

        std::shared_ptr<fcl::BVHModel<fcl::AABBd>> bvhModel = std::make_shared<fcl::BVHModel<fcl::AABBd>>();
        bvhModel->beginModel();

        for (auto &polygon : mesh->polygons)
        {
            if (polygon.vertices.size() != 3)
            {
                continue; 
            }

            auto &p0 = meshVertices[polygon.vertices[0]];
            auto &p1 = meshVertices[polygon.vertices[1]];
            auto &p2 = meshVertices[polygon.vertices[2]];

            bvhModel->addTriangle({p0.x, p0.y, p0.z}, {p1.x, p1.y, p1.z}, {p2.x, p2.y, p2.z});
        }

        if (!bvhModel->endModel())
        {
            throw std::runtime_error("Failed to build FCL BVH model");
        }

        envManager->registerObject(std::make_shared<fcl::CollisionObjectd>(bvhModel).get());
        envManager->update();

        return envManager;
    }
}
