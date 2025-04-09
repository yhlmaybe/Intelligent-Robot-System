#include "PointCloudTool.h"

namespace PointCloudTool
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr Date::StatisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, int neighbourPoints)
    {
        if(cloudPoints->size() == 0) return cloudPoints;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloudPoints);
        sor.setMeanK(neighbourPoints); 
        sor.setStddevMulThresh(1.0); 
        sor.filter(*cloudFiltered);
        return cloudFiltered;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr Date::RadiusOutlierRemoval(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, double radius, int minNeighbourPoints)
    {
        if(cloudPoints->size() == 0) return cloudPoints;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
        ror.setInputCloud(cloudPoints);
        ror.setRadiusSearch(radius);      
        ror.setMinNeighborsInRadius(minNeighbourPoints); 
        ror.filter(*cloudFiltered);
        return cloudFiltered;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr Date::VoxelGridFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float scale)
    {
        if(cloudPoints->size() == 0) return cloudPoints;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDownsampled = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloudPoints);
        vg.setLeafSize(scale, scale, scale);
        vg.filter(*cloudDownsampled);
        return cloudDownsampled;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr Date::UniformSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float radius)
    {
        if(cloudPoints->size() == 0) return cloudPoints;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDownsampled = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::UniformSampling<pcl::PointXYZ> us;
        us.setInputCloud(cloudPoints);
        us.setRadiusSearch(radius);
        us.filter(*cloudDownsampled);
        return cloudDownsampled;
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr Date::MovingLeastSquaresSmoothed(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float radius)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr smoothedCloud = pcl::make_shared<pcl::PointCloud<pcl::PointNormal>>();
        if(cloudPoints->size() == 0) return smoothedCloud;
        pcl::PointCloud<pcl::Normal>::Ptr normals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();

        pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
        mls.setComputeNormals(true);
        mls.setInputCloud(cloudPoints);
        mls.setSearchRadius(0.03);
        mls.setPolynomialOrder(2);
        mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::NONE);
        mls.setDilationIterations(1);
        mls.setUpsamplingRadius(radius);
        mls.setUpsamplingStepSize(0.005);
        mls.process(*smoothedCloud);
        return smoothedCloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr Date::GaussianFilterSmoothed(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float sigmaS)
    {
        if(cloudPoints->size() == 0) return cloudPoints;
        pcl::PointCloud<pcl::PointXYZ>::Ptr smoothedCloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::FastBilateralFilter<pcl::PointXYZ> fb;
        fb.setInputCloud(cloudPoints);
        fb.setSigmaS(sigmaS); 
        fb.setSigmaR(0.01); 
        fb.applyFilter(*smoothedCloud);
        return smoothedCloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr Date::RemoveNaNF(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints)
    {
        if(cloudPoints->size() == 0) return cloudPoints;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloudPoints, *cloudFiltered, indices);
        return cloudFiltered;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr Date::ConditionsFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float minx, float miny, float minz, float maxx, float maxy, float maxz)
    {
        if(cloudPoints->size() == 0) return cloudPoints;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        pcl::ConditionAnd<pcl::PointXYZ>::Ptr condition(pcl::make_shared<pcl::ConditionAnd<pcl::PointXYZ>>());
        condition->addComparison(pcl::make_shared<pcl::FieldComparison<pcl::PointXYZ>>("x", pcl::ComparisonOps::GT, minx));
        condition->addComparison(pcl::make_shared<pcl::FieldComparison<pcl::PointXYZ>>("x", pcl::ComparisonOps::LT, maxx));
        condition->addComparison(pcl::make_shared<pcl::FieldComparison<pcl::PointXYZ>>("y", pcl::ComparisonOps::GT, miny));
        condition->addComparison(pcl::make_shared<pcl::FieldComparison<pcl::PointXYZ>>("y", pcl::ComparisonOps::LT, maxy));
        condition->addComparison(pcl::make_shared<pcl::FieldComparison<pcl::PointXYZ>>("z", pcl::ComparisonOps::GT, minz));
        condition->addComparison(pcl::make_shared<pcl::FieldComparison<pcl::PointXYZ>>("z", pcl::ComparisonOps::LT, maxz));

        pcl::ConditionalRemoval<pcl::PointXYZ> cr;
        cr.setCondition(condition);
        cr.setInputCloud(cloudPoints);
        cr.setKeepOrganized(false);
        cr.filter(*cloudFiltered);
        return cloudFiltered;
    }

    std::shared_ptr<pcl::PolygonMesh> Date::GenerateMesh(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints)
    {
        std::shared_ptr<pcl::PolygonMesh> mesh = std::make_shared<pcl::PolygonMesh>();

        if(cloudPoints->size() == 0) return mesh;

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
        ne.setInputCloud(cloudPoints);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(0.03); 

        pcl::PointCloud<pcl::Normal>::Ptr normals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
        ne.compute(*normals);

        pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormals = pcl::make_shared<pcl::PointCloud<pcl::PointNormal>>();
        pcl::concatenateFields(*cloudPoints, *normals, *cloudWithNormals);

        pcl::GreedyProjectionTriangulation<pcl::PointNormal> gpt;
        gpt.setInputCloud(cloudWithNormals);
        gpt.setSearchRadius(0.025);           
        gpt.setMu(2.5);                       
        gpt.setMaximumNearestNeighbors(100);  
        gpt.setMaximumSurfaceAngle(M_PI / 4); 
        gpt.setMinimumAngle(M_PI / 18);       
        gpt.setMaximumAngle(2 * M_PI / 3);    
        gpt.setNormalConsistency(false);      

        gpt.reconstruct(*mesh);

        return mesh;
    }
}