#ifndef POINTCLOUDTOOL_H
#define POINTCLOUDTOOL_H

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/surface/gp3.h>

namespace PointCloudTool
{

    class Date
    {
        public:

        static pcl::PointCloud<pcl::PointXYZ>::Ptr StatisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, int neighbourPoints = 50);

        static pcl::PointCloud<pcl::PointXYZ>::Ptr RadiusOutlierRemoval(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, double radius = 0.1, int minNeighbourPoints = 5);
    
        static pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelGridFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float scale = 0.01);
    
        static pcl::PointCloud<pcl::PointXYZ>::Ptr UniformSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float radius = 0.02);

        static pcl::PointCloud<pcl::PointNormal>::Ptr MovingLeastSquaresSmoothed(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float radius = 0.03);
    
        static pcl::PointCloud<pcl::PointXYZ>::Ptr GaussianFilterSmoothed(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float sigmaS= 0.05);

        static pcl::PointCloud<pcl::PointXYZ>::Ptr RemoveNaNF(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints);

        static pcl::PointCloud<pcl::PointXYZ>::Ptr ConditionsFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints, float minx, float miny, float minz, float maxx, float maxy, float maxz);
    
        static std::shared_ptr<pcl::PolygonMesh> GenerateMesh(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPoints);
    };


}

#endif