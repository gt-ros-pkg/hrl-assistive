#ifndef PCL_UPSAMPLE_EXTRACTOR_H_
#define PCL_UPSAMPLE_EXTRACTOR_H_

// ROS
#include <ros/ros.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h> 
#include <pcl/filters/conditional_removal.h> 
/* #include <pcl_ros/transforms.h> */
#include <pcl/surface/mls.h>

// Message
/* #include <sensor_msgs/PointCloud.h> */
#include <sensor_msgs/PointCloud2.h>

// Boost
#include <boost/thread.hpp>

typedef pcl::PointXYZRGB PointType;
/* typedef pcl::PointXYZRGBA PointType; */
/* typedef pcl::PointXYZI KeyType; */
/* typedef pcl::PointXYZ PointType; */
/* typedef pcl::PointXYZI KeyType; */
using namespace std;

class UPSAMPLE
{
public:
    UPSAMPLE(const ros::NodeHandle &nh);
    ~UPSAMPLE();

private:
    bool initComms();
    bool initFilter();

    void cameraCallback(const sensor_msgs::PointCloud2ConstPtr& input);

public:
    void runFilter();

    ros::Publisher pcl_filtered_pub_;
    ros::Subscriber camera_sub_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    // Common major variables
    ros::NodeHandle nh_;

    // PCL
    pcl::PointCloud<PointType>::Ptr cloud_ptr_; 
    pcl::PointCloud<PointType>::Ptr cloud_filtered_ptr_; 

    // filtering
    pcl::ConditionAnd<PointType>::Ptr range_cond_;
    pcl::MovingLeastSquares<PointType, PointType>::Ptr mls_filter_;

    boost::mutex cloud_mtx_; // mutex for contact cost subscribers

};

#endif

