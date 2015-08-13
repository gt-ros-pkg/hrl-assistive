#ifndef AD_PCL_FILTER_H_
#define AD_PCL_FILTER_H_

// System stuff
#include <math.h>
#include <vector>
#include <cstdlib> 

// ROS
#include <ros/ros.h>
#include <ros/package.h> // for getting file path for loading images
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

class PCLfilter
{
public:
    PCLfilter(const ros::NodeHandle &nh);
    ~PCLfilter();

    /* EIGEN_MAKE_ALIGNED_OPERATOR_NEW */

private:
    void getParams();   
    void initComms();

    void depthPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    ros::NodeHandle nh_;

    // pass filtering range
    double x_min_, x_max_; 
    double y_min_, y_max_;
    double z_min_, z_max_;
    
    // 
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_ptr_;


    // ROS Publisher and subscribers
    ros::Publisher filtered_pcl_pub_;
    ros::Subscriber pcl_sub_;
};



#endif
