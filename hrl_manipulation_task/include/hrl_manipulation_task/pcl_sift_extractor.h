#ifndef PCL_SIFT_EXTRACTOR_H_
#define PCL_SIFT_EXTRACTOR_H_

// ROS
#include <ros/ros.h>
/* #include "tf/LinearMath/Transform.h" */
/* #include "tf/transform_listener.h" */

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h> 
#include <pcl/keypoints/sift_keypoint.h>
/* #include <pcl/filters/conditional_removal.h> */

// Message
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

// Boost
#include <boost/thread.hpp>
/* #include <boost/math/distributions/normal.hpp> */

const float min_scale = 0.01;
const int nr_octaves = 3;
const int nr_scales = 3;
const float contrast = 10;

typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointXYZI KeyType;

class SIFT
{
public:
    SIFT(const ros::NodeHandle &nh);
    ~SIFT();

private:
    bool getParams();   
    bool initComms();

    void cameraCallback(const sensor_msgs::PointCloud2ConstPtr& input);
    void jointStateCallback(const sensor_msgs::JointStateConstPtr &jointState);

public:
    void pubSiftMarkers();
    void extractSIFT();

    ros::Publisher sift_markers_pub_;

    ros::Subscriber camera_sub_;
    ros::Subscriber joint_state_sub_;


private:

    // Common major variables
    ros::NodeHandle nh_;
    boost::mutex camera_mtx; // mutex for contact cost subscribers

    // PCL
    pcl::PointCloud<PointType>::Ptr cloud_ptr_; 
    pcl::PointCloud<PointType>::Ptr kpts_ptr_; 


    /* // Robot */
    /* boost::shared_ptr<Robot> robot_ptr_; */
    /* std::string base_frame_; */
    /* std::string ee_frame_; */
    /* pcl::PointCloud<pcl::PointXYZ> arm_mesh_; */
    /* int robot_dimensions_; /// Number of dimensions in the robot space, eg 1->n */

    /* // current info */
    /* sensor_msgs::JointState joint_state_;  */
    /* std::vector<std::string> joint_names_; */
    /* std::vector<double> joint_angles_; // Current joint angles. */
    /* bool has_current_; */
    /* KDL::Frame current_ee_frame_; */


    /* tf::TransformListener tf_lstnr_; */
    /* tf::StampedTransform tr_; */

    boost::mutex mtx; // mutex for contact cost subscribers

};


/* long long getTimestamp(void) */
/* { */
/*     static __thread struct timeval tv; */
/*     gettimeofday(&tv, NULL); */
/*     return 1000000LL * tv.tv_sec + tv.tv_usec; */
/* } */



#endif

