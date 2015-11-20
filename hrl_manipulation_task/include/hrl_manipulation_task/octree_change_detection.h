#ifndef OCTREE_CHANGE_DETECTION_H_
#define OCTREE_CHANGE_DETECTION_H_

// ROS
#include <ros/ros.h>
/* #include "tf/LinearMath/Transform.h" */
#include "tf/transform_listener.h"

#include "hrl_manipulation_task/robot.h"

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h> 
#include <pcl/filters/conditional_removal.h> 
#include <pcl_ros/transforms.h>
#include <pcl/octree/octree.h>

// Message
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

// Boost
#include <boost/thread.hpp>
/* #include <boost/math/distributions/normal.hpp> */


typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZI KeyType;
using namespace std;

class changeDetector
{
public:
    changeDetector(const ros::NodeHandle &nh);
    ~changeDetector();

private:
    bool getParams();   
    bool initComms();
    bool initFilter();
    bool initRobot();

    void cameraCallback(const sensor_msgs::PointCloud2ConstPtr& input);
    void jointStateCallback(const sensor_msgs::JointStateConstPtr &jointState);

public:
    void pubChangetMarkers();
    void runDetector();

private:
    ros::Publisher change_pub_;

    ros::Subscriber camera_sub_;
    ros::Subscriber joint_state_sub_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    // Common major variables
    ros::NodeHandle nh_;

    // PCL
    pcl::PointCloud<PointType>::Ptr cloud_ptr_; 
    pcl::PointCloud<PointType>::Ptr cloud_filtered_ptr_; 


    // tf
    tf::StampedTransform head_transform_;

    // filtering
    pcl::ConditionAnd<PointType>::Ptr range_cond_;

    // Robot
    boost::shared_ptr<Robot> robot_ptr_;
    std::string base_frame_;
    std::string ee_frame_;
    int robot_dimensions_; /// Number of dimensions in the robot space, eg 1->n 

    // current info
    sensor_msgs::JointState joint_state_;
    std::vector<std::string> joint_names_;
    std::vector<double> joint_angles_; // Current joint angles.
    bool has_current_;

    KDL::Frame current_ee_frame_;
    KDL::Frame current_wrist_frame_;
    KDL::Frame current_elbow_frame_;
    KDL::Frame current_shoulder_frame_;

    // flag
    bool has_tf_;
    bool has_joint_state_;

    boost::mutex cloud_mtx_; // mutex for contact cost subscribers
    /* boost::mutex points_mutex_; */
    /* boost::mutex camera_mtx; // mutex for contact cost subscribers */

};


/* long long getTimestamp(void) */
/* { */
/*     static __thread struct timeval tv; */
/*     gettimeofday(&tv, NULL); */
/*     return 1000000LL * tv.tv_sec + tv.tv_usec; */
/* } */



#endif

