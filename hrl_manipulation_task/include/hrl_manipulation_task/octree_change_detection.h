#ifndef OCTREE_CHANGE_DETECTION_H_
#define OCTREE_CHANGE_DETECTION_H_

// ROS
#include <ros/ros.h>
/* #include "tf/LinearMath/Transform.h" */
#include "tf/transform_listener.h"

#include "hrl_manipulation_task/robot.h"
#include "hrl_manipulation_task/util.h"

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h> 
#include <pcl/filters/conditional_removal.h> 
#include <pcl_ros/transforms.h>
#include <pcl/octree/octree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

// For custom PCL filter
#include <pcl/filters/filter_indices.h>
#include <pcl/search/pcl_search.h>
#include <pcl/common/io.h>

// Message
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/JointState.h>
#include "hrl_anomaly_detection/pclChange.h"
/* #include <visualization_msgs/MarkerArray.h> */
/* #include <visualization_msgs/Marker.h> */

// Boost
#include <boost/thread.hpp>
/* #include <boost/math/distributions/normal.hpp> */

// Octree resolution - side length of octree voxels
const float resolution = 0.05;

typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZI KeyType;
typedef typename pcl::search::Search<PointType>::Ptr SearcherPtr;

using namespace std;

class changeDetector
{
public:
    changeDetector(const ros::NodeHandle &nh);
    ~changeDetector();

    void pubFilteredPCL();
    void pubChangesPCL();
    void pubChanges();
    void runDetector();

private:
    bool getParams();   
    bool initComms();
    bool initFilter();
    bool initRobot();

    void cameraCallback(const sensor_msgs::PointCloud2ConstPtr& input);
    void jointStateCallback(const sensor_msgs::JointStateConstPtr &jointState);

    void robotBodyFilter(const pcl::PointCloud<PointType>::Ptr& pcl_cloud);
    void noiseFilter(const pcl::PointCloud<PointType>::Ptr& pcl_cloud, int mean_k, 
                     double distance_threshold, double std_mul);

private:
    ros::Publisher pcl_filtered_pub_;
    ros::Publisher pcl_changes_pub_;
    ros::Publisher changes_pub_;

    ros::Subscriber camera_sub_;
    ros::Subscriber joint_state_sub_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    // Common major variables
    ros::NodeHandle nh_;

    // PCL
    pcl::PointCloud<PointType>::Ptr cloud_ptr_; 
    pcl::PointCloud<PointType>::Ptr cloud_filtered_ptr_; 
    pcl::PointCloud<PointType>::Ptr cloud_changes_ptr_; 
    // Instantiate octree-based point cloud change detection class
    boost::shared_ptr<pcl::octree::OctreePointCloudChangeDetector<PointType> > octree_ptr_;

    // Create the filtering object
    pcl::ExtractIndices<PointType>::Ptr extract_ptr_;
    pcl::PointIndices::Ptr inliers_ptr_;
    pcl::StatisticalOutlierRemoval<PointType>::Ptr sorfilter_ptr_;
    pcl::VoxelGrid<PointType> voxelfilter_;

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

    /* KDL::Frame current_ee_frame_; */
    std::vector<KDL::Frame*> cur_frames_;
    std::vector<KDL::Frame*> last_frames_;
    std::queue<std::vector<KDL::Frame*> > frame_seq_;
    std::vector<double> radius_;

    std_msgs::Header header_;

    // flag
    bool has_tf_;
    bool has_joint_state_;
    bool has_robot_;
    int counter_;
    int max_frame_check_step_;

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

