#ifndef PCL_BASIC_H
#define PCL_BASIC_H
#include <ros/ros.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>

#include <boost/format.hpp>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZRGB PRGB;
typedef pcl::PointCloud<PRGB> PCRGB;

void transformPC(const PCRGB &in_pc, PCRGB &out_pc, 
                 const Eigen::Affine3d& transform);
void pubLoop(PCRGB::Ptr& pc, const std::string& topic, double rate = 1.0, int number_runs=10000);
void pubLoop(vector<PCRGB::Ptr> &pcs, const vector<std::string>& topics, double rate, 
             int number_runs=10000);
void boxFilter(const PCRGB &in_pc, PCRGB &out_pc,
               double min_x, double max_x, double min_y, double max_y, double min_z, double max_z);
void extractIndices(const PCRGB::Ptr& in_pc, pcl::IndicesPtr& inds, PCRGB::Ptr& out_pc, bool is_negative = false);
void readPCBag(const string& filename, PCRGB::Ptr& pc);
void savePCBag(const string& filename, PCRGB::Ptr& pc);

#endif // PCL_BASIC_H
