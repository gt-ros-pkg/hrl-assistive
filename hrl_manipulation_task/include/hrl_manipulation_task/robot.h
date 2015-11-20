#ifndef ROBOT_H_
#define ROBOT_H_ 

// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <kdl_parser/kdl_parser.hpp>

// KDL
#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/frames_io.hpp>

// msg
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/JointState.h>

#include <vector>

#include <stdio.h>
#include <iostream>

#define MESH_SURFACE    1
#define MESH_VOLUME     2

#define LEFT 1
#define RIGHT 2

using namespace std;


class Robot
{
public:
  
    Robot(ros::NodeHandle &nh, string base_link, string target_link);
    ~Robot();

    bool getParams();  
    bool setupKDL();  

    bool forwardKinematics(vector<double> &joint_angles, KDL::Frame &cartpos, int link_number) const;
    bool setAllJoints(const sensor_msgs::JointState joint_state);

    vector<string> getJointNames() const;

/* private: */
    /* void jointStateCallback(const sensor_msgs::JointStateConstPtr &jointState);  */

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    ros::NodeHandle nh_;

    KDL::Chain chain_;
    KDL::Tree tree_;

    // Planning model variables
    vector<string> left_arm_joint_names_;
    vector<string> left_arm_link_names_;
    string origin_frame_; 

    /* urdf::Model urdf_model_;    */

    string base_link_;
    string target_link_;

    int arm_;
    map<string, double> joint_values_;
    /* vector<planning_models::KinematicModel::MultiDofConfig> multi_dof_configs_; */

public:
    KDL::JntArray qmin_;
    KDL::JntArray qmax_;

    double radius_;

 
};

#endif
