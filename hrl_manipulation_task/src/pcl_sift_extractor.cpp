#include "hrl_manipulation_task/pcl_sift_extractor.h"


SIFT::SIFT(const ros::NodeHandle &nh): nh_(nh)
{
    cloud_ptr_.reset (new pcl::PointCloud<PointType>());
    cloud_filtered_ptr_.reset (new pcl::PointCloud<PointType>());
    kpts_ptr_.reset (new pcl::PointCloud<PointType>);

    tf::TransformListener listener;
    has_tf_ = false;
    has_joint_state_ = false;

    getParams();
    initComms();
    initFilter();
    initRobot();

    std::cout << "Waiting tf and joint state" << std::endl;
    ros::Rate rate(10.0);
    while (nh.ok()){
        try{
            listener.lookupTransform("torso_lift_link", "head_mount_kinect_rgb_optical_frame",  
                                     ros::Time(0), head_transform_);
            has_tf_ = true;
        }
        catch (tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }

        if (has_tf_ && has_joint_state_) break;

        rate.sleep();
    }
    std::cout << "Loaded tf and joint state" << std::endl;
    

}

SIFT::~SIFT()
{
}

bool SIFT::getParams()
{

    joint_names_.push_back("l_shoulder_pan_joint");
    joint_names_.push_back("l_shoulder_lift_joint");
    joint_names_.push_back("l_upper_arm_roll_joint");
    joint_names_.push_back("l_elbow_flex_joint");
    joint_names_.push_back("l_forearm_roll_joint");
    joint_names_.push_back("l_wrist_flex_joint");
    joint_names_.push_back("l_wrist_roll_joint");

    return true;
}

bool SIFT::initComms()
{
    sift_markers_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/hrl_manipulation_task/sift_markers", 10, true);
    
    camera_sub_ = nh_.subscribe("/head_mount_kinect/depth_registered/points", 1, &SIFT::cameraCallback, this);
    joint_state_sub_ = nh_.subscribe("/joint_states", 10, &SIFT::jointStateCallback, this);    
    ROS_INFO("Comms Initialized!!");
    return true;
}

bool SIFT::initFilter()
{
    range_cond_.reset (new pcl::ConditionAnd<PointType> ());

    // location condition
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
     pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::GT, 0.25)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::LT, 1.2)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
     pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::GT, -0.8)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::LT, 0.2)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::GT, -0.5)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::LT, 0.1)));

    // // build the filter
    // condrem_ptr_.reset (new pcl::ConditionalRemoval<PointType>(range_cond) );    
}

bool SIFT::initRobot()
{
    base_frame_ = "torso_lift_link";
    ee_frame_   = "l_gripper_tool_frame";
    robot_ptr_.reset(new Robot(nh_, base_frame_, ee_frame_) );

    // Wait current joint update from robot_state
    bool ret;
    while (ros::ok()){    
        ret = robot_ptr_->setAllJoints(joint_state_);
        if (ret == true) break;
        
        ros::Duration(2.0).sleep();
        ros::spinOnce();
    }
    ROS_INFO("Robot Initialized!!");

    return true;
}

void SIFT::cameraCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
    // boost::mutex::scoped_lock lock(camera_mtx);
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(*input, pcl_pc);

    pcl::fromPCLPointCloud2(pcl_pc, *cloud_ptr_);
}

void SIFT::jointStateCallback(const sensor_msgs::JointStateConstPtr &jointState) 
{
    if (jointState->name.size() != jointState->position.size() || 
        jointState->name.size() !=jointState->velocity.size()) {
        ROS_ERROR("Planning environment received invalid joint state");
        return;
    }
    else if (jointState->name.size() == 0) {
        ROS_WARN("Planning environment is waiting joint state");
        return;
    }

    joint_state_ = *jointState;

    std::vector<int> jt_idx_list;
    jt_idx_list.resize(joint_names_.size());

    for (unsigned int i=0 ; i<joint_names_.size() ; i++) {
      for (unsigned int j=0 ; j<joint_state_.name.size() ; j++) {
        if (joint_names_[i] == (joint_state_.name)[j]) {
          jt_idx_list[i] = j;
        }
      }
    }
             
    for (unsigned int i=0 ; i<jt_idx_list.size() ; i++)
      joint_angles_[i] = joint_state_.position[jt_idx_list[i]];

    // Get end-effector
    // robot_ptr_->forwardKinematics(joint_angles_, current_ee_frame_, robot_dimensions_);

    has_joint_state_ = true;
}



// Publish voxelized & entire surface map
void SIFT::pubSiftMarkers()
{
    // visualization_msgs::MarkerArray siftMarkers;
    sensor_msgs::PointCloud2 msg;

    pcl::toROSMsg(*kpts_ptr_, msg);
    msg.header.frame_id = "torso_lift_link";
    sift_markers_pub_.publish(msg);

}

void SIFT::extractSIFT()
{
    // transform
    tf::Transform head_transform(head_transform_.getRotation(), head_transform_.getOrigin());
    pcl_ros::transformPointCloud(*cloud_ptr_, *cloud_filtered_ptr_, head_transform);

    // Apply point cloud filter
    pcl::ConditionalRemoval<PointType> condrem (range_cond_);
    condrem.setInputCloud (cloud_filtered_ptr_);
    condrem.setKeepOrganized(true);
    condrem.filter (*cloud_filtered_ptr_);    

    // Exclude arm
    


    //Compute Keypoints
    pcl::PointCloud<KeyType>::Ptr keypoints (new pcl::PointCloud<KeyType>);
    pcl::SIFTKeypoint<PointType, KeyType> sift;
    sift.setScales(min_scale, nr_octaves, nr_scales);
    sift.setMinimumContrast(contrast);

    sift.setInputCloud(cloud_ptr_);
    sift.setSearchSurface(cloud_ptr_);
    sift.compute(*keypoints);
    kpts_ptr_->points.resize(keypoints->points.size());
    pcl::copyPointCloud(*keypoints, *kpts_ptr_);
    std::cout << "Found " << keypoints->points.size() << " keypoints." << std::endl;

    
    // visualization
    pubSiftMarkers();
}



int main(int argc, char **argv)
{
    ROS_INFO("SIFT extractor main()");
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;

    // Initialize a SIFT extractor object.
    SIFT extractor(n);


    ROS_INFO("SIFT extractor: Loop Start!!");
    ros::Rate loop_rate(1.0); // 1Hz
    while (ros::ok())
    { 
        extractor.extractSIFT();

        // Ros loop stuff
        ros::spinOnce();
        loop_rate.sleep();        
    }

    return 0;
}


