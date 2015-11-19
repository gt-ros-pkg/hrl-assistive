#include "hrl_manipulation_task/pcl_sift_extractor.h"


SIFT::SIFT(const ros::NodeHandle &nh)
{
    cloud_ptr_.reset (new pcl::PointCloud<PointType>());
    kpts_ptr_.reset (new pcl::PointCloud<PointType>);

    getParams();
    initComms();
}

SIFT::~SIFT()
{
}

bool SIFT::getParams()
{

}

bool SIFT::initComms()
{
    sift_markers_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/hrl_manipulation_task/sift_markers", 10, true);
    
    camera_sub_ = nh_.subscribe("/head_mount_kinect/depth_registered/points", 1, &SIFT::cameraCallback, this);
    joint_state_sub_ = nh_.subscribe("/joint_states", 10, &SIFT::jointStateCallback, this);    
    ROS_INFO("Comms Initialized!!");
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
    // if (jointState->name.size() != jointState->position.size() || 
    //     jointState->name.size() !=jointState->velocity.size()) {
    //     ROS_ERROR("Planning environment received invalid joint state");
    //     return;
    // }
    // else if (jointState->name.size() == 0) {
    //     ROS_WARN("Planning environment is waiting joint state");
    //     return;
    // }

    // joint_state_ = *jointState;

    // std::vector<int> jt_idx_list;
    // jt_idx_list.resize(joint_names_.size());

    // for (unsigned int i=0 ; i<joint_names_.size() ; i++) {
    //   for (unsigned int j=0 ; j<joint_state_.name.size() ; j++) {
    //     if (joint_names_[i] == (joint_state_.name)[j]) {
    //       jt_idx_list[i] = j;
    //     }
    //   }
    // }
             
    // for (unsigned int i=0 ; i<jt_idx_list.size() ; i++)
    //   joint_angles_[i] = joint_state_.position[jt_idx_list[i]];

    // // Get end-effector
    // robot_ptr_->forwardKinematics(joint_angles_, current_ee_frame_, robot_dimensions_);

    // has_current_ = true;
}



// Publish voxelized & entire surface map
void SIFT::pubSiftMarkers()
{
    // visualization_msgs::MarkerArray siftMarkers;
    sensor_msgs::PointCloud2 msg;

    pcl::toROSMsg(*kpts_ptr_, msg);
    msg.header.frame_id = "head_mount_kinect_rgb_optical_frame";
    sift_markers_pub_.publish(msg);

}

void SIFT::extractSIFT()
{
    // Filter cloud


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


