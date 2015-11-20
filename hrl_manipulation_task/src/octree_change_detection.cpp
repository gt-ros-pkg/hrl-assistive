#include "hrl_manipulation_task/pcl_sift_extractor.h"


changeDetector::changeDetector(const ros::NodeHandle &nh): nh_(nh)
{
    cloud_ptr_.reset (new pcl::PointCloud<PointType>());
    cloud_filtered_ptr_.reset (new pcl::PointCloud<PointType>());
    kpts_ptr_.reset (new pcl::PointCloud<PointType>);

#ifndef USE_changeDetector
    // PyramidalKLT
    tracker_.reset (new pcl::tracking::PyramidalKLTTracker<PointType>);
#endif

    has_tf_ = false;
    has_joint_state_ = false;

    getParams();
    initFilter();
    initComms();    
    initRobot();

}

changeDetector::~changeDetector()
{
}

bool changeDetector::getParams()
{
    robot_dimensions_ = 7; // 7 link pr2 arm

    joint_names_.push_back("l_shoulder_pan_joint");
    joint_names_.push_back("l_shoulder_lift_joint");
    joint_names_.push_back("l_upper_arm_roll_joint");
    joint_names_.push_back("l_elbow_flex_joint");
    joint_names_.push_back("l_forearm_roll_joint");
    joint_names_.push_back("l_wrist_flex_joint");
    joint_names_.push_back("l_wrist_roll_joint");

    joint_angles_.resize(robot_dimensions_);

    return true;
}

bool changeDetector::initComms()
{
    sift_markers_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/hrl_manipulation_task/sift_markers", 10, true);
    com_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/hrl_manipulation_task/com_markers", 
                                                                      10, true);
    
    camera_sub_ = nh_.subscribe("/head_mount_kinect/depth_registered/points", 1, &changeDetector::cameraCallback, this);
    joint_state_sub_ = nh_.subscribe("/joint_states", 10, &changeDetector::jointStateCallback, this);    
    ROS_INFO("Comms Initialized!!");

    tf::TransformListener listener;
    std::cout << "Waiting tf and joint state" << std::endl;
    ros::Rate rate(1.0);
    while (nh_.ok()){
        try{
            listener.lookupTransform("torso_lift_link", "head_mount_kinect_rgb_optical_frame",  
                                     ros::Time(0), head_transform_);
            has_tf_ = true;
        }
        catch (tf::TransformException ex){
            ROS_WARN("%s",ex.what());
        }

        if (has_tf_ && has_joint_state_) break;
        ros::spinOnce();
        rate.sleep();
    }
    std::cout << "Loaded tf and joint state" << std::endl;

    return true;
}

bool changeDetector::initFilter()
{
    range_cond_.reset (new pcl::ConditionAnd<PointType> ());

    // location condition
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
     pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::GT, 0.4)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("x", pcl::ComparisonOps::LT, 1.2)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
     pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::GT, -0.2)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::LT, 0.6)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::GT, -0.4)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::LT, 0.4)));

    // // build the filter
    // condrem_ptr_.reset (new pcl::ConditionalRemoval<PointType>(range_cond) );    

    sift_.reset(new pcl::changeDetectorKeypoint<PointType, KeyType>);
    sift_->setScales(min_scale, nr_octaves, nr_scales);
    sift_->setMinimumContrast(contrast);

}

bool changeDetector::initRobot()
{
    ROS_INFO("Start to initialize robot!!");
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

void changeDetector::cameraCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
    boost::mutex::scoped_lock lock(cloud_mtx_);
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(*input, pcl_pc);

    pcl::fromPCLPointCloud2(pcl_pc, *cloud_ptr_);

    if (has_tf_ && has_joint_state_)
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
    
    }
    
}

void changeDetector::jointStateCallback(const sensor_msgs::JointStateConstPtr &jointState) 
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
             
    for (unsigned int i=0 ; i<jt_idx_list.size() ; i++){
      joint_angles_[i] = joint_state_.position[jt_idx_list[i]];
    }

    // Get end-effector
    robot_ptr_->forwardKinematics(joint_angles_, current_ee_frame_, robot_dimensions_);
    cout << current_ee_frame_ << endl;

    has_joint_state_ = true;
}



// Publish 
void changeDetector::pubSiftMarkers()
{
    // visualization_msgs::MarkerArray siftMarkers;
    sensor_msgs::PointCloud2 msg;

    pcl::toROSMsg(*kpts_ptr_, msg);
    msg.header.frame_id = "torso_lift_link";
    sift_markers_pub_.publish(msg);

}

// Publish 
void changeDetector::pubCoMMarkers(double x, double y, double z)
{
    visualization_msgs::Marker marker;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = "torso_lift_link"; //base_link_;
    marker.header.stamp = ros::Time::now();

    marker.ns = "basic_shapes";
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;

    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration();

    marker.id = 101;

    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    visualization_msgs::MarkerArray COM_markers;
    COM_markers.markers.push_back(marker);    
    com_markers_pub_.publish(COM_markers);
}


void changeDetector::extractchangeDetector()
{
    // ROS_INFO("Start to extract changeDetector features");

#ifdef USE_changeDetector
    // changeDetector
    pcl::PointCloud<KeyType>::Ptr keypoints (new pcl::PointCloud<KeyType>);
    sift_->setInputCloud(cloud_filtered_ptr_);
    sift_->setSearchSurface(cloud_filtered_ptr_);
    sift_->compute(*keypoints);
    kpts_ptr_->points.resize(keypoints->points.size());
    pcl::copyPointCloud(*keypoints, *kpts_ptr_);
    // std::cout << "Found " << keypoints->points.size() << " keypoints." << std::endl;

    // Get mean
    double x = 0;
    double y = 0;
    double z = 0;
    for(std::size_t i = 0 ; i < kpts_ptr_->size() ; ++i)
    {
        x += kpts_ptr_->points[i].x/(double)kpts_ptr_->points.size();
        y += kpts_ptr_->points[i].y/(double)kpts_ptr_->points.size();
        z += kpts_ptr_->points[i].z/(double)kpts_ptr_->points.size();
    }

    KeyType point = KeyType();
    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = 100;
    // this->kpts_ptr_->push_back(point);
    ROS_INFO("%f %f %f", x,y,z);

    //visualization
    pubSiftMarkers();
    pubCoMMarkers(x,y,z);
#else
    // PyramidalKLT    
    tracker_->setInputCloud (cloud_filtered_ptr_);
    if (!points_ || true)
    {
        detect_keypoints (cloud_);
        tracker_->setPointsToTrack (points_);
    }
    tracker_->compute ();

    if (tracker_->getInitialized () && cloud_)
    {
        if (points_mutex_.try_lock ())
        {
            keypoints_ = tracker_->getTrackedPoints ();
            points_status_ = tracker_->getPointsToTrackStatus ();
            points_mutex_.unlock ();
        }        
    }
    

#endif
}

#ifndef USE_changeDetector
void changeDetector::detect_keypoints (const CloudConstPtr& cloud)
{
    pcl::HarrisKeypoint2D<PointType, KeyType> harris;
    harris.setInputCloud (cloud);
    harris.setNumberOfThreads (6);
    harris.setNonMaxSupression (true);
    harris.setRadiusSearch (0.01);
    harris.setMethod (pcl::HarrisKeypoint2D<PointType, KeyType>::TOMASI);
    harris.setThreshold (0.05);
    harris.setWindowWidth (5);
    harris.setWindowHeight (5);
    pcl::PointCloud<KeyType>::Ptr response (new pcl::PointCloud<KeyType>);
    harris.compute (*response);
    points_ = harris.getKeypointsIndices ();
}
#endif


int main(int argc, char **argv)
{
    ROS_INFO("changeDetector main()");
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;

    // Initialize a changeDetector object.
    changeDetector detector(n);


    ROS_INFO("changeDetector: Loop Start!!");
    ros::Rate loop_rate(1.0); // 1Hz
    while (ros::ok())
    { 
        detector.runDetector();

        // Ros loop stuff
        ros::spinOnce();
        loop_rate.sleep();        
    }

    return 0;
}


