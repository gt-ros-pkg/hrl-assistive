#include "hrl_manipulation_task/octree_change_detection.h"


changeDetector::changeDetector(const ros::NodeHandle &nh): nh_(nh)
{
    cloud_ptr_.reset (new pcl::PointCloud<PointType>());
    cloud_filtered_ptr_.reset (new pcl::PointCloud<PointType>());
    octree_ptr_.reset (new pcl::octree::OctreePointCloudChangeDetector<PointType>(resolution));
    extract_ptr_.reset(new pcl::ExtractIndices<PointType>());
    inliers_ptr_.reset(new pcl::PointIndices ());

    has_tf_ = false;
    has_joint_state_ = false;

    getParams();
    initFilter();
    initComms();    
    initRobot();
}

changeDetector::~changeDetector()
{
    std::vector<KDL::Frame> T;
    frames_.swap(T);
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
    pcl_filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/hrl_manipulation_task/pcl_filtered", 10, true);
    octree_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/hrl_manipulation_task/octree_changes",
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

        // Switch octree buffers: This resets octree but keeps previous tree structure in memory.
        if (counter_>0) octree_ptr_->switchBuffers ();

        // Add points from cloud to octree
        octree_ptr_->setInputCloud (cloud_filtered_ptr_);
        octree_ptr_->addPointsFromInputCloud ();
        counter_++;
        if (counter_ > 65534) counter_ = 1;    
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
    for (unsigned int i=0 ; i<frames_.size() ; i++)
    {
        robot_ptr_->forwardKinematics(joint_angles_, frames_[i], i);
    }
    // 
    // cout << current_ee_frame_ << endl;

    has_joint_state_ = true;
}


// Publish 
void changeDetector::pubFilteredPCL()
{
    sensor_msgs::PointCloud2 msg;

    pcl::toROSMsg(*cloud_filtered_ptr_, msg);
    msg.header.frame_id = "/torso_lift_link";
    pcl_filtered_pub_.publish(msg);    
}

// Publish 
// void changeDetector::pubChangeMarkers(const std::vector<int> &newPointIdxVector)
void changeDetector::pubChangeMarkers()
{

    ros::Time rostime = ros::Time::now();
    // COLOR c = {1.0,1.0,1.0}; // white
    // COLOR cr = getColor(value, FAKE_LOGODD, 0.9); // need parameterize
    std_msgs::ColorRGBA cubeColor;
    cubeColor.r = 0.0f;
    cubeColor.g = 1.0f;
    cubeColor.b = 0.0f;
    cubeColor.a = 1.0f;

    visualization_msgs::MarkerArray octreeChangeVis;
    octreeChangeVis.markers.resize(1);

    for (size_t i = 0; i < cloud_filtered_ptr_->points.size (); ++i)
    {
        // Cubes
        geometry_msgs::Point cubeCenter;
        cubeCenter.x = cloud_filtered_ptr_->points[i].x;
        cubeCenter.y = cloud_filtered_ptr_->points[i].y;
        cubeCenter.z = cloud_filtered_ptr_->points[i].z;
        
        octreeChangeVis.markers[0].points.push_back(cubeCenter);
        octreeChangeVis.markers[0].colors.push_back(cubeColor);
    }

    double size =  resolution; //cloud_filtered_ptr_->getVoxelSquaredDiameter();

    // For occupied voxel
    octreeChangeVis.markers[0].header.frame_id = "/torso_lift_link";
    octreeChangeVis.markers[0].header.stamp = rostime;
    octreeChangeVis.markers[0].ns = "leaf";
    octreeChangeVis.markers[0].id = 1000;
    octreeChangeVis.markers[0].type = visualization_msgs::Marker::CUBE_LIST;
    octreeChangeVis.markers[0].scale.x = size;
    octreeChangeVis.markers[0].scale.y = size;
    octreeChangeVis.markers[0].scale.z = size;
    octreeChangeVis.markers[0].color = cubeColor;
    octreeChangeVis.markers[0].lifetime = ros::Duration();
    
    if ( octreeChangeVis.markers[0].points.size() > 0 )                               
        octreeChangeVis.markers[0].action = visualization_msgs::Marker::ADD;
    else
        octreeChangeVis.markers[0].action = visualization_msgs::Marker::DELETE;        

    octree_marker_pub_.publish(octreeChangeVis);
}



void changeDetector::runDetector()
{
    // ROS_INFO("Start to extract changeDetector features");

    ROS_INFO("changeDetector: Loop Start!!");
    ros::Rate loop_rate(10.0); // 1Hz
    while (ros::ok())
    { 
        if (counter_==0) continue;

        // changeDetector
        std::vector<int> newPointIdxVector;

        // Get vector of point indices from octree voxels which did not exist in previous buffer
        octree_ptr_->getPointIndicesFromNewVoxels (newPointIdxVector);
        inliers_ptr_->indices = newPointIdxVector;

        // Extract the inliers
        extract_ptr_->setInputCloud (cloud_filtered_ptr_);
        extract_ptr_->setIndices (inliers_ptr_);
        extract_ptr_->setNegative (false);
        extract_ptr_->filter (*cloud_filtered_ptr_);

        // Output points
        cout << "The Number of change voxels " <<  newPointIdxVector.size() << " " << 
            octree_ptr_->getVoxelSquaredDiameter() << endl;
        // std::cout << "Output from getPointIndicesFromNewVoxels:" << std::endl;
        // for (size_t i = 0; i < newPointIdxVector.size (); ++i)
        //     std::cout << i << "# Index:" << newPointIdxVector[i]
        //               << "  Point:" << cloud_filtered_ptr_->points[newPointIdxVector[i]].x << " "
        //               << cloud_filtered_ptr_->points[newPointIdxVector[i]].y << " "
        //               << cloud_filtered_ptr_->points[newPointIdxVector[i]].z << std::endl;


        filterRobotBody();
        dist_Point_to_Segment();

        //visualization
        // pubCurOctree(octree);
        pubFilteredPCL();
        pubChangeMarkers();


        // Ros loop stuff
        ros::spinOnce();
        loop_rate.sleep();        
    }
}

void changeDetector::filterRobotBody(const pcl::PointCloud<PointType>::Ptr& pcl_cloud)
{
    for (unsigned int i = 0; i < pcl_cloud->size(); i++)
    {

    }    
}

double changeDetector::dist_Point_to_Segment( KDL::Vector p0, KDL::Vector p1, Segment S)
{
    KDL::Vector v = S.P1 - S.P0;
    KDL::Vector w = P - S.P0;

    double c1 = dot(w,v);
    if ( c1 <= 0 )
        return d(P, S.P0);

    double c2 = dot(v,v);
    if ( c2 <= c1 )
        return d(P, S.P1);

    double b = c1 / c2;
    Point Pb = S.P0 + b * v;
    return d(P, Pb);
}


int main(int argc, char **argv)
{
    ROS_INFO("changeDetector main()");
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;

    // Initialize a changeDetector object.
    changeDetector detector(n);
    detector.runDetector();

    return 0;
}


