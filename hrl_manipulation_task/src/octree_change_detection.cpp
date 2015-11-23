#include "hrl_manipulation_task/octree_change_detection.h"


changeDetector::changeDetector(const ros::NodeHandle &nh): nh_(nh)
{
    cloud_ptr_.reset (new pcl::PointCloud<PointType>());
    cloud_filtered_ptr_.reset (new pcl::PointCloud<PointType>());
    cloud_changes_ptr_.reset (new pcl::PointCloud<PointType>());
    octree_ptr_.reset (new pcl::octree::OctreePointCloudChangeDetector<PointType>(resolution));
    extract_ptr_.reset(new pcl::ExtractIndices<PointType>());
    sorfilter_ptr_.reset(new pcl::StatisticalOutlierRemoval<PointType>(false));
    voxelfilter_.setSaveLeafLayout(true);

    has_tf_ = false;
    has_joint_state_ = false;
    has_robot_ = false;
    counter_ = 0;
    time_gap_counter_=0;

    getParams();
    initFilter();
    initComms();    
    initRobot();
}

changeDetector::~changeDetector()
{
    std::vector<KDL::Frame*> T;
    cur_frames_.swap(T);
    last_frames_.swap(T);
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
    pcl_changes_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/hrl_manipulation_task/pcl_changes", 10, true);
    changes_pub_ = nh_.advertise<hrl_anomaly_detection::pclChange>("/hrl_manipulation_task/changes", 10, true);
    
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
      pcl::FieldComparison<PointType> ("y", pcl::ComparisonOps::LT, 0.7)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::GT, -0.4)));
    range_cond_->addComparison (pcl::FieldComparison<PointType>::ConstPtr (new
      pcl::FieldComparison<PointType> ("z", pcl::ComparisonOps::LT, 0.4)));

    // // build the filter
    // condrem_ptr_.reset (new pcl::ConditionalRemoval<PointType>(range_cond) );    

}

bool changeDetector::initRobot()
{
    cur_frames_.push_back(new KDL::Frame);    
    cur_frames_.push_back(new KDL::Frame);    
    cur_frames_.push_back(new KDL::Frame);    
    cur_frames_.push_back(new KDL::Frame);    

    last_frames_.push_back(new KDL::Frame);    
    last_frames_.push_back(new KDL::Frame);    
    last_frames_.push_back(new KDL::Frame);    
    last_frames_.push_back(new KDL::Frame);    

    last1_frames_.push_back(new KDL::Frame);    
    last1_frames_.push_back(new KDL::Frame);    
    last1_frames_.push_back(new KDL::Frame);    
    last1_frames_.push_back(new KDL::Frame);    

    last2_frames_.push_back(new KDL::Frame);    
    last2_frames_.push_back(new KDL::Frame);    
    last2_frames_.push_back(new KDL::Frame);    
    last2_frames_.push_back(new KDL::Frame);    

    last3_frames_.push_back(new KDL::Frame);    
    last3_frames_.push_back(new KDL::Frame);    
    last3_frames_.push_back(new KDL::Frame);    
    last3_frames_.push_back(new KDL::Frame);    

    last4_frames_.push_back(new KDL::Frame);    
    last4_frames_.push_back(new KDL::Frame);    
    last4_frames_.push_back(new KDL::Frame);    
    last4_frames_.push_back(new KDL::Frame);    

    last5_frames_.push_back(new KDL::Frame);    
    last5_frames_.push_back(new KDL::Frame);    
    last5_frames_.push_back(new KDL::Frame);    
    last5_frames_.push_back(new KDL::Frame);    

    last6_frames_.push_back(new KDL::Frame);    
    last6_frames_.push_back(new KDL::Frame);    
    last6_frames_.push_back(new KDL::Frame);    
    last6_frames_.push_back(new KDL::Frame);    


    radius_.push_back(0.10); // upper arm
    radius_.push_back(0.10); // forearm
    radius_.push_back(0.08); // hand

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
    
    has_robot_=true;
    ROS_INFO("Robot Initialized!!");

    return true;
}

void changeDetector::cameraCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
    boost::mutex::scoped_lock lock(cloud_mtx_);
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(*input, pcl_pc);

    pcl::fromPCLPointCloud2(pcl_pc, *cloud_ptr_);
    header_ = input->header;

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


        for (unsigned int i=0 ; i<cur_frames_.size() ; i++)
        {
            last_frames_[i]  = last1_frames_[i];
            last1_frames_[i] = last2_frames_[i];
            last2_frames_[i] = last3_frames_[i];
            last3_frames_[i] = last4_frames_[i];
            last4_frames_[i] = last5_frames_[i];
            last5_frames_[i] = last6_frames_[i];
            last6_frames_[i] = cur_frames_[i];
        }

        // Get end-effector
        for (unsigned int i=0 ; i<cur_frames_.size() ; i++)
            robot_ptr_->forwardKinematics(joint_angles_, *cur_frames_[i], i);

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
void changeDetector::pubChangesPCL()
{
    sensor_msgs::PointCloud2 msg;

    pcl::toROSMsg(*cloud_changes_ptr_, msg);
    msg.header.frame_id = "/torso_lift_link";
    pcl_changes_pub_.publish(msg);    
}

// Publish 
void changeDetector::pubChanges()
{
    hrl_anomaly_detection::pclChange msg;
    msg.header = header_;
    msg.header.frame_id = "/torso_lift_link";

    for (unsigned int i = 0; i < cloud_changes_ptr_->size(); i++)
    {
        msg.centers_x.push_back(cloud_changes_ptr_->points[i].x);
        msg.centers_y.push_back(cloud_changes_ptr_->points[i].y);
        msg.centers_z.push_back(cloud_changes_ptr_->points[i].z);
    }

    changes_pub_.publish(msg);
}


void changeDetector::runDetector()
{
    // ROS_INFO("Start to extract changeDetector features");
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    ROS_INFO("changeDetector: Loop Start!!");
    ros::Rate loop_rate(10.0); // 1Hz
    while (ros::ok())
    { 
        if (counter_==0) continue;

        // changeDetector
        std::vector<int> newPointIdxVector;

        // Get vector of point indices from octree voxels which did not exist in previous buffer
        octree_ptr_->getPointIndicesFromNewVoxels (newPointIdxVector);
        inliers->indices = newPointIdxVector;

        // Extract the inliers
        extract_ptr_->setInputCloud (cloud_filtered_ptr_);
        extract_ptr_->setIndices (inliers);
        extract_ptr_->setNegative (false);
        extract_ptr_->filter (*cloud_changes_ptr_);        
        if (cloud_changes_ptr_->points.size() == 0){
            ros::spinOnce();
            continue;
        }

        // Create the filtering object
        voxelfilter_.setInputCloud (cloud_changes_ptr_);
        voxelfilter_.setLeafSize (0.01f, 0.01f, 0.01f);
        voxelfilter_.filter (*cloud_changes_ptr_);
        if (cloud_changes_ptr_->points.size() == 0){
            ros::spinOnce();
            continue;
        }

        // Output points
        // cout << "The Number of change voxels " <<  newPointIdxVector.size() << " " << 
        //     octree_ptr_->getVoxelSquaredDiameter() << endl;
        // std::cout << "Output from getPointIndicesFromNewVoxels:" << std::endl;
        // for (size_t i = 0; i < newPointIdxVector.size (); ++i)
        //     std::cout << i << "# Index:" << newPointIdxVector[i]
        //               << "  Point:" << cloud_filtered_ptr_->points[newPointIdxVector[i]].x << " "
        //               << cloud_filtered_ptr_->points[newPointIdxVector[i]].y << " "
        //               << cloud_filtered_ptr_->points[newPointIdxVector[i]].z << std::endl;

        robotBodyFilter(cloud_changes_ptr_);        
        if (cloud_changes_ptr_->points.size() == 0){
            ros::spinOnce();
            continue;
        }

        // Statistical outlier filter
        sorfilter_ptr_->setInputCloud (cloud_changes_ptr_);
        sorfilter_ptr_->setMeanK (8);
        sorfilter_ptr_->setStddevMulThresh (-0.0);
        sorfilter_ptr_->filter (*cloud_changes_ptr_);
        if (cloud_changes_ptr_->points.size() == 0){
            ros::spinOnce();
            continue;
        }

        // int mean_k = 8;
        // double dist = 0.005;
        // double std_mul = 1.0;
        // noiseFilter(cloud_filtered_ptr_, mean_k, dist, std_mul);

        //visualization
        // pubFilteredPCL();
        pubChangesPCL();
        pubChanges();
        // pubChangeMarkers();


        // Ros loop stuff
        ros::spinOnce();
        loop_rate.sleep();        
    }
}

void changeDetector::robotBodyFilter(const pcl::PointCloud<PointType>::Ptr& pcl_cloud)
{
    pcl::PointIndices::Ptr outliers (new pcl::PointIndices ());


    for (unsigned int i=0 ; i<cur_frames_.size()-1 ; i++)
    {
        std::vector<int> outlier_idx;
        for (unsigned int j = 0; j < pcl_cloud->size(); j++)
        {
            KDL::Vector p( pcl_cloud->points[j].x, pcl_cloud->points[j].y, pcl_cloud->points[j].z );
            double dist = dist_Point_to_Segment(p, cur_frames_[i]->p, cur_frames_[i+1]->p);
            if ( dist < radius_[i])
            {
                outlier_idx.push_back(j);
                continue;
            }
            dist = dist_Point_to_Segment(p, last_frames_[i]->p, last_frames_[i+1]->p);
            if ( dist < radius_[i])
            {
                outlier_idx.push_back(j);
                continue;
            }
            // else
            //     cout << dist << " / " << "outlier: " << j << " / " << pcl_cloud->points.size()<< endl;

        }    
        outliers->indices = outlier_idx;
        // cout << i << endl;
        
        extract_ptr_->setInputCloud (pcl_cloud);
        extract_ptr_->setIndices (outliers);
        extract_ptr_->setNegative (true);
        extract_ptr_->filter (*pcl_cloud);
    }
}

void changeDetector::noiseFilter(const pcl::PointCloud<PointType>::Ptr& pcl_cloud, int mean_k, 
                                 double distance_threshold, double std_mul)
{
    SearcherPtr searcher_;
    // KdTreePtr tree_;
    std::vector<int> indices, indices_;
    bool negative_ = false;

    
    indices_.resize(pcl_cloud->points.size() );


    // Initialize the search class
    if (!searcher_)
    {
        if (pcl_cloud->isOrganized ())
            searcher_.reset (new pcl::search::OrganizedNeighbor<PointType> ());
        else
            searcher_.reset (new pcl::search::KdTree<PointType> (false));
    }
    searcher_->setInputCloud (pcl_cloud);

   // The arrays to be used
   std::vector<int> nn_indices (mean_k);
   std::vector<float> nn_dists (mean_k);
   std::vector<float> distances (indices_.size ());
   std::vector<float> max_distances (indices_.size ());
   indices.resize (indices_.size ());
   int oii = 0, rii = 0;  // oii = output indices iterator, rii = removed indices iterator


   // First pass: Compute the mean distances for all points with respect to their k nearest neighbors
   int valid_distances = 0;
   for (int iii = 0; iii < static_cast<int> (indices_.size ()); ++iii)  // iii = input indices iterator
   {
       if (!pcl_isfinite (pcl_cloud->points[indices_[iii]].x) ||
           !pcl_isfinite (pcl_cloud->points[indices_[iii]].y) ||
           !pcl_isfinite (pcl_cloud->points[indices_[iii]].z))
       {
           distances[iii] = 0.0;
           max_distances[iii] = 100000.0;
           continue;
       }
       
       // Perform the nearest k search
       if (searcher_->nearestKSearch (indices_[iii], mean_k + 1, nn_indices, nn_dists) == 0)
       {
           distances[iii] = 0.0;
           max_distances[iii] = 100000.0;
           ROS_WARN ("Searching for the closest %d neighbors failed.\n", mean_k);
           continue;
       }
  
       // Calculate the mean distance to its neighbors
       double dist_sum = 0.0;
       double dist;
       for (int k = 1; k < mean_k + 1; ++k){  // k = 0 is the query point
           dist = sqrt (nn_dists[k]);
           dist_sum += dist;
           if (k==1) max_distances[iii] = static_cast<float>(dist);
           else if ( max_distances[iii] < dist) max_distances[iii] = static_cast<float>(dist);
       }
       
       distances[iii] = static_cast<float> (dist_sum / mean_k);
       valid_distances++;
   }

   // Estimate the mean and the standard deviation of the distance vector
   double sum = 0, sq_sum = 0;
   for (size_t i = 0; i < distances.size (); ++i)
   {
       sum += distances[i];
       sq_sum += distances[i] * distances[i];
   }
   double mean = sum / static_cast<double>(valid_distances);
   double variance = (sq_sum - sum * sum / static_cast<double>(valid_distances)) / (static_cast<double>(valid_distances) - 1);
   double stddev = sqrt (variance);

   // cout << "Current mean: " << mean << endl;
   //getMeanStd (distances, mean, stddev);
   // double distance_threshold = mean + std_mul_ * stddev;
   double dist_thres;
   // if (distance_threshold > mean + std_mul * stddev)
   //     dist_thres = mean + std_mul * stddev;
   // else
   //     dist_thres = distance_threshold;
   dist_thres = distance_threshold;

   // Second pass: Classify the points on the computed distance threshold
   for (int iii = 0; iii < static_cast<int> (indices_.size ()); ++iii)  // iii = input indices iterator
   {
       // Points having a too high average distance are outliers and are passed to removed indices
       // Unless negative was set, then it's the opposite condition
       // if ((!negative_ && distances[iii] > dist_thres) || (negative_ && distances[iii] <= dist_thres))
       // {
       //     continue;
       // }
       if ((!negative_ && max_distances[iii] > dist_thres) || (negative_ && max_distances[iii] <= dist_thres))
       {
           continue;
       }

       cout << max_distances[iii] << endl;
 
       // Otherwise it was a normal point for output (inlier)
       indices[oii++] = indices_[iii];
   }
   
   // Resize the output arrays
   indices.resize (oii);


   // Extract the inliers
   pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
   extract_ptr_->setInputCloud (pcl_cloud);
   inliers->indices = indices;
   extract_ptr_->setIndices (inliers);
   extract_ptr_->setNegative (false);
   extract_ptr_->filter (*pcl_cloud);
   
       
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


