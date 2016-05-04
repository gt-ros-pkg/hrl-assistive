#include "hrl_manipulation_task/pcl_upsample.h"

UPSAMPLE::UPSAMPLE(const ros::NodeHandle &nh): nh_(nh)
{
    cloud_ptr_.reset (new pcl::PointCloud<PointType>());
    cloud_filtered_ptr_.reset (new pcl::PointCloud<PointType>());

    initFilter();
    initComms();    
}

UPSAMPLE::~UPSAMPLE()
{
}

bool UPSAMPLE::initFilter()
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

    mls_filter_.reset (new pcl::MovingLeastSquares<PointType,PointType> ());
    return true;
}

bool UPSAMPLE::initComms()
{
    pcl_filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/pcl_filter/upsample", 10, true);
    camera_sub_ = nh_.subscribe("/head_mount_kinect/sd/points", 1, &UPSAMPLE::cameraCallback, this);

    ROS_INFO("Comms Initialized!!");
    return true;
}

void UPSAMPLE::cameraCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
    boost::mutex::scoped_lock lock(cloud_mtx_);
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(*input, pcl_pc);

    pcl::fromPCLPointCloud2(pcl_pc, *cloud_ptr_);

    // Apply point cloud filter
    // pcl::ConditionalRemoval<PointType> condrem (range_cond_);
    // condrem.setInputCloud (cloud_ptr_);
    // condrem.setKeepOrganized(true);
    // condrem.filter (*cloud_filtered_ptr_);    


    // Objects for storing the point clouds.
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Filtering object.
    mls_filter_->setInputCloud(cloud_ptr_);
    // Object for searching.
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree;
    mls_filter_->setSearchMethod(kdtree);
    // Use all neighbors in a radius of 2cm.
    mls_filter_->setSearchRadius(0.02);
    // Upsampling method. Other possibilites are DISTINCT_CLOUD, RANDOM_UNIFORM_DENSITY
    // and VOXEL_GRID_DILATION. NONE disables upsampling. Check the API for details.
    mls_filter_->setUpsamplingMethod(pcl::MovingLeastSquares<PointType, PointType>::SAMPLE_LOCAL_PLANE);
    // Radius around each point, where the local plane will be sampled.
    mls_filter_->setUpsamplingRadius(0.015);
    // Sampling step size. Bigger values will yield less (if any) new points.
    mls_filter_->setUpsamplingStepSize(0.005);
    
    mls_filter_->process(*cloud_filtered_ptr_);
}



int
main(int argc, char** argv)
{
    ROS_INFO("UPSAMPLE extractor main()");
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;


    ROS_INFO("UPSAMPLE extractor: Loop Start!!");
    ros::Rate loop_rate(20); // 
    while (ros::ok())
    { 
        // extractor.extractUPSAMPLE();
        // Ros loop stuff
        ros::spinOnce();
        loop_rate.sleep();        
    }

    return 0;


}
