/*
 * This program's backbone is based on the pixel_2_3d program.
 * This program must be used with the clickable UI.
 * 
 */



//#include <numeric>
#include <ros/ros.h>
#include <algorithm>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/conditional_removal.h>
#include <tf/transform_listener.h>
#include <opencv/cv.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <pcl_ros/transforms.h>

#include <cup_finder/CupFinder.h>

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/io.h>
#include "pcl_ros/point_cloud.h"

#include <pcl/filters/voxel_grid.h>

//#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

#include "std_msgs/String.h"
#include "std_msgs/Bool.h"


#define DIST3(x1,y1,z1,x2,y2,z2) (std::sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))
#define SQ(x) ((x) * (x))
typedef pcl::PointXYZRGB PRGB;

namespace cup_finder {

    class CupFinderServer {
        public:
            ros::NodeHandle nh;
            tf::TransformListener tf_listener;
            ros::Subscriber pc_sub, l_click_sub, confirm_sub, action_sub;
            ros::Publisher pt3d_pub, cupConfirm_pub;
            ros::ServiceServer pix_srv;
            image_transport::ImageTransport img_trans;
            image_transport::CameraSubscriber camera_sub;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_pc;
            double normal_search_radius;
            std::string output_frame;
            uint32_t img_width, img_height;
            bool cam_called, pc_called, use_closest_pixel, cup_found, working, cup_est;
            geometry_msgs::PoseStamped cupLoc;


            CupFinderServer();
            void onInit();
            void cameraCallback(const sensor_msgs::ImageConstPtr& img_msg,
                                const sensor_msgs::CameraInfoConstPtr& info_msg);
            void pcCallback(sensor_msgs::PointCloud2::ConstPtr pc_msg);
            bool pixCallback(CupFinder::Request& req, CupFinder::Response& resp);
            void lClickCallback(const geometry_msgs::PointStamped& click_msg);
	    void confirmCallback(const std_msgs::String::ConstPtr& confirm_msg);
	    void actionCallback (const std_msgs::String::ConstPtr& action_msg);
	    float tableZ, cupX, cupY, cupZ;

    };

    CupFinderServer::CupFinderServer() : nh("~"), img_trans(nh),
                                       cur_pc(new pcl::PointCloud<pcl::PointXYZRGB>),
                                       cam_called(false), pc_called(false) {
        onInit();
    }

    void CupFinderServer::onInit() {
        nh.param<double>("normal_radius", normal_search_radius, 0.03);
        nh.param<bool>("use_closest_pixel", use_closest_pixel, false);
        nh.param<std::string>("output_frame", output_frame, "");
        camera_sub = img_trans.subscribeCamera<CupFinderServer>
                                              ("/image", 1, 
                                               &CupFinderServer::cameraCallback, this);
        pc_sub = nh.subscribe("/point_cloud", 1, &CupFinderServer::pcCallback, this);
//      TODO Why service activated??? Problem Here?
        pix_srv = nh.advertiseService("/finding_bowl_service", &CupFinderServer::pixCallback, this);
        pt3d_pub = nh.advertise<geometry_msgs::PoseStamped>("/RYDS_CupLocation", 1);
//	cupConfirm_pub = nh.advertise<std_msgs::String>("/RYDS_CupConfrimation", 1);
        l_click_sub = nh.subscribe("/l_mouse_click", 1, &CupFinderServer::lClickCallback, this);
        confirm_sub = nh.subscribe("/RYDS_Confirm", 1, &CupFinderServer::confirmCallback, this);
        action_sub = nh.subscribe("/RYDS_Action", 1, &CupFinderServer::actionCallback, this);
        ROS_INFO("[cup_finder] CupFinderServer loaded");
	cup_found = false;
	working = false;
	cup_est = false;
    }

    void CupFinderServer::actionCallback(const std_msgs::String::ConstPtr& action_msg) {
        std::string checker ="RYDS_FindingCup";
	std::string container = action_msg->data.c_str();
//	std::string CupFoundS = "CupConfirmed";
	if (container.compare(checker)==0) {

                std::cerr << "Action order received. Finding the bowl." <<std::endl;
		working = true;
        cup_found = false;
		if (cup_est){
			pt3d_pub.publish(cupLoc);
/*			if(cup_found) {
				cupConfirm_pub.publish(CupFoundS);
			}*/
		}
	}
	else {
		working = false;
		if (cup_est){
			pt3d_pub.publish(cupLoc);
		}
	}
    }


    void CupFinderServer::cameraCallback(const sensor_msgs::ImageConstPtr& img_msg,
                                         const sensor_msgs::CameraInfoConstPtr& info_msg) {
        if(!info_msg)
            return;
        img_width = info_msg->width;
        img_height = info_msg->height;
        cam_called = true;
        camera_sub.shutdown();
    }

    void CupFinderServer::pcCallback(sensor_msgs::PointCloud2::ConstPtr pc_msg) {
        pcl::fromROSMsg(*pc_msg, *cur_pc);
        pc_called = true;
    }

    bool CupFinderServer::pixCallback(CupFinder::Request& req, CupFinder::Response& resp) {
        if (!working)
	{
	  return true;
	} 

	if (cup_found)
	{
	  return true;
	}

        resp.pixel3d.pose.position.x = -10000.0;
        resp.pixel3d.pose.position.y = -10000.0;
        resp.pixel3d.pose.position.z = -10000.0;

        if(!cam_called) {
            ROS_WARN("No camera_info message received.");
            resp.error_flag = resp.NO_CAMERA_INFO;
            return true;
        }
        if(!pc_called) {
            ROS_WARN("No point cloud message received.");
            resp.error_flag = resp.NO_POINT_CLOUD;
            return true;
        }

        int64_t pc_ind = req.pixel_u + req.pixel_v * img_width;
        if(req.pixel_u < 0 || req.pixel_v < 0 || 
           req.pixel_u >= (int32_t) img_width || req.pixel_v >= (int32_t) img_height) {
            ROS_WARN("Pixel requested is outside image size.");
            resp.error_flag = resp.OUTSIDE_IMAGE;
            return true;
        }
        geometry_msgs::PointStamped pt3d, pt3d_trans;
        pt3d_trans.header.frame_id = cur_pc->header.frame_id;
        pt3d_trans.header.stamp = ros::Time::now();
        if(cur_pc->points[pc_ind].x != cur_pc->points[pc_ind].x) {
            if(use_closest_pixel) {
                // find the closest pixel that has a point in the PC
                std::vector<double> dists(img_width * img_height, 1e30);
                int64_t cur_pc_ind;
                for(int64_t i=0;i<img_height;i++) {
                    for(int64_t j=0;j<img_width;j++) {
                        cur_pc_ind = j + i * img_width;
                        if(cur_pc->points[cur_pc_ind].x == cur_pc->points[cur_pc_ind].x)
                            dists[cur_pc_ind] = SQ(req.pixel_u - j) + SQ(req.pixel_v - i);
                    }
                }
                pc_ind = std::min_element(dists.begin(), dists.end()) - dists.begin();
            } else {
                ROS_WARN("Point cloud not defined for this region.");
                resp.error_flag = resp.OUTSIDE_POINT_CLOUD;
                return true;
            }
        }
        pt3d_trans.point.x = cur_pc->points[pc_ind].x;
        pt3d_trans.point.y = cur_pc->points[pc_ind].y;
        pt3d_trans.point.z = cur_pc->points[pc_ind].z;

        // Filter to only points in small voxel range
        pcl::ConditionAnd<PRGB>::Ptr near_cond(new pcl::ConditionAnd<PRGB>());
        pcl::PointCloud<PRGB>::Ptr near_pts(new pcl::PointCloud<PRGB>());
        pcl::ConditionalRemoval<PRGB> near_extract;
        double voxel_size = normal_search_radius*2.1;
        near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                                 "x", pcl::ComparisonOps::GT, pt3d_trans.point.x - voxel_size/2)));
        near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                                 "x", pcl::ComparisonOps::LT, pt3d_trans.point.x + voxel_size/2)));
        near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                                 "y", pcl::ComparisonOps::GT, pt3d_trans.point.y - voxel_size/2)));
        near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                                 "y", pcl::ComparisonOps::LT, pt3d_trans.point.y + voxel_size/2)));
        near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                                 "z", pcl::ComparisonOps::GT, pt3d_trans.point.z - voxel_size/2)));
        near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                                 "z", pcl::ComparisonOps::LT, pt3d_trans.point.z + voxel_size/2)));
        near_extract.setCondition(near_cond);
        near_extract.setKeepOrganized(false);
        near_extract.setInputCloud(cur_pc);
        near_extract.filter(*near_pts);
        std::vector<int> inds;
        pcl::removeNaNFromPointCloud<PRGB>(*near_pts, *near_pts, inds);

        uint32_t closest_ind;
        double closest_dist = 1000, cur_dist;
        for(uint32_t i=0;i<inds.size();i++) {
            cur_dist = DIST3(pt3d_trans.point.x, pt3d_trans.point.y, pt3d_trans.point.z,
                             near_pts->points.at(i).x, near_pts->points.at(i).y, 
                             near_pts->points.at(i).z);
            if(cur_dist < closest_dist) {
                closest_dist = cur_dist;
                closest_ind = i;
            }
        }

        // Compute normals
        pcl::PointCloud<pcl::Normal>::Ptr normals_ptr(new pcl::PointCloud<pcl::Normal>());
        pcl::search::KdTree<PRGB>::Ptr normals_tree (new pcl::search::KdTree<PRGB> ());
        pcl::PointCloud<PRGB> mls_points;
        pcl::MovingLeastSquares<PRGB, pcl::Normal> mls;
        normals_tree->setInputCloud(near_pts);
        mls.setOutputNormals(normals_ptr);
        mls.setInputCloud(near_pts);
        mls.setPolynomialFit(true);
        mls.setSearchMethod(normals_tree);
        mls.setSearchRadius(normal_search_radius);
        mls.reconstruct(mls_points);

        // convert normal to quaternion
        double nx = normals_ptr->points[closest_ind].normal[0];
        double ny = normals_ptr->points[closest_ind].normal[1];
        double nz = normals_ptr->points[closest_ind].normal[2];
        double dot = nx*pt3d_trans.point.x + ny*pt3d_trans.point.y + nz*pt3d_trans.point.z;
        if(dot > 0) { nx = -nx; ny = -ny; nz = -nz; }
       
        //Phil's update, now returns direction of pose (x-axis) along normal
        btVector3 normal_vec(nx,ny,nz);
        btVector3 x_axis(1.0,0.0,0.0);
        btVector3 axis = x_axis.cross(normal_vec);
        double angle = x_axis.angle(normal_vec);
        btQuaternion quat(axis, angle);
        
        //Kelsey's solution, returns Z-axis of quaternion along normal
        //double j = std::sqrt(1/(1+ny*ny/(nz*nz)));
        //double k = -ny*j/nz;
        //btMatrix3x3 M (0,  ny*k - nz*j,  nx,      
        //               j,  -nx*k,        ny,      
        //               k,  nx*j,         nz);
        //btQuaternion quat;
        //M.getRotation(quat);

        geometry_msgs::PoseStamped pt3d_pose;
        pt3d_pose.header.frame_id = cur_pc->header.frame_id;
        pt3d_pose.header.stamp = ros::Time(0);
        pt3d_pose.pose.position.x = pt3d_trans.point.x;
        pt3d_pose.pose.position.y = pt3d_trans.point.y;
        pt3d_pose.pose.position.z = pt3d_trans.point.z;
        pt3d_pose.pose.orientation.x = quat.getX();
        pt3d_pose.pose.orientation.y = quat.getY();
        pt3d_pose.pose.orientation.z = quat.getZ();
        pt3d_pose.pose.orientation.w = quat.getW();
        

	float centerX = pt3d_trans.point.x;
	float centerY = pt3d_trans.point.y;
	float centerZ = pt3d_trans.point.z;

        if(output_frame == "")
            output_frame = cur_pc->header.frame_id;
        tf_listener.transformPose(output_frame, pt3d_pose, pt3d_pose);
        resp.pixel3d.header.frame_id = output_frame;
        resp.pixel3d.header.stamp = ros::Time::now();
        resp.pixel3d.pose.position.x = pt3d_pose.pose.position.x;
        resp.pixel3d.pose.position.y = pt3d_pose.pose.position.y;
        resp.pixel3d.pose.position.z = pt3d_pose.pose.position.z;
        resp.pixel3d.pose.orientation.x = pt3d_pose.pose.orientation.x;
        resp.pixel3d.pose.orientation.y = pt3d_pose.pose.orientation.y;
        resp.pixel3d.pose.orientation.z = pt3d_pose.pose.orientation.z;
        resp.pixel3d.pose.orientation.w = pt3d_pose.pose.orientation.w;

/*
        pt3d_pub.publish(pt3d_pose);
        ROS_INFO("[pixel_2_3d] Pixel (%d, %d) converted to pose (%f, %f, %f), (%f, %f, %f, %f) in %s",
                 req.pixel_u, req.pixel_v, 
                 pt3d_pose.pose.position.x, pt3d_pose.pose.position.y, pt3d_pose.pose.position.z,
                 pt3d_pose.pose.orientation.x, pt3d_pose.pose.orientation.y,
                 pt3d_pose.pose.orientation.z, pt3d_pose.pose.orientation.w,
                 output_frame.c_str());
*/


	// Create the filtering object
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_pc_sample_box (new pcl::PointCloud<pcl::PointXYZRGB>);
    	pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_pc (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr sample_box (new pcl::ConditionAnd<pcl::PointXYZRGB> ());
    	sample_box->addComparison (pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZRGB> ("x", pcl::ComparisonOps::GT, centerX-0.20)));
    	sample_box->addComparison (pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZRGB> ("x", pcl::ComparisonOps::LT, centerX+0.20)));
    	sample_box->addComparison (pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZRGB> ("y", pcl::ComparisonOps::GT, centerY-0.20)));
    	sample_box->addComparison (pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZRGB> ("y", pcl::ComparisonOps::LT, centerY+0.20)));
    	sample_box->addComparison (pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZRGB> ("z", pcl::ComparisonOps::GT, centerZ-0.20)));
    	sample_box->addComparison (pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZRGB> ("z", pcl::ComparisonOps::LT, centerZ+0.20)));
//    	sample_box->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::LT, 130)));
//    	sample_box->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::LT, 130)));
//    	sample_box->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::GT, 150)));

    	// build the filter
    	pcl::ConditionalRemoval<pcl::PointXYZRGB> sample_box_rem (sample_box);
    	sample_box_rem.setInputCloud (cur_pc); //TODO I changed here.
    	sample_box_rem.setKeepOrganized(true);
    	// apply filter
    	sample_box_rem.filter (*filter_pc);



	//Transform PCL
	//cur_pc_sample_box.header.stamp = ros::Time::now();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cupZone (new pcl::PointCloud<pcl::PointXYZRGB>);
//	tf_listener.transformPointCloud(output_frame, cur_pc_sample_box, cupZone);
	pcl_ros::transformPointCloud(output_frame, *filter_pc, *cupZone, tf_listener); //TODO fix const problem. 
        std::cerr << "Found this many points   :   " << cupZone->points.size()<< "points" <<std::endl;





	float Test;
	float dataZ = 0;
	float numberT = 0;
	float dataX = 0;
	float dataY = 0;
	int numberR = 0;



	for (size_t i = 0; i < cupZone->points.size (); ++i)
		  {
		  Test = cupZone->points[i].x;
		  	if (Test != Test) {
		 	 }
			  else {
				dataX = dataX + cupZone->points[i].x;
				dataY = dataY + cupZone->points[i].y;
				dataZ = dataZ + cupZone->points[i].z;
				numberT= numberT+1;
				numberR= numberR+1;
			  }
		  }
	int tableFinder[numberR];
	int counter = 0;
	int modeCounter[numberR];

	for (size_t i = 0; i < cupZone->points.size (); ++i)
		{
		Test = cupZone->points[i].x;
			if(Test != Test) {
			}
			 else {
			tableFinder[counter] = (int)(cupZone->points[i].z * 100);
			counter = counter +1;
			}
		}

	for (int i = 0; i < numberR; ++i)
		{
			modeCounter[i] = 0;
			for (int j = 0; j < numberR; ++j)
				{
					if ( i != j && tableFinder[i] == tableFinder[j]) 
					{
						modeCounter[i] = modeCounter[i] + 1;
					}
				}
		}

	int modeLocation = 0;
	int modeNumber = modeCounter[modeLocation];
	for (int i = 0; i < numberR; ++i)
	{
		if (modeNumber < modeCounter[i]) 
			{
				modeLocation = i;
				modeNumber = modeCounter[i];
			}    

	}

	float tableZ;

	// Determining TableZ TODO I kinda screwed around here, too.
//	tableZ = dataZ/numberT;
	tableZ = ((float) tableFinder[modeLocation] )/ 100;




	//TODO starting of the XY finder viz table
	//Cup segmentation by Z
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cupPCL (new pcl::PointCloud<pcl::PointXYZRGB>);
    	pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr sample_box2 (new pcl::ConditionAnd<pcl::PointXYZRGB> ());
    	sample_box2->addComparison (pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZRGB> ("z", pcl::ComparisonOps::GT, tableZ+0.01)));
    	sample_box2->addComparison (pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZRGB> ("z", pcl::ComparisonOps::LT, tableZ+0.15)));
    	sample_box2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::LT, 140)));
    	sample_box2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::LT, 140)));
    	sample_box2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::ConstPtr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::GT, 140)));

    	// build the filter
    	pcl::ConditionalRemoval<pcl::PointXYZRGB> sample_box_rem2 (sample_box2);
    	sample_box_rem2.setInputCloud (cupZone);
    	sample_box_rem2.setKeepOrganized(true);
    	// apply filter
    	sample_box_rem2.filter (*cupPCL);


	//TODO using this data to find better XY

	dataX = 0;
	dataY = 0;
	float numberS = 0;
	for (size_t i =0; i < cupPCL->points.size (); ++i)
		{
		
			Test = cupPCL->points[i].x;
			if (Test != Test) {
			}
			 else {
				dataX = dataX + cupPCL->points[i].x;
				dataY = dataY + cupPCL->points[i].y;
				numberS = numberS+1;
			}
		}
	
	float cupPointZ = tableZ;
	float cupPointX = dataX/numberS;
	float cupPointY = dataY/numberS;
	


	//Publishing it as topic/.
	//TODO fix to parameter?
        cupLoc.header.frame_id = output_frame;
        cupLoc.header.stamp = ros::Time::now();
        cupLoc.pose.position.x = cupPointX;
        cupLoc.pose.position.y = cupPointY;
        cupLoc.pose.position.z = cupPointZ;
        cupLoc.pose.orientation.x = 0;
        cupLoc.pose.orientation.y = 0;
        cupLoc.pose.orientation.z = 0;
        cupLoc.pose.orientation.w = 1;
        pt3d_pub.publish(cupLoc);
    std::cerr << "Cup Location published. It is..  : (" << cupPointX << ", " << cupPointY << ", " << cupPointZ<< ")" << std::endl;


	    cup_est = true;

        return true;
    }

    void CupFinderServer::lClickCallback(const geometry_msgs::PointStamped& click_msg) {
        CupFinder::Request req; CupFinder::Response resp;
        req.pixel_u = click_msg.point.x;
        req.pixel_v = click_msg.point.y;
        if (!cup_found && working) {
	    pixCallback(req, resp);
	}
    }

    void CupFinderServer::confirmCallback(const std_msgs::String::ConstPtr& confirm_msg) {
        std::string checker ="RYDS_BowlRegConfirm";
	std::string container = confirm_msg->data.c_str();
	if (container.compare(checker) == 0) {
		if(working) {
			cup_found = true;
		}
	}
    }

};


using namespace cup_finder;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cup_finder");
    CupFinderServer p3d;
    ros::spin();
    return 0;
}
