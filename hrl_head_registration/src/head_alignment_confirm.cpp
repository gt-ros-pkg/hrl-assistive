#include <ros/ros.h>

#include <tf/transform_listener.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv/cv.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>

#include <hrl_head_registration/hsl_rgb_conversions.h>

typedef pcl::PointXYZRGB PRGB;
typedef pcl::PointCloud<PRGB> PCRGB;

PCRGB::Ptr aligned_pc;
ros::Subscriber align_sub;

image_transport::CameraSubscriber camera_sub;
image_transport::Publisher overlay_pub;
image_geometry::PinholeCameraModel cam_model;
boost::shared_ptr<tf::TransformListener> tf_list;

void subAlignCallback(const PCRGB::Ptr& aligned_pc_)
{
    tf::StampedTransform transform;
    ros::Time pc_time;
    pc_time.fromNSec(aligned_pc_->header.stamp * 1000ull);
    if(!tf_list->waitForTransform("/base_link", aligned_pc_->header.frame_id,
                                  pc_time, ros::Duration(3)))
        return;
    tf_list->lookupTransform("/base_link", aligned_pc_->header.frame_id, 
                             pc_time, transform);
    aligned_pc = boost::shared_ptr<PCRGB>(new PCRGB());
    pcl_ros::transformPointCloud<PRGB>(*aligned_pc_, *aligned_pc, transform);
}

void doOverlay(const sensor_msgs::ImageConstPtr& img_msg,
               const sensor_msgs::CameraInfoConstPtr& info_msg) {

    // convert camera image into opencv
    cam_model.fromCameraInfo(info_msg);
    cv_bridge::CvImagePtr cv_img = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::RGB8);

    double alpha_mult;
    ros::param::param<double>("~alpha_mult", alpha_mult, 0.5);

    uint8_t r, g, b;
    if(aligned_pc) {
        if(!tf_list->waitForTransform(img_msg->header.frame_id, "/base_link",
                                     img_msg->header.stamp, ros::Duration(3)))
            return;
        tf::StampedTransform transform;
        tf_list->lookupTransform(img_msg->header.frame_id, "/base_link", 
                                img_msg->header.stamp, transform);
        PCRGB::Ptr tf_pc(new PCRGB());
        pcl_ros::transformPointCloud<PRGB>(*aligned_pc, *tf_pc, transform);
        for(uint32_t i=0;i<tf_pc->size();i++) {
            cv::Point3d proj_pt_cv(tf_pc->points[i].x, tf_pc->points[i].y, 
                                   tf_pc->points[i].z);
            cv::Point pt2d = cam_model.project3dToPixel(proj_pt_cv);
            extractRGB(tf_pc->points[i].rgb, r, g, b);
            if(pt2d.x >= 0 && pt2d.y >= 0 && 
               pt2d.x < cv_img->image.rows && pt2d.y < cv_img->image.cols) {
                double old_r = cv_img->image.at<cv::Vec3b>(pt2d.y, pt2d.x)[0];
                double old_g = cv_img->image.at<cv::Vec3b>(pt2d.y, pt2d.x)[1];
                double old_b = cv_img->image.at<cv::Vec3b>(pt2d.y, pt2d.x)[2];
                cv_img->image.at<cv::Vec3b>(pt2d.y, pt2d.x)[0] = 
                    (uint8_t) min(alpha_mult*old_r+r, 255.0);
                cv_img->image.at<cv::Vec3b>(pt2d.y, pt2d.x)[1] = 
                    (uint8_t) min(alpha_mult*old_g+g, 255.0);
                cv_img->image.at<cv::Vec3b>(pt2d.y, pt2d.x)[2] = 
                    (uint8_t) min(alpha_mult*old_b+b, 255.0);
            }
        }
    }
    
    overlay_pub.publish(cv_img->toImageMsg());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "head_registration");
    ros::NodeHandle nh;
    image_transport::ImageTransport img_trans(nh);
    tf_list = boost::shared_ptr<tf::TransformListener>(new tf::TransformListener());

    align_sub = nh.subscribe("/head_registration/aligned_pc", 1, &subAlignCallback);
    camera_sub = img_trans.subscribeCamera("/kinect_camera", 1, 
                                           &doOverlay);
    overlay_pub = img_trans.advertise("/head_registration/confirmation", 1);

    ros::Rate r(5);
    while(ros::ok()) {
        cv::Mat image(480, 640, CV_8UC3);
        ros::spinOnce();
        r.sleep();
    }

    return 0;
}
