#include <ros/ros.h>

#include <geometry_msgs/PointStamped.h>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>

#include <hrl_head_registration/head_registration.h>

#define USE_COLOR_MODEL

char *orig_filename, *face_filename;

ros::Subscriber pc_sub;
PCRGB::Ptr cur_pc;

void subPCCallback(const PCRGB::Ptr& pc_msg);
static void mouseClickCallback(int event, int x, int y, int flags, void* data);

void subPCCallback(const PCRGB::Ptr& pc_msg)
{
    cv::Mat image(pc_msg->height, pc_msg->width, CV_8UC3);
            
    uint8_t r, g, b;
    for(uint32_t i=0;i<pc_msg->size();i++) {
        if(PT_IS_NOT_NAN(pc_msg, i)) {
            extractRGB(pc_msg->points[i].rgb, r, g, b);
            image.at<cv::Vec3b>(i/pc_msg->width, i%pc_msg->width)[2] = r;
            image.at<cv::Vec3b>(i/pc_msg->width, i%pc_msg->width)[1] = g;
            image.at<cv::Vec3b>(i/pc_msg->width, i%pc_msg->width)[0] = b;
        } else {
            image.at<cv::Vec3b>(i/pc_msg->width, i%pc_msg->width)[2] = 0;
            image.at<cv::Vec3b>(i/pc_msg->width, i%pc_msg->width)[1] = 0;
            image.at<cv::Vec3b>(i/pc_msg->width, i%pc_msg->width)[0] = 0;
        }
    }
    cv::imshow("Capture PC", image);
    cv::waitKey(3);
    cur_pc = pc_msg;
}

void mouseClickCallback(int event, int x, int y, int flags, void* param) 
{
    if(event == CV_EVENT_LBUTTONDOWN) {

        savePCBag(orig_filename, cur_pc);
        ROS_INFO("Saving full pointcloud to %s", orig_filename);

        PCRGB::Ptr skin_pc(new PCRGB());

#ifdef USE_COLOR_MODEL
        extractFaceColorModel(cur_pc, skin_pc, x, y);
#else
        extractFace(cur_pc, skin_pc, x, y);
#endif
        skin_pc->header.frame_id = cur_pc->header.frame_id;
        savePCBag(face_filename, skin_pc);
        ROS_INFO("Saving face pointcloud to %s", face_filename);

        ros::shutdown();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clickable_display");

    if (argc != 3) 
    {
        ROS_ERROR("Usage: rosrun hrl_head_registration <file_to_save_full_pointcloud.bag> <file_to_save_face_pointcloud.bag> input_pc:=<pointcloud_topic>");
         return 0;
    }

    orig_filename = argv[1];
    face_filename = argv[2];

    ros::NodeHandle nh;
    cv::namedWindow("Capture PC", 1);
    cv::setMouseCallback("Capture PC", &mouseClickCallback);
    pc_sub = nh.subscribe("/input_pc", 1, &subPCCallback);

    ros::spin();
}
