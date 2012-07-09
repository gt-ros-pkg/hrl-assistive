
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
    cv::Mat image(480, 640, CV_8UC3);
            
    uint8_t r, g, b;
    for(uint32_t i=0;i<pc_msg->size();i++) {
        if(PT_IS_NOT_NAN(pc_msg, i)) {
            extractRGB(pc_msg->points[i].rgb, r, g, b);
            image.at<cv::Vec3b>(i/640, i%640)[2] = r;
            image.at<cv::Vec3b>(i/640, i%640)[1] = g;
            image.at<cv::Vec3b>(i/640, i%640)[0] = b;
        }
    }
    cv::imshow("Capture PC", image);
    cv::waitKey(3);
    cur_pc = pc_msg;
}

void mouseClickCallback(int event, int x, int y, int flags, void* param) 
{
    if(event == CV_EVENT_LBUTTONDOWN) {
        /*
        FILE* click_file = fopen(click_filename, "w");
        fprintf(click_file, "%d,%d\n", x, y);
        fclose(click_file);
        */

        savePCBag(orig_filename, cur_pc);

        PCRGB::Ptr skin_pc(new PCRGB());

#ifdef USE_COLOR_MODEL
        extractFaceColorModel(cur_pc, skin_pc, x, y);
#else
        extractFace(cur_pc, skin_pc, x, y);
#endif
        skin_pc->header.frame_id = "/openni_rgb_optical_frame";
        savePCBag(face_filename, skin_pc);

        ros::shutdown();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clickable_display");

    //click_filename = string(argv[1]) + ".click";
    orig_filename = argv[1];
    face_filename = argv[2];

    ros::NodeHandle nh;
    cv::namedWindow("Capture PC", 1);
    cv::setMouseCallback("Capture PC", &mouseClickCallback);
    pc_sub = nh.subscribe("/kinect_head/rgb/points", 1, &subPCCallback);

    ros::spin();
}
