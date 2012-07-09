#include <ros/ros.h>

#include <geometry_msgs/PointStamped.h>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>

#include <hrl_head_registration/head_registration.h>

class ClickableDisplay {
    public:
        ros::NodeHandle nh, nh_priv;
        ros::Publisher l_click_pub, r_click_pub;
        std::string img_frame;
        bool have_img;

        ClickableDisplay();
        ~ClickableDisplay() {}
        void onInit();
        //void imgCallback(const sensor_msgs::ImageConstPtr& img_msg);
        void showImg(cv::Mat& img);
        static void mouseClickCallback(int event, int x, int y, int flags, void* data);

};

ClickableDisplay::ClickableDisplay() : nh_priv("~"), 
                                       have_img(false) {
}

void ClickableDisplay::onInit() {
    cv::namedWindow("Clickable World", 1);
    cv::setMouseCallback("Clickable World", &ClickableDisplay::mouseClickCallback, this);
    ROS_INFO("[clickable_display] ClickableDisplay loaded.");
    ros::Duration(1).sleep();
}

//void ClickableDisplay::imgCallback(const sensor_msgs::ImageConstPtr& img_msg) {
//}

void ClickableDisplay::showImg(cv::Mat& img) {
    ros::Rate r(30);
    while(ros::ok()) {
        ros::spinOnce();
        cv::imshow("Clickable World", img);
        cv::waitKey(3);
        r.sleep();
    }
}

char* file_out;

void ClickableDisplay::mouseClickCallback(int event, int x, int y, int flags, void* param) {
    ClickableDisplay* this_ = reinterpret_cast<ClickableDisplay*>(param);
    if(event == CV_EVENT_LBUTTONDOWN) {
        FILE* file = fopen(file_out, "w");
        fprintf(file, "%d,%d\n", x, y);
        fclose(file);
        ros::shutdown();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clickable_display");
    ClickableDisplay cd;
    cv::Mat image(480, 640, CV_8UC3);

    PCRGB::Ptr input_pc;
    readPCBag(argv[1], input_pc);
    file_out = argv[2];
    uint8_t r, g, b;
    for(uint32_t i=0;i<input_pc->size();i++) {
        if(PT_IS_NOT_NAN(input_pc, i)) {
            extractRGB(input_pc->points[i].rgb, r, g, b);
            image.at<cv::Vec3b>(i/640, i%640)[2] = r;
            image.at<cv::Vec3b>(i/640, i%640)[1] = g;
            image.at<cv::Vec3b>(i/640, i%640)[0] = b;
        }
    }

    cd.onInit();
    cd.showImg(image);
    ros::spin();
    return 0;
}
