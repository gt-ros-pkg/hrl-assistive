//
///////////////////////////////////////////////////////////////////////////
//
//manipulation_task_image_overlayer
//
//The following code takes in the image from Kinect2, and data from TF, and action.
//It overlay TF information and action message on the image for the user feedback.
//
//Currently this code provides feedback for : 
// * The area of consideration near tool frame
// * Current action status of the program
// * Anomaly Detection
//
// 
// 
// 
//
//////////////////////////////////////////////////////////////////////////
//
// Created by You Keun Kim
//
/////////////////////////////////////////////////////////////////////////


#include <ros/ros.h>

#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv/cv.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>

#include "std_msgs/String.h"

using namespace cv;

ros::Subscriber action_sub, actionLoc_sub, actionLoc2_sub, actionLoc3_sub, abnorm_sub;
bool actionLocExist, actionLoc2Exist, actionLoc3Exist, abnormForce, abnormSound, abnormTouch;
geometry_msgs::PoseStamped actionLoc, actionLoc2, actionLoc3;
std::string actionArr[10];

image_transport::CameraSubscriber camera_sub;
image_transport::Publisher overlay_pub;
image_geometry::PinholeCameraModel cam_model;
boost::shared_ptr<tf::TransformListener> tf_list;

void subActionCallback(const std_msgs::String::ConstPtr& action_msg){

    std::string container = action_msg->data,c_str();
    actionArr[0] = container;

}

void subAbnormCallback(const std_msgs::String::ConstPtr& action_msg){

    std::string container = action_msg->data,c_str();
   
    if (container.find("Sound")!= std::string::npos) {
    abnormSound = true;
    }
    else {
    abnormSound = false;
    }

    if (container.find("Touch")!= std::string::npos) {
    abnormTouch = true;
    }
    else {
    abnormTouch = false;
    }

    if (container.find("Force")!= std::string::npos) {
    abnormForce = true;
    }
    else {
    abnormForce = false;
    }

}

void subLocCallback(const geometry_msgs::PoseStamped& Loc_msg){
	actionLoc.header.frame_id = Loc_msg.header.frame_id;
        actionLoc.header.stamp = ros::Time::now();
        actionLoc.pose.position.x = Loc_msg.pose.position.x;
        actionLoc.pose.position.y = Loc_msg.pose.position.y;
        actionLoc.pose.position.z = Loc_msg.pose.position.z;
        actionLoc.pose.orientation.x = 0;
        actionLoc.pose.orientation.y = 0;
        actionLoc.pose.orientation.z = 0;
        actionLoc.pose.orientation.w = 1;
	actionLocExist = true;
}

void subLoc2Callback(const geometry_msgs::PoseStamped& Loc_msg2){
	actionLoc2.header.frame_id = Loc_msg2.header.frame_id;
        actionLoc2.header.stamp = ros::Time::now();
        actionLoc2.pose.position.x = Loc_msg2.pose.position.x;
        actionLoc2.pose.position.y = Loc_msg2.pose.position.y;
        actionLoc2.pose.position.z = Loc_msg2.pose.position.z;
        actionLoc2.pose.orientation.x = 0;
        actionLoc2.pose.orientation.y = 0;
        actionLoc2.pose.orientation.z = 0;
        actionLoc2.pose.orientation.w = 1;
	actionLoc2Exist = true;
}

void subLoc3Callback(const geometry_msgs::PoseStamped& Loc_msg3){
	actionLoc3.header.frame_id = Loc_msg3.header.frame_id;
        actionLoc3.header.stamp = ros::Time::now();
        actionLoc3.pose.position.x = Loc_msg3.pose.position.x;
        actionLoc3.pose.position.y = Loc_msg3.pose.position.y;
        actionLoc3.pose.position.z = Loc_msg3.pose.position.z;
        actionLoc3.pose.orientation.x = 0;
        actionLoc3.pose.orientation.y = 0;
        actionLoc3.pose.orientation.z = 0;
        actionLoc3.pose.orientation.w = 1;
	actionLoc3Exist = true;
}


void doOverlay(const sensor_msgs::ImageConstPtr& img_msg,
               const sensor_msgs::CameraInfoConstPtr& info_msg) {

    // convert camera image into opencv
    cam_model.fromCameraInfo(info_msg);
    cv_bridge::CvImagePtr cv_img = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::RGB8);

    // Preparing the Status and abnormality box.

    cv::Mat roi = cv_img->image(cv::Rect(20, 10, 920, 100));
    cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(10, 10, 10)); 
    double alpha = 0.5;
    cv::addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi); 
    
    //TODO: Change the text as aparamer? as actionArr[0]?
    if(abnormSound || abnormTouch || abnormForce) {
        cv::putText(cv_img->image, "Halting", cvPoint(70,70), FONT_HERSHEY_PLAIN, 2, cvScalar(250,0,0), 2, CV_AA);
    }
    else if(actionArr[0].compare("Scooping")==0) {
        cv::putText(cv_img->image, "Scooping", cvPoint(50,70), FONT_HERSHEY_PLAIN, 2, cvScalar(250,250,250), 2, CV_AA);
    }
    else if(actionArr[0].compare("Feeding")==0) {
        cv::putText(cv_img->image, "Feeding", cvPoint(70,70), FONT_HERSHEY_PLAIN, 2, cvScalar(250,250,250), 2, CV_AA);
    }
    else {
        cv::putText(cv_img->image, "Repositioning", cvPoint(20,70), FONT_HERSHEY_PLAIN, 2, cvScalar(250,250,250), 2, CV_AA);
    }

    ///TODO: change cvScalar Only?

if(abnormSound) {
        cv::putText(cv_img->image, "Sound", cvPoint(255,40), FONT_HERSHEY_PLAIN, 1.5, cvScalar(250,0,0), 2, CV_AA);
        cv::rectangle(cv_img->image, cvPoint (245, 15), cvPoint(345, 50), cvScalar(250, 0, 0), 1, 8, 0);
    }
    else {
        cv::putText(cv_img->image, "Sound", cvPoint(255,40), FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,250,0), 2, CV_AA);
        cv::rectangle(cv_img->image, cvPoint (245, 15), cvPoint(345, 50), cvScalar(0, 250, 0), 1, 8, 0);
    }

    if(abnormForce) {
        cv::putText(cv_img->image, "Force", cvPoint(375,40), FONT_HERSHEY_PLAIN, 1.5, cvScalar(250,0,0), 2, CV_AA);
        cv::rectangle(cv_img->image, cvPoint (365, 15), cvPoint(465, 50), cvScalar(250, 0, 0), 1, 8, 0);
    }
    else {
        cv::putText(cv_img->image, "Force", cvPoint(375,40), FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,250,0), 2, CV_AA);
        cv::rectangle(cv_img->image, cvPoint (365, 15), cvPoint(465, 50), cvScalar(0, 250, 0), 1, 8, 0);
    }

    if(abnormTouch) {
        cv::putText(cv_img->image, "Touch", cvPoint(495,40), FONT_HERSHEY_PLAIN, 1.5, cvScalar(250,0,0), 2, CV_AA);
        cv::rectangle(cv_img->image, cvPoint (485, 15), cvPoint(585, 50), cvScalar(250, 0, 0), 1, 8, 0);
    }
    else {
        cv::putText(cv_img->image, "Touch", cvPoint(495,40), FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,250,0), 2, CV_AA);
        cv::rectangle(cv_img->image, cvPoint (485, 15), cvPoint(585, 50), cvScalar(0, 250, 0), 1, 8, 0);
    }

    // TODO: This are empty boxes.

    cv::rectangle(cv_img->image, cvPoint (245, 60), cvPoint(345, 95), cvScalar(0, 250, 0), 1, 8, 0);

    cv::rectangle(cv_img->image, cvPoint (365, 60), cvPoint(465, 95), cvScalar(0, 250, 0), 1, 8, 0);

    cv::rectangle(cv_img->image, cvPoint (485, 60), cvPoint(585, 95), cvScalar(0, 250, 0), 1, 8, 0);




    // Circling Starts here.

    cv::Mat copyImage = cv_img->image.clone();
    double alpha2 = 0.3;

    if(actionLocExist) {
     //TODO: Should be /torso_lift_link
        if(!tf_list->waitForTransform(img_msg->header.frame_id, "/torso_lift_link",
                                     ros::Time::now(), ros::Duration(3)))
            return;
	    tf_list->transformPose(img_msg->header.frame_id, actionLoc, actionLoc);
	cv::Point3d proj_pt_cv2(actionLoc.pose.position.x, actionLoc.pose.position.y, actionLoc.pose.position.z);
        cv::Point pt2d2 = cam_model.project3dToPixel(proj_pt_cv2);

        cv::circle(copyImage, pt2d2, 50, cvScalar(0,0,250), -1, 8, 0);
        cv::putText(copyImage, "Action Area", cvPoint(pt2d2.x-40,pt2d2.y+120), FONT_HERSHEY_PLAIN, 1, cvScalar(0,0,250), 1, CV_AA);
    }
    
    if(actionLoc2Exist) {
     //std::cout <<"Worked!" << '\n';
     //TODO: Should be /torso_lift_link
        if(!tf_list->waitForTransform(img_msg->header.frame_id, "/torso_lift_link",
                                     ros::Time::now(), ros::Duration(3)))
            return;
	    tf_list->transformPose(img_msg->header.frame_id, actionLoc2, actionLoc2);
	cv::Point3d proj_pt_cv3(actionLoc2.pose.position.x, actionLoc2.pose.position.y, actionLoc2.pose.position.z);
        cv::Point pt2d2_3 = cam_model.project3dToPixel(proj_pt_cv3);

        cv::circle(copyImage, pt2d2_3, 50, cvScalar(0,250, 0), -1, 8, 0);
    }

    if(actionLoc3Exist) {
     //TODO: Should be /torso_lift_link
        if(!tf_list->waitForTransform(img_msg->header.frame_id, "/torso_lift_link",
                                     ros::Time::now(), ros::Duration(3)))
            return;
	    tf_list->transformPose(img_msg->header.frame_id, actionLoc3, actionLoc3);
	cv::Point3d proj_pt_cv4(actionLoc3.pose.position.x, actionLoc3.pose.position.y, actionLoc3.pose.position.z);
        cv::Point pt2d2_4 = cam_model.project3dToPixel(proj_pt_cv4);

        cv::circle(copyImage, pt2d2_4, 50, cvScalar(250,0,0), -1, 8, 0);
    }
   
    cv::addWeighted(copyImage, alpha2, cv_img->image, 1.0 - alpha2 , 0.0, cv_img->image);

    //std::cout <<"Worked!" << '\n';
    overlay_pub.publish(cv_img->toImageMsg());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_overlayer");
    ros::NodeHandle nh;
    image_transport::ImageTransport img_trans(nh);
    tf_list = boost::shared_ptr<tf::TransformListener>(new tf::TransformListener());

    abnorm_sub = nh.subscribe("/manipulation_task_action/abnormality", 1, &subAbnormCallback);
    action_sub = nh.subscribe("/manipulation_task_action", 1, &subActionCallback);
    actionLoc_sub = nh.subscribe("/manipulation_task_action_location", 1, &subLocCallback);
    actionLoc2_sub = nh.subscribe("/manipulation_task_action_location_2", 1, &subLoc2Callback);
    actionLoc3_sub = nh.subscribe("/manipulation_task_action_location_3", 1, &subLoc3Callback);

    sleep(5);
//    cout << "If the Imager_transport warning shows, change the buffer size in the //head_bowl_confirm program to 3." << endl;
//    camera_sub = img_trans.subscribeCamera("/kinect_camera", 10, 
//                                           &doOverlay);

    camera_sub = img_trans.subscribeCamera("/head_mount_kinect/rgb_lowres/image", 3, 
                                           &doOverlay);


    overlay_pub = img_trans.advertise("/manipulation_task/overlay", 1);

    ros::Rate r(5);
    while(ros::ok()) {
        cv::Mat image(540, 960, CV_8UC3);
        ros::spinOnce();
        r.sleep();
    }

    return 0;
}


