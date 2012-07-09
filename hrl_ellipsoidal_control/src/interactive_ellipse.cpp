#include <ros/ros.h>
#include <interactive_markers/interactive_marker_server.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Pose.h>
#include <hrl_ellipsoidal_control/EllipsoidParams.h>
#include <hrl_ellipsoidal_control/LoadEllipsoidParams.h>
//#include <hrl_ellipsoidal_control/utils.h>

class InteractiveEllipse 
{
private:
    interactive_markers::InteractiveMarkerServer im_server_;
    tf::TransformBroadcaster tf_broad_;
    ros::Publisher params_pub;
    ros::Subscriber params_cmd;
    std::string parent_frame_, child_frame_;
    double rate_;
    ros::Timer tf_timer_;
    geometry_msgs::Pose marker_pose_;
    double z_axis_, y_axis_, old_z_axis_, old_y_axis_;
    geometry_msgs::Transform old_marker_tf_;
    geometry_msgs::TransformStamped tf_msg_;
    hrl_ellipsoidal_control::EllipsoidParams cur_e_params_;
    ros::ServiceServer load_param_srv_;
public:
    InteractiveEllipse(const std::string& parent_frame, const std::string& child_frame, double rate = 100);
    void processTFControl(const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback);
    void processEllipseControlY(const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback);
    void processEllipseControlZ(const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback);
    void addTFMarker(const geometry_msgs::Transform& mkr_tf);
    void addEllipseMarker();
    void publishTF(const ros::TimerEvent& event);
    void loadEllipsoidParams(const hrl_ellipsoidal_control::EllipsoidParams& e_params);
    void bagTF(const string& bag_name);
    bool loadParamCB(hrl_ellipsoidal_control::LoadEllipsoidParams::Request& req,
                     hrl_ellipsoidal_control::LoadEllipsoidParams::Response& resp);
};

InteractiveEllipse::InteractiveEllipse(const std::string& parent_frame, 
                                       const std::string& child_frame, double rate) :
    im_server_("transform_marker"),
    parent_frame_(parent_frame),
    child_frame_(child_frame),
    rate_(rate), z_axis_(0.0), y_axis_(0.0) 
    {
    ros::NodeHandle nh;
    marker_pose_.orientation.w = 1;
    params_pub = nh.advertise<hrl_ellipsoidal_control::EllipsoidParams>("/ellipsoid_params", 1);
    params_cmd = nh.subscribe("/ell_params_cmd", 1, &InteractiveEllipse::loadEllipsoidParams, this);
    load_param_srv_ = nh.advertiseService("/load_ellipsoid_params", 
                                          &InteractiveEllipse::loadParamCB, this);
}

void InteractiveEllipse::addTFMarker(const geometry_msgs::Transform& mkr_tf) 
{
    ros::NodeHandle nh;
    visualization_msgs::InteractiveMarker tf_marker;
    tf_marker.header.frame_id = parent_frame_;
    tf_marker.pose.position.x = mkr_tf.translation.x;
    tf_marker.pose.position.y = mkr_tf.translation.y;
    tf_marker.pose.position.z = mkr_tf.translation.z;
    tf_marker.pose.orientation.x = mkr_tf.rotation.x;
    tf_marker.pose.orientation.y = mkr_tf.rotation.y;
    tf_marker.pose.orientation.z = mkr_tf.rotation.z;
    tf_marker.pose.orientation.w = mkr_tf.rotation.w;
    tf_marker.name = "tf_marker";
    tf_marker.scale = 0.2;
    visualization_msgs::InteractiveMarkerControl tf_control;
    tf_control.orientation_mode = visualization_msgs::InteractiveMarkerControl::INHERIT;
    // x
    tf_control.orientation.x = 1; tf_control.orientation.y = 0;
    tf_control.orientation.z = 0; tf_control.orientation.w = 1;
    tf_control.name = "rotate_x";
    tf_control.interaction_mode = visualization_msgs::InteractiveMarkerControl::ROTATE_AXIS;
    tf_marker.controls.push_back(tf_control);
    tf_control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;
    tf_marker.controls.push_back(tf_control);
    // y
    tf_control.orientation.x = 0; tf_control.orientation.y = 1;
    tf_control.orientation.z = 0; tf_control.orientation.w = 1;
    tf_control.name = "rotate_y";
    tf_control.interaction_mode = visualization_msgs::InteractiveMarkerControl::ROTATE_AXIS;
    tf_marker.controls.push_back(tf_control);
    tf_control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;
    tf_marker.controls.push_back(tf_control);
    // z
    tf_control.orientation.x = 0; tf_control.orientation.y = 0;
    tf_control.orientation.z = 1; tf_control.orientation.w = 1;
    tf_control.name = "rotate_z";
    tf_control.interaction_mode = visualization_msgs::InteractiveMarkerControl::ROTATE_AXIS;
    tf_marker.controls.push_back(tf_control);
    tf_control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;
    tf_marker.controls.push_back(tf_control);
    im_server_.insert(tf_marker, boost::bind(&InteractiveEllipse::processTFControl, this, _1));
    im_server_.applyChanges();
    tf_timer_ = nh.createTimer(ros::Duration(1.0 / rate_), &InteractiveEllipse::publishTF, this);
}

void InteractiveEllipse::processTFControl(
        const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback) 
        {
    ROS_INFO_STREAM(tf_msg_.transform.translation.x << " " << 
                    tf_msg_.transform.translation.y << " " <<
                    tf_msg_.transform.translation.z << " " <<
                    tf_msg_.transform.rotation.x << " " <<
                    tf_msg_.transform.rotation.y << " " <<
                    tf_msg_.transform.rotation.z << " " <<
                    tf_msg_.transform.rotation.w);
    marker_pose_ = feedback->pose;
}

void InteractiveEllipse::addEllipseMarker() 
{
    visualization_msgs::InteractiveMarker tf_marker;
    tf_marker.header.frame_id = child_frame_;
    tf_marker.name = "ellipse_marker_y";
    tf_marker.scale = 0.4;
    visualization_msgs::InteractiveMarkerControl tf_control;
    tf_control.orientation_mode = visualization_msgs::InteractiveMarkerControl::INHERIT;
    // y
    tf_control.orientation.x = 0; tf_control.orientation.y = 1;
    tf_control.orientation.z = 0; tf_control.orientation.w = 1;
    tf_control.name = "shift_y";
    tf_control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;
    tf_marker.controls.push_back(tf_control);
    im_server_.insert(tf_marker, boost::bind(&InteractiveEllipse::processEllipseControlY, this, _1));
    tf_marker.controls.clear();
    // z
    tf_marker.name = "ellipse_marker_z";
    tf_control.orientation.x = 0; tf_control.orientation.y = 0;
    tf_control.orientation.z = 1; tf_control.orientation.w = 1;
    tf_control.name = "shift_z";
    tf_control.interaction_mode = visualization_msgs::InteractiveMarkerControl::MOVE_AXIS;
    tf_marker.controls.push_back(tf_control);
    im_server_.insert(tf_marker, boost::bind(&InteractiveEllipse::processEllipseControlZ, this, _1));
    im_server_.applyChanges();
}

void InteractiveEllipse::processEllipseControlY(
        const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback) 
        {
    z_axis_ = feedback->pose.position.z;
}

void InteractiveEllipse::processEllipseControlZ(
        const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback) 
        {
    y_axis_ = feedback->pose.position.y;
}

void InteractiveEllipse::publishTF(const ros::TimerEvent& event) 
{
    tf_msg_.header.stamp = ros::Time::now();
    tf_msg_.header.frame_id = parent_frame_;
    tf_msg_.child_frame_id = child_frame_;
    tf_msg_.transform.translation.x = marker_pose_.position.x;
    tf_msg_.transform.translation.y = marker_pose_.position.y;
    tf_msg_.transform.translation.z = marker_pose_.position.z;
    tf_msg_.transform.rotation.x = marker_pose_.orientation.x;
    tf_msg_.transform.rotation.y = marker_pose_.orientation.y;
    tf_msg_.transform.rotation.z = marker_pose_.orientation.z;
    tf_msg_.transform.rotation.w = marker_pose_.orientation.w;
    tf::Transform cur_tf, old_tf;
    tf::transformMsgToTF(tf_msg_.transform, cur_tf);
    tf::transformMsgToTF(old_marker_tf_, old_tf);
    Eigen::Affine3d cur_tf_eig, old_tf_eig;
    tf::TransformTFToEigen(cur_tf, cur_tf_eig);
    tf::TransformTFToEigen(old_tf, old_tf_eig);
    cur_tf_eig = cur_tf_eig;
    tf::TransformEigenToTF(cur_tf_eig, cur_tf);
    tf::transformTFToMsg(cur_tf, tf_msg_.transform);
    tf_broad_.sendTransform(tf_msg_);
    hrl_ellipsoidal_control::EllipsoidParams e_params;
    e_params.e_frame = tf_msg_;
    e_params.height = y_axis_ + old_y_axis_;
    e_params.E = z_axis_ + old_z_axis_;
    params_pub.publish(e_params);
    cur_e_params_ = e_params;
}

void InteractiveEllipse::loadEllipsoidParams(const hrl_ellipsoidal_control::EllipsoidParams& e_params) 
{
    geometry_msgs::Pose pose, empty_pose;
    pose.position.x = e_params.e_frame.transform.translation.x;
    pose.position.y = e_params.e_frame.transform.translation.y;
    pose.position.z = e_params.e_frame.transform.translation.z;
    pose.orientation.x = e_params.e_frame.transform.rotation.x;
    pose.orientation.y = e_params.e_frame.transform.rotation.y;
    pose.orientation.z = e_params.e_frame.transform.rotation.z;
    pose.orientation.w = e_params.e_frame.transform.rotation.w;
    marker_pose_ = pose;
    empty_pose.orientation.w = 1;
    im_server_.setPose("tf_marker", pose);
    im_server_.setPose("ellipse_marker_y", empty_pose);
    im_server_.setPose("ellipse_marker_z", empty_pose);
    im_server_.applyChanges();

    old_y_axis_ = e_params.height;
    old_z_axis_ = e_params.E;
}

void InteractiveEllipse::bagTF(const string& bag_name) 
{
    rosbag::Bag bag;
    bag.open(bag_name, rosbag::bagmode::Write);
    bag.write("/ellipsoid_params", ros::Time::now(), cur_e_params_);
    bag.close();
}

bool InteractiveEllipse::loadParamCB(hrl_ellipsoidal_control::LoadEllipsoidParams::Request& req,
                                     hrl_ellipsoidal_control::LoadEllipsoidParams::Response& resp)
{
    loadEllipsoidParams(req.params);
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "interative_ellipse");
    if(argc < 3 || argc > 7) {
        printf("Usage: interative_ellipse parent_frame child_frame rate [inital_params] [remove_ells]\n");
        return 1;
    }
    if(argc >= 4) {
        InteractiveEllipse itf(argv[1], argv[2], atof(argv[3]));
        geometry_msgs::Transform mkr_tf;
        mkr_tf.rotation.w = 1;
        itf.addTFMarker(mkr_tf);
        if(!(argc >= 7 && atoi(argv[6])))
            itf.addEllipseMarker();
        if(argc >= 6) {
            // load params
            std::vector<hrl_ellipsoidal_control::EllipsoidParams::Ptr> params;
            readBagTopic<hrl_ellipsoidal_control::EllipsoidParams>(argv[5], params, "/ellipsoid_params");
            itf.loadEllipsoidParams(*params[0]);
        }
        ros::spin();
        if(argc >= 5)
            itf.bagTF(argv[4]);
    } 

    return 0;
}
