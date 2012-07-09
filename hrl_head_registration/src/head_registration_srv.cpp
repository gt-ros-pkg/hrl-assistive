

#include <hrl_head_registration/head_registration.h>
#include <hrl_head_registration/HeadRegistration.h>
#include <geometry_msgs/TransformStamped.h>

typedef hrl_head_registration::HeadRegistration HeadRegSrv;

PCRGB::Ptr cur_pc, template_pc;

ros::Subscriber pc_sub;
ros::ServiceServer reg_srv;
ros::Publisher aligned_pub; 
        
void subPCCallback(const PCRGB::Ptr& cur_pc_);
bool regCallback(HeadRegSrv::Request& req, HeadRegSrv::Response& resp);

void subPCCallback(const PCRGB::Ptr& cur_pc_)
{
    cur_pc = cur_pc_;
}

void readTFBag(const string& filename, geometry_msgs::TransformStamped::Ptr& tf_msg) 
{
    rosbag::Bag bag;
    bag.open(filename, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery("/itf_transform"));
    BOOST_FOREACH(rosbag::MessageInstance const msg, view) {
        tf_msg = msg.instantiate< geometry_msgs::TransformStamped >();
        break;
    }
    bag.close();
}

bool regCallback(HeadRegSrv::Request& req, HeadRegSrv::Response& resp)
{
    if(!cur_pc) {
        ROS_ERROR("No point cloud received.");
        return false;
    }
    Eigen::Affine3d tf_mat;
    if(!findFaceRegistration(template_pc, cur_pc, req.u, req.v, tf_mat)) {
        ROS_ERROR("Bad initialization pixel.");
        return false;
    }
    resp.tf_reg.header.frame_id = cur_pc->header.frame_id;
    resp.tf_reg.header.stamp = ros::Time::now();
    tf::poseEigenToMsg(tf_mat, resp.tf_reg.pose);

    // TODO SHAVING_SIDE
    string shaving_side;
    ros::param::param<string>("/shaving_side", shaving_side, "r");
    double new_hue = (shaving_side[0] == 'r') ? 120 : 240;
    // TODO END SHAVING_SIDE

    PCRGB::Ptr aligned_pc(new PCRGB());
    transformPC(*template_pc, *aligned_pc, tf_mat.inverse());
    double h, s, l;
    for(uint32_t i=0;i<aligned_pc->size();i++) {
        extractHSL(aligned_pc->points[i].rgb, h, s, l);
        writeHSL(new_hue, 50, l, aligned_pc->points[i].rgb);
    }
    aligned_pc->header.frame_id = "/openni_rgb_optical_frame";
    aligned_pub.publish(aligned_pc);

#if 0
    vector<PCRGB::Ptr> pcs;
    vector<string> pc_topics;

    PCRGB::Ptr skin_pc(new PCRGB());
    extractFaceColorModel(cur_pc, skin_pc, req.u, req.v);
    skin_pc->header.frame_id = "/openni_rgb_optical_frame";
    pcs.push_back(skin_pc);
    pc_topics.push_back("/target_pc");

    PCRGB::Ptr tf_pc(new PCRGB());
    transformPC(*template_pc, *tf_pc, tf_mat);
    tf_pc->header.frame_id = "/openni_rgb_optical_frame";
    pcs.push_back(tf_pc);
    pc_topics.push_back("/template_pc");

    pubLoop(pcs, pc_topics, 5);
#endif
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "head_registration");
    ros::NodeHandle nh;

    readPCBag(argv[1], template_pc);

    aligned_pub = nh.advertise<PCRGB>("/head_registration/aligned_pc", 1, true);
    pc_sub = nh.subscribe("/kinect_head/rgb/points", 1, &subPCCallback);
    reg_srv = nh.advertiseService("/head_registration", &regCallback);
    ros::spin();

    return 0;
}
