#include <hrl_head_registration/pcl_basic.h>
#include <limits>

void transformPC(const PCRGB &in_pc, PCRGB &out_pc, 
                 const Eigen::Affine3d& transform) 
                 {
    MatrixXd pt_mat = MatrixXd::Constant(4, in_pc.points.size(), 1.0);
    uint32_t i = 0;
    vector<bool> nans_list(in_pc.points.size());
    BOOST_FOREACH(PRGB const pt, in_pc.points) {
        if(pt.x == pt.x && pt.y == pt.y && pt.z == pt.z) {
            pt_mat(0, i) = pt.x; pt_mat(1, i) = pt.y; pt_mat(2, i) = pt.z; 
        }
        else
            nans_list[i] = true;
        i++;
    }
    MatrixXd trans_pt_mat = transform.matrix() * pt_mat;
    for(i=0;i<in_pc.points.size();i++) {
        PRGB pt;
        if(!nans_list[i]) {
            pt.x = trans_pt_mat(0, i); pt.y = trans_pt_mat(1, i); pt.z = trans_pt_mat(2, i); 
        } else {
            pt.x = numeric_limits<float>::quiet_NaN();
            pt.y = numeric_limits<float>::quiet_NaN();
            pt.z = numeric_limits<float>::quiet_NaN();
        }
        pt.rgb = in_pc.points[i].rgb;
        out_pc.points.push_back(pt);
    }
}

void extractIndices(const PCRGB::Ptr& in_pc, pcl::IndicesPtr& inds, PCRGB::Ptr& out_pc, bool is_negative) 
{
    vector<bool> inds_bool(in_pc->points.size(), false);
    for(size_t i=0;i<inds->size();i++) {
        inds_bool[inds->operator[](i)] = true;
        if(inds->operator[](i) >= in_pc->points.size()) 
            ROS_ERROR("Bad index: %d", inds->operator[](i));
    }
    for(size_t i=0;i<in_pc->points.size();i++) {
        if(inds_bool[i] && is_negative || !inds_bool[i] && !is_negative)
            continue;
        const PRGB pt = in_pc->points[i];
        PRGB npt;
        npt.x = pt.x; npt.y = pt.y; npt.z = pt.z; npt.rgb = pt.rgb;
        out_pc->points.push_back(npt);
    }
}

void pubLoop(PCRGB::Ptr& pc, const std::string& topic, double rate, int number_runs) 
{
    ros::NodeHandle nh;
    ros::Publisher pub_pc = nh.advertise<PCRGB>(topic, 1);
    ros::Rate r(rate);
    while(ros::ok() && number_runs-- > 0) {
        //Hack for Hydro to adapt time formats:
        ros::Time time_stamp = ros::Time::now();
        pc->header.stamp = time_stamp.toNSec()/1e3;
        pub_pc.publish(pc);
        r.sleep();
    }
}

void pubLoop(vector<PCRGB::Ptr> &pcs, const vector<std::string>& topics, double rate, int number_runs) 
{
    ros::NodeHandle nh;
    vector<ros::Publisher> pub_pcs;
    for(size_t i=0;i<topics.size();i++) 
        pub_pcs.push_back(nh.advertise<PCRGB>(topics[i], 1));
    ros::Rate r(rate);
    while(ros::ok() && number_runs-- > 0) {
        for(size_t i=0;i<pcs.size();i++) {
            //Hack for Hydro to adapt time formats:
            ros::Time time_stamp = ros::Time::now();
            pcs[i]->header.stamp = time_stamp.toNSec()/1e3;
            pub_pcs[i].publish(*pcs[i]);
        }
        r.sleep();
    }
}

void boxFilter(const PCRGB &in_pc, PCRGB &out_pc,
               double min_x, double max_x, double min_y, double max_y, double min_z, double max_z) 
               {
    pcl::ConditionAnd<PRGB>::Ptr near_cond(new pcl::ConditionAnd<PRGB>());
    PCRGB::Ptr near_pts(new PCRGB());
    pcl::ConditionalRemoval<PRGB> near_extract;
    near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                             "x", pcl::ComparisonOps::GT, min_x)));
    near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                             "x", pcl::ComparisonOps::LT, max_x)));
    near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                             "y", pcl::ComparisonOps::GT, min_y)));
    near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                             "y", pcl::ComparisonOps::LT, max_y)));
    near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                             "z", pcl::ComparisonOps::GT, min_z)));
    near_cond->addComparison(pcl::FieldComparison<PRGB>::Ptr(new pcl::FieldComparison<PRGB>(
                             "z", pcl::ComparisonOps::LT, max_z)));
    near_extract.setCondition(near_cond);
    near_extract.setKeepOrganized(true);
    near_extract.setInputCloud(in_pc.makeShared());
    near_extract.filter(out_pc);
}

void readPCBag(const string& filename, PCRGB::Ptr& pc) 
{
    rosbag::Bag bag;
    bag.open(filename, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery("/kinect_head/rgb/points"));
    BOOST_FOREACH(rosbag::MessageInstance const msg, view) {
        pc = msg.instantiate< PCRGB >();
        break;
    }
    bag.close();
}

void savePCBag(const string& filename, PCRGB::Ptr& pc)
{
    rosbag::Bag bag;
    bag.open(filename, rosbag::bagmode::Write);
    bag.write("/kinect_head/rgb/points", ros::Time::now(), pc);
    bag.close();
}
