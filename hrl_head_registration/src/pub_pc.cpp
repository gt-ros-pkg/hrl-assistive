
#include <hrl_head_registration/pcl_basic.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pub_pc");
    ros::NodeHandle nh;
    PCRGB::Ptr input_pc;
    readPCBag(argv[1], input_pc);

    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>(argv[2], 100);
    while(ros::ok()) {
        input_pc->header.stamp = ros::Time().now();
        pc_pub.publish(input_pc);
        ros::Duration(0.1).sleep();
    }
    return 0;
}
