#include <hrl_head_registration/pcl_basic.h>

int main(int argc, char **argv)
{
    if (argc != 3) {
        ROS_ERROR("Usage: rosrun hrl_head_registration pub_pc <path/to/pointcloud/in/bagfile.bag> <topic_name>");
        return 0;
    }
    
    ros::init(argc, argv, "pub_pc");
    ros::NodeHandle nh;

    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>(argv[2], 100, true);
    PCRGB::Ptr input_pc;
    readPCBag(argv[1], input_pc);

    ros::Duration(1.0).sleep(); //Give extra time for publisher to be spun-up
    ros::Time time_stamp = ros::Time().now();
    input_pc->header.stamp = time_stamp.toNSec()/1e3;
    pc_pub.publish(input_pc);
    ROS_INFO("[pub_pc] Publishing pointcloud from %s to topic: %s", argv[1], argv[2]);
    ros::spin();
    return 0;
}
