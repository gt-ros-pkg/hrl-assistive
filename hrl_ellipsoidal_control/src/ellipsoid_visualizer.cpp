#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <hrl_ellipsoidal_control/EllipsoidParams.h>
//#include <hrl_ellipsoidal_control/utils.h>
#include <hrl_ellipsoidal_control/ellipsoid_space.h>

typedef pcl::PointXYZRGB PRGB;
typedef pcl::PointCloud<PRGB> PCRGB;

void sampleEllipse(double A, double B, double height, PCRGB::Ptr& pc);
void publishEllipsoid(hrl_ellipsoidal_control::EllipsoidParams::ConstPtr e_params);

ros::Publisher pub_pc;

void sampleEllipse(double A, double B, double height, PCRGB::Ptr& pc) 
{
    bool is_prolate;
    ros::param::param<bool>("~is_prolate", is_prolate, true);
    Ellipsoid e(A, B, is_prolate);
    double lat, lon;
    int numlat = 8, numlon = 16;
    lat = (is_prolate) ? 0 : -PI/2;
    for(int i=0;i<numlat;i++) {
        lon = (is_prolate) ? 0 : -PI;
        lat += PI / numlat;
        for(int j=0;j<600;j++) {
            lon += 2 * PI / 600;
            PRGB pt;
            ((uint32_t*) &pt.rgb)[0] = 0xffffffff;
            double x, y, z;
            e.ellipsoidalToCart(lat, lon, height, x, y, z);
            pt.x = x; pt.y = y; pt.z = z;
            pc->points.push_back(pt);
        }
    }
    lon = (is_prolate) ? 0 : -PI;
    for(int i=0;i<numlon;i++) {
        lat = (is_prolate) ? 0 : -PI/2;
        lon += 2 * PI / numlon;
        for(int j=0;j<600;j++) {
            lat += PI / 600;
            PRGB pt;
            ((uint32_t*) &pt.rgb)[0] = 0xffffffff;
            double x, y, z;
            e.ellipsoidalToCart(lat, lon, height, x, y, z);
            pt.x = x; pt.y = y; pt.z = z;
            pc->points.push_back(pt);
        }
    }
}

void publishEllipsoid(hrl_ellipsoidal_control::EllipsoidParams::ConstPtr e_params) 
{
    PCRGB::Ptr pc(new PCRGB());
    double A = 1;
    double E = e_params->E;
    double B = sqrt(1 - SQ(E));
    sampleEllipse(A, B, e_params->height, pc);
    pc->header.frame_id = "/ellipse_frame";
    pc->header.stamp = ros::Time::now();
    pub_pc.publish(pc);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ellipsoid_visualizer");
    ros::NodeHandle nh;

    pub_pc = nh.advertise<sensor_msgs::PointCloud2>(argv[1], 1);
    ros::Subscriber sub_e_params = nh.subscribe("/ellipsoid_params", 1, &publishEllipsoid);

    ros::spin();
    return 0;
}
