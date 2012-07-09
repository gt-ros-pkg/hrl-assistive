
void extractSkinPC(PCRGB::Ptr& pc) 
{
    uint8_t r, g, b; 
    double skin_like, h, s, l;
    for(uint32_t i=0;i<pc->size();i++) {
        if(pc->points[i].x == pc->points[i].x && 
           pc->points[i].y == pc->points[i].y && 
           pc->points[i].z == pc->points[i].z) {
            extractRGB(pc->points[i].rgb, r, g, b);
            skin_like = skin_likelihood(r, g, b);
            extractHSL(pc->points[i].rgb, h, s, l);
            writeHSL(0, min(skin_like*30, 100.0), l, pc->points[i].rgb);
        }
    }
}

#if 0
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/pose", 100);
    geometry_msgs::PoseStamped ps;
    ps.pose.position.x = input_pc->points[closest_ind].x; 
    ps.pose.position.y = input_pc->points[closest_ind].y; 
    ps.pose.position.z = input_pc->points[closest_ind].z; 
    ps.pose.orientation.w = 1.0;
    ps.header.frame_id = "/base_link";
    while(ros::ok()) {
        pose_pub.publish(ps);
        ros::Duration(0.1).sleep();
    }
#endif
