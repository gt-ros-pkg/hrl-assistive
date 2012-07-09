#!/usr/bin/python

import roslib; roslib.load_manifest('web_teleop_trunk')
import rospy
import math
from std_msgs.msg import Bool
from tf import TransformBroadcaster
from web_teleop_trunk.srv import FrameUpdate

class Left_Utility_Frame():
  
    frame = 'base_footprint'
    px = py = pz = 0;
    qx = qy = qz = 0;
    qw = 1;

    def __init__(self):
        rospy.init_node('left_utilitiy_frame_source')
        
        self.updater = rospy.Service('l_utility_frame_update', FrameUpdate, self.update_frame)
        
        self.tfb = TransformBroadcaster()

    def update_frame(self, req):
        ps = req.pose
        if not (math.isnan(ps.pose.orientation.x) or 
                math.isnan(ps.pose.orientation.y) or
                math.isnan(ps.pose.orientation.z) or
                math.isnan(ps.pose.orientation.w)):
            self.frame = ps.header.frame_id
            self.px = ps.pose.position.x    
            self.py = ps.pose.position.y    
            self.pz = ps.pose.position.z    
            self.qx = ps.pose.orientation.x
            self.qy = ps.pose.orientation.y
            self.qz = ps.pose.orientation.z
            self.qw = ps.pose.orientation.w
        else:
            rospy.logerr("NAN's sent to l_utility_frame_source")

        self.tfb.sendTransform((self.px,self.py,self.pz),(self.qx,self.qy,self.qz,self.qw), rospy.Time.now(), "lh_utility_frame", self.frame)
        rsp = Bool()
        rsp.data = True
        return rsp

if __name__ == '__main__':
    LUF = Left_Utility_Frame()

    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        LUF.tfb.sendTransform((LUF.px,LUF.py,LUF.pz),(LUF.qx,LUF.qy,LUF.qz,LUF.qw), rospy.Time.now(), "lh_utility_frame", LUF.frame)
        r.sleep()
