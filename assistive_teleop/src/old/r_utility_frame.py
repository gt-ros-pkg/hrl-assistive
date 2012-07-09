#!/usr/bin/python

import roslib; roslib.load_manifest('web_teleop_trunk')
import rospy
from std_msgs.msg import Bool
from tf import TransformBroadcaster
from web_teleop_trunk.srv import FrameUpdate

class Right_Utility_Frame():
  
    frame = 'base_footprint'
    px = py = pz = 0;
    qx = qy = qz = 0;
    qw = 1;

    def __init__(self):
        rospy.init_node('right_utilitiy_frame_source')
        
        self.updater = rospy.Service('r_utility_frame_update', FrameUpdate, self.update_frame)
        
        self.tfb = TransformBroadcaster()

    def update_frame(self, req):
        ps = req.pose
        self.frame = ps.header.frame_id
        self.px = ps.pose.position.x    
        self.py = ps.pose.position.y    
        self.pz = ps.pose.position.z    
        self.qx = ps.pose.orientation.x
        self.qy = ps.pose.orientation.y
        self.qz = ps.pose.orientation.z
        self.qw = ps.pose.orientation.w

        self.tfb.sendTransform((self.px,self.py,self.pz),(self.qx,self.qy,self.qz,self.qw), rospy.Time.now(), "rh_utility_frame", self.frame)
        rsp = Bool()
        rsp.data = True
        return rsp

if __name__ == '__main__':
    RUF = Right_Utility_Frame()

    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        RUF.tfb.sendTransform((RUF.px,RUF.py,RUF.pz),(RUF.qx,RUF.qy,RUF.qz,RUF.qw), rospy.Time.now(), "rh_utility_frame", RUF.frame)
        r.sleep()
