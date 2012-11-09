#!/usr/bin/env python

import sys.argv

import roslib;roslib.load_manifest('hrl_face_adls')
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from tf import transformations as trans

class ClickedPoseRelay(object):
    def __init__(self, offset_x=0., offset_y=0., offset_z=0., rot_z=0., rot_y=0., rot_z=0.)
        """Setup pub/subs and transform parameters"""
        self.pose_sub = rospy.Subscriber('pose_in', PoseStamped, self.pose_in_cb)
        self.pose_sub = rospy.Publisher('pose_out', PoseStamped)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.quat_offset = trans.quaternion_from_euler(rot_x, rot_y, rot_z)

    def self.pose_in_cb(self, ps_in):
        """Apply transform to received pose and republish"""
        ps_out = PoseStamped()
        ps_out.header.frame_id = ps_in.header.frame_id
        ps_out.header.stamp = ps_in.header.stamp

        ps_out.pose.position.x = ps_in.pose.position.x + self.offset_x
        ps_out.pose.position.y = ps_in.pose.position.y + self.offset_y
        ps_out.pose.position.z = ps_in.pose.position.z + self.offset_z

        quat_in = (ps_in.pose.orientation.x, ps_in.pose.orientation.y
                   ps_in.pose.orientation.z, ps_in.pose.orientation.w)
        quat_out = trans.quaternion_multiply(quat_in, self.quat_offset)
        ps_out.pose.orientation = Quaternion(*quat_out)

        self.pose_pub.publish(ps_out)

if __name__=='__main__':
    rospy.init_node('clicked_pose_relay')
    relay = ClickedPoseRelay(rot_z=3.14159)
    rospy.spin()


