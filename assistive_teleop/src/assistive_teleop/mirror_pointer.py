#!/usr/bin/env python

import math
import numpy as np

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Point, PointStamped
from tf import TransformListener, TransformBroadcaster, transformations as tft
import pose_utils as pu
from assistive_teleop.srv import PointMirror, PointMirrorResponse

class MirrorPointer(object):
    def __init__(self):
        self.tf = TransformListener()
        self.tfb = TransformBroadcaster()
        self.active = True
        self.head_pose = PoseStamped()
        self.goal_pub = rospy.Publisher('goal_pose', PoseStamped)
        rospy.Subscriber('/head_center', PoseStamped, self.head_pose_cb)
        rospy.Service('/point_mirror', PointMirror, self.point_mirror_cb)

    def head_pose_cb(self, msg):
        """Save update head pose, transforming to torso frame if necessary"""
        msg.header.stamp = rospy.Time(0)
        if not (msg.header.frame_id.lstrip('/') == 'torso_lift_link'):
            self.head_pose = self.tf.transformPose('/torso_lift_link', msg)
        else:
            self.head_pose = msg

    def get_current_mirror_pose(self):
        """Get the current pose of the mirror (hardcoded relative to tool frame"""
        mp = PoseStamped()
        mp.header.frame_id = "/r_gripper_tool_frame"
        mp.pose.position = Point(0.15, 0, 0)
        mp.pose.orientation = Quaternion(0,0,0,1)
        mirror_pose = self.tf.transformPose("torso_lift_link", mp)
        return mirror_pose

    def get_pointed_mirror_pose(self):
        """Get the pose of the mirror pointet at the goal location"""
        target_pt = PointStamped(self.head_pose.header, self.head_pose.pose.position)
        mp = self.get_current_mirror_pose()
        pu.aim_pose_to(mp, target_pt, (0,1,0))
        return mp

    def trans_mirror_to_wrist(self, mp):
        """Get the wrist location from a mirror pose"""
        mp.header.stamp = rospy.Time(0)
        try:
            mp_in_mf = self.tf.transformPose('mirror',mp)
        except:
            return
        mp_in_mf.pose.position.x -= 0.15
        try:
            wp = self.tf.transformPose('torso_lift_link',mp_in_mf)
        except:
            return
        return wp

    def head_pose_sensible(self, ps):
        """Set a bounding box on reasonably expected head poses"""
        if ((ps.pose.position.x < 0.35) or
            (ps.pose.position.x > 1.0) or
            (ps.pose.position.y < -0.2) or
            (ps.pose.position.y > 0.85) or
            (ps.pose.position.z < -0.3) or
            (ps.pose.position.z > 1) ):
            return False
        else:
            return True

    def point_mirror_cb(self, req):
        rospy.loginfo("Mirror Adjust Request Received")
        if not self.head_pose_sensible(self.head_pose):
            rospy.logwarn("Registered Head Position outside of expected region: %s" %self.head_pose)
            return PoseStamped()
        mp = self.get_pointed_mirror_pose()
        goal = self.trans_mirror_to_wrist(mp)
        goal.header.stamp = rospy.Time.now()
        resp = PointMirrorResponse()
        resp.wrist_pose = goal
        print "Head Pose: "
        print self.head_pose
        self.goal_pub.publish(goal)
        return resp

    def broadcast_mirror_tf(self):
        self.tfb.sendTransform((0.15,0,0),
                               (0,0,0,1),
                               rospy.Time.now(),
                               "mirror",
                               "r_gripper_tool_frame")

if __name__=='__main__':
    rospy.init_node('mirror_pointer')
    mp = MirrorPointer()
    rospy.sleep(1)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        mp.broadcast_mirror_tf()
        mp.trans_mirror_to_wrist(mp.get_current_mirror_pose())
