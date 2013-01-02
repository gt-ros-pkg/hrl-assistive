#!/usr/bin/env python

import argparse

import roslib;roslib.load_manifest('hrl_face_adls')
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from tf import transformations as trans

import pose_utils as pu

class ClickedPoseRelay(object):
    def __init__(self, translation, quaternion):
        """Setup pub/subs and transform parameters"""
        self.pose_sub = rospy.Subscriber('pose_in',
                                         PoseStamped,
                                         self.pose_in_cb)
        self.pose_pub = rospy.Publisher('pose_out', PoseStamped)
        self.offset_x = translation[0]
        self.offset_y = translation[1]
        self.offset_z = translation[2]
        self.quat_offset = quaternion
        rospy.loginfo('['+rospy.get_name()[1:]+']'+
                      'Pose relay node started with:'+
                      '\r\nTranslation: '+ str(translation) +
                      '\r\nRotation: '+ str(quaternion))

    def pose_in_cb(self, ps_in):
        """Apply transform to received pose and republish"""
        trans_pose = pu.pose_relative_trans(ps_in, self.offset_x,
                                            self.offset_y, self.offset_z)

        ps_out = PoseStamped()
        ps_out.header.frame_id = ps_in.header.frame_id
        ps_out.header.stamp = ps_in.header.stamp
        ps_out.pose.position = trans_pose.pose.position

        quat_in = (ps_in.pose.orientation.x, ps_in.pose.orientation.y,
                   ps_in.pose.orientation.z, ps_in.pose.orientation.w)
        quat_out = trans.quaternion_multiply(quat_in, self.quat_offset)
        ps_out.pose.orientation = Quaternion(*quat_out)

        self.pose_pub.publish(ps_out)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Apply a transform "+
                                     "(in the 'frame' of the pose) "+
                                     "to an incoming pose and republish")
    parser.add_argument('-t','--translation', nargs=3, default=(0.,0.,0.),
                        type=float, help='The translation to be applied to'+
                                         ' the pose (with respect to itself)')
    parser.add_argument('-q','--quaternion', nargs=4, default=(0.,0.,0.,1.),
                        type=float, help='The rotation quaterion to '+
                                         'be applied to the pose')
    parser.add_argument('-r','-rot','-rpy','--rotation', nargs=3,
                        type=float, help='The roll-pitch-yaw rotation '+
                                         '(with respect to itself) '+
                                         'to be applied to the pose')
    args = parser.parse_known_args()

    if args[0].rotation:
        rpy = args[0].rotation
        if args[0].quaternion != (0.,0.,0.,1.):
            parser.exit(status=1, message="Please specify only one rotation")
        else:
            quaternion = trans.quaternion_from_euler(rpy[0],rpy[1],rpy[2])
    else:
        quaternion = args[0].quaternion

    rospy.init_node('clicked_pose_relay')
    relay = ClickedPoseRelay(args[0].translation, quaternion)
    rospy.spin()
