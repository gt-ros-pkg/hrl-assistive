#!/usr/bin/env python

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion
from tf import (TransformListener, ExtrapolationException,
                LookupException, ConnectivityException)
from tf import transformations as trans

class TwistToPoseConverter(object):
    def __init__(self, ee_frame='/l_gripper_tool_frame'):
        self.twist_sub = rospy.Subscriber('/twist_in',
                                          TwistStamped,
                                          self.twist_cb)
        self.pose_pub = rospy.Publisher('/pose_out', PoseStamped)
        self.tf_listener = TransformListener()
        self.ee_frame = ee_frame

    def get_ee_pose(self, link, frame='/torso_lift_link'):
        """Get current end effector pose from tf."""
        try:
            now = rospy.Time.now()
            self.tf_listener.waitForTransform(frame, link,
                                              now,   rospy.Duration(8.0))
            pos, quat = self.tf_listener.lookupTransform(frame, link, now)
        except (LookupException, ConnectivityException,
                ExtrapolationException, Exception) as e:
            rospy.logwarn("[Node: "+rospy.get_name()[1:]+"]"+
                          "TF Failure getting current "+
                          "end-effector pose:\r\n %s" %e)
            return None, None
        return pos, quat 

    def twist_cb(self, ts):
        """Get current end effector pose and augment with twist command."""
        cur_pos, cur_quat = self.get_ee_pose(self.ee_frame, ts.header.frame_id)
        ps = PoseStamped()
        ps.header.frame_id = ts.header.frame_id
        ps.header.stamp = ts.header.stamp
        ps.pose.position.x = cur_pos[0] + ts.twist.linear.x
        ps.pose.position.y = cur_pos[1] + ts.twist.linear.y
        ps.pose.position.z = cur_pos[2] + ts.twist.linear.z

        twist_quat = trans.quaternion_from_euler(ts.twist.angular.x,
                                                 ts.twist.angular.y,
                                                 ts.twist.angular.z)

        final_quat = trans.quaternion_multiply(cur_quat, twist_quat)
        ps.pose.orientation = Quaternion(*final_quat)

        try:
            self.tf_listener.waitForTransform('/torso_lift_link',
                                              ps.header.frame_id,
                                              ps.header.stamp,
                                              rospy.Duration(8.0))
            pose_out = self.tf_listener.transformPose('/torso_lift_link', ps)
        except (LookupException, ConnectivityException, 
                ExtrapolationException, Exception) as e:
            rospy.logwarn("[Node: "+rospy.get_name()[1:]+"]"+
                          "TF Failure transforming goal pose"+
                          "back to torso_lift_link:\r\n %s" %e)
            return

        self.pose_pub.publish(pose_out)

if __name__=='__main__':
    rospy.init_node('twist_to_pose_node')
    converter = TwistToPoseConverter()
    rospy.spin()
