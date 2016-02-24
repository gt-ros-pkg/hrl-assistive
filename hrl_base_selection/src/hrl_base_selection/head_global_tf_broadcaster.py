#!/usr/bin/env python

import roslib
roslib.load_manifest('hrl_base_selection')
import rospy
import tf
from helper_functions import createBMatrix, Bmat_to_pos_quat
from geometry_msgs.msg import PoseStamped


class AutobedGlobalTFBroadcaster(object):
    def __init__(self):
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        rospy.sleep(2)
        rate = rospy.Rate(10.0)
        self.map_B_head = None
        self.out_trans = None
        self.out_rot = None
        self.ar_tag_head_sub = rospy.Subscriber('/ar_tag_tracking/head_pose', PoseStamped, self.ar_tag_head_cb)
        try:
            while (not self.tf_listener.canTransform('odom_combined', 'torso_lift_link', rospy.Time(0))):
                try:
                    print 'Waiting for head localization in world.'
                    rospy.sleep(1)
                except:
                    print 'Head TF broadcaster crashed!'
                    break
            rospy.sleep(1)
            while not rospy.is_shutdown():
                if self.map_B_bed is not None:
                    try:
                        self.tf_broadcaster.sendTransform(self.out_trans, self.out_rot,
                                                          rospy.Time.now(),
                                                          'odom_combined',
                                                          'user_head_link')
                        rate.sleep()
                    except:
                        print 'Head TF broadcaster crashed!'
                        break
                else:
                    print 'Waiting to detect head AR tag at least once.'
                    rospy.sleep(1)
        except:
            print 'Head TF broadcaster crashed!'


    def ar_tag_head_cb(self, msg):
        trans = [msg.pose.position.x,
                 msg.pose.position.y,
                 msg.pose.position.z]
        rot = [msg.pose.orientation.x,
               msg.pose.orientation.y,
               msg.pose.orientation.z,
               msg.pose.orientation.w]
        now = rospy.Time.now()
        pr2_B_head = createBMatrix(trans, rot)
        self.listener.waitForTransform('odom_combined', 'torso_lift_link', now, rospy.Duration(1))
        (trans, rot) = self.tf_listener.lookupTransform('odom_combined', 'torso_lift_link', now)
        map_B_pr2 = createBMatrix(trans, rot)
        self.map_B_head = map_B_pr2*pr2_B_head
        (self.out_trans, self.out_rot) = Bmat_to_pos_quat(self.map_B_head)


if __name__ == '__main__':
    rospy.init_node('autobed_global_tf_broadcaster')
    autobed_global_tf_broadcaster = AutobedGlobalTFBroadcaster()
    rospy.spin()








