#!/usr/bin/env python

import roslib
roslib.load_manifest('hrl_base_selection')
import rospy
import tf
from helper_functions import createBMatrix, Bmat_to_pos_quat
from geometry_msgs.msg import PoseStamped
import threading


class AutobedGlobalTFBroadcaster(object):
    def __init__(self):
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        self.frame_lock = threading.RLock()
        rospy.sleep(10)
        rate = rospy.Rate(50.0)
        self.map_B_bed = None
        self.out_trans = None
        self.out_rot = None

        while (not self.tf_listener.canTransform('map', 'torso_lift_link', rospy.Time(0))):
            try:
                print 'Waiting for PR2 localization in world.'
                rospy.sleep(1)
            except:
                print 'Bed TF broadcaster crashed!'
                break
        rospy.sleep(1)
        self.ar_tag_autobed_sub = rospy.Subscriber('/ar_tag_tracking/autobed_pose', PoseStamped, self.ar_tag_autobed_cb)
        rospy.sleep(1)
        print 'Now have the map-to-pr2 transform. Will now try to starting broadcasting Autobeds position globally!'
        while not rospy.is_shutdown():
            if self.map_B_bed is not None:
                try:
                    self.tf_broadcaster.sendTransform(self.out_trans, self.out_rot,
                                                      rospy.Time.now(),
                                                      'autobed/base_link',
                                                      'map')
                    rate.sleep()
                except:
                    print 'Bed TF broadcaster crashed trying to broadcast!'
                    break
            else:
                print 'Waiting to detect bed AR tag at least once.'
                rospy.sleep(2)
        print 'Bed TF broadcaster crashed!'

    def ar_tag_autobed_cb(self, msg):
        with self.frame_lock:
            trans = [msg.pose.position.x,
                     msg.pose.position.y,
                     msg.pose.position.z]
            rot = [msg.pose.orientation.x,
                   msg.pose.orientation.y,
                   msg.pose.orientation.z,
                   msg.pose.orientation.w]
            now = rospy.Time.now()
            pr2_B_bed = createBMatrix(trans, rot)
            self.tf_listener.waitForTransform('map', msg.header.frame_id, now, rospy.Duration(3))
            (newtrans, newrot) = self.tf_listener.lookupTransform('map', msg.header.frame_id, now)
            map_B_pr2 = createBMatrix(newtrans, newrot)
            self.map_B_bed = map_B_pr2*pr2_B_bed
            (self.out_trans, self.out_rot) = Bmat_to_pos_quat(self.map_B_bed)

if __name__ == '__main__':
    rospy.init_node('autobed_global_tf_broadcaster')
    autobed_global_tf_broadcaster = AutobedGlobalTFBroadcaster()
    rospy.spin()








