#!/usr/bin/env python

##############
## Borrows heavily from pr2_manipulation_controllers/scripts/posture.py in ROS Fuerte.
##
##############
import sys
import copy

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
from std_msgs.msg import Bool, Float64MultiArray


POSTURES = {
    'off': [],
    'mantis': [0., 1., 0.,  -1., 3.14, -1., 3.14],
    'elbowupr': [-0.79, 0, -1.6, 9999, 9999, 9999, 9999],
    'elbowupl': [0.79, 0, 1.6, 9999, 9999, 9999, 9999],
    'old_elbowupr': [-0.79, 0, -1.6, -0.79, 3.14, -0.79, 5.49],
    'old_elbowupl': [0.79, 0, 1.6, -0.79, 3.14, -0.79,5.49],
    'elbowdownr': [-0.02826, 1.29463, -0.257856, -1.549888, -31.2789138, -1.0527644, -1.8127318],
    'elbowdownl': [-0.00882, 1.28343, 0.2033844, -1.556527, -0.09634002, -1.0235018, 1.79908930]
}

class JTTaskPostureControl(object):
    def __init__(self):
        controller_ns = rospy.get_param("~controller_ns", "r_cart")
        self.posture = rospy.get_param("~posture", "elbowupr")
        if not self.posture in POSTURES:
            rospy.logerr("[%s] Default posture must be in list of known postures.")
            rospy.logerr("Known Postures:")
            for posture in POSTURES.iterkeys():
                rospy.logerr("\t"+posture)
            sys.exit()

        self.shakeout_posture = rospy.get_param("~shakeout_posture","elbowdownr")
        if not self.shakeout_posture in POSTURES:
            rospy.logerr("[%s] Shakeout posture must be in list of known postures.")
            rospy.logerr("Known Postures:")
            for posture in POSTURES.iterkeys():
                rospy.logerr("\t"+posture)
            sys.exit()

        self.posture_pub = rospy.Publisher("%s/command_posture" %controller_ns, Float64MultiArray)
        self.shakeout_sub = rospy.Subscriber("shakeout_cmd", Bool, self.shakeout_cb)


    def shakeout_cb(self, msg):
        if msg.data:
            orig_posture = copy.copy(self.posture)
            self.posture = self.shakeout_posture
            self.posture_pub.publish(Float64MultiArray(data = POSTURES[self.posture]))
            def timer_cb(event):
                self.posture = orig_posture
                self.posture_pub.publish(Float64MultiArray(data = POSTURES[self.posture]))
            self.timer = rospy.Timer(rospy.Duration(1.0), timer_cb, oneshot=True)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.posture_pub.publish(Float64MultiArray(data = POSTURES[self.posture]))
            rate.sleep()

if __name__=='__main__':
    rospy.init_node('posture_node')
    jtpc = JTTaskPostureControl()
    jtpc.run()
