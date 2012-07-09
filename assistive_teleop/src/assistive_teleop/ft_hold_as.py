#!/usr/bin/env python

import math

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
import actionlib
from std_msgs.msg import Int8 # kelsey
from geometry_msgs.msg import WrenchStamped
from assistive_teleop.msg import FtHoldAction, FtHoldFeedback, FtHoldResult


class FtHolder:
    
    def __init__(self):
        self.hold_server = actionlib.SimpleActionServer('ft_hold_action', FtHoldAction, self.hold, False) 
        self.trans_pub = rospy.Publisher('shaving_state', Int8, latch=True) # kelsey
        rospy.Subscriber('/netft_gravity_zeroing/wrench_zeroed', WrenchStamped, self.get_netft_state)
        self.hold_server.start()

    def get_netft_state(self, ws):
        self.netft_wrench = ws
        self.ft_mag = math.sqrt(ws.wrench.force.x**2 + ws.wrench.force.y**2 + ws.wrench.force.z**2)

    def hold(self, goal):
        rospy.loginfo("Holding Position: Will retreat with inactivity or high forces")
        ft_active = rospy.Time.now()
        hold_rate=rospy.Rate(200)
        result=FtHoldResult()
        print "Check ft mag init"
        print self.ft_mag
        self.trans_pub.publish(15) # kelsey
        while not rospy.is_shutdown():

            if self.hold_server.is_preempt_requested():
                print "FtHolder: Preempt requested"
                self.hold_server.set_preempted()
                rospy.loginfo("Hold Action Server Goal Preempted")
                break
            
            if self.ft_mag > goal.activity_thresh:
                ft_active = rospy.Time.now()
            elif rospy.Time.now()-ft_active > goal.timeout:
                result.timeout = True
                self.hold_server.set_succeeded(result, "Inactive for %ss, finished holding" %goal.timeout.secs)
                rospy.loginfo("Inactive for %ss, finished holding" %goal.timeout.secs)
                break
            
            if self.ft_mag > goal.mag_thresh:
                result.unsafe = True
                self.hold_server.set_aborted(result, "Unsafe Force Magnitude! Felt %f, Safety Threshold is %f" \
                                                     %(self.ft_mag, goal.mag_thresh))
                rospy.loginfo("Unsafe Force Magnitude! Felt %f, Safety Threshold is %f"\
                              %(self.ft_mag, goal.mag_thresh))
                break

            if self.netft_wrench.wrench.force.z > goal.z_thresh:
                result.unsafe = True
                self.hold_server.set_aborted(result, "Unsafe Normal Forces! Felt %f, Safety Threshold is %f" \
                                                     %(self.netft_wrench.wrench.force.z, goal.z_thresh))
                rospy.loginfo("Unsafe Normal Forces! Felt %f, Safety Threshold is %f"\
                              %(self.netft_wrench.wrench.force.z, goal.z_thresh))
                break


            hold_rate.sleep()

if __name__ == '__main__':
    rospy.init_node('hold_action')
    fth = FtHolder()
    while not rospy.is_shutdown():
        rospy.spin()
