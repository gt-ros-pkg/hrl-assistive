#!/usr/bin/env python

import sys
import argparse
from collections import deque

import rospy
from std_msgs.msg import Bool

from pr2_controllers_msgs.msg import JointControllerState
import pr2_gripper_sensor_msgs.msg as gsm


class GripperSensorGraspMonitor(object):
    def __init__(self, side, fully_closed_dist):
        self.side = side
        self.fully_closed_dist = fully_closed_dist
        gsa_ns = '/' + self.side[0] + '_gripper_sensor_controller'

        self.grasping = None
        self.grasp_state_deque = deque([None]*24, 24)
        self.grasp_state_pub = rospy.Publisher('/grasping/'+self.side+'_gripper', Bool, queue_size=10, latch=True)
        self.fully_closed = False
        self.cannot_close = False
        self.both_contact = False
        self.last_empty_position = 0.1
        self.opening_from_empty = False

        self.gripper_state_sub = rospy.Subscriber(gsa_ns+'/state', JointControllerState, self.gripper_state_cb)
        self.contact_state_sub = rospy.Subscriber(gsa_ns+'/contact_state', gsm.PR2GripperFindContactData, self.contact_state_cb)

    def gripper_state_cb(self, msg):
        self.fully_closed = bool(msg.process_value < self.fully_closed_dist)  # Gripper completely closed
        self.openness = msg.process_value

        if abs(msg.process_value_dot) < 0.001:
            self.cannot_close = bool(msg.error > 0.001)  # Can't close to setpoint
        else:
            self.cannot_close = False  # still moving, don't make any assumptions

        if not self.grasping:
            if msg.process_value_dot > 0.0:  # Opening
                self.opening_from_empty = True
            elif msg.process_value_dot < 0.0:  # Closing
                self.last_empty_position = msg.process_value
                self.opening_from_empty = False

    def contact_state_cb(self, msg):
        left = msg.left_fingertip_pad_contact
        right = msg.right_fingertip_pad_contact
        if left and right:
            self.both_contact = True  # Contact on both pads likely means we're holding something, don't change for only 1 contact
        elif not left and not right:
            self.both_contact = False  # No contacts, not grasping...

    def update_grasp_state(self):
#        msg = ''
        grasping_now = None
        if self.fully_closed:
#            msg += " Fully Closed "
            grasping_now = False
        if self.opening_from_empty:
#            msg += " Opening "
            grasping_now = False if grasping_now is None else grasping_now
        if self.cannot_close:
#            msg += " Stuck "
            grasping_now = True if grasping_now is None else grasping_now
        if self.both_contact:
#            msg += " Contact "
            grasping_now = True if grasping_now is None else grasping_now
        if (not self.cannot_close and not self.both_contact):
#            msg += " Neither "
            grasping_now = False if grasping_now is None else grasping_now
        if grasping_now is None:
            return  # Nothing happening, skip ahead

#        print grasping_now, msg, self.openness
        self.grasp_state_deque.append(grasping_now)
        if None in self.grasp_state_deque:
            return
#        print "Deque:\n", self.grasp_state_deque
        filtered_grasping = True if sum(self.grasp_state_deque) > self.grasp_state_deque.maxlen/2 else False
#        print "Filtered Grasping: ", filtered_grasping

        if filtered_grasping != self.grasping:
            if filtered_grasping:
                print "%s Gripper Grasped" % self.side.capitalize()
            else:
                print "%s Gripper Released" % self.side.capitalize()
            self.grasping = filtered_grasping
            self.grasp_state_pub.publish(filtered_grasping)


def main():
    rospy.init_node('gripper_sensor_grasp_detection')
    parser = argparse.ArgumentParser(description="Report when a gripper grasps or releases an object")
    parser.add_argument('side', choices=["left", "right"], help="The side of the gripper to monitor ('left' or 'right')")
    parser.add_argument('--closed', '-c', type=float, default=0.004, help="The position at which the gripper is fully closed (m)")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = GripperSensorGraspMonitor(args.side, args.closed)
    rospy.sleep(1.5)
    rate = rospy.Rate(12)
    while not rospy.is_shutdown():
        monitor.update_grasp_state()
        rate.sleep()
