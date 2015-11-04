#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool

from pr2_controllers_msgs.msg import JointControllerState
import pr2_gripper_sensor_msgs.msg as gsm


class GripperSensorGraspMonitor(object):
    def __init__(self, side):
        self.side = side
        gsa_ns = '/' + self.side[0] + '_gripper_sensor_controller'

        self.grasping = None
        self.grasp_state_pub = rospy.Publisher('/grasping/'+self.side+'_gripper', Bool, latch=True)
        self.fully_closed = False
        self.cannot_close = False
        self.both_contact = False
        self.last_empty_position = 0.1
        self.opening_from_empty = False

        self.gripper_state_sub = rospy.Subscriber(gsa_ns+'/state', JointControllerState, self.gripper_state_cb)
        self.contact_state_sub = rospy.Subscriber(gsa_ns+'/contact_state', gsm.PR2GripperFindContactData, self.contact_state_cb)

    def gripper_state_cb(self, msg):
        self.fully_closed = bool(msg.process_value < 0.0015)  # Gripper completely closed

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
        if (left and right):
            self.both_contact = True  # Contact on both pads likely means we're holding something, don't change for only 1 contact
        elif (not left and not right):
            self.both_contact = False  # No contacts, not grasping...

    def update_grasp_state(self):
        # msg = ''
        grasping_now = None
        if self.fully_closed:
            # msg += " Fully Closed "
            grasping_now = False
        elif self.opening_from_empty:
            grasping_now = False
            # msg += " Opening "
        elif self.cannot_close:
            # msg += " Stuck "
            grasping_now = True
        elif self.both_contact:
            # msg += " Contact "
            grasping_now = True
        elif (not self.cannot_close and not self.both_contact):
            # msg += " Neither "
            grasping_now = False
        else:
            return  # Nothing happening, skip ahead

        if grasping_now != self.grasping:
            if grasping_now:
                print "%s Gripper Grasped" % self.side.capitalize()
            else:
                print "%s Gripper Released" % self.side.capitalize()
            self.grasping = grasping_now
            self.grasp_state_pub.publish(grasping_now)


def main():
    rospy.init_node('gripper_sensor_grasp_detection')
    left_monitor = GripperSensorGraspMonitor('left')
    right_monitor = GripperSensorGraspMonitor('right')
    rate = rospy.Rate(25)
    while not rospy.is_shutdown():
        left_monitor.update_grasp_state()
        right_monitor.update_grasp_state()
        rate.sleep()
