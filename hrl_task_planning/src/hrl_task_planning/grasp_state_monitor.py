from collections import deque
import numpy as np

import rospy
from std_msgs.msg import Bool, Float32
from pr2_msgs.msg import PressureState
from pr2_controllers_msgs.msg import JointControllerState


class GraspPressureMonitor(object):
    """ A class for tracking if a gripper is currently grasping something. """

    def __init__(self, side):
        self.side = side
        self.gripper_name = '_'.join([side, "gripper"])
        self.state_pub = rospy.Publisher('/'.join(["/grasping", self.gripper_name]), Bool, latch=True)
        self.pressure_sub = rospy.Subscriber(''.join(['/pressure/', side[0], "_gripper_motor"]),
                                             PressureState,
                                             self.pressure_cb)
        self.gripper_state_sub = rospy.Subscriber('_'.join([side[0], "gripper_controller/state"]),
                                                  JointControllerState,
                                                  self.gripper_state_cb)
        self.pressure_grad_pub = rospy.Publisher('/pressure/gradient', Float32)
        self.l_pressure_sum_low = 0.0
        self.r_pressure_sum_low = 0.0
        self.l_pressure_sum_deque = deque([0]*25, 25)
        self.l_pressure_grad_deque = deque([0]*25, 25)
        self.r_pressure_sum_deque = deque([0]*25, 25)
        self.r_pressure_grad_deque = deque([0]*25, 25)
        self.grasp_state = False  # Default assumption is that the hand is empty... (this may or may not be valid)
        self.motion_state = "STEADY"
        self.pressure_state = "STEADY"
        self.state_pub.publish(self.grasp_state)  # Always have something latched, even if it's a bad guess...
        rospy.loginfo("[%s] %s Grasp State Monitor Ready" % (rospy.get_name(), side.capitalize()))

    def _sum_sensors(self, data):
        return sum(data[7:])

    def pressure_cb(self, pressure_msg):
        """ Record data from pressure sensors: Sum of pressure on pads of fingers, and gradient, of ~1 sec. """
        l_sum = self._sum_sensors(pressure_msg.l_finger_tip)
        self.l_pressure_sum_deque.append(l_sum)
        l_grad = np.mean(np.gradient(np.array(self.l_pressure_sum_deque)))
        self.l_pressure_grad_deque.append(l_grad)
        l_grad_trend = np.mean(self.l_pressure_grad_deque)
        if l_grad_trend > 200:
            l_pressure_state = 1
        elif l_grad_trend < -200:
            l_pressure_state = -1
        else:
            l_pressure_state = 0

        r_sum = self._sum_sensors(pressure_msg.r_finger_tip)
        self.r_pressure_sum_deque.append(r_sum)
        r_grad = np.mean(np.gradient(np.array(self.r_pressure_sum_deque)))
        self.r_pressure_grad_deque.append(r_grad)
        r_grad_trend = np.mean(self.r_pressure_grad_deque)
        self.pressure_grad_pub.publish(r_grad_trend)
        if r_grad_trend > 200:
            r_pressure_state = 1
        elif r_grad_trend < -200:
            r_pressure_state = -1
        else:
            r_pressure_state = 0

        net = l_pressure_state + r_pressure_state
        if net > 0:
            self.pressure_state = "RISING"
        elif net < 0:
            self.pressure_state = "FALLING"
        else:
            self.pressure_state = "STEADY"

    def gripper_state_cb(self, gs_msg):
        delta = gs_msg.process_value_dot
        if abs(delta) < 0.0025:
            self.motion_state = "STEADY"
        elif delta < 0:
            self.motion_state = "CLOSING"
        else:
            self.motion_state = "OPENING"

        self.update_grasp_state()

    def update_grasp_state(self):
        grasp_state = self.grasp_state
        if self.motion_state == "OPENING":
            grasp_state = False
        elif self.motion_state == "STEADY":
            if self.pressure_state == "RISING":
                grasp_state = self.grasp_state
            elif self.pressure_state == "FALLING":
                grasp_state = False
        elif self.motion_state == "CLOSING":
            if self.pressure_state == "RISING":
                grasp_state = True
            else:
                grasp_state = False

        if grasp_state != self.grasp_state:
            self.state_pub.publish(grasp_state)
            self.grasp_state = grasp_state


def main():
    rospy.init_node('move_object_manager')
    left_grasp_monitor = GraspPressureMonitor("left")
    right_grasp_monitor = GraspPressureMonitor("right")
    rospy.spin()
