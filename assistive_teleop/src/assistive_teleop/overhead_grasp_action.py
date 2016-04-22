#!/usr/bin/env python

from copy import copy

import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Bool


class HapticMpcArmWrapper(object):
    def __init__(self, side):
        self.side = side
        self.pose_state = None
        self.in_deadzone = None
        self.state_sub = rospy.Subscriber('/'+side+'_arm/haptic_mpc/gripper_pose', PoseStamped, self.state_cb)
        self.at_goal_sub = rospy.Subscriber('/'+side+'_arm/haptic_mpc/in_deadzone', Bool, self.deadzone_cb)
        self.goal_pub = rospy.Publisher('/'+side+'_arm/haptic_mpc/goal_pose', PoseStamped, queue_size=3)

    def state_cb(self, ps_msg):
        self.state = ps_msg

    def deadzone_cb(self, msg):
        self.in_deadzone = msg.data

    def move_arm(self, ps_msg, wait=False):
        ps_msg.header.stamp = rospy.Time.now()
        self.goal_pub.publish(ps_msg)
        if not wait:
            return
        rospy.sleep(0.5)  # make sure the msg has time to arrive...
        while not self.in_deadzone:
            rospy.sleep(0.1)


from pr2_controllers_msgs.msg import Pr2GripperCommandAction, Pr2GripperCommandGoal
from pr2_gripper_sensor_msgs.msg import PR2GripperGrab, PR2GripperGrabCommand


class GripperGraspControllerWrapper(object):
    def __init__(self, side):
        self.side = side
        self.gripper_client = actionlib.SimpleActionClient('/'+side[0]+'_gripper_sensor_controller/gripper_action', Pr2GripperCommandAction)
        self.grab_client = actionlib.SimpleActionClient('/'+side[0]+'_gripper_sensor_controller/grab', PR2GripperGrab)
        self.gripper_client.wait_for_server()
        self.grab_client.wait_for_server()

    def open_gripper(self):
        goal = Pr2GripperCommandGoal()
        goal.position = 0.09
        goal.max_effort = -1.0
        self.gripper_client.send_goal(goal)

    def grasp(self):
        goal = PR2GripperGrabCommand(0.03)
        self.grab_client.send_goal(goal)


class OverheadGraspAction(object):
    def __init__(self, side, overhead_offset=0.2):
        self.overhead_offset = overhead_offset
        self.side = side
        self.arm = HapticMpcArmWrapper(side)
        self.gripper = GripperGraspControllerWrapper(side)
        self.action_server = actionlib.SimpleActionServer('overhead_grasp', OverheadGraspAction, self.execute, False)
        self.action_server.start()

    def exectute(self, goal):
        (setup_pose, overhead_pose, goal_pose) = self.process_path(goal.goal_pose)
        self.arm.move_arm(setup_pose, wait=True)
        self.arm.move_arm(overhead_pose, wait=True)
        self.gripper.open_gripper()
        self.arm.move_arm(goal_pose, wait=True)
        self.gripper.grasp()

    def process_path(self, goal_pose):
        while self.arm.state is None:
            rospy.sleep(0.1)
            rospy.loginfo("[%s] Waiting for %s arm state", rospy.get_name(), self.side)
        current_pose = copy(self.arm.state)
        goal_pose.pose.orientation = Quaternion(0.7071067, 0, -0.7071067, 0)

        overhead_height = goal_pose.pose.position.z + self.overhead_offset
        setup_pose = copy(current_pose)
        setup_pose.pose.position.z = overhead_height
        overhead_pose = copy(goal_pose)
        overhead_pose.pose.position.z = overhead_height


if __name__ == '__main__':
    rospy.init_node('overhead_grasp')
    r_overhead_grasp = HapticMpcArmWrapper('right')
    l_overhead_grasp = HapticMpcArmWrapper('left')
    rospy.spin()
