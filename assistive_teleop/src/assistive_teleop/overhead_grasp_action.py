#!/usr/bin/env python

from copy import copy, deepcopy

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
        self.in_deadzone = False
        rospy.sleep(1.0)  # make sure the msg has time to arrive...
        while not self.in_deadzone:
            rospy.sleep(0.1)


from pr2_controllers_msgs.msg import Pr2GripperCommandAction, Pr2GripperCommandGoal, Pr2GripperCommand
from pr2_gripper_sensor_msgs.msg import (PR2GripperGrabAction, PR2GripperGrabGoal, PR2GripperGrabCommand,
                                         PR2GripperReleaseAction, PR2GripperReleaseGoal, PR2GripperReleaseCommand)


class GripperGraspControllerWrapper(object):
    def __init__(self, side):
        self.side = side
        self.gripper_client = actionlib.SimpleActionClient('/'+side[0]+'_gripper_sensor_controller/gripper_action', Pr2GripperCommandAction)
        self.grab_client = actionlib.SimpleActionClient('/'+side[0]+'_gripper_sensor_controller/grab', PR2GripperGrabAction)
        self.contact_release_client = actionlib.SimpleActionClient('/'+side[0]+'_gripper_sensor_controller/release', PR2GripperReleaseAction)
        self.gripper_client.wait_for_server()
        self.grab_client.wait_for_server()

    def open_gripper(self, wait=False):
        cmd = Pr2GripperCommand(0.09, -1.0)
        goal = Pr2GripperCommandGoal(cmd)
        print "Gripper Goal: ", goal
        self.gripper_client.send_goal(goal)
        if wait:
            self.contact_release_client.wait_for_result()

    def grasp(self, wait=False):
        cmd = PR2GripperGrabCommand(0.03)
        goal = PR2GripperGrabGoal(cmd)
        self.grab_client.send_goal(goal)
        if wait:
            self.contact_release_client.wait_for_result()

    def release_on_contact(self, wait=False):
        cmd = PR2GripperReleaseCommand()
        cmd.event.trigger_conditions = 2  # Slip, impact, or acceleration
        cmd.event.acceleration_trigger_magnitude = 2.0
        cmd.event.slip_trigger_magnitude = 0.005
        goal = PR2GripperReleaseGoal(cmd)
        self.contact_release_client.send_goal(goal)
        if wait:
            self.contact_release_client.wait_for_result()


from assistive_teleop.msg import OverheadGraspAction, OverheadGraspResult, OverheadGraspFeedback


class OverheadGrasp(object):
    def __init__(self, side, overhead_offset=0.15):
        self.overhead_offset = overhead_offset
        self.side = side
        self.arm = HapticMpcArmWrapper(side)
        self.gripper = GripperGraspControllerWrapper(side)
        self.action_server = actionlib.SimpleActionServer('/%s_arm/overhead_grasp' % self.side, OverheadGraspAction, self.execute, False)
        self.action_server.start()
        rospy.loginfo("[%s] %s Overhead Grasp Action Started", rospy.get_name(), self.side.capitalize())

    def execute(self, goal):
        print "received goal"
        self.action_server.publish_feedback(OverheadGraspFeedback("Processing Goal Pose"))
        (setup_pose, overhead_pose, goal_pose) = self.process_path(goal.goal_pose)
        print "moving arm to setup"
        self.action_server.publish_feedback(OverheadGraspFeedback("Moving to Setup Position"))
        self.arm.move_arm(setup_pose, wait=True)
        print "moving arm to overhead"
        self.action_server.publish_feedback(OverheadGraspFeedback("Moving to Overhead Position"))
        self.arm.move_arm(overhead_pose, wait=True)
        print "opening gripper"
        self.action_server.publish_feedback(OverheadGraspFeedback("Opening Gripper"))
        self.gripper.open_gripper()
        rospy.sleep(2.0)
        print "moving arm to goal"
        self.action_server.publish_feedback(OverheadGraspFeedback("Moving to goal"))
        self.arm.move_arm(goal_pose, wait=True)
        print "closing gripper"
        self.action_server.publish_feedback(OverheadGraspFeedback("Closing Gripper"))
        self.gripper.grasp()
        self.action_server.publish_feedback(OverheadGraspFeedback("finished"))
        self.action_server.set_succeeded(OverheadGraspResult('finished'))

    def process_path(self, goal_pose):
        while self.arm.state is None:
            rospy.sleep(0.1)
            rospy.loginfo("[%s] Waiting for %s arm state", rospy.get_name(), self.side)
        current_pose = deepcopy(self.arm.state)
        goal_pose.pose.orientation = Quaternion(0.7071067, 0, -0.7071067, 0)
        overhead_height = goal_pose.pose.position.z + self.overhead_offset
        setup_pose = deepcopy(current_pose)
        setup_pose.header.frame_id = '/base_link'
        setup_pose.pose.position.z = overhead_height
        overhead_pose = deepcopy(goal_pose)
        overhead_pose.pose.position.z = overhead_height
        goal_pose.pose.position.z += 0.03
        return (setup_pose, overhead_pose, goal_pose)


from assistive_teleop.msg import OverheadPlaceAction, OverheadPlaceResult, OverheadPlaceFeedback


class OverheadPlace(object):
    def __init__(self, side, overhead_offset=0.1):
        self.overhead_offset = overhead_offset
        self.side = side
        self.arm = HapticMpcArmWrapper(side)
        self.gripper = GripperGraspControllerWrapper(side)
        self.action_server = actionlib.SimpleActionServer('/%s_arm/overhead_place' % self.side, OverheadPlaceAction, self.execute, False)
        self.action_server.start()
        rospy.loginfo("[%s] %s Overhead Place Action Started", rospy.get_name(), self.side.capitalize())

    def execute(self, goal):
        print "received goal"
        self.action_server.publish_feedback(OverheadGraspFeedback("Processing Goal Pose"))
        (setup_pose, overhead_pose, goal_pose) = self.process_path(goal.goal_pose)
        print "moving arm to setup"
        self.action_server.publish_feedback(OverheadGraspFeedback("Moving to Setup Position"))
        self.arm.move_arm(setup_pose, wait=True)
        print "moving arm to overhead"
        self.action_server.publish_feedback(OverheadGraspFeedback("Moving to Overhead Position"))
        self.arm.move_arm(overhead_pose, wait=True)
        self.gripper.release_on_contact()
        print "moving arm to goal"
        self.action_server.publish_feedback(OverheadGraspFeedback("Moving to goal"))
        self.arm.move_arm(goal_pose, wait=True)
        print "opening gripper"
        self.action_server.publish_feedback(OverheadGraspFeedback("Opening Gripper"))
        self.gripper.open_gripper()
        self.action_server.publish_feedback(OverheadGraspFeedback("finished"))
        self.action_server.set_succeeded(OverheadGraspResult('finished'))

    def process_path(self, goal_pose):
        while self.arm.state is None:
            rospy.sleep(0.1)
            rospy.loginfo("[%s] Waiting for %s arm state", rospy.get_name(), self.side)
        current_pose = deepcopy(self.arm.state)
        goal_pose.pose.orientation = Quaternion(0.7071067, 0, -0.7071067, 0)
        overhead_height = goal_pose.pose.position.z + self.overhead_offset
        setup_pose = deepcopy(current_pose)
        setup_pose.header.frame_id = '/base_link'
        setup_pose.pose.position.z = overhead_height
        overhead_pose = deepcopy(goal_pose)
        overhead_pose.pose.position.z = overhead_height
        goal_pose.pose.position.z += 0.03
        return (setup_pose, overhead_pose, goal_pose)


def main():
    rospy.init_node('overhead_grasp')
    r_overhead_grasp = OverheadGrasp('right')
    l_overhead_grasp = OverheadGrasp('left')
    r_overhead_place = OverheadPlace('right')
    l_overhead_place = OverheadPlace('left')
    rospy.spin()
