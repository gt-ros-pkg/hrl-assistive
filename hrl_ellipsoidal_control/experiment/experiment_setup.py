#! /usr/bin/python

import numpy as np
import sys

import roslib
roslib.load_manifest('hrl_pr2_arms')
import rospy
import actionlib
from pr2_controllers_msgs.msg import SingleJointPositionAction, SingleJointPositionGoal
from pr2_controllers_msgs.msg import Pr2GripperCommandAction, Pr2GripperCommandGoal
from pr2_controllers_msgs.msg import PointHeadAction, PointHeadGoal

from hrl_generic_arms.pose_converter import PoseConverter
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher
from hrl_pr2_arms.pr2_arm import create_pr2_arm, PR2ArmJointTrajectory
from hrl_pr2_arms.pr2_arm import PR2ArmJTransposeTask

joint_deltas = [0.01, 0.01, 0.01, 0.012, 0.01, 0.01, 0.01]
SETUP_ANGLES = [0, 0, np.pi/2, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2]


class ExperimentSetup(object):
    def __init__(self):
        self.ctrl_switcher = ControllerSwitcher()
        self.torso_sac = actionlib.SimpleActionClient('torso_controller/position_joint_action',
                                                      SingleJointPositionAction)
        self.torso_sac.wait_for_server()
        self.head_point_sac = actionlib.SimpleActionClient(
                                                '/head_traj_controller/point_head_action',
                                                PointHeadAction)
        self.head_point_sac.wait_for_server()
        rospy.loginfo("[experiment_setup] ExperimentSetup ready.")

    def point_head(self):
        print "Pointing head"
        head_goal = PointHeadGoal()
        head_goal.target = PoseConverter.to_point_stamped_msg('/torso_lift_link',
                                                              np.mat([1., 0.4, 0.]).T,
                                                              np.mat(np.eye(3)))
        head_goal.target.header.stamp = rospy.Time()
        head_goal.min_duration = rospy.Duration(3.)
        head_goal.max_velocity = 0.2
        self.head_point_sac.send_goal_and_wait(head_goal)

    def adjust_torso(self):
        # move torso up
        tgoal = SingleJointPositionGoal()
        tgoal.position = 0.238  # all the way up is 0.300
        tgoal.min_duration = rospy.Duration( 2.0 )
        tgoal.max_velocity = 1.0
        self.torso_sac.send_goal_and_wait(tgoal)

    def mirror_arm_setup(self):
        self.ctrl_switcher.carefree_switch('r', 'r_joint_controller_mirror', 
                                "$(find hrl_ellipsoidal_control)/params/mirror_params.yaml")
        rospy.sleep(1)
        arm = create_pr2_arm('r', PR2ArmJointTrajectory, 
                             controller_name="r_joint_controller_mirror")
        arm.set_ep([-0.26880036055585677, 0.71881299774143248, 
                    -0.010187938126536471, -1.43747589322259, 
                    -12.531293698878677, -0.92339874393497123, 
                    3.3566322715405432], 5)
        rospy.sleep(6)

    def tool_arm_setup(self):
        self.ctrl_switcher.carefree_switch('l', '%s_arm_controller', None)
        rospy.sleep(1)
        joint_arm = create_pr2_arm('l', PR2ArmJointTrajectory)
        setup_angles = SETUP_ANGLES
        joint_arm.set_ep(setup_angles, 5)
        rospy.sleep(6)

    def mirror_mannequin(self):
        arm = create_pr2_arm('r', PR2ArmJointTrajectory, 
                             controller_name="r_joint_controller_mirror")
        r = rospy.Rate(10)
        q_act_last = arm.get_joint_angles()
        while not rospy.is_shutdown():
            q_act = arm.get_joint_angles()
            q_ep = arm.get_ep()
            new_ep = q_ep.copy()
            for i in range(7):
                if np.fabs(q_act[i] - q_act_last[i]) > joint_deltas[i]:
                    new_ep[i] = q_act[i]
            arm.set_ep(new_ep, 0.1)
            q_act_last = q_act
            r.sleep()

    def cart_controller_setup(self):
        self.ctrl_switcher.carefree_switch('l', '%s_cart_jt_task', 
                                           "$(find hrl_rfh_fall_2011)/params/l_jt_task_shaver45.yaml") 
        self.cart_arm = create_pr2_arm('l', PR2ArmJTransposeTask, 
                                       controller_name='%s_cart_jt_task', 
                                       end_link="%s_gripper_shaver45_frame")
        rospy.sleep(2)
        setup_angles = SETUP_ANGLES
        self.cart_arm.set_posture(setup_angles)
        self.cart_arm.set_gains([200, 800, 800, 80, 80, 80], [15, 15, 15, 1.2, 1.2, 1.2])

def main():
    rospy.init_node("experiment_setup")

    setup = ExperimentSetup()
    if sys.argv[1] == 'setup':
        setup.point_head()
        setup.adjust_torso()
        setup.mirror_arm_setup()
        setup.tool_arm_setup()
        setup.cart_controller_setup()
        return

    if sys.argv[1] == 'mirror':
        setup.mirror_mannequin()
        return


if __name__ == "__main__":
    main()
