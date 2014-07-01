#!/usr/bin/env python

import copy
import numpy as np

import roslib; roslib.load_manifest('hrl_base_selection')
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
from tf import transformations as tft
from sensor_msgs.msg import JointState

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_controllers_msgs.msg import JointTrajectoryControllerState

from hrl_haptic_manipulation_in_clutter_msgs.msg import HapticMpcWeights
from hrl_msgs.msg import FloatArray

SETUP_POSE = ( (0.38, 0.14, 0.10), (0., 0., -0.7170, 0.7170) )
#SETUP_POSTURE = [ 0.6352, -0.7099, 0.8311, 4.5575, -2.286, -2.190, 3.1477]
SETUP_POSTURE = [-0.231, -0.4998, 0.64603, -1.28553, -3.6191, -1.8328, 2.31741]


class ReachAction(object):
    def __init__(self):
        self.mpc_goal_pub = rospy.Publisher("/haptic_mpc/goal_pose", PoseStamped)
        self.haptic_weights = rospy.Publisher("haptic_mpc/weights", HapticMpcWeights)
        self.fdbk_pub = rospy.Publisher("wt_log_out", String)
        self.goal_posture = rospy.Publisher("haptic_mpc/joint_trajectory", JointTrajectory,latch=True)

        self.ee_pose_sub = rospy.Subscriber("/haptic_mpc/gripper_pose", PoseStamped, self.ee_pose_cb)
        self.joint_state_sub = rospy.Subscriber('/l_arm_controller/state', JointTrajectoryControllerState, self.joint_state_cb)
        self.ee_pose = None
        self.goal_pose_sub = rospy.Subscriber("~goal_pose", PoseStamped, self.goal_cb)
        self.goal_pose = None

        self.last_progress_time = rospy.Time(0)
        self.last_progress_err = [np.inf, np.pi/2]
        self.progress_timeout = rospy.Duration(10.)
        self.dist_err_thresh = 0.01
        self.posture_err_thresh = 0.1
        self.theta_err_thresh = np.pi/18.

        self.setup_1_passed = False
        self.setup_2_passed = False
        self.preempted = False

        #self.setup_pose = PoseStamped()
        #self.setup_pose.header.frame_id = 'torso_lift_link'
        #self.setup_pose.pose.position = Point(*SETUP_POSE[0])
        #self.setup_pose.pose.orientation = Quaternion(*SETUP_POSE[1])

        self.setup_posture = FloatArray()
        self.setup_posture.data = SETUP_POSTURE

        self.posture_weight = HapticMpcWeights()
        self.posture_weight.position_weight = 0
        self.posture_weight.orient_weight = 0
        self.posture_weight.posture_weight = 1

        self.orient_weight = HapticMpcWeights()
        self.orient_weight.position_weight = 1
        self.orient_weight.orient_weight = 1
        self.orient_weight.posture_weight = 0
        rospy.loginfo("[%s] Ready" %rospy.get_name())


    def pub_feedback(self, msg):
        rospy.loginfo("[%s] %s" % (rospy.get_name(), msg))
        self.fdbk_pub.publish(msg);

    def ee_pose_cb(self, msg):
        self.ee_pose = msg

    def joint_state_cb(self, msg):
        #TODO: Gets joint positions from arm controller state
        self.joint_posture = copy.copy(msg.actual.positions)

    def goal_cb(self, ps_msg):
        self.pub_feedback("Received new goal for arm.")
        self.goal_pose = copy.copy(ps_msg)
        self.preempted = True

    def pose_err(self, ps1, ps2):
        p1 = [ps1.pose.position.x, ps1.pose.position.y, ps1.pose.position.z]
        p2 = [ps2.pose.position.x, ps2.pose.position.y, ps2.pose.position.z]
        d_err = np.linalg.norm(np.subtract(p1,p2))

        q1 = [ps1.pose.orientation.x, ps1.pose.orientation.y,
              ps1.pose.orientation.z, ps1.pose.orientation.w]
        q2 = [ps2.pose.orientation.x, ps2.pose.orientation.y,
              ps2.pose.orientation.z, ps2.pose.orientation.w]
        q1_inv = tft.quaternion_inverse(q1)
        q_err = tft.quaternion_multiply(q2, q1_inv)
        euler_angles = tft.euler_from_quaternion(q_err)
        theta_err = np.linalg.norm(euler_angles)
        return d_err, theta_err

    def posture_err(self, p1, p2):
        print 'p1 is: \n' ,p1
        print 'p2 is: \n', p2
        err = np.linalg.norm(np.subtract(p1,p2))
        print 'posture error is: \n', err
        return err

    def at_goal(self, goal):
        curr = copy.copy(self.ee_pose)
        d_err, ang_err = self.pose_err(curr, goal)
        if (d_err < self.dist_err_thresh) and (ang_err < self.theta_err_thresh):
            return True
        else:
            return False

    def at_start(self, goal):
        curr = copy.copy(self.joint_posture)
        err = self.posture_err(curr, goal)
        return (err < self.posture_err_thresh)

    def progressing(self, goal):
        curr = copy.copy(self.ee_pose)
        d_err, ang_err = self.pose_err(curr, goal)
        if ((d_err < 0.9 * self.last_progress_err[0]) or
            (ang_err < 0.9 * self.last_progress_err[1])):
            self.last_progress_time = rospy.Time.now()
            self.last_progress_err = [d_err, ang_err]
            return True
        elif rospy.Time.now() < self.last_progress_time + self.progress_timeout:
            return True
        return False

    def progressing_to_start(self, goal):
        curr = copy.copy(self.joint_posture)
        err = self.posture_err(curr, goal)
        if (err < 0.97 * self.last_progress_err_posture):
            self.last_progress_time = rospy.Time.now()
            self.last_progress_err_posture = err
            print 'Posture movement made progress! \n'
            return True
        elif rospy.Time.now() < self.last_progress_time + self.progress_timeout:
            print 'Posture movement made no progress, but I havent timed out \n'
            return True
        print 'Posture movement just timed out \n'
        return False

    def run(self):
        while not rospy.is_shutdown() and self.ee_pose is None:
            rospy.sleep(0.5)
        rate_limit = rospy.Rate(20)
        print 'Ready and waiting for a goal! \n'
        while not rospy.is_shutdown():
            rate_limit.sleep()
            if self.goal_pose is None:
                continue
            self.preempted = False
#            s1_goal = copy.copy(self.setup_pose)
#            s1_goal.pose.position.z = self.ee_pose.pose.position.z
#            if not self.reach_to_goal(s1_goal):
#                 print "Past 1st goal"
            self.haptic_weights.publish(self.posture_weight)
            print 'Starting to reach toward setup position \n'
            if not self.reach_to_start(self.setup_posture):
                print "Past 2nd goal"
            self.haptic_weights.publish(self.orient_weight)
            print 'Starting to reach toward goal position \n'
            if self.reach_to_goal(self.goal_pose):
                print "[%s] Reached Goal" % rospy.get_name()
            else:
                print "[%s] Failed to reach Goal" % rospy.get_name()
                self.goal_pose = None

    def reach_to_start(self, goal, freq=50):
        point = JointTrajectoryPoint()
        point.positions = goal.data
        trajectory = JointTrajectory()
        trajectory.points.append(point)
        trajectory.joint_names = ['l_upper_arm_roll_joint',
                                  'l_shoulder_pan_joint',
                                  'l_shoulder_lift_joint',
                                  'l_forearm_roll_joint',
                                  'l_elbow_flex_joint',
                                  'l_wrist_flex_joint',
                                  'l_wrist_roll_joint']
        rate = rospy.Rate(freq)
        self.last_progress_time = rospy.Time.now()
        self.last_progress_err_posture = self.posture_err(goal.data, self.joint_posture)
        trajectory.header.stamp = rospy.Time.now()
        self.goal_posture.publish(trajectory)
        while not rospy.is_shutdown():
            if self.preempted:
                print "[%s] Preempted" % rospy.get_name()
                return False
            if self.at_start(goal.data):
                return True
            if not self.progressing_to_start(goal.data):
                print "[%s] Stopped Progressing" % rospy.get_name()
                return False
            rate.sleep()

    def reach_to_goal(self, goal, freq=50):
        rate = rospy.Rate(freq)
        self.last_progress_time = rospy.Time.now()
        self.last_progress_err = self.pose_err(goal, self.ee_pose)
        goal.header.stamp = rospy.Time.now()
        self.mpc_goal_pub.publish(goal)
        while not rospy.is_shutdown():
            if self.preempted:
                print "[%s] Preempted" % rospy.get_name()
                return False
            if self.at_goal(goal):
                return True
            if not self.progressing(goal):
                print "[%s] Stopped Progressing" % rospy.get_name()
                return False
            rate.sleep()

if __name__=='__main__':
    rospy.init_node("arm_reacher")
    reach = ReachAction()
    reach.run()
