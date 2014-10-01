#!/usr/bin/env python

import sys
from threading import Lock

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
import actionlib
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from  kinematics_msgs.srv import (GetKinematicSolverInfo,
                                  GetPositionIK,
                                  GetPositionIKRequest)
from pr2_controllers_msgs.msg import (JointTrajectoryAction,
                                      JointTrajectoryGoal,
                                      JointTrajectoryControllerState)
import tf

class IKGoalSender(object):
    """A class for performing IK with the PR2 """
    def __init__(self, arm):
        """Initialize IK Services, arm trajectory client, state, and
        feedback"""
        listener = tf.TransformListener()
        self.arm = arm
        self.joint_state_lock = Lock()
        #Setup action client for arm joint trajectories
        self.traj_client = actionlib.SimpleActionClient(self.arm[0]+
                '_arm_controller/joint_trajectory_action',
                JointTrajectoryAction)
        if not self.traj_client.wait_for_server(rospy.Duration(5)):
            rospy.loginfo('[IKGoalSender] Timed Out waiting for '+
                          'JointTrajectoryAction Server')
        #Connect to IK services
        prefix =  'pr2_'+self.arm+'_arm_kinematics/'
        try:
            rospy.loginfo('Waiting for IK services')
            rospy.wait_for_service(prefix+'get_ik_solver_info')
            rospy.wait_for_service(prefix+'get_ik')
            rospy.loginfo("Found IK services")
            self.ik_info_proxy = rospy.ServiceProxy(prefix+"get_ik_solver_info",
                                                    GetKinematicSolverInfo)
            self.ik_info = self.ik_info_proxy().kinematic_solver_info
            self.ik_client = rospy.ServiceProxy(prefix+"get_ik",
                                                GetPositionIK,
                                                True)
        except:
            rospy.logerr("Could not find IK services")

        self.log_pub = rospy.Publisher('log_out', String)
        self.joint_state = {'positions':[],'velocities':[]}
        self.joint_state_received = False
        self.joint_state_sub = rospy.Subscriber(self.arm[0]+
                                                '_arm_controller/state',
                                                JointTrajectoryControllerState,
                                                self.joint_state_cb)
        #Need to come up with a better way of doing this, should have FeedPos
        try:
            self.hold = listener.lookupTransform('/base_link', '/FeedPos', rospy.Time(0))
            self.pose_sub = rospy.Subscriber(self.hold, PoseStamped, self.pose_cb)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("Could not get hold")


    def joint_state_cb(self, js_msg):
        """Update joint positions and velocities"""
        with self.joint_state_lock:
            self.joint_state['positions'] = js_msg.actual.positions[:]
            self.joint_state['velocities'] = js_msg.actual.velocities[:]
        self.joint_state_received = True

    def pose_cb(self, ps):
        """ Perform IK for a goal pose, and send action goal if acheivable"""
        rospy.loginfo('made it to pose_cb')
        if not self.joint_state_received:
            rospy.loginfo('[IKGoalSender] No Joint State Received')
            return
        req = self.form_ik_request(ps)
        ik_goal = self.ik_client(req)
        if ik_goal.error_code.val == 1:
           traj_point = JointTrajectoryPoint()
           traj_point.positions = ik_goal.solution.joint_state.position
           traj_point.velocities = ik_goal.solution.joint_state.velocity
           traj_point.time_from_start = rospy.Duration(1)

           traj = JointTrajectory()
           traj.joint_names = self.ik_info.joint_names
           traj.points.append(traj_point)

           traj_goal = JointTrajectoryGoal()
           traj_goal.trajectory = traj
           #self.traj_client.send_goal(traj_goal)
        else:
            rospy.loginfo('[IKGoalSender] IK Failed: Cannot reach goal')
            self.log_pub.publish(String('IK Failed: Cannot reach goal'))

    def form_ik_request(self, ps):
        """Compose an IK request from ik solver info and pose goal"""
        req = GetPositionIKRequest()
        req.timeout = rospy.Duration(5)
        req.ik_request.ik_link_name = self.ik_info.link_names[-1]
        req.ik_request.pose_stamped = ps
        req.ik_request.ik_seed_state.joint_state.name = self.ik_info.joint_names
        with self.joint_state_lock:
            positions = self.joint_state['positions'][:]
            velocities = self.joint_state['velocities'][:]
        req.ik_request.ik_seed_state.joint_state.position = positions
        req.ik_request.ik_seed_state.joint_state.velocity = velocities
        return req

if __name__=='__main__':
    rospy.init_node('ik_goal_relay')
    sender = IKGoalSender(sys.argv[sys.argv.index('--arm')+1])
    rospy.spin()
