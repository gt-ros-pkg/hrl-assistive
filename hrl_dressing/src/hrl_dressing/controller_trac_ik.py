import os, sys, roslib, rospy, tf, actionlib, PyKDL

import numpy as np
import math as m
from geometry_msgs.msg import PointStamped, PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectoryPoint
from pr2_controllers_msgs.msg import JointTrajectoryGoal, JointTrajectoryAction, JointTrajectoryControllerState, JointControllerState, Pr2GripperCommandAction, Pr2GripperCommandGoal, PointHeadAction, PointHeadGoal
from pykdl_utils.kdl_kinematics import create_kdl_kin # TODO: Remove this dependency
from trac_ik_python.trac_ik import IK


class Controller:
    def __init__(self, frame, verbose=False):
        self.frame = frame
        self.verbose = verbose
        self.tf = tf.TransformListener()
        self.leftGripperAngle = 0

        # Set all arm movement parameters
        self.rightJointLimitsMax = np.radians([32.349, 74.2725, 37.242, -0.01, 360.0, 0.01, 360.0])
        self.rightJointLimitsMin = np.radians([-122.349, -20.26, -214.859, -121.54, -360.0, -114.59, -360.0])
        #self.rightJointLimitsMax = np.radians([26.0, 68.0, 41.0, 0.01, 180.0, 0.01, 180.0])
        #self.rightJointLimitsMin = np.radians([-109.0, -24.0, -220.0, -132.0, -180.0, -120.0, -180.0])
        self.leftJointLimitsMax = np.radians([122.349, 74.2725, 214.859, 0.01, 360.0, 0.01, 360.0]) # TODO: Update based on new soft limits for right arm
        self.leftJointLimitsMin = np.radians([-32.349, -20.26, -37.242, -121.54, -360.0, -114.59, -360.0]) # TODO: Update based on new soft limits for right arm
        # self.initRightJointGuess = np.array([-0.236, 0.556, -0.091, -1.913, -1.371, -1.538, -3.372])
        # self.initLeftJointGuess = np.array([0.203, 0.846, 1.102, -1.671, 5.592, -1.189, -3.640])
        self.initRightJointGuess = np.array([0.13, 0.2, 0.63, -3.23, -2.0, -0.96, -0.1]) # TODO: incorrect
        self.initLeftJointGuess = np.array([0.74, 0.03, 1.05, -0.32, -1.63, -1.28, 4.22]) # TODO: incorrect

        self.initJoints()

        self.rightArmClient = actionlib.SimpleActionClient('/r_arm_controller/joint_trajectory_action', JointTrajectoryAction)
        self.rightArmClient.wait_for_server()
        self.rightGripperClient = actionlib.SimpleActionClient('/r_gripper_controller/gripper_action', Pr2GripperCommandAction)
        self.rightGripperClient.wait_for_server()
        self.leftArmClient = actionlib.SimpleActionClient('/l_arm_controller/joint_trajectory_action', JointTrajectoryAction)
        self.leftArmClient.wait_for_server()
        self.leftGripperClient = actionlib.SimpleActionClient('/l_gripper_controller/gripper_action', Pr2GripperCommandAction)
        self.leftGripperClient.wait_for_server()
        self.pointHeadClient = actionlib.SimpleActionClient('/head_traj_controller/point_head_action', PointHeadAction)
        self.pointHeadClient.wait_for_server()

        # Initialize KDL for inverse kinematics       
        #self.rightArmKdl.joint_safety_lower = self.rightJointLimitsMin
        #self.rightArmKdl.joint_safety_upper = self.rightJointLimitsMax
        self.rightArmKdl_dist = IK(self.frame, "r_gripper_tool_frame", timeout=0.04, solve_type='Distance')
        self.rightArmKdl_speed = IK(self.frame, "r_gripper_tool_frame", timeout=0.01, solve_type='Speed')

        #self.leftArmKdl.joint_safety_lower = self.leftJointLimitsMin
        #self.leftArmKdl.joint_safety_upper = self.leftJointLimitsMax
        self.leftArmKdl_dist = IK(self.frame, "l_gripper_tool_frame", timeout=0.04, solve_type='Distance')
        self.leftArmKdl_speed = IK(self.frame, "l_gripper_tool_frame", timeout=0.01, solve_type='Speed')

        self.rightJointPositions = None
        self.leftJointPositions = None
        rospy.Subscriber('/r_arm_controller/state', JointTrajectoryControllerState, self.rightArmState)
        rospy.Subscriber('/l_arm_controller/state', JointTrajectoryControllerState, self.leftArmState)

    def rightArmState(self, msg):
        self.rightJointPositions = msg.actual.positions

    def leftArmState(self, msg):
        self.leftJointPositions = msg.actual.positions

    def initJoints(self):
        msg = rospy.wait_for_message('/r_arm_controller/state', JointTrajectoryControllerState)
        self.rightJointNames = msg.joint_names
        self.initRightJointPositions = msg.actual.positions
        msg = rospy.wait_for_message('/l_arm_controller/state', JointTrajectoryControllerState)
        self.leftJointNames = msg.joint_names
        self.initLeftJointPositions = msg.actual.positions
        if self.verbose:
            print 'Right joint names:', self.rightJointNames
            print 'Left joint names:', self.leftJointNames

    def position_quat_to_pose(self, pos, quat):
        ps = Pose()
        ps.position.x, ps.position.y, ps.position.z = pos
        ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w = quat
        return ps

    def arrayToPose(self, poseArray):
        frameData = PyKDL.Frame()
        # Pose array to KDL frame
        p = PyKDL.Vector(*poseArray[:3])
        M = PyKDL.Rotation.RPY(*poseArray[3:])
        poseFrame = PyKDL.Frame(M,p)

        # Frame conversion
        pos = frameData * poseFrame.p
        rot = frameData.M * poseFrame.M
        poseFrame = PyKDL.Frame(rot, pos)

        # KDL frame to Pose
        ps = Pose()
        ps.position.x, ps.position.y, ps.position.z = poseFrame.p
        ps.orientation.x, ps.orientation.y, ps.orientation.z, ps.orientation.w = poseFrame.M.GetQuaternion()
        return ps

    def setJointGuesses(self, rightGuess=None, leftGuess=None):
        if rightGuess is not None:
            self.initRightJointGuess = rightGuess
        if leftGuess is not None:
            self.initLeftJointGuess = leftGuess

    def getGripperPosition(self, rightArm=True, this_frame=None):
        # now = rospy.Time.now()
        # self.tf.waitForTransform(self.frame, ('r' if rightArm else 'l') + '_gripper_tool_frame', now, rospy.Duration(10.0))
        # Return the most revent transformation
        if this_frame is None:
            this_frame = self.frame
        if rightArm:
            return self.tf.lookupTransform(this_frame, 'r_gripper_tool_frame', rospy.Time(0))
        else:
            return self.tf.lookupTransform(this_frame, 'l_gripper_tool_frame', rospy.Time(0))

    def grip(self, openGripper=True, maxEffort=200.0, rightArm=True, miniOpen=False, stopMovement=False):
        msg = Pr2GripperCommandGoal()
        if stopMovement:
            pos = rospy.wait_for_message('/r_gripper_controller/state', JointControllerState).process_value
            msg.command.position = pos
        else:
            msg.command.position = 0.1 if openGripper else (0.005 if miniOpen else 0.0)
        msg.command.max_effort = maxEffort if stopMovement or not openGripper else -1.0
        if rightArm:
            self.rightGripperClient.send_goal(msg)
        else:
            self.leftGripperClient.send_goal(msg)
        # self.rightGripperClient.send_goal_and_wait(msg)

    def rotateGripperWrist(self, angle):
        self.leftGripperAngle += angle
        rotatedWristAngles = list(self.initLeftJointPositions)
        rotatedWristAngles[-1] += self.leftGripperAngle
        self.moveToJointAngles(rotatedWristAngles, timeout=2.0, wait=True, rightArm=False)

    # Move using IK and joint trajectory controller
    # Attach new pose to a frame
    def moveGripperTo(self, position, quaternion=None, rollpitchyaw=None, timeout=1, wait=False, rightArm=True, useInitGuess=False, ret=False, this_frame=None):
        # TODO: Repalce this with standard PR2 inverse kinematics
        # TODO: Repalce this with standard PR2 inverse kinematics

        # Create a pose message from the desired position and orientation
        if rollpitchyaw is None and quaternion is None:
            rollpitchyaw = [-np.pi, 0.0, 0.0]
            rollpitchyaw = [0.0, 0.0, 0.0]
            poseData = list(position) + list(rollpitchyaw)
            pose = self.arrayToPose(poseData)
        elif quaternion is not None:
            pose = self.position_quat_to_pose(position, quaternion)
        elif rollpitchyaw is not None:
            poseData = list(position) + list(rollpitchyaw)
            pose = self.arrayToPose(poseData)
        if this_frame is None:
            this_frame = self.frame

        # Create a PoseStamped message and perform transformation to given frame
        ps = PoseStamped()
        ps.header.frame_id = this_frame
        ps.pose.position = pose.position
        ps.pose.orientation = pose.orientation
        #print 'ps before transform', ps
        #ps = self.tf.transformPose('torso_lift_link', ps)
        ps = self.tf.transformPose(self.frame, ps)
        #print 'ps after transform\n', ps
        
               # get_ik(qinit,
               #        x, y, z,
               #        rx, ry, rz, rw,
               #        bx=1e-5, by=1e-5, bz=1e-5,
               #        brx=1e-3, bry=1e-3, brz=1e-3)

        # Perform IK
        if rightArm:
            ikGoal = self.rightArmKdl_dist.get_ik(self.initRightJointGuess if useInitGuess else self.rightJointPositions,
                                                  ps.pose.position.x, ps.pose.position.y, ps.pose.position.z,
                                                  ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w,
                                                  0.00001, 0.00001, 0.00001, 0.001, 0.001, 0.001            )
            if ikGoal is None and False:
                print 'using speed'
                ikGoal = self.rightArmKdl_speed.get_ik(self.initRightJointGuess if useInitGuess else self.rightJointPositions,
                                                       ps.pose.position.x, ps.pose.position.y, ps.pose.position.z,
                                                       ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w)

            # ikGoal = self.rightArmKdl.inverse(ps.pose, q_guess=self.initRightJointGuess if useInitGuess else self.rightJointPositions, min_joints=self.rightJointLimitsMin, max_joints=self.rightJointLimitsMax)
            #print 'rightjointpositions\n', self.rightJointPositions
#            print 'ikGoal1', ikGoal
#            ikGoal = self.rightArmKdl.inverse_search(ps.pose, timeout=timeout)
#            print 'ikgoal2:', ikGoal
        else:
            ikGoal = self.leftArmKdl_dist.get_ik(self.initLeftJointGuess if useInitGuess else self.rightJointPositions,
                                                 ps.pose.position.x, ps.pose.position.y, ps.pose.position.z,
                                                 ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w)
            if ikGoal is None and False:
                ikGoal = self.leftArmKdl_speed.get_ik(self.initLeftJointGuess if useInitGuess else self.rightJointPositions,
                                                      ps.pose.position.x, ps.pose.position.y, ps.pose.position.z,
                                                      ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w)
            
            # ikGoal = self.leftArmKdl.inverse(ps.pose, q_guess=self.initLeftJointGuess if useInitGuess else self.leftJointPositions, min_joints=self.leftJointLimitsMin, max_joints=self.leftJointLimitsMax)

        if ikGoal is not None:
            if not ret:
                #print 'current joints:', self.rightJointPositions
                #print 'ikGoal:', ikGoal
                #print 'diff:', ikGoal - self.rightJointPositions
                # max_diff = np.degrees(np.max(np.abs(ikGoal - self.rightJointPositions)[[0,1,2,3,5]]))  # This is to ignore the infinite roll joints
                max_diff = np.degrees(np.max(np.abs(np.array(ikGoal) - np.array(self.rightJointPositions))))  # This considers all joints
                #print 'max joint angle (degrees) change requested:', max_diff
                #if max_diff > 5.:
                #    if max_diff/20. > timeout:
                #        print 'slowing down move'
                timeout = np.max([timeout, max_diff/20.])
                self.moveToJointAngles(ikGoal, timeout=timeout, wait=wait, rightArm=rightArm)
                return list(ikGoal), timeout
            else:
                return list(ikGoal), 0.
        else:
            print 'IK failed'
            return None, 0.

    def moveToJointAngles(self, jointStates, timeout=1, wait=False, rightArm=True):
        # Create and send trajectory message for new joint angles
        trajMsg = JointTrajectoryGoal()
        trajPoint = JointTrajectoryPoint()
        trajPoint.positions = jointStates
        trajPoint.time_from_start = rospy.Duration(timeout)
        trajMsg.trajectory.points.append(trajPoint)
        trajMsg.trajectory.joint_names = self.rightJointNames if rightArm else self.leftJointNames
        if not wait:
            if rightArm:
                self.rightArmClient.send_goal(trajMsg)
            else:
                self.leftArmClient.send_goal(trajMsg)
        else:
            if rightArm:
                self.rightArmClient.send_goal_and_wait(trajMsg)
            else:
                self.leftArmClient.send_goal_and_wait(trajMsg)

    def lookAt(self, pos, sim=False):
        goal = PointHeadGoal()

        point = PointStamped()
        point.header.frame_id = self.frame
        point.point.x = pos[0]
        point.point.y = pos[1]
        point.point.z = pos[2]
        goal.target = point

        # Point using kinect frame
        goal.pointing_frame = 'head_mount_kinect_rgb_link'
        if sim:
            goal.pointing_frame = 'high_def_frame'
        goal.pointing_axis.x = 1
        goal.pointing_axis.y = 0
        goal.pointing_axis.z = 0
        goal.min_duration = rospy.Duration(1.0)
        goal.max_velocity = 1.0

        self.pointHeadClient.send_goal_and_wait(goal)

    def printJointStates(self):
        try:
            # now = rospy.Time.now()
            # self.tf.waitForTransform(self.frame, 'r_gripper_tool_frame', now, rospy.Duration(10.0))
            self.tf.waitForTransform(self.frame, 'r_gripper_tool_frame', rospy.Time(), rospy.Duration(10.0))
            currentRightPos, currentRightOrient = self.tf.lookupTransform(self.frame, 'r_gripper_tool_frame', rospy.Time(0))

            print 'Right positions:', currentRightPos, currentRightOrient
            print 'Right joint positions:', self.rightJointPositions

            # now = rospy.Time.now()
            # self.tf.waitForTransform(self.frame, 'l_gripper_tool_frame', now, rospy.Duration(10.0))
            currentLeftPos, currentLeftOrient = self.tf.lookupTransform(self.frame, 'l_gripper_tool_frame', rospy.Time(0))

            print 'Left positions:', currentLeftPos, currentLeftOrient
            print 'Left joint positions:', self.leftJointPositions

            # print 'Right gripper state:', rospy.wait_for_message('/r_gripper_controller/state', JointControllerState)
        except tf.ExtrapolationException:
            print 'No transformation available! Failing to record this time step.'


