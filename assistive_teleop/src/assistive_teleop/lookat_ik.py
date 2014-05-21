#!/usr/bin/env python

import sys
import copy

import numpy as np
import openravepy as op

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_controllers_msgs.msg import JointTrajectoryControllerState
from tf import TransformListener, LookupException, ConnectivityException, ExtrapolationException

from assistive_teleop.srv import LookatIk, LookatIkResponse

FORCE_REBUILD = False

class CameraLookatIk(object):
    def __init__(self, robotfile, manipname, freeindices):
        self.side = side
        self.env = op.Environment()
        self.tfl = TransformListener()
        #env.SetViewer('qtcoin')
        self.env.Load(robotfile)
        self.robot = self.env.GetRobots()[0]
        self.base_frame = self.robot.GetLinks()[0].GetName()
        self.manip = self.robot.SetActiveManipulator(manipname)
        self.arm_joint_names = [self.robot.GetJoints()[ind].GetName() for ind in self.manip.GetArmJoints()]
        self.arm_indices = self.manip.GetArmIndices()
        self.joint_angles = None
        freejoints = [self.robot.GetJoints()[ind].GetName() for ind in freeindices]
        self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot,
                                                                             iktype=op.IkParameterization.Type.Lookat3D,
                                                                             forceikfast=True,
                                                                             freeindices=freeindices,
                                                                             freejoints=freejoints)
        self.ikmodel.freeinc = [0.15, 0.15, 0.15]

        if not self.ikmodel.load():
            rospy.loginfo("[%s] LookatIK Model not available: Generating" %rospy.get_name())
            self.ikmodel.generate()
            self.ikmodel.save()
        elif FORCE_REBUILD:
            rospy.loginfo("[%s] Force rebuild of LookatIK Model: Generating" %rospy.get_name())
            self.ikmodel.generate()
            self.ikmodel.save()
#            rospy.loginfo("[%s] Finished rebuilding and saving new LookatIK Model. Closing." %rospy.get_name())
#            sys.exit()
        else:
            rospy.loginfo("[%s] Lookat IK Model Loaded" %rospy.get_name())
        rospy.Subscriber("/joint_states", JointState, self.joint_state_cb)
        rospy.Service("~/%s/lookat_ik" %manipname, LookatIk, self.lookat_ik_cb)

    def joint_state_cb(self, msg):
        self.joint_angles = [msg.position[msg.name.index(jnt)] for jnt in self.arm_joint_names]
        self.robot.SetDOFValues(self.joint_angles, self.arm_indices)

    def lookat_ik(self, pt):
        if (pt.header.frame_id != self.base_frame):
            try:
                now = rospy.Time.now() + rospy.Duration(0.10)
                self.tfl.waitForTransform(pt.header.frame_id, self.base_frame, now, rospy.Duration(1.0))
                pt.header.stamp = now
                pt = self.tfl.transformPoint(self.base_frame, pt)
            except (LookupException, ConectivityException, ExtrapolationException):
                rospy.logwarn("[%s] TF Error tranforming point from %s to %s" %(rospy.get_name(),
                                                                                pt.header.frame_id,
                                                                                self.base_frame))
        target = np.array([pt.point.x,
                           pt.point.y,
                           pt.point.z])
       # sol = self.manip.FindIKSolution(op.IkParameterization(target, op.IkParameterizationType.Lookat3D),
       #                                                          op.IkFilterOptions.CheckEnvCollisions)
        sols = self.manip.FindIKSolutions(op.IkParameterization(target, op.IkParameterizationType.Lookat3D),
                                                                 op.IkFilterOptions.CheckEnvCollisions)
        print "Found %d solutions" %len(sols)
        curr_angles = copy.copy(self.joint_angles)
        sol = sols[np.argmin(np.sum(np.subtract(curr_angles, sols)**2, 1))]
        rospy.loginfo("[%s] Found LookatIK solution." %rospy.get_name())
        return sol

    def lookat_ik_cb(self, req):
        sol = self.lookat_id(req.point)
        resp = LookatIkResponse()
        resp.joint_names = self.arm_joint_names
        resp.joint_angles = sol
        return resp

class CameraPointer(object):
    def __init__(self, side, robotfile, manipname, freeindices):
        self.side = side
        self.joint_names = None
        self.joint_angles_act = None
        self.joint_angles_des = None
        self.camera_ik = CameraLookatIk(robotfile, manipname, freeindices)

        self.joint_state_sub = rospy.Subscriber('/%s_arm_controller/state' %self.side, JointTrajectoryControllerState, self.joint_state_cb)
        self.joint_traj_pub = rospy.Publisher('/%s_arm_controller/command' %self.side, JointTrajectory)
        while self.joint_names is None and not rospy.is_shutdown():
            rospy.sleep(0.5)
            rospy.loginfo("[%s] Waiting for joint state from arm controller." %rospy.get_name())
        self.target_sub = rospy.Subscriber('%s/lookat_ik/goal' %rospy.get_name(), PointStamped, self.goal_cb)
        rospy.loginfo("[%s] Ready." %rospy.get_name())

    def goal_cb(self, pt_msg):
        if self.joint_names is None or self.joint_angles_des is None:
            rospy.logwarn("[%s] Joint state for arm not received yet. Ignoring." %rospy.get_name())
            return
        rospy.loginfo("[%s] New LookatIK goal received." %rospy.get_name())
        iksol = self.camera_ik.lookat_ik(pt_msg) #Get IK Solution
        # Start with current angles, then replace angles in camera IK with IK solution
        jtp = JointTrajectoryPoint()
        jtp.positions = list(copy.copy(self.joint_angles_des))
        for i, joint_name in enumerate(self.camera_ik.arm_joint_names):
            jtp.positions[self.joint_names.index(joint_name)] = iksol[i]
        jtp.time_from_start = rospy.Duration(2.0)
        jt = JointTrajectory()
        jt.joint_names = copy.copy(self.joint_names)
        jt.points.append(jtp)
        self.joint_traj_pub.publish(jt)
        rospy.loginfo("[%s] Sending Joint Angles." %rospy.get_name())

    def joint_state_cb(self, jtcs):
        if self.joint_names is None:
            self.joint_names = jtcs.joint_names
        self.joint_angles_act = jtcs.actual.positions
        self.joint_angles_des = jtcs.desired.positions

if __name__=='__main__':
    robot = "PR2"
    side = 'r'
    freeindices = [27,28,30]

    if robot == "PR2":
        robotfile = "robots/pr2-beta-static.zae"
        manipname = "rightarm_camera" if side == 'r' else "leftarm_camera"
    else:
        rospy.logerr("[%s] Only the PR2 is currently supported")
        sys.exit()

    rospy.init_node(manipname)

    camera_pointer = CameraPointer(side, robotfile, manipname, freeindices)
    rospy.spin()
