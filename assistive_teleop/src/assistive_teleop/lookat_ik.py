#!/usr/bin/env python

import sys

import numpy as np
import openravepy as op

import roslib; roslib.load_manifest('assistive_teleop')
import rospy
from sensor_msgs.msg import JointState

from assistive_teleop.srv import LookatIk, LookatIkResponse

class CameraLookatIk(object):
    def __init__(self, robotfile, manipname, freeindices):
        self.side = side
        self.env = op.Environment()
        #env.SetViewer('qtcoin')
        self.env.Load(robotfile)
        self.robot = self.env.GetRobots()[0]
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
        if not self.ikmodel.load():
            rospy.loginfo("[%s] LookatIK Model not available: Generating" %rospy.get_name())
            self.ikmodel.generate()
            self.ikmodel.save()
        else:
            rospy.loginfo("[%s] Lookat IK Model Loaded" %rospy.get_name())
        rospy.Subscriber("/joint_states", JointState, self.joint_state_cb)
        rospy.Service("~/%s" %self.manipname, LookatIk, self.lookat_ik_cb)

    def joint_state_cb(self, msg):
        self.joint_angles = [msg.position[msg.name.index(jnt)] for jnt in self.arm_joint_names]
        self.robot.SetDOFValues(self.joint_angles, self.arm_indices)

    def lookat_ik_cb(self, req):
        if (req.header.frame_id != self.robot.GetLinks()[0].GetName()):
            return [0]*5
        target = np.array([req.point.x,
                           req.point.y,
                           req.point.z])
        sol = ikmodel.manip.FindIKSolution(op.IkParameterization(target, op.IkParameterizationType.Lookat3D),
                                                                 op.IkFilterOptions.CheckEnvCollisions)
        rospy.loginfo("[%s] Found LookatIK solution.")
        return sol

if __name__=='__main__':
    rospy.init_node("lookat_ik")
    robot = "PR2"
    side = 'r'
    freeindices = [27,28,30]

    if robot == "PR2":
        robotfile = "robots/pr2-beta-static.zae"
        manipname = "rightarm_camera" if side == 'r' else "leftarm_camera"
    else:
        rospy.logerr("[%s] Only the PR2 is currently supported")
        sys.exit()

    camera_ik = CameraLookatIk(robotfile, manipname, freeindices)
    while not rospy.is_shutdown():
        print camera_ik.joint_angles
        rospy.sleep(1)
