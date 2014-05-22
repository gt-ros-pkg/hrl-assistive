#!/usr/bin/env python

import sys
import copy

import numpy as np
import openravepy as op

import roslib
roslib.load_manifest('assistive_teleop')
import rospy
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_controllers_msgs.msg import JointTrajectoryControllerState
from tf import TransformListener, LookupException, ConnectivityException, ExtrapolationException

from assistive_teleop.srv import LookatIk, LookatIkResponse

FORCE_REBUILD = False
TEST = False

class IkError(Exception):
    pass


class CameraLookatIk(object):
    def __init__(self, robotfile, manipname, freeindices,
                 freeinc=[0.75, 0.75, 0.75],
                 weights = [1.0, 0.95, 0.8, 0.66, 0.2]):
        self.side = side
        self.weights=weights
        self.env = op.Environment()
        self.env.Load(robotfile)
        self.robot = self.env.GetRobots()[0]
        self.base_frame = self.robot.GetLinks()[0].GetName()
        self.manip = self.robot.SetActiveManipulator(manipname)
        self.arm_joint_names = [self.robot.GetJoints()[ind].GetName() for ind in self.manip.GetArmJoints()]
        self.arm_indices = self.manip.GetArmIndices()
        self.joint_angles = [0]*len(self.arm_indices)
        freejoints = [self.robot.GetJoints()[ind].GetName() for ind in freeindices]
        self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot,
                                                                             iktype=op.IkParameterization.Type.Lookat3D,
                                                                             forceikfast=True,
                                                                             freeindices=freeindices,
                                                                             freejoints=freejoints)
        self.ikmodel.freeinc = freeinc

        if not self.ikmodel.load():
            rospy.loginfo("[%s] LookatIK Model not available: Generating" % rospy.get_name())
            self.ikmodel.generate()
            self.ikmodel.save()
        elif FORCE_REBUILD:
            rospy.loginfo("[%s] Force rebuild of LookatIK Model: Generating" % rospy.get_name())
            self.ikmodel.generate()
            self.ikmodel.save()
        else:
            rospy.loginfo("[%s] Lookat IK Model Loaded" % rospy.get_name())

    def verify_solution(self, target, solution):
        self.robot.SetDOFValues(solution, self.arm_indices)
        cam_tf = self.manip.GetEndEffectorTransform()
        target_in_cam = op.transformInversePoints(cam_tf, np.array([target]))[0]
        err = np.linalg.norm(target_in_cam[:2])
        return True if err < 0.001 else False

    def lookat_ik(self, target, current_angles=None):
        # Update robot state to current angles (or best guess from current model position)
        if current_angles is None:
            current_angles = self.robot.GetDOFValues(self.arm_indices)
        self.robot.SetDOFValues(current_angles, self.arm_indices)
        # Solve IK
        sols = self.manip.FindIKSolutions(op.IkParameterization(target, op.IkParameterizationType.Lookat3D),
                                                                 op.IkFilterOptions.CheckEnvCollisions)
        if not np.any(sols):
            raise IkError("[{0}] Could not find an IK solution.".format(rospy.get_name()))
        # Weed out solutions which don't point directly at the target point
        good_sols = np.array([s for s in sols if self.verify_solution(target, s)])
        # Get the solution with the (weighted) nearest joint angles to the current angles
        # (Argmin over weighted sum of squared error)
        rospy.loginfo("[%s] Found LookatIK solution." % rospy.get_name())
        return good_sols[np.argmin(np.sum(self.weights * np.sqrt((good_sols-current_angles) ** 2), 1))]


class CameraPointer(object):
    def __init__(self, side, camera_ik):
        self.side = side
        self.camera_ik = camera_ik
        self.joint_names = self.joint_angles_act = self.joint_angles_des = None
        self.tfl = TransformListener()
        self.joint_state_sub = rospy.Subscriber('/{0}_arm_controller/state'.format(self.side), JointTrajectoryControllerState, self.joint_state_cb)
        self.joint_traj_pub = rospy.Publisher('/{0}_arm_controller/command'.format(self.side), JointTrajectory)
        # Wait for joint information to become available
        while self.joint_names is None and not rospy.is_shutdown():
            rospy.sleep(0.5)
            rospy.loginfo("[{0}] Waiting for joint state from arm controller.".format(rospy.get_name()))
        #Set rate limits on a per-joint basis
        self.max_vel_rot = [np.pi]*len(self.joint_names)
        self.target_sub = rospy.Subscriber('{0}/lookat_ik/goal'.format(rospy.get_name()), PointStamped, self.goal_cb)
        rospy.loginfo("[{0}] Ready.".format(rospy.get_name()))

    def joint_state_cb(self, jtcs):
        if self.joint_names is None:
            self.joint_names = jtcs.joint_names
        self.joint_angles_act = jtcs.actual.positions
        self.joint_angles_des = jtcs.desired.positions

    def goal_cb(self, pt_msg):
        rospy.loginfo("[{0}] New LookatIK goal received.".format(rospy.get_name()))
        if (pt_msg.header.frame_id != self.camera_ik.base_frame):
            try:
                now = pt_msg.header.stamp
                self.tfl.waitForTransform(pt_msg.header.frame_id,
                                          self.camera_ik.base_frame,
                                          now, rospy.Duration(1.0))
                pt_msg = self.tfl.transformPoint(self.camera_ik.base_frame, pt_msg)
            except (LookupException, ConnectivityException, ExtrapolationException):
                rospy.logwarn("[{0}] TF Error tranforming point from {1} to {2}".format(rospy.get_name(),
                                                                                        pt_msg.header.frame_id,
                                                                                        self.camera_ik.base_frame))
        target = np.array([pt_msg.point.x, pt_msg.point.y, pt_msg.point.z])
        # Get IK Solution
        current_angles = list(copy.copy(self.joint_angles_act))
        iksol = self.camera_ik.lookat_ik(target, current_angles[:len(self.camera_ik.arm_indices)])
        # Start with current angles, then replace angles in camera IK with IK solution
        # Use desired joint angles to avoid sagging over time
        jtp = JointTrajectoryPoint()
        jtp.positions = list(copy.copy(self.joint_angles_des))
        for i, joint_name in enumerate(self.camera_ik.arm_joint_names):
            jtp.positions[self.joint_names.index(joint_name)] = iksol[i]
        deltas = np.abs(np.subtract(jtp.positions, current_angles))
        duration = np.max(np.divide(deltas, self.max_vel_rot))
        jtp.time_from_start = rospy.Duration(duration)
        jt = JointTrajectory()
        jt.joint_names = self.joint_names
        jt.points.append(jtp)
        rospy.loginfo("[{0}] Sending Joint Angles.".format(rospy.get_name()))
        self.joint_traj_pub.publish(jt)



def test(camera_ik, ntrials):
    camera_ik.env.SetViewer('qtcoin')
    pointer_vec = np.array([0,0,5])
    xbounds, ybounds, zbounds = [-3, 3], [-3, 3], [-3, 3]
    for i in xrange(ntrials):
        target = np.array([np.random.uniform(*xbounds),
                           np.random.uniform(*ybounds),
                           np.random.uniform(*zbounds)])
        print "Target #{0:d}: ({1:.2f}, {2:.2f}, {3:.2f})".format(i,*target)
        # Get IK Solution
        try:
            iksol = camera_ik.lookat_ik(target)
        except IkError as e:
            rospy.logerr("[{0}] Failed: {1}.".format(rospy.get_name(), e.message))
            continue
        # Put robot in configuration, draw dots at camera and target
        # Draw arrows from camera to target, and on camera z axis (should align through target)
        camera_ik.robot.SetDOFValues(iksol, camera_ik.arm_indices)
        cam_tf = camera_ik.manip.GetEndEffectorTransform()
        cam_pt = cam_tf[:3,3]
        target_handle = camera_ik.env.plot3(np.array([target]), 15, np.array([[1.,0.,0.,1.]]))
        cam_handle = camera_ik.env.plot3(np.array([cam_pt]), 15, np.array([[0.,1.,0.,1.]]))
        pointed_vec = op.transformPoints(cam_tf, np.array([pointer_vec]))[0]
        arrow_handle = camera_ik.env.drawarrow(cam_pt, pointed_vec, 0.01, np.array([0.,0.,1.,1.]))
        arrow_handle2 = camera_ik.env.drawarrow(cam_pt, target, 0.02, np.array([0.,1.,0.,1.]))
        inp = raw_input("Check camera pointing. Enter 'q' to quit, or just [Enter] to continue.");
        if inp in ['q','Q', 'quit','Quit']:
            break


if __name__ == '__main__':
    robot = "PR2"
    side = 'r'
    freeindices = [27, 28, 30]

    if robot == "PR2":
        robotfile = "robots/pr2-beta-static.zae"
        manipname = "rightarm_camera" if side == 'r' else "leftarm_camera"
    else:
        raise NotImplemetedError("[%s] Only the PR2 is currently supported" % rospy.get_name())

    rospy.init_node(manipname)
    camera_ik = CameraLookatIk(robotfile, manipname, freeindices)
    rospy.on_shutdown(camera_ik.env.Destroy)

    if TEST:
        test(camera_ik, 50)
        rospy.loginfo("[{0}] Test completed".format(rospy.get_name()))
    else:
        camera_pointer = CameraPointer(side, camera_ik)
        rospy.spin()

