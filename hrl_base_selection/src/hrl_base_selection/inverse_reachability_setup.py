#!/usr/bin/env python



import numpy as np
import math as m
import openravepy as op
from openravepy.misc import InitOpenRAVELogging
import random

import rospy
import roslib; roslib.load_manifest('hrl_haptic_mpc')
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospkg

roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle


class InverseReachabilitySetup(object):
    def __init__(self, visualize=False, redo_ik=False, redo_reachability=False, redo_ir=False, manip='leftarm'):
        self.visualize = visualize
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')
        self.data_path = '/home/ari/svn/robot1_data/usr/ari/data/base_selection'

        self.pointscale = [0., 0.]

        self.setup_openrave(redo_ik=redo_ik, redo_reachability=redo_reachability, redo_ir=redo_ir, manip=manip)

    # Base pose is [x, y, theta, z_height] in world frame
    # Goal grasp is homogeneous transform in world frame
    def find_reachability_of_grasp_from_pose(self, goal_grasps, base_pose):
        # print 'goal_grasps',goal_grasps
        # print 'base_pose',base_pose
        goal_grasps = np.array(goal_grasps)
        if len(goal_grasps.shape) == 2:
            goal_grasps = np.array([np.array(goal_grasps)])
        if not len(goal_grasps.shape) == 3:
            print 'The size of the grasps matrix is wrong. Should be a list of 4x4 homogeneous transforms!'
            return None
        # print 'goal_B_gripper\n', goal_grasps
        # print 'base_pose\n', base_pose

        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = base_pose[3]
        self.robot.SetActiveDOFValues(v, 1)

        self.robot.SetTransform(np.eye(4))
        pr2_B_arm_base = np.eye(4)
        pr2_B_arm_base[0:3, 3] = self.rmodel.getOrderedArmJoints()[0].GetAnchor()

        origin_B_pr2 = np.matrix([[m.cos(base_pose[2]), -m.sin(base_pose[2]), 0., base_pose[0]],
                                  [m.sin(base_pose[2]), m.cos(base_pose[2]), 0., base_pose[1]],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
        base_B_grasps = np.array([np.matrix(origin_B_pr2).I*np.matrix(i) for i in goal_grasps])
        # print 'base_B_grasps\n', base_B_grasps
        self.robot.SetTransform(np.array(origin_B_pr2))
        # print 'robot location\n', self.robot.GetTransform()



        # arm_base = self.rmodel.getOrderedArmJoints()[0].GetAnchor()
        # world_B_arm_base = np.eye(4)
        # world_B_arm_base[0:3, 3] = arm_base
        arm_base = np.array(origin_B_pr2*np.matrix(pr2_B_arm_base))[0:3, 3]

        # print 'arm_base\n', arm_base
        # print base_B_grasps[:, 0:3, 3]
        world_B_grasps = goal_grasps
        # grasp_relative_translation = base_B_grasps[:, 0:3, 3] - np.tile(baseanchor, (len(base_B_grasps),1))
        grasp_relative_translation = world_B_grasps[:, 0:3, 3] - np.tile(arm_base, (len(base_B_grasps), 1))
        # print 'grasp_relative_translation\n', grasp_relative_translation
        # print np.tile(baseanchor, (len(base_B_grasps),1))
        # print self.pointscale[0]*(base_B_grasps[:, 0:3, 3] - np.tile(baseanchor, (len(base_B_grasps), 1)))
        # reach_score_location = np.rint(self.pointscale[0]*(base_B_grasps[:, 0:3, 3] - np.tile(arm_base, (len(base_B_grasps),1))) + self.pointscale[1]).astype(int)

        reach_score_locations = np.rint(self.pointscale[0] * (world_B_grasps[:, 0:3, 3] - np.tile(arm_base, (len(world_B_grasps), 1))) +
            self.pointscale[1]).astype(int)
        # for reach_score_location in reach_score_locations:
        #     if np.any(reach_score_location>51) or np.any(reach_score_location<0):
        #         print 'The goal is outside the workspace of the robot.'
        #         return None

        reach3d = np.array(self.rmodel.reachability3d)
        # np.unravel_index(a.argmax(), a.shape)
        # reach_score = np.zeroes([len(goal_grasps), 3])
        # for i in xrange(reach_score_location):
        #     reach_score = [reach3d[i[0], i[1],i[2] for i in reach_score_location]
        # print reach_score_locations
        # print 'reachability scores:',[reach3d[i[0], i[1], i[2]] if np.all(i<=51) and np.all(i>=0) else 0. for i in reach_score_locations]
        return [reach3d[i[0], i[1], i[2]] if np.all(i<=51) and np.all(i>=0) else 0. for i in reach_score_locations]

    # goal_grasps should be a list of goal grasps. Each grasp is a 4x4 homogeneous transform as a numpy array.
    def get_inverse_reachability(self, goal_grasps):
        # I'm using aggregate inverse reachability which takes in a list of goal grasps. Here I check if I only
        # received a single grasp, not as a list of grasps. If so, I stick it into a new list.
        # if len(np.shape(goal_grasps)) == 2:
        #     goal_grasps = [goal_grasps]
        # for grasp in goal_grasps:
        #     grasp = np.array(grasp)
        densityfn, samplerfn, bounds = None, None, None
        # computeBaseDistribution
        Tgrasp = goal_grasps[0][0]
        densityfn, samplerfn, bounds = self.irmodel[0].computeBaseDistribution(Tgrasp,
                                                                                     logllthresh=-100000000000.0)
        # densityfn, samplerfn, bounds = self.irmodel.computeAggregateBaseDistribution(goal_grasps, logllthresh=-100000000000.0)
        if not densityfn:
            print 'I have no results for the density function. Maybe try setting the log likelihood threshold ' \
                  'lower or something else is wrong'
        result = op.databases.inversereachability.InverseReachabilityModel.showBaseDistribution(self.env, densityfn, bounds, thresh=-10000000)
        print 'here1'
        print densityfn
        print 'here2'
        print bounds
        print 'here3'

        goals = []
        numfailures = 0
        import time
        starttime = time.time()
        timeout = 100.
        N=100
        j = 4
        with self.robot:
            while len(goals) < N:
                if time.time() - starttime > timeout:
                    break
                poses, jointstate = samplerfn(N - len(goals))
                print 'here'+str(j)
                j+=1
                for pose in poses:
                    self.robot.SetTransform(pose)
                    self.robot.SetDOFValues(*jointstate)
                    # print jointstate
                    # validate that base is not in collision
                    if not self.manip.CheckIndependentCollision(op.CollisionReport()):
                        q = self.manip.FindIKSolution(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                        if q is not None:
                            values = self.robot.GetDOFValues()
                            values[self.manip.GetArmIndices()] = q
                            goals.append((Tgrasp, pose, values))
                        elif self.manip.FindIKSolution(Tgrasp, 0) is None:
                            numfailures += 1
        print 'showing %d results' % N
        print 'best?\n', densityfn(np.array([[1., 0., 0., 0., 0., 0., 0.]]))
        self.robot.SetTransform(np.array([1., 0., 0., 0., 0., 0., 0.]))
        print 'best?\n', densityfn(np.array([[1., 0., 0., 0., -0.19, -0.748, 0.]]))
        self.robot.SetTransform(np.array([1., 0., 0., 0., -0.19, -0.748, 0.]))
        # [0.19, 0.748, 0.682175]
        self.env.UpdatePublishedBodies()
        sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
        # print 'sols:\n', sols
        for solution in sols:
            self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
            self.env.UpdatePublishedBodies()
            rospy.sleep(1.0)

        for ind, goal in enumerate(goals):
            raw_input('press ENTER to show goal %d' % ind)
            Tgrasp, pose, values = goal
            print pose
            l_tool = self.robot.GetLink('l_gripper_tool_frame')
            print 'tool\n', np.round(l_tool.GetTransform(),4)
            print 'Tgrasp\n',Tgrasp
            self.robot.SetTransform(pose)
            self.robot.SetDOFValues(values)
            print 'densityfn value:\n', densityfn(np.array([pose]))

    def setup_openrave(self, redo_ik=False, redo_reachability=False, redo_ir=False, manip='leftarm'):
        starting_to_setup_openrave_time = rospy.Time.now()
        InitOpenRAVELogging()
        self.env = op.Environment()
        if self.visualize:
            self.env.SetViewer('qtcoin')
        self.env.Load('robots/pr2-beta-static.zae')
        self.robot = self.env.GetRobots()[0]

        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint('l_gripper_l_finger_joint').GetDOFIndex()] = .34
        v[self.robot.GetJoint('r_gripper_l_finger_joint').GetDOFIndex()] = .34
        self.robot.SetActiveDOFValues(v, 1)
        self.env.UpdatePublishedBodies()

        lims = self.robot.GetDOFLimits()
        v = self.robot.GetActiveDOFValues()

        min_check = np.argmin(zip(lims[0], v), axis=1)
        max_check = np.argmax(zip(lims[1], v), axis=1)
        out_of_range_indices = np.hstack([min_check.nonzero(), max_check.nonzero()])[0]
        print 'Joint(s) out of range is: '
        for ind in out_of_range_indices:
            print self.robot.GetJointFromDOFIndex(ind)

        print 'Setting joint limits of PR2 to soft-limits from URDF'
        # Set lower limits
        lims[0][self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = 0.0

        lims[0][self.robot.GetJoint('l_shoulder_pan_joint').GetDOFIndex()] = -0.564601836603
        lims[0][self.robot.GetJoint('l_shoulder_lift_joint').GetDOFIndex()] = -0.3536
        lims[0][self.robot.GetJoint('l_upper_arm_roll_joint').GetDOFIndex()] = -0.65
        lims[0][self.robot.GetJoint('l_elbow_flex_joint').GetDOFIndex()] = -2.1213
        lims[0][self.robot.GetJoint('l_forearm_roll_joint').GetDOFIndex()] = -10.
        lims[0][self.robot.GetJoint('l_wrist_flex_joint').GetDOFIndex()] = -2.0
        lims[0][self.robot.GetJoint('l_wrist_roll_joint').GetDOFIndex()] = -10.

        lims[0][self.robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -2.1353981634
        lims[0][self.robot.GetJoint('r_shoulder_lift_joint').GetDOFIndex()] = -0.3536
        lims[0][self.robot.GetJoint('r_upper_arm_roll_joint').GetDOFIndex()] = -3.75
        lims[0][self.robot.GetJoint('r_elbow_flex_joint').GetDOFIndex()] = -2.1213
        lims[0][self.robot.GetJoint('r_forearm_roll_joint').GetDOFIndex()] = -10.
        lims[0][self.robot.GetJoint('r_wrist_flex_joint').GetDOFIndex()] = -2.0
        lims[0][self.robot.GetJoint('r_wrist_roll_joint').GetDOFIndex()] = -10.

        # Set upper limits
        lims[1][self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = 0.325

        lims[1][self.robot.GetJoint('l_shoulder_pan_joint').GetDOFIndex()] = 2.1353981634
        lims[1][self.robot.GetJoint('l_shoulder_lift_joint').GetDOFIndex()] = 1.2963
        lims[1][self.robot.GetJoint('l_upper_arm_roll_joint').GetDOFIndex()] = 3.75
        lims[1][self.robot.GetJoint('l_elbow_flex_joint').GetDOFIndex()] = 0.0
        lims[1][self.robot.GetJoint('l_forearm_roll_joint').GetDOFIndex()] = 10.
        lims[1][self.robot.GetJoint('l_wrist_flex_joint').GetDOFIndex()] = 0.0
        lims[1][self.robot.GetJoint('l_wrist_roll_joint').GetDOFIndex()] = 10.

        lims[1][self.robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = 0.564601836603
        lims[1][self.robot.GetJoint('r_shoulder_lift_joint').GetDOFIndex()] = 1.2963
        lims[1][self.robot.GetJoint('r_upper_arm_roll_joint').GetDOFIndex()] = 0.65
        lims[1][self.robot.GetJoint('r_elbow_flex_joint').GetDOFIndex()] = 0.0
        lims[1][self.robot.GetJoint('r_forearm_roll_joint').GetDOFIndex()] = 10.
        lims[1][self.robot.GetJoint('r_wrist_flex_joint').GetDOFIndex()] = 0.0
        lims[1][self.robot.GetJoint('r_wrist_roll_joint').GetDOFIndex()] = 10.
        self.robot.SetDOFLimits(lims[0], lims[1])

        min_check = np.argmin(zip(lims[0], v), axis=1)
        max_check = np.argmax(zip(lims[1], v), axis=1)
        out_of_range_indices = np.hstack([min_check.nonzero(), max_check.nonzero()])[0]
        print 'Joint(s) out of range is: '
        for ind in out_of_range_indices:
            print self.robot.GetJointFromDOFIndex(ind)
        self.reset_pr2_arms()

        ## Set robot manipulators
        self.robot.SetActiveManipulator(manip)
        self.manip = self.robot.GetActiveManipulator()
        self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot,
                                                                        iktype=op.IkParameterization.Type.Transform6D)
        if not self.ikmodel.load() or redo_ik:
            print 'Will now generate an IK model. This will take a while!'
            self.ikmodel.autogenerate()
            self.ikmodel.load()
        print 'I have loaded the IK model'
        self.rmodel = op.databases.kinematicreachability.ReachabilityModel(self.robot)
        if not self.rmodel.load() or redo_reachability:
            print 'Will now generate a kinematic reachability model (aka capability map). This will take a while!'
            self.rmodel.autogenerate()
            self.rmodel.load()
        print 'I have loaded the Kinematic Reachability (capability map) model'

        armjoints = self.rmodel.getOrderedArmJoints()
        baseanchor = armjoints[0].GetAnchor()
        eetrans = self.rmodel.manip.GetEndEffectorTransform()[0:3, 3]
        armlength = 0
        for j in armjoints[::-1]:
            armlength += np.sqrt(np.sum((eetrans - j.GetAnchor()) ** 2))
            eetrans = j.GetAnchor()
        maxradius = armlength + self.rmodel.xyzdelta * np.sqrt(3.0) * 1.05
        allpoints, insideinds, shape, self.pointscale = self.rmodel.UniformlySampleSpace(maxradius, delta=self.rmodel.xyzdelta)

        # print len(np.array(self.rmodel.reachabilitystats))
        # print np.size(np.array(self.rmodel.reachabilitystats))
        # print np.array(self.rmodel.reachabilitystats)[0]
        # print np.array(self.rmodel.reachabilitystats)[:,7]
        #
        # self.irmodel = dict()
        # for z_height in np.arange(0.0, 0.32+0.001, 0.30):
        #     self.irmodel[z_height] = op.databases.inversereachability.InverseReachabilityModel(self.robot)
        #     self.irmodel[z_height].id = int(z_height*100.)
        #     self.irmodel[z_height].jointvalues = np.array([z_height])
        #     if not self.irmodel[z_height].load() or redo_ir:
        #         print 'Will now generate a inverse-reachability model for height', z_height
        #         print 'This will take a while!'
        #         self.irmodel[z_height].autogenerate()
        #         self.irmodel[z_height].load()
        #     else:
        #         print 'I already have an inverse-reachability model for height', z_height
        # print 'I have loaded the Inverse Reachability model'

        starting_to_setup_openrave_elapsed = rospy.Time.now() - starting_to_setup_openrave_time
        print 'Time elapsed to load everything for openrave was:', starting_to_setup_openrave_elapsed.to_sec()

    def reset_pr2_arms(self):
        self.set_pr2_arms('left', [3.14 / 2, -0.3536, 0., -3.14 * 2 / 3, 0., 0., 0.])
        self.set_pr2_arms('right', [-3.14 / 2, -0.3536, 0., -3.14 * 2 / 3, 0., 0., 0.])

    # arm is 'left' vs 'right', but only the first letter is used.
    # 'q' is joint angles in order: shoulder_pan, shoulder_lift, upper_roll,
    # elbow_flex, forearm_roll, wrist_flex, wrist_roll
    def set_pr2_arms(self, arm, q):
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint(arm[0] + '_shoulder_pan_joint').GetDOFIndex()] = q[0]
        v[self.robot.GetJoint(arm[0] + '_shoulder_lift_joint').GetDOFIndex()] = q[1]
        v[self.robot.GetJoint(arm[0] + '_upper_arm_roll_joint').GetDOFIndex()] = q[2]
        v[self.robot.GetJoint(arm[0] + '_elbow_flex_joint').GetDOFIndex()] = q[3]
        v[self.robot.GetJoint(arm[0] + '_forearm_roll_joint').GetDOFIndex()] = q[4]
        v[self.robot.GetJoint(arm[0] + '_wrist_flex_joint').GetDOFIndex()] = q[5]
        v[self.robot.GetJoint(arm[0] + '_wrist_roll_joint').GetDOFIndex()] = q[6]
        self.robot.SetActiveDOFValues(v, 1)

        lims = self.robot.GetDOFLimits()

        min_check = np.argmin(zip(lims[0], v), axis=1)
        max_check = np.argmax(zip(lims[1], v), axis=1)
        if np.count_nonzero(min_check) > 0 or np.count_nonzero(max_check) > 0:
            print 'Some joints are out of range.'
            print 'I will set them to be at the limit. This may not be a good idea.'
            out_of_range_indices = np.hstack([min_check.nonzero(), max_check.nonzero()])[0]
            print out_of_range_indices
            print 'Joint(s) out of range is: '
            for ind in out_of_range_indices:
                print self.robot.GetJointFromDOFIndex(ind)
            for ind in min_check.nonzero():
                v[ind] = lims[0][ind]
            for ind in max_check.nonzero():
                v[ind] = lims[1][ind]
            self.robot.SetActiveDOFValues(v, 1)

        self.env.UpdatePublishedBodies()

if __name__ == "__main__":
    rospy.init_node('inverse_reachability_setup')
    irs = InverseReachabilitySetup(visualize=False, redo_ik=False, redo_reachability=False, redo_ir=False, manip='leftarm')
    # irs = InverseReachabilitySetup(visualize=False, redo_ik=False, redo_reachability=False, redo_ir=,
    #                                manip='leftarm')
    # irs_la = InverseReachabilitySetup(visualize=False, redo_ik=True, redo_reachability=True, redo_ir=True,
    #                                manip='leftarm')
    print 'done with leftarm'

    # irs_lat = InverseReachabilitySetup(visualize=False, redo_ik=True, redo_reachability=True, redo_ir=True,
    #                                manip='leftarm_torso')
    # print 'done with leftarm torso'
    goal_grasp = np.eye(4)
    goal_grasp[2, 3] = 1.0
    goal_grasp[2, 3] = 0.682175
    # goal_grasp[0:3, 3] = [ 0.19    ,  0.748   ,  0.682175]
    goal_grasp[0:3, 3] = np.array([1.19, 0.748, 0.682175])
    goal_grasp[0:3, 3] = np.array([0.69, 0.748, 0.682175])
    goal_grasp2 = np.eye(4)
    goal_grasp2[0:3, 3] = np.array([0.19, 0.748, 0.682175])
    goal_B_gripper = np.matrix([[0., 0., 1., 0.0],
                                [0., 1., 0., 0.0],
                                [-1., 0., 0., 0.0],
                                [0., 0., 0., 1.0]])
    tgrasp = [np.array(np.matrix(goal_grasp)*goal_B_gripper), 0]
    # out = irs.get_inverse_reachability([tgrasp])
    out = irs.find_reachability_of_grasp_from_pose([np.array(np.matrix(goal_grasp)*goal_B_gripper), np.array(np.matrix(goal_grasp2)*goal_B_gripper)], [0.5, 0., 0., 0.])
    # out = irs.find_reachability_of_grasp_from_pose(np.array(np.matrix(goal_grasp) * goal_B_gripper), [0., 0., 0., 0.])
    print out
    # print out
    rospy.spin()
    #







