#!/usr/bin/env python

import rospy
import openravepy as op


if __name__=='__main__':
    env = op.Environment()
 #   env.SetViewer('qtcoin')
    env.Load('robots/pr2-beta-static.zae')
    pr2 = env.GetRobots()[0]
    pr2.SetActiveManipulator('rightarm_camera')
    ikmodel = op.databases.inversekinematics.InverseKinematicsModel(pr2, iktype=op.IkParameterization.Type.Lookat3D)
    if not ikmodel.load():
        print "Generating new IK Model"
        ikmodel.autogenerate()
    print " Model Generated "
