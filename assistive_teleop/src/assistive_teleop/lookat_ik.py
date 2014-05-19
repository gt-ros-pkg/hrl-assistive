#!/usr/bin/env python

import rospy
import openravepy as op

if __name__=='__main__':
    env = op.Environment()
 #   env.SetViewer('qtcoin')
    env.Load('robots/pr2-beta-static.zae')
    pr2 = env.GetRobots()[0]
    manip = pr2.SetActiveManipulator('rightarm_camera')
    freeindices = [27,28,30]
    freejoints = [pr2.GetJoints()[ind].GetName() for ind in freeindices]
    ikmodel = op.databases.inversekinematics.InverseKinematicsModel(pr2,
                                                                    iktype=op.IkParameterization.Type.Lookat3D,
                                                                    forceikfast=True,
                                                                    freeindices=freeindices,
                                                                    freejoints=freejoints)
    if not ikmodel.load():
        print "Generating Lookat IK Model"
        ikmodel.generate(freejoints=freejoints)
        ikmodel.save()
    else:
        print "Lookat IK Model Loaded"

