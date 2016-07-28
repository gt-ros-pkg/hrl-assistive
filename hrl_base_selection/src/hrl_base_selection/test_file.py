#!/usr/bin/env python

import openravepy as op
import numpy

import os

from openravepy.misc import InitOpenRAVELogging 
InitOpenRAVELogging() 

a = os.path.dirname(os.path.abspath(__file__))
b = a.split('/')
print a
print b

env = op.Environment() # create the environment
env.SetViewer('qtcoin') # start the viewer
env.Load('data/pr2test1.env.xml') # load a scene
robot = env.GetRobots()[0] # get the first robot

manip = robot.SetActiveManipulator('leftarm_torso') # set the manipulator to leftarm
ikmodel = op.databases.inversekinematics.InverseKinematicsModel(robot,iktype=op.IkParameterization.Type.Transform6D)
#ikmodel.autogenerate()
if not ikmodel.load():
    ikmodel.autogenerate()

with env: # lock environment
    Tgoal = numpy.array([[0,-1,0,-0.21],[-1,0,0,0.04],[0,0,-1,0.92],[0,0,0,1]])
    sol = manip.FindIKSolution(Tgoal, op.IkFilterOptions.CheckEnvCollisions) # get collision-free solution
    with robot: # save robot state
        robot.SetDOFValues(sol,manip.GetArmIndices()) # set the current solution
        Tee = manip.GetEndEffectorTransform()
        env.UpdatePublishedBodies() # allow viewer to update new robot
        raw_input('press any key')

    op.raveLogInfo('Tee is: '+repr(Tee))
