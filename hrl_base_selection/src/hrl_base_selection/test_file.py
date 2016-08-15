#!/usr/bin/env python

import openravepy as op
import numpy

import os

#from openravepy.misc import InitOpenRAVELogging 
#InitOpenRAVELogging() 

from openravepy import *
import numpy, time
env = Environment() # create openrave environment
env.SetViewer('qtcoin') # attach viewer (optional)
with env:
    body = RaveCreateKinBody(env,'')
    body.SetName('testbody')
    body.InitFromBoxes(numpy.array([[0.0,0,0,0.2,0.2,.2]]),True) # set geometry as one box of extents 0.1, 0.2, 0.3
    env.AddKinBody(body)

#time.sleep(4) # sleep 4 seconds
with env:
    env.Remove(body)
    body.InitFromBoxes(numpy.array([[0.5,0.5,0.5,0.2,0.2,0.2],[0.5,0.25,0.5,0.05,0.05,0.05],[0.,0.,0,0.,0.0,0.0]]),True) # set geometry as two boxes
    env.AddKinBody(body)
time.sleep(18) # sleep 4 seconds
