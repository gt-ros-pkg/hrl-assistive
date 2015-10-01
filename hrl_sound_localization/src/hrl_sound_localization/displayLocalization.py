#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)
import harkpython.harkbasenode as harkbasenode

import numpy as np, math
import roslib
import rospy

# ROS message
from std_msgs.msg import Bool, Empty, Int32, Int64, Float32, Float64
import jsk_hark_msgs

def cartesian2polar(xyz):
    """cartesian2polar(x, y, z)
converts coordinate from xyz to r\theta\phi"""
    x, y, z = xyz
    if z == 0:
        raise exceptions.ValueError("cartesian " + str(xyz) + " z must be nonzero.")

    r = pylab.sqrt(sum(map(lambda x: x**2, [x, y, z])))
    theta = pylab.arccos(z/r)
    phi = pylab.arccos(x / pylab.sqrt(sum(map(lambda x: x**2, [x, y]))))
    if y < 0: phi *= -1

    return (r, theta, phi)

def r2d(r):
    return 180.0 / numpy.pi * r

class HarkNode(harkbasenode.HarkBaseNode):
    def __init__(self):
        print "Initialization"
        rospy.init_node('hark_src')
        
        self.outputNames=("output",)  # one output terminal named "output"
        self.outputTypes=("prim_float",)  # the type is primitive float.

        self.frame_count = {}
        self.src_r       = {}
        self.src_theta   = {}
        self.src_phi     = {}
        

    def initComms():
        ''' Initialization of publishers and subscribers'''
        self.id_pub = rospy.Publisher('/sound_localization/id', Int32)

    def calculate(self):
        ''' Run this code per each input '''
        for src in self.SOURCES:
            if src.has_key("x"):
                self.frame_count[src["id"]].append(self.count)
                r, theta, phi = cartesian2polar(src["x"])
                theta = r2d(theta)
                phi   = r2d(phi)
                self.src_r[src["id"]].append(r)
                self.src_theta[src["id"]].append(theta)
                self.src_phi[src["id"]].append(phi)
                
        if self.count % 10 == 0:
            for i, srcid in enumerate(self.src_r.keys()):
                msg = Int32()
                msg.data = srcid
                self.id_pub.publish(msg)

        self.outputValues["OUTPUT"] = 0

        


