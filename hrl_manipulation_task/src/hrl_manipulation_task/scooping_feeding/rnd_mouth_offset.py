#!/usr/bin/env python
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

# System
import sys, time, copy
import random
import numpy as np

# ROS
import rospy

# Msg
from hrl_msgs.msg import FloatArray
QUEUE_SIZE = 10

class mouth_noise():
    def __init__(self):
        
        self.initComms()
        
        return

    def initComms(self):
        self.mouth_noise_pub = rospy.Publisher('/hrl_manipulation_task/mouth_noise', FloatArray, \
                                                queue_size=QUEUE_SIZE, latch=True)

    def run(self):
    
        ## rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():

            print "1: miss "
            print "2: collision (side movement)"
            print "3: collision (depth, too risky)"            
            print " else: disable)"

            val = raw_input('Enable random mouth pose?')
            ps = FloatArray()

            # miss
            if val == '1':
                val2 = random.randint(0,1)
                # side
                if val2 == 0:
                    print "Miss - transition (mainly side movement)"
                    ps.data = [np.random.normal(scale=0.04),
                               random.choice([-1.,1.])*random.uniform(0.05,0.1),
                               np.random.normal(scale=0.04)]
                else:
                    print "Miss - transition (mainly not enough depth)"
                    ps.data = [np.random.normal(scale=0.04),
                               np.random.normal(scale=0.04),
                               random.uniform(-0.06,-0.1)]
            # collision
            elif val == '2':
                # side
                val2 = random.choice([0,0,1])
                if val2 == 0:
                    print "Collision - transition (mainly side movement)"
                    ps.data = [0.0,
                               random.choice([-1.,1.])*random.uniform(0.02,0.04),
                               abs(np.random.normal(scale=0.04))]
                else:
                    print "Collision - transition (mainly lowering movement)"
                    ps.data = [-random.uniform(0.02,0.04),
                               0.0,
                               abs(np.random.normal(scale=0.04))]
            # collision but too risky
            elif val == '3':
                print "Collision - transition (depth)"
                ps.data = [0.0,
                           0.0,
                           random.uniform(0.03,0.08)]
                
            else:
                ps.data = [0.,0.,0.]
            self.mouth_noise_pub.publish(ps)
                
if __name__ == '__main__':
    
    rospy.init_node('mouth_noise')
    mo = mouth_noise()
    mo.run()
