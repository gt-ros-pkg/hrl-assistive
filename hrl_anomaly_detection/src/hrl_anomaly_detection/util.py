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

# system
import rospy
import roslib
roslib.load_manifest('hrl_anomaly_detection')
import os, sys, copy

# util
import numpy as np
import hrl_lib.util as ut

from scipy import interpolate


def loadData(fileNames, isTrainingData=False, downSampleSize=100, verbose=False):

    for idx, fileName in enumerate(fileNames):
        if os.path.isdir(fileName):
            continue

        # sound
        audio_time    = d['audio_time']
        audio_azimuth = d['audio_azimuth']
        audio_power   = d['audio_power']

        # kinematics
        kin_time = d['kinematics_time']
        kin_pos  = d['kinematics_ee_pos'] # 3xN

        # ft
        ft_time        = d['ft_time']
        ft_force_array = d['ft_force']

        # vision
        vision_time = d['vision_time']
        vision_pos  = d['vision_pos']
        vision_quat = d['vision_quat']

        # pps
        pps_skin_time  = d['pps_skin_time']
        pps_skin_left  = d['pps_skin_left']
        pps_skin_right = d['pps_skin_right']

        
        newTimes = np.linspace(0.01, audio_time[-1], downSampleSize)

        forceInterp = interpolate.splrep(forceTimes, forces, s=0)
                
        
    
