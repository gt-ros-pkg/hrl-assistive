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
from hrl_multimodal_anomaly_detection.hmm import util
import matplotlib.pyplot as plt


class data_viz:
    def __init__(self, subject=None, task=None, verbose=False):
        rospy.loginfo('log data visualization..')

        self.subject = subject
        self.task    = task
        self.verbose = verbose
        
        self.initParams()

    def initParams(self):
        '''
        # load parameters
        '''        
        self.record_root_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016'
        self.folderName = os.path.join(self.record_root_path, self.subject + '_' + self.task)
        
    def audio_test(self):
        
        success_list, failure_list = util.getSubjectFileList(self.record_root_path, [self.subject], self.task)

        for fileName in failure_list:
            d = ut.load_pickle(fileName)
            print d.keys()


            time_max = np.amax(d['audio_time'])
            time_min = np.amin(d['audio_time'])

            azimuth_max = 90.0
            azimuth_min = -90.0

            power_max = np.amax(d['audio_power'])
            power_min = np.amin(d['audio_power'])

            time_list    = d['audio_time']
            azimuth_list = np.arange(azimuth_min, azimuth_max, 1.0)
            
            audio_image = np.zeros( (len(time_list), len(azimuth_list)) )

            print "image size ", audio_image.shape

            for idx, p in enumerate(d['audio_power']):

                azimuth_idx = min(range(len(azimuth_list)), key=lambda i: \
                                  abs(azimuth_list[i]-d['audio_azimuth'][idx]))
                
                audio_image[idx][azimuth_idx] = (p - power_min)/(power_max - power_min)

                
            fig = plt.figure()            
            ax1 = fig.add_subplot(211)
            ax1.imshow(audio_image.T)
            ax1.set_aspect('auto')
            ax1.set_ylabel('azimuth angle', fontsize=18)

            y     = np.arange(0.0, len(azimuth_list), 30.0)
            new_y = np.arange(azimuth_min, azimuth_max, 30.0)
            plt.yticks(y,new_y)
            #------------------------------------------------------------

            n,m = np.shape(d['audio_feature'])

            last_feature = np.hstack([ np.zeros((n,1)), d['audio_feature'][:,:-1] ])            
            feature_delta = d['audio_feature'] - last_feature
            
            ax2 = fig.add_subplot(212)
            ax2.imshow( feature_delta[:n/2] )
            ax2.set_aspect('auto')
            ax2.set_xlabel('time', fontsize=18)
            ax2.set_ylabel('MFCC derivative', fontsize=18)

            #------------------------------------------------------------
            plt.suptitle('Auditory features', fontsize=18)            
            plt.show()
            

    def ft_test(self):
        
        success_list, failure_list = util.getSubjectFileList(self.record_root_path, [self.subject], self.task)

        for fileName in failure_list:
            d = ut.load_pickle(fileName)
            print d.keys()


            fig = plt.figure()            

            

if __name__ == '__main__':


    subject = 'gatsbii'
    task    = 'scooping'
    verbose = True
    

    l = data_viz(subject, task, verbose=verbose)
    ## l.audio_test()
    l.ft_test()
