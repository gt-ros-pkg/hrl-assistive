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
roslib.load_manifest('hrl_manipulation_task')
import os, sys, threading, copy
import gc

# util
import numpy as np

# 
from hrl_anomaly_detection import util
import hrl_lib.util as ut

# msgs and srvs
from hrl_manipulation_task.msg import MultiModality
from hrl_srvs.srv import Bool_None, Bool_NoneResponse, String_None, String_NoneResponse

# Sensors
from sensor.kinect_audio import kinect_audio
from sensor.robot_kinematics import robot_kinematics
from sensor.tool_ft import tool_ft
from sensor.artag_vision import artag_vision
from sensor.pps_skin import pps_skin


class logger:
    def __init__(self, ft=False, audio=False, kinematics=False, vision=False, pps=False, \
                 subject=None, task=None, verbose=False):
        rospy.logout('ADLs_log node subscribing..')

        self.subject  = subject
        self.task     = task
        self.verbose  = verbose
        
        self.initParams()
        
        self.audio      = kinect_audio() if audio else None
        self.kinematics = robot_kinematics() if kinematics else None
        self.ft         = tool_ft() if ft else None
        self.vision     = artag_vision(False, viz=False) if vision else None
        self.pps_skin   = pps_skin(True) if pps else None

        self.waitForReady()
        self.initComms()
        
    def initParams(self):
        '''
        # load parameters
        '''        
        # File saving
        self.record_root_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016'
        self.folderName = os.path.join(self.record_root_path, self.subject + '_' + self.task)

        
    def initComms(self):
        '''
        Record data and publish raw data
        '''        
        self.rawDataPub = rospy.Publisher('/hrl_manipulation_task/raw_data', MultiModality)
        
        ## self.log_start_service = rospy.Service('/data_record/log_start', String_None, self.logStartCallback)

        
    ## def logStartCallback(self, msg):
    ##     if msg.data == True: self.log_start()            
    ##     else: self.close_log_file()            
    ##     return Bool_NoneResponse()

        
    def log_start(self):
        self.init_time = rospy.get_time()
        if self.audio is not None:
            self.audio.init_time = self.init_time
            self.audio.start()
            self.audio.reset(self.init_time)

        if self.kinematics is not None:
            self.kinematics.init_time = self.init_time
            self.kinematics.start()
            self.kinematics.reset(self.init_time)

        if self.ft is not None:
            self.ft.init_time = self.init_time
            self.ft.start()
            self.ft.reset(self.init_time)

        if self.vision is not None:
            self.vision.init_time = self.init_time
            self.vision.start()
            self.vision.reset(self.init_time)

        if self.pps_skin is not None:
            self.pps_skin.init_time = self.init_time
            self.pps_skin.start()
            self.pps_skin.reset(self.init_time)
            
    def close_log_file(self):
        data = {}
        data['init_time'] = self.init_time

        if self.audio is not None:
            self.audio.cancel()            
            data['audio_time']        = self.audio.time_data
            data['audio_feature']     = self.audio.audio_feature
            data['audio_power']       = self.audio.audio_power
            data['audio_azimuth']     = self.audio.audio_azimuth
            data['audio_head_joints'] = self.audio.audio_head_joints
            data['audio_cmd']         = self.audio.audio_cmd

        if self.kinematics is not None:
            self.kinematics.cancel()
            data['kinematics_time']    = self.kinematics.time_data
            data['kinematics_ee_pos']  = self.kinematics.kinematics_ee_pos
            data['kinematics_ee_quat'] = self.kinematics.kinematics_ee_quat
            data['kinematics_jnt_pos'] = self.kinematics.kinematics_jnt_pos
            data['kinematics_jnt_vel'] = self.kinematics.kinematics_jnt_vel
            data['kinematics_jnt_eff'] = self.kinematics.kinematics_jnt_eff
            data['kinematics_target_pos']  = self.kinematics.kinematics_target_pos
            data['kinematics_target_quat'] = self.kinematics.kinematics_target_quat
            
        if self.ft is not None:
            self.ft.cancel()
            data['ft_time']   = self.ft.time_data
            data['ft_force']  = self.ft.force_array
            data['ft_torque'] = self.ft.torque_array

        if self.vision is not None:
            self.vision.cancel()
            data['vision_time'] = self.vision.time_data
            data['vision_pos']  = self.vision.vision_tag_pos
            data['vision_quat'] = self.vision.vision_tag_quat

        if self.pps_skin is not None:
            self.pps_skin.cancel()
            data['pps_skin_time'] = self.pps_skin.time_data
            data['pps_skin_left']  = self.pps_skin.pps_skin_left
            data['pps_skin_right'] = self.pps_skin.pps_skin_right
            
        flag = raw_input('Enter trial\'s status (e.g. 1:success, 2:failure, 3: skip): ')
        if flag == '1':   status = 'success'
        elif flag == '2': status = 'failure'
        elif flag == '3': status = 'skip'
        else: status = flag

        if status == 'success' or status == 'failure':
            if status == 'failure':
                failure_class = raw_input('Enter failure reason if there is: ')

            if not os.path.exists(self.folderName): os.makedirs(self.folderName)

            # get next file id
            if status == 'success':
                fileList = util.getSubjectFileList(self.record_root_path, [self.subject], self.task)[0]
            else:
                fileList = util.getSubjectFileList(self.record_root_path, [self.subject], self.task)[1]
            test_id = -1
            if len(fileList)>0:
                for f in fileList:            
                    if test_id < int((os.path.split(f)[1]).split('_')[1]):
                        test_id = int((os.path.split(f)[1]).split('_')[1])

            if status == 'failure':        
                fileName = os.path.join(self.folderName, 'iteration_%d_%s_%s.pkl' % (test_id+1, status, failure_class))
            else:
                fileName = os.path.join(self.folderName, 'iteration_%d_%s.pkl' % (test_id+1, status))

            print 'Saving to', fileName
            ut.save_pickle(data, fileName)

        # Reinitialize all sensors
        if self.audio is not None: self.audio = kinect_audio()
        if self.kinematics is not None: self.kinematics = robot_kinematics() #.initVars() #
        if self.ft is not None: self.ft = tool_ft()
        if self.vision is not None: self.vision = artag_vision()
        if self.pps_skin is not None: self.pps_skin = pps_skin()

        gc.collect()
        rospy.sleep(1.0)


    def enableDetector(enableFlag):
            
        s   = rospy.ServiceProxy('anomaly_detector_enable/'+self.task, Bool_None)
        ret = s(enableFlag)

        

    def waitForReady(self):

        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():
            rate.sleep()

            if self.audio is not None:
                if self.audio.isReady() is False: 
                    print "audio is not ready"
                    continue

            if self.kinematics is not None:
                if self.kinematics.isReady() is False: 
                    print "kinematics is not ready"                    
                    continue

            if self.ft is not None:
                if self.ft.isReady() is False: 
                    print "ft is not ready"                                        
                    continue

            if self.vision is not None:
                if self.vision.isReady() is False: 
                    print "AR tag is not ready"                                        
                    continue

            if self.pps_skin is not None:
                if self.pps_skin.isReady() is False: 
                    print "pps is not ready"                                        
                    continue
                
            break

        if self.verbose: print "record_data>> completed to wait sensing data"
            

    def run(self):

        self.log_start()

        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():
            rate.sleep()

        self.close_log_file()

    def runDataPub(self):
        '''
        Publish collected data
        '''
        
        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():

            msg = MultiModality()
            msg.header.stamp      = rospy.Time.now()

            if self.audio is not None:            
                msg.audio_feature     = np.squeeze(self.audio.feature.T).tolist()
                msg.audio_power       = self.audio.power
                msg.audio_azimuth     = self.audio.azimuth+self.audio.base_azimuth
                msg.audio_head_joints = [self.audio.head_joints[0], self.audio.head_joints[1]]
                msg.audio_cmd         = self.audio.recog_cmd if type(self.audio.recog_cmd)==str() else 'None'

            if self.kinematics is not None:
                ee_pos, ee_quat           = self.kinematics.getEEFrame()
                jnt_pos, jnt_vel, jnt_eff = self.kinematics.return_joint_state()
                target_pos, target_quat   = self.kinematics.getTargetFrame()
            
                msg.kinematics_ee_pos  = np.squeeze(ee_pos.T).tolist()
                msg.kinematics_ee_quat = np.squeeze(ee_quat.T).tolist()
                msg.kinematics_jnt_pos = np.squeeze(jnt_pos.T).tolist()
                msg.kinematics_jnt_vel = np.squeeze(jnt_vel.T).tolist()
                msg.kinematics_jnt_eff = np.squeeze(jnt_eff.T).tolist()
                msg.kinematics_target_pos  = np.squeeze(target_pos.T).tolist()
                msg.kinematics_target_quat = np.squeeze(target_quat.T).tolist()

            if self.ft is not None:
                msg.ft_force  = np.squeeze(self.ft.force_raw.T).tolist()
                msg.ft_torque = np.squeeze(self.ft.torque_raw.T).tolist()

            if self.vision is not None:
                msg.vision_pos  = np.squeeze(self.vision.artag_pos.T).tolist()
                msg.vision_quat = np.squeeze(self.vision.artag_quat.T).tolist()
            
            if self.pps_skin is not None:
                msg.pps_skin_left  = np.squeeze(self.pps_skin.data_left.T).tolist()
                msg.pps_skin_right = np.squeeze(self.pps_skin.data_right.T).tolist()
            
            self.rawDataPub.publish(msg)
            rate.sleep()
        
                
if __name__ == '__main__':

    subject = 'gatsbii'
    task    = '10'
    verbose = True

    rospy.init_node('record_data')
    log = logger(ft=True, audio=True, kinematics=True, vision=True, pps=False, \
                 subject=subject, task=task, verbose=verbose)

    rospy.sleep(1.0)
    ## log.run()
    log.runDataPub()
    
