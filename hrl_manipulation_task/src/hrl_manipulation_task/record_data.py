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
import rospy, roslib
import os, sys, threading, copy
import gc

# util
import numpy as np

# 
from hrl_anomaly_detection import util
import hrl_lib.util as ut

# msgs and srvs
from hrl_anomaly_detection.msg import MultiModality
from hrl_srvs.srv import Bool_None, Bool_NoneResponse, String_None, String_NoneResponse

# Sensors
from sensor.kinect_audio import kinect_audio
from sensor.robot_kinematics import robot_kinematics
from sensor.tool_ft import tool_ft
from sensor.artag_vision import artag_vision
from sensor.pps_skin import pps_skin
from sensor.fabric_skin import fabric_skin


class logger:
    def __init__(self, ft=False, audio=False, kinematics=False, vision=False, pps=False, skin=False, \
                 subject=None, task=None, data_pub= False, verbose=False):
        rospy.logout('ADLs_log node subscribing..')

        self.subject  = subject
        self.task     = task
        self.data_pub = data_pub
        self.verbose  = verbose
        
        self.initParams()
        
        self.audio       = kinect_audio() if audio else None
        self.kinematics  = robot_kinematics() if kinematics else None
        self.ft          = tool_ft() if ft else None
        self.vision      = artag_vision(False, viz=False) if vision else None
        self.pps_skin    = pps_skin(True) if pps else None
        self.fabric_skin = fabric_skin(True) if skin else None

        self.waitForReady()
        self.initComms()

        if self.data_pub:
            t = threading.Thread(target=self.runDataPub)
            t.setDaemon(True)
            t.start()
        
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

        
    def log_start(self):
        self.init_time = rospy.get_rostime().to_sec()
        self.data = {}
        self.data['init_time'] = self.init_time

        ## # logging by callback
        ## if self.audio is not None:
        ##     self.audio.reset(self.init_time)
        ##     self.audio.enable_log = True
        ## if self.kinematics is not None:
        ##     self.kinematics.reset(self.init_time)
        ##     self.kinematics.enable_log = True
            
        ## if self.ft is not None:
        ## if self.vision is not None:
        ## if self.pps_skin is not None:
        ## if self.fabric_skin is not None:

        # logging by thread
        self.enable_log_thread = True
        self.logger = threading.Thread(target=self.runDataLogger)
        self.logger.setDaemon(True)
        self.logger.start()
                    
    def close_log_file(self):

        ## # logging by callback
        ## # disable logging
        ## if self.audio is not None: self.audio.enable_log = False
        ## if self.kinematics is not None: self.kinematics.enable_log = False
        
        ## # log into data
        ## if self.audio is not None: 
        ##     self.data['audio_time']    = self.audio.time_data            
        ##     self.data['audio_azimuth'] = self.audio.audio_azimuth
        ##     self.data['audio_power']   = self.audio.audio_power
        
        ## if self.kinematics is not None:
        ##     self.data['kinematics_time']        = self.kinematics.time_data
        ##     self.data['kinematics_ee_pos']      = self.kinematics.kinematics_ee_pos
        ##     self.data['kinematics_ee_quat']     = self.kinematics.kinematics_ee_quat
        ##     self.data['kinematics_jnt_pos']     = self.kinematics.kinematics_main_jnt_pos
        ##     self.data['kinematics_jnt_vel']     = self.kinematics.kinematics_main_jnt_vel
        ##     self.data['kinematics_jnt_eff']     = self.kinematics.kinematics_main_jnt_eff
        ##     self.data['kinematics_target_pos']  = self.kinematics.kinematics_target_pos
        ##     self.data['kinematics_target_quat'] = self.kinematics.kinematics_target_quat                    

        # logging by thread 
        self.enable_log_thread = False


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
            ut.save_pickle(self.data, fileName)

        gc.collect()
        rospy.sleep(1.0)


    def enableDetector(self, enableFlag):
        print "Wait anomaly detector service"
        rospy.wait_for_service('anomaly_detector_enable/'+self.task)
        s   = rospy.ServiceProxy('anomaly_detector_enable/'+self.task, Bool_None)
        ret = s(enableFlag)

        
    def waitForReady(self):

        print "-------------------------------------"
        print "Wait for sensor ready"
        print "-------------------------------------"

        print self.audio.isReady()
        
        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():
            rate.sleep()

            if self.audio is not None:
                if self.audio.isReady() is False:
                    print "-------------------------------------"
                    print "audio is not ready"
                    print "-------------------------------------"
                    continue

            if self.kinematics is not None:
                if self.kinematics.isReady() is False: 
                    print "-------------------------------------"
                    print "kinematics is not ready"                    
                    print "-------------------------------------"
                    continue

            if self.ft is not None:
                if self.ft.isReady() is False: 
                    print "-------------------------------------"
                    print "ft is not ready"                                        
                    print "-------------------------------------"
                    continue

            if self.vision is not None:
                if self.vision.isReady() is False: 
                    print "-------------------------------------"
                    print "AR tag is not ready"                                        
                    print "-------------------------------------"
                    continue

            if self.pps_skin is not None:
                if self.pps_skin.isReady() is False: 
                    print "-------------------------------------"
                    print "pps is not ready"                                        
                    print "-------------------------------------"
                    continue

            if self.fabric_skin is not None:
                if self.fabric_skin.isReady() is False: 
                    print "-------------------------------------"
                    print "fabric skin is not ready"                                        
                    print "-------------------------------------"
                    continue
                
            break

        if self.verbose: print "record_data>> completed to wait sensing data"
            

    def run(self):

        self.log_start()

        count = 0
        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():
            count += 1
            print count
            if count > 100: break
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
                if self.audio.feature is not None:           
                    msg.audio_feature     = np.squeeze(self.audio.feature.T).tolist()
                    msg.audio_power       = self.audio.power
                    msg.audio_azimuth     = self.audio.azimuth+self.audio.base_azimuth
                msg.audio_head_joints     = [self.audio.head_joints[0], self.audio.head_joints[1]]
                msg.audio_cmd             = self.audio.recog_cmd if type(self.audio.recog_cmd)==str() else 'None'

            if self.kinematics is not None:
                msg.kinematics_ee_pos  = np.squeeze(self.kinematics.ee_pos.T).tolist()
                msg.kinematics_ee_quat = np.squeeze(self.kinematics.ee_quat.T).tolist()
                msg.kinematics_jnt_pos = np.squeeze(self.kinematics.main148_jnt_positions.T).tolist()
                msg.kinematics_jnt_vel = np.squeeze(self.kinematics.main_jnt_velocities.T).tolist()
                msg.kinematics_jnt_eff = np.squeeze(self.kinematics.main_jnt_efforts.T).tolist()
                msg.kinematics_target_pos  = np.squeeze(self.kinematics.target_pos.T).tolist()
                msg.kinematics_target_quat = np.squeeze(self.kinematics.target_quat.T).tolist()

            if self.ft is not None:
                msg.ft_force  = np.squeeze(self.ft.force_raw.T).tolist()
                msg.ft_torque = np.squeeze(self.ft.torque_raw.T).tolist()

            if self.vision is not None:
                if self.vision.artag_pos is not None:
                    msg.vision_pos  = np.squeeze(self.vision.artag_pos.T).tolist()
                    msg.vision_quat = np.squeeze(self.vision.artag_quat.T).tolist()
            
            if self.pps_skin is not None:
                msg.pps_skin_left  = np.squeeze(self.pps_skin.data_left.T).tolist()
                msg.pps_skin_right = np.squeeze(self.pps_skin.data_right.T).tolist()

            if self.fabric_skin is not None:
                msg.fabric_skin_centers_x = self.fabric_skin.centers_x
                msg.fabric_skin_centers_y = self.fabric_skin.centers_y
                msg.fabric_skin_centers_z = self.fabric_skin.centers_z

                msg.fabric_skin_normals_x = self.fabric_skin.normals_x
                msg.fabric_skin_normals_y = self.fabric_skin.normals_y
                msg.fabric_skin_normals_z = self.fabric_skin.normals_z

                msg.fabric_skin_values_x = self.fabric_skin.values_x
                msg.fabric_skin_values_y = self.fabric_skin.values_y
                msg.fabric_skin_values_z = self.fabric_skin.values_z
                
            self.rawDataPub.publish(msg)
            rate.sleep()
        

    def runDataLogger(self):
        '''
        Publish collected data
        '''
        
        rate = rospy.Rate(100) # 25Hz, nominally.
        while not rospy.is_shutdown():

            if self.audio is not None: 
                if 'audio_time' not in self.data.keys():
                    self.data['audio_time']    = [self.audio.time]
                    self.data['audio_azimuth'] = [self.audio.azimuth]
                    self.data['audio_power']   = [self.audio.power]
                else:
                    self.data['audio_time'].append(self.audio.time)
                    self.data['audio_azimuth'].append(self.audio.azimuth)
                    self.data['audio_power'].append(self.audio.power)
                    
            if self.kinematics is not None:
                if 'kinematics_time' not in self.data.keys():
                    self.data['kinematics_time'] = [self.kinematics.time]
                    self.data['kinematics_ee_pos'] = self.kinematics.ee_pos
                    self.data['kinematics_ee_quat'] = self.kinematics.ee_quat
                    self.data['kinematics_jnt_pos'] = self.kinematics.main_jnt_positions
                    self.data['kinematics_jnt_vel'] = self.kinematics.main_jnt_velocities
                    self.data['kinematics_jnt_eff'] = self.kinematics.main_jnt_efforts
                    self.data['kinematics_target_pos']  = self.kinematics.target_pos
                    self.data['kinematics_target_quat'] = self.kinematics.target_quat                    
                else:
                    self.data['kinematics_time'].append(self.kinematics.time)
                    self.data['kinematics_ee_pos'] = np.hstack([self.data['kinematics_ee_pos'], \
                                                           self.kinematics.ee_pos]) 
                    self.data['kinematics_ee_quat'] = np.hstack([self.data['kinematics_ee_quat'], \
                                                            self.kinematics.ee_quat])
                    self.data['kinematics_jnt_pos'] = np.hstack([self.data['kinematics_jnt_pos'], \
                                                            self.kinematics.main_jnt_positions])
                    self.data['kinematics_jnt_vel'] = np.hstack([self.data['kinematics_jnt_vel'], \
                                                            self.kinematics.main_jnt_velocities])
                    self.data['kinematics_jnt_eff'] = np.hstack([self.data['kinematics_jnt_eff'], \
                                                            self.kinematics.main_jnt_efforts])
                    self.data['kinematics_target_pos']  = np.hstack([self.data['kinematics_target_pos'], \
                                                               self.kinematics.target_pos])
                    self.data['kinematics_target_quat'] = np.hstack([self.data['kinematics_target_quat'], \
                                                               self.kinematics.target_quat])
                
            if self.ft is not None:
                if 'ft_time' not in self.data.keys():
                    self.data['ft_time']   = [rospy.get_rostime().to_sec()-self.init_time]
                    self.data['ft_force']  = self.ft.force_raw
                    self.data['ft_torque'] = self.ft.torque_raw
                else:                    
                    self.data['ft_time'].append(rospy.get_rostime().to_sec()-self.init_time)
                    self.data['ft_force']  = np.hstack([self.data['ft_force'], self.ft.force_raw])
                    self.data['ft_torque'] = np.hstack([self.data['ft_torque'], self.ft.torque_raw])
                    
            if self.vision is not None:
                if 'vision_time' not in self.data.keys():
                    self.data['vision_time'] = [rospy.get_rostime().to_sec()-self.init_time]
                    self.data['vision_pos']  = self.vision.artag_pos
                    self.data['vision_quat'] = self.vision.artag_quat
                else:                    
                    self.data['vision_time'].append(rospy.get_rostime().to_sec()-self.init_time)
                    self.data['vision_pos']  = np.hstack([self.data['vision_pos'], self.vision.artag_pos])
                    self.data['vision_quat'] = np.hstack([self.data['vision_quat'], self.vision.artag_quat])
                                
            if self.pps_skin is not None:
                if 'pps_skin_time' not in self.data.keys():
                    self.data['pps_skin_time']  = [rospy.get_rostime().to_sec()-self.init_time]
                    self.data['pps_skin_left']  = self.pps_skin.data_left
                    self.data['pps_skin_right'] = self.pps_skin.data_right
                else:                    
                    self.data['pps_skin_time'].append(rospy.get_rostime().to_sec()-self.init_time)
                    self.data['pps_skin_left']  = np.hstack([self.data['pps_skin_left'], \
                                                             self.pps_skin.data_left])
                    self.data['pps_skin_right'] = np.hstack([self.data['pps_skin_right'], \
                                                             self.pps_skin.data_right])

            if self.fabric_skin is not None:
                if 'fabric_skin_time' not in self.data.keys():
                    self.data['fabric_skin_time'] = [rospy.get_rostime().to_sec()-self.init_time]
                    self.data['fabric_skin_centers_x'] = []
                    self.data['fabric_skin_centers_y'] = []
                    self.data['fabric_skin_centers_z'] = []
                    self.data['fabric_skin_normals_x'] = []
                    self.data['fabric_skin_normals_y'] = []
                    self.data['fabric_skin_normals_z'] = []
                    self.data['fabric_skin_values_x'] = []
                    self.data['fabric_skin_values_y'] = []
                    self.data['fabric_skin_values_z'] = []
                else:                    
                    self.data['fabric_skin_time'].append(rospy.get_rostime().to_sec()-self.init_time)
                self.data['fabric_skin_centers_x'] += self.fabric_skin.centers_x
                self.data['fabric_skin_centers_y'] += self.fabric_skin.centers_y
                self.data['fabric_skin_centers_z'] += self.fabric_skin.centers_z
                self.data['fabric_skin_normals_x'] += self.fabric_skin.normals_x
                self.data['fabric_skin_normals_y'] += self.fabric_skin.normals_y
                self.data['fabric_skin_normals_z'] += self.fabric_skin.normals_z
                self.data['fabric_skin_values_x'] += self.fabric_skin.values_x
                self.data['fabric_skin_values_y'] += self.fabric_skin.values_y
                self.data['fabric_skin_values_z'] += self.fabric_skin.values_z
                    
                                                             
            if self.enable_log_thread == False: break
            rate.sleep()
        
                

            
if __name__ == '__main__':

    subject = 'gatsbii'
    task    = 'test'
    verbose = True

    rospy.init_node('record_data')
    log = logger(ft=False, audio=True, kinematics=True, vision=False, pps=False, skin=False,\
                 subject=subject, task=task, verbose=verbose)

    rospy.sleep(1.0)
    log.run()
    ## log.runDataPub()
    
