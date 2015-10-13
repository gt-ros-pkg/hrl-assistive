#!/usr/bin/env python

# system
import rospy
import roslib
roslib.load_manifest('hrl_manipulation_task')
import os, sys, threading, copy
import gc

# util
import numpy as np

# 
from hrl_multimodal_anomaly_detection.hmm import util
import hrl_lib.util as ut

#
from sensor.kinect_audio import kinect_audio
from sensor.robot_kinematics import robot_kinematics
from sensor.tool_ft import tool_ft
## from vision.tool_vision import tool_vision


class logger:
    def __init__(self, ft=False, audio=False, kinematics=False, subject=None, task=None, verbose=False):
        rospy.logout('ADLs_log node subscribing..')

        self.subject = subject
        self.task    = task
        self.verbose = verbose
        
        self.initParams()
        
        self.audio      = kinect_audio(True) if audio else None
        self.kinematics = robot_kinematics() if kinematics else None
        self.ft         = tool_ft() if ft else None


    def initParams(self):
        '''
        # load parameters
        '''        
        # File saving
        self.record_root_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016'
        self.folderName = os.path.join(self.record_root_path, self.subject + '_' + self.task)
        
        
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
            
    def close_log_file(self):
        data = {}
        data['init_time'] = self.init_time

        if self.audio is not None:
            self.audio.cancel()            
            data['audio_time']  = self.audio.time_data
            data['audio_feature']  = self.audio.audio_feature
            data['audio_power'] = self.audio.audio_power
            data['audio_azimuth']  = self.audio.audio_azimuth
            data['audio_cmd']  = self.audio.audio_cmd

        if self.kinematics is not None:
            self.kinematics.cancel()
            data['kinematics_time']    = self.kinematics.time_data
            data['kinematics_ee_pos']  = self.kinematics.kinematics_ee_pos
            data['kinematics_ee_quat'] = self.kinematics.kinematics_ee_quat
            data['kinematics_jnt_pos'] = self.kinematics.kinematics_jnt_pos
            data['kinematics_jnt_vel'] = self.kinematics.kinematics_jnt_vel
            data['kinematics_jnt_eff'] = self.kinematics.kinematics_jnt_eff

        if self.ft is not None:
            self.ft.cancel()
            data['ft_time']   = self.ft.time_data
            data['ft_force']  = self.ft.force_array
            data['ft_torque'] = self.ft.torque_array
            
        flag = raw_input('Enter trial\'s status (e.g. 1:success, 2:failure, 3: exit): ')
        if flag == '1':   status = 'success'
        elif flag == '2': status = 'failure'
        elif flag == '3': sys.exit(0)
        else: status = flag

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
        ## if self.ft is not None:
        ##     self.ft = tool_ft('/netft_data')
        if self.audio is not None: self.audio = kinect_audio()
        if self.kinematics is not None: self.kinematics = robot_kinematics()
        if self.ft is not None: self.ft = tool_ft()

        gc.collect()


    def waitForReady(self):

        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():

            flag = 1.0
            if self.audio is not None:
                if self.audio.isReady(): flag *= 1.0
                else: flag *= -1.0

            if self.kinematics is not None:
                if self.kinematics.isReady(): flag *= 1.0
                else: flag *= -1.0

            if self.ft is not None:
                if self.ft.isReady(): flag *= 1.0
                else: flag *= -1.0

            if flag == 1.0: break                    
            rate.sleep()

        if self.verbose: print "record_data>> completed to wait sensing data"
            

    def run(self):

        self.waitForReady()
        self.log_start()

        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():
            rate.sleep()

        self.close_log_file()

    
        
                
if __name__ == '__main__':

    subject = 'gatsbii'
    task    = '10'
    actor   = '2'
    manip   = True
    verbose = True

    rospy.init_node('record_data')
    log = logger(ft=True, audio=True, kinematics=True, subject=subject, task=task, verbose=verbose)

    rospy.sleep(1.0)
    log.run()
    
