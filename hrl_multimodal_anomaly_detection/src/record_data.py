#!/usr/bin/env python

# System
import os
import sys
import glob
import time
import getpass
from pylab import *

from audio.tool_audio import tool_audio
from vision.tool_vision import tool_vision
from kinematics.robot_kinematics import robot_kinematics
from forces.tool_ft import tool_ft

# ROS
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import rospy, optparse
import tf

# HRL
import hrl_lib.util as ut

def log_parse():
    parser = optparse.OptionParser('Input the Pose node name and the ft sensor node name')

    parser.add_option("-t", "--tracker", action="store", type="string",\
    dest="tracker_name", default="adl2")
    parser.add_option("-f", "--force" , action="store", type="string",\
    dest="ft_sensor_name",default="/netft_data")

    (options, args) = parser.parse_args()

    return options.tracker_name, options.ft_sensor_name


class ADL_log:
    def __init__(self, ft=True, audio=False, vision=False, kinematics=False, subject=None, task=None):
        self.init_time = 0
        self.tool_tracker_name, self.ft_sensor_topic_name = log_parse()
        self.tf_listener = tf.TransformListener()
        rospy.logout('ADLs_log node subscribing..')

        self.ft = tool_ft('/netft_data') if ft else None
        self.audio = tool_audio() if audio else None
        self.vision = tool_vision(self.tf_listener) if vision else None
        self.kinematics = robot_kinematics(self.tf_listener) if kinematics else None

        # File saving
        self.iteration = 0
        self.subject = subject.replace(' ', '')
        self.task = 'scooping' if task == 's' else 'feeding'

        # raw_input('press Enter to reset')
        # if ft: self.ft.reset()
        # if audio: self.audio.reset()

    def log_start(self):
        self.init_time = rospy.get_time()
        if self.ft is not None:
            self.ft.init_time = self.init_time
            self.ft.start()
        if self.audio is not None:
            self.audio.init_time = self.init_time
            self.audio.start()
        if self.vision is not None:
            self.vision.init_time = self.init_time
            self.vision.start()
        if self.kinematics is not None:
            self.kinematics.init_time = self.init_time
            self.kinematics.start()

    def close_log_file(self):
        data = dict()
        data['init_time'] = self.init_time

        if self.ft is not None:
            self.ft.cancel()
            ## data['force'] = self.ft.force_data
            ## data['torque'] = self.ft.torque_data
            data['ft_force_raw']  = self.ft.force_raw_data
            data['ft_torque_raw'] = self.ft.torque_raw_data
            data['ft_time']       = self.ft.time_data

        if self.audio is not None:
            self.audio.cancel()
            data['audio_data']  = self.audio.audio_data
            data['audio_amp']   = self.audio.audio_amp
            data['audio_freq']  = self.audio.audio_freq
            data['audio_chunk'] = self.audio.CHUNK
            data['audio_time']  = self.audio.time_data
            data['audio_data_raw'] = self.audio.audio_data_raw

        if self.vision is not None:
            self.vision.cancel()
            data['visual_points'] = self.vision.visual_points
            data['visual_time'] = self.vision.time_data

        if self.kinematics:
            self.kinematics.cancel()
            data['kinematics_time']  = self.kinematics.time_data
            data['kinematics_joint'] = self.kinematics.joint_data
            data['l_end_effector_pos'] = self.kinematics.l_end_effector_pos
            data['l_end_effector_quat'] = self.kinematics.l_end_effector_quat
            data['r_end_effector_pos'] = self.kinematics.r_end_effector_pos
            data['r_end_effector_quat'] = self.kinematics.r_end_effector_quat

        flag = raw_input('Enter trial\'s status (e.g. 1:success, 2:failure, 3: exit): ')
        if flag == '1': status = 'success'
        elif flag == '2': status = 'failure'
        elif flag == '3': sys.exit()
        else: status = flag

        directory = os.path.join(os.path.dirname(__file__), '../recordings/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        folderName = os.path.join(directory, self.subject + '_' + self.task + '_' + time.strftime('%m-%d-%Y_%H-%M-%S/'))
        fileName = os.path.join(folderName, 'iteration_%d_%s.pkl' % (self.iteration, status))
        ut.save_pickle(data, fileName)
        self.iteration += 1

        # Reinitialize all sensors
        if self.ft is not None:
            self.ft = tool_ft('/netft_data')
        if self.audio is not None:
            self.audio = tool_audio()
        if self.vision is not None:
            self.vision = tool_vision(self.tf_listener)
        if self.kinematics is not None:
            self.kinematics = robot_kinematics(self.tf_listener)


if __name__ == '__main__':
    subject = 'gatsbii'
    task = '10'
    actor = '2'
    manip = True

    ## log = ADL_log(audio=True, ft=True, manip=manip, test_mode=False)
    log = ADL_log(ft=True, audio=True, vision=True, kinematics=True, subject=subject, task=task)

    log.log_start()

    rate = rospy.Rate(1000) # 25Hz, nominally.
    while not rospy.is_shutdown():
        rate.sleep()

    log.close_log_file()

