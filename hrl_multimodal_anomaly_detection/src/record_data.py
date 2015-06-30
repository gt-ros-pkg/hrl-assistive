#!/usr/bin/env python

# System
import os
import sys
import glob
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
from hrl_srvs.srv import None_Bool, None_BoolResponse
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
    def __init__(self, ft=True, audio=False, audioRecord=False, vision=False, kinematics=False, manip=False, test_mode=False,
                    subject=None, task=None, actor=None):
        #rospy.init_node('ADLs_log', anonymous = True)

        self.manip = manip
        self.test_mode = test_mode
        self.audioRec = audioRecord

        self.sub_name = subject
        self.task_name = task
        self.actor = actor

        self.init_time = 0.
        self.file_name = 'test'
        self.tool_tracker_name, self.ft_sensor_topic_name = log_parse()
        self.tf_listener = tf.TransformListener()
        rospy.logout('ADLs_log node subscribing..')

        if self.manip:
            rospy.wait_for_service("/arm_reach_enable")
            self.armReachAction = rospy.ServiceProxy("/arm_reach_enable", None_Bool)
            rospy.loginfo("arm reach server connected!!")

        self.ft = tool_ft('/netft_data') if ft else None
        self.audio = tool_audio(self.audioRec) if audio else None
        self.vision = tool_vision(self.tf_listener) if vision else None
        self.kinematics = robot_kinematics(self.tf_listener) if kinematics else None

        # raw_input('press Enter to reset')
        # if ft: self.ft.reset()
        # if audio: self.audio.reset()

    def log_start(self, trial_name):

        raw_input('press Enter to begin the test')
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

        if self.manip:
            rospy.sleep(1.0)
            ret = self.armReachAction()
            print ret

            self.close_log_file(trial_name)
            sys.exit()


    def close_log_file(self, trial_name):
        # Finish data collection
        if self.ft: self.ft.cancel()
        if self.audio: self.audio.cancel()
        if self.vision: self.vision.cancel()
        if self.kinematics: self.kinematics.cancel()


        d = dict()
        d['init_time'] = self.init_time

        if self.ft:
            ## dict['force'] = self.ft.force_data
            ## dict['torque'] = self.ft.torque_data
            d['ft_force_raw']  = self.ft.force_raw_data
            d['ft_torque_raw'] = self.ft.torque_raw_data
            d['ft_time']       = self.ft.time_data

        if self.audio:
            d['audio_data']  = self.audio.audio_data
            d['audio_amp']   = self.audio.audio_amp
            d['audio_freq']  = self.audio.audio_freq
            d['audio_chunk'] = self.audio.CHUNK
            d['audio_time']  = self.audio.time_data
            #d['audio_data_raw'] = self.audio.audio_data_raw

        if self.vision:
            d['visual_points'] = self.vision.visual_points
            d['visual_time'] = self.vision.time_data

        if self.kinematics:
            d['kinematics_time']  = self.kinematics.time_data
            d['kinematics_joint'] = self.kinematics.joint_data
            d['l_end_effector_pos'] = self.kinematics.l_end_effector_pos
            d['l_end_effector_quat'] = self.kinematics.l_end_effector_quat
            d['r_end_effector_pos'] = self.kinematics.r_end_effector_pos
            d['r_end_effector_quat'] = self.kinematics.r_end_effector_quat


        ## if trial_name is not None: self.trial_name = trial_name
        ## else:
        flag = raw_input("Enter trial's name (e.g. 1:success, 2:failure_reason, 3: exit): ")
        #flag = "1"
        if flag == "1": trial_name = 'success'
        elif flag == "2": trial_name = trial_name
        elif flag == "3": sys.exit()
        else: trial_name = flag

        current_user = getpass.getuser()
        folder_name = '/home/'+ current_user + '/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/'
        print "Current save folder is: " + folder_name
        change_folder = raw_input("Change save folder? [y/n]")
        if change_folder == 'y':
            folder_name = raw_input("Enter new save folder: [ex: ../recordings/]")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.file_name = folder_name + self.sub_name + '_' + self.task_name + '_' + self.actor + '_' + trial_name

        #SAVING AS PICKLE FILE!#
        #OLD METHOD, USING PANDAS INSTEAD!#
        pkl_list = glob.glob('*.pkl')
        max_num = 0

        for pkl in pkl_list:
            if pkl.find(self.file_name)>=0:
                num = int(pkl.split('_')[-1].split('.')[0])
                if max_num < num:
                    max_num = num
        max_num = int(max_num)+1
        self.pkl = self.file_name+'_'+str(max_num)+'.pkl'

        print "Pickle file name: ", self.pkl
        ut.save_pickle(d, self.pkl)

        #NOT USED Pandas File formating saving!
     #    csv_list = glob.glob('*.csv')
     #    max_num = 0

     #    for csv in csv_list:
     #        if csv.find(self.file_name)>=0:
     #            num = int(csv.split('_')[-1].split('.')[0])
     #            if max_num < num:
     #                max_num = num
     #    max_num = int(max_num)+1
     #    self.csv_file_name = self.file_name+'_'+str(max_num)+'.csv'


	    # self.csv_file_name_FT = self.file_name+'_FT_'+str(max_num)+'.csv'
     #    print "CSV (Pandas) file name: ", self.csv_file_name
     #    df.to_csv(self.csv_file_name)

        ## self.tool_tracker_log_file.close()
        ## self.tooltip_log_file.close()
        ## self.head_tracker_log_file.close()
        ## self.gen_log_file.close()
        print 'Closing...  log files have saved...'

        if self.audioRec:
            self.audio.stopWav(self.file_name)


if __name__ == '__main__':

    subject = 'gatsbii'
    task = '10'
    actor = '2'
    trial_name = 'success'
    manip=True

    ## log = ADL_log(audio=True, ft=True, manip=manip, test_mode=False)
    log = ADL_log(audio=True, ft=True, vision=True, kinematics=True,  manip=manip, test_mode=False, subject=subject, task=task, actor=actor)

    log.log_start(trial_name)

    rate = rospy.Rate(1000) # 25Hz, nominally.
    while not rospy.is_shutdown():
        rate.sleep()

    log.close_log_file(trial_name)

