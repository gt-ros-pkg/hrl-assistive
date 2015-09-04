#!/usr/bin/env python

# System
import os
import gc
import sys
import time
from pylab import *
import cPickle as pickle

from audio.tool_audio_slim import tool_audio_slim
from vision.tool_vision import tool_vision
from kinematics.tool_kinematics import tool_kinematics
from forces.tool_ft import tool_ft

# ROS
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
from geometry_msgs.msg import PoseStamped
import rospy, optparse
import tf

from hrl_srvs.srv import None_Bool, None_BoolResponse, Int_Int
from hrl_multimodal_anomaly_detection.srv import String_String
from hmm.util import *

def log_parse():
    parser = optparse.OptionParser('Input the Pose node name and the ft sensor node name')

    parser.add_option("-t", "--tracker", action="store", type="string",\
    dest="tracker_name", default="adl2")
    parser.add_option("-f", "--force" , action="store", type="string",\
    dest="ft_sensor_name",default="/netft_data")

    (options, args) = parser.parse_args()

    return options.tracker_name, options.ft_sensor_name


class ADL_log:
    def __init__(self, ft=True, audio=False, kinematics=False, subject=None, task=None):
        self.init_time = 0
        self.tool_tracker_name, self.ft_sensor_topic_name = log_parse()
        self.tf_listener = tf.TransformListener()
        rospy.logout('ADLs_log node subscribing..')
        self.isScooping = True if task == 's' else False

        # self.audio = rospy.ServiceProxy('/audio_server', String_String) if audio else None
        # self.audioTrialName = rospy.ServiceProxy('/audio_server_trial_name', String_String) if audio else None

        self.ft = tool_ft('/netft_data') if ft else None
        self.audio = tool_audio_slim() if audio else None
        self.kinematics = tool_kinematics(self.tf_listener, targetFrame='/torso_lift_link', isScooping=self.isScooping) if kinematics else None

        # File saving
        self.iteration = 0
        self.subject = subject.replace(' ', '')
        self.task = 'scooping' if task == 's' else 'feeding'

        directory = os.path.join(os.path.dirname(__file__), '../recordings/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        sensors = ''
        if self.ft is not None: sensors += 'f'
        if self.audio is not None: sensors += 'a'
        if self.kinematics is not None: sensors += 'k'

        self.record_root_path = directory
        self.folderName = os.path.join(directory, self.subject + '_' + self.task)
        ## self.folderName = os.path.join(directory, self.subject + '_' + self.task + '_' + sensors + '_' + time.strftime('%m-%d-%Y_%H-%M-%S/'))

        self.scooping_steps_times = []

        self.scoopingStepsService = rospy.Service('/scooping_steps_service', None_Bool, self.scoopingStepsTimesCallback)

    def log_start(self):
        self.init_time = rospy.get_time()
        if self.ft is not None:
            self.ft.init_time = self.init_time
            self.ft.start()
        if self.audio is not None:
            self.audio.init_time = self.init_time
            self.audio.start()
            self.audio.reset(self.init_time)
        if self.kinematics is not None:
            self.kinematics.init_time = self.init_time
            self.kinematics.start()

    def close_log_file(self):
        data = dict()
        data['init_time'] = self.init_time

        if self.kinematics:
            self.kinematics.cancel()
            data['kinematics_time']  = self.kinematics.time_data
            data['kinematics_data'] = self.kinematics.kinematics_data

        time.sleep(0.01)

        if self.ft is not None:
            self.ft.cancel()
            ## data['force'] = self.ft.force_data
            ## data['torque'] = self.ft.torque_data
            data['ft_force_raw']  = self.ft.force_raw_data
            data['ft_torque_raw'] = self.ft.torque_raw_data
            data['ft_time']       = self.ft.time_data

        if self.audio is not None:
            self.audio.cancel()
            
            data['audio_chunk'] = self.audio.CHUNK
            data['audio_sample_time'] = self.audio.UNIT_SAMPLE_TIME
            data['audio_time']  = self.audio.time_data
            data['audio_data_raw'] = self.audio.audio_data_raw

        data['scooping_steps_times'] = self.scooping_steps_times
        self.scooping_steps_times = []

        flag = raw_input('Enter trial\'s status (e.g. 1:success, 2:failure, 3: exit): ')
        if flag == '1': status = 'success'
        elif flag == '2': status = 'failure'
        elif flag == '3': sys.exit(0)
        else: status = flag

        if status == 'failure':
            failure_class = raw_input('Enter failure reason if there is: ')
                
        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)

        # get next file id
        if status == 'success':
            fileList = getSubjectFileList(self.record_root_path, [self.subject], self.task)[0]
        else:
            fileList = getSubjectFileList(self.record_root_path, [self.subject], self.task)[1]
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

        #Send trial name to audio recording server!
        # if self.audio is not None:
        #     self.audioTrialName(fileName)

        with open(fileName, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print 'Data saved to', fileName

        self.iteration += 1

        # Reinitialize all sensors
        if self.ft is not None:
            self.ft = tool_ft('/netft_data')
        if self.audio is not None:
            self.audio = tool_audio_slim()
        if self.kinematics is not None:
            self.kinematics = tool_kinematics(self.tf_listener, targetFrame='/torso_lift_link', isScooping=self.isScooping)

        gc.collect()

    def scoopingStepsTimesCallback(self, data):
        self.scooping_steps_times.append(rospy.get_time() - self.init_time)
        return None_BoolResponse(True)

if __name__ == '__main__':
    subject = 'gatsbii'
    task = '10'
    actor = '2'
    manip = True

    ## log = ADL_log(audio=True, ft=True, manip=manip, test_mode=False)
    log = ADL_log(ft=True, audio=True, kinematics=True, subject=subject, task=task)

    log.log_start()

    rate = rospy.Rate(1000) # 25Hz, nominally.
    while not rospy.is_shutdown():
        rate.sleep()

    log.close_log_file()

