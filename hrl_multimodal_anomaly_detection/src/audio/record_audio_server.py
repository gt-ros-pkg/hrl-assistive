#!/usr/bin/env python

import roslib
# roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import rospy
import numpy as np
import time
import gc
import os
import sys
from pylab import *

import cPickle as pickle


import hrl_haptic_mpc.haptic_mpc_util as haptic_mpc_util
from audio.tool_audio import tool_audio

from hrl_srvs.srv import None_Bool, None_BoolResponse, Int_Int
from hrl_multimodal_anomaly_detection.srv import String_String
from std_msgs.msg import String

class audioServer():

	def __init__(self):

		self.audio = tool_audio()

		self.audio_server = rospy.Service('/audio_server', String_String, self.audioCallback)
		self.audio_server_trial_name = rospy.Service('/audio_server_trial_name', String_String, self.audioTrialNameCallback)

		self.trial_name = None

	def audioCallback(self, req):
		req = req.data

		if req == "start"
			self.init_time = rospy.get_time()
			self.audio.init_time = self.init_time
			self.audio.start()

		if req == "cancel"
			self.audio.cancel()

			data['audio_chunk'] = self.audio.CHUNK
            data['audio_sample_time'] = self.audio.UNIT_SAMPLE_TIME
            data['audio_time']  = self.audio.time_data
            data['audio_data_raw'] = self.audio.audio_data_raw

		if req == "close_log_file"

			filename = self.trial_name
			with open(fileName, 'wb') as f:
            	pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        	print 'Data saved to', fileName

	def audioTrialNameCallback(self, req):

		req = req.data
		self.trial_name = req

		self.trial_name = self.trial_name[:-4]
		self.trial_name = self.trial_name + "_AUDIO_ONLY.pkl" 

	def start(self):
		print "Nothing"
	def cancel(self):
		print "Nothing"
	def run(self):
		print "Nothing"



if __name__ == '__main__':
	rospy.init_node('audio_server')

	audio = audioServer()
	rospy.spin()
