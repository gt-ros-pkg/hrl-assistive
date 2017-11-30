#!/usr/bin/env python

# Subscribes to Messages and caches them into circular buffers
# Preprocess data to mfcc and position and save

import rospy
from std_msgs.msg import String, Float64, Float64MultiArray
import collections
from sklearn.preprocessing import MinMaxScaler
import librosa
import config as cf
from hrl_anomaly_detection.msg import audio
from visualization_msgs.msg import Marker
import numpy as np

# This class simply listens to publishers and collect data into a global circular buffer
# Saves relative position data
class predict_subscriber():
	MAX_LEN = 10
	# Audio_Buffer = collections.deque(maxlen=MAX_LEN)
	# RelPos_Buffer = collections.deque(maxlen=MAX_LEN)
	Audio_Buffer = None
	RelPos_Buffer = None 
	static_ar = None
	dynamic_ar = None
	audio_pub = None
	relpos_pub = None
	MUTEX_A = False
	MUTEX_M = False

	def __init__(self):
		self.audio_pub = rospy.Publisher('preprocessed_audio', Float64MultiArray, queue_size=10)
		self.relpos_pub = rospy.Publisher('preprocessed_relpos', Float64MultiArray, queue_size=10)
		# self.Audio_Buffer = None
		# self.RelPos_Buffer = None
		# self.MUTEX_M = False
		# self.MUTEX_A = False

	def m_callback(self, data):
		if data.id == 0: #id 0 = static
			# print np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]).shape
			self.static_ar = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
		elif data.id == 9: #id 9 = dynamic
			self.dynamic_ar = [data.pose.position.x, data.pose.position.y, data.pose.position.z]

		if self.static_ar is not None and self.dynamic_ar is not None:
			self.static_ar = np.array(self.static_ar, dtype=np.float64)
			self.dynamic_ar = np.array(self.dynamic_ar, dtype=np.float64)
			self.RelPos_Buffer = self.static_ar - self.dynamic_ar
			self.MUTEX_M = True
			self.static_ar, self.dynamic_ar = None, None
		print 'relpos'
		print self.RelPos_Buffer

	def a_callback(self, data):
		# will read one msg at a time
		# print np.array(msg.audio_data, dtype=np.int16).shape
		audio = np.array(data.audio_data, dtype=np.int16)
		# audio_store = np.hstack(audio_store)
		# audio_store = np.array(audio_store, dtype=np.float64)
		mfccs = librosa.feature.mfcc(y=audio, sr=cf.RATE, hop_length=cf.HOP_LENGTH, n_fft=cf.N_FFT, n_mfcc=cf.N_MFCC)
		mfccs = np.array(mfccs)
		self.Audio_Buffer = mfccs
		self.MUTEX_A = True
		print 'audio'
		print self.Audio_Buffer

	def run(self):
		rospy.init_node('predict_preprocessor', anonymous=True)
		# while not rospy.is_shutdown():
		rospy.Subscriber('hrl_manipulation_task/wrist_audio', audio, self.a_callback)
		rospy.Subscriber('visualization_marker', Marker, self.m_callback)
		rospy.spin()
			# self.audio_pub.publish(self.Audio_Buffer)
			# self.relpos_pub.publish(self.RelPos_Buffer)

		# rospy.spin()
		# 	self.audio_pub.publish(self.Audio_Buffer)
		# 	self.relpos_pub.publish(self.RelPos_Buffer)
		
def main():
	p_subs = predict_subscriber()
	print 'data being preprocessed and published'
	p_subs.run()

if __name__ == '__main__':
	main()    
