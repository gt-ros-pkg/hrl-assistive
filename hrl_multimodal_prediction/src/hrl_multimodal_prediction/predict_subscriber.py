#!/usr/bin/env python

# Subscribes to Messages and caches them into circular buffers
# Preprocess data to mfcc and position and save

import rospy
import collections
from sklearn.preprocessing import MinMaxScaler
import librosa
import numpy as np

import config as cf
from std_msgs.msg import String, Float64, Float64MultiArray, MultiArrayLayout
from hrl_multimodal_prediction.msg import audio, pub_relpos, pub_mfcc
# from hrl_anomaly_detection.msg import audio
from visualization_msgs.msg import Marker


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

	def __init__(self):
		print 'init called'
		self.audio_pub = rospy.Publisher('preprocessed_audio', pub_mfcc, queue_size=10)
		self.relpos_pub = rospy.Publisher('preprocessed_relpos', pub_relpos, queue_size=10)

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
			self.static_ar, self.dynamic_ar = None, None
		
		msg = pub_relpos()
		msg.header.stamp = data.header.stamp
		msg.relpos = self.RelPos_Buffer
		self.relpos_pub.publish(msg)
		# print 'relpos'
		# print self.RelPos_Buffer

	def a_callback(self, data):
		# will read one msg at a time
		# print np.array(msg.audio_data, dtype=np.int16).shape
		audio = np.array(data.audio_data, dtype=np.int16)
		# audio_store = np.hstack(audio_store)
		# audio_store = np.array(audio_store, dtype=np.float64)
		mfccs = librosa.feature.mfcc(y=audio, sr=cf.RATE, hop_length=cf.HOP_LENGTH, n_fft=cf.N_FFT, n_mfcc=cf.N_MFCC)
		mfccs = np.array(mfccs, dtype=np.float64) # shape=(n_mfcc,t)
		mfccs = np.swapaxes(mfccs,0,1) # shape=(t, n_mfcc)
		self.Audio_Buffer = mfccs 
		
		msg = pub_mfcc()
		msg.header.stamp = data.header.stamp #processed data time stamp has delay for processing time
		msg.mfcc = self.Audio_Buffer.flatten() #Can read from subscribe and reconstruct into 2D
		self.audio_pub.publish(msg)
		# print 'audio'
		# print self.Audio_Buffer.shape

	def run(self):
		# rospy.init_node('predict_preprocessor', anonymous=True)
		# while not rospy.is_shutdown():
		rospy.Subscriber('hrl_manipulation_task/wrist_audio', audio, self.a_callback)
		rospy.Subscriber('visualization_marker', Marker, self.m_callback)
		rospy.spin()
		
def main():
	rospy.init_node('predict_preprocessor', anonymous=True)
	p_subs = predict_subscriber()
	p_subs.run()

if __name__ == '__main__':
	main()    
