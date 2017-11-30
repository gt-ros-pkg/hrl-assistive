#!/usr/bin/env python

# Subscribes to Messages and caches them into circular buffers
# Preprocess data to mfcc and position and save

import rospy
from std_msgs.msg import String, Int16MultiArray
import collections
from sklearn.preprocessing import MinMaxScaler
import librosa
import config as cfz
from hrl_multimodal_prediction.msg import audio
from visualization_msgs.msg import Marker
import numpy as np

# This class simply listens to publishers and collect data into a global circular buffer
# Saves relative position data
class predict_subscriber():
	MAX_LEN = 10
	Audio_Buffer = collections.deque(maxlen=MAX_LEN)
	RelPos_Buffer = collections.deque(maxlen=MAX_LEN)
	static_ar = None
	dynamic_ar = None

	def callback(self, data):
		# will read one msg at a time
		if data.msg._type == 'hrl_anomaly_detection/audio':
			# print np.array(msg.audio_data, dtype=np.int16).shape
			audio = np.array(msg.audio_data, dtype=np.int16)
			# audio_store = np.hstack(audio_store)
			# audio_store = np.array(audio_store, dtype=np.float64)
			mfccs = librosa.feature.mfcc(y=audio, sr=cf.RATE, hop_length=cf.HOP_LENGTH, n_fft=cf.N_FFT, n_mfcc=cf.N_MFCC)
			mfccs = np.array(mfccs)
			Audio_Buffer.append(mfccs)
		# elif data.msg._type == 'visualization_msgs/Marker':
		if data.id == 0: #id 0 = static
			# print np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]).shape
			self.static_ar = [data.pose.position.x, data.pose.position.y, data.pose.position.z]
		elif data.id == 9: #id 9 = dynamic
			self.dynamic_ar = [data.pose.position.x, data.pose.position.y, data.pose.position.z]

		if self.static_ar is not None and self.dynamic_ar is not None:
			self.static_ar = np.array(self.static_ar, dtype=np.float64)
			self.dynamic_ar = np.array(self.dynamic_ar, dtype=np.float64)
			self.RelPos_Buffer.append(self.static_ar - self.dynamic_ar)
			self.static_ar, self.dynamic_ar = None, None

	def run(self):
		rospy.init_node('listener', anonymous=True)
		rospy.Subscriber('hrl_manipulation_task/wrist_audio', audio, self.callback)
		rospy.Subscriber('visualization_marker', Marker, self.callback)
		rospy.spin()

def main():
	p_subs = predict_subscriber()
	p_subs.run()

if __name__ == '__main__':
	main()    
