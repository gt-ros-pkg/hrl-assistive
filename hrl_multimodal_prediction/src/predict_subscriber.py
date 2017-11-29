#!/usr/bin/env python

# Subscribes to Messages and caches them into circular buffers
# Preprocess data to mfcc and position and save

import rospy
from std_msgs.msg import String
import collections
from sklearn.preprocessing import MinMaxScaler
import librosa
import config as cf
# This class simply listens to publishers and collect data into a global circular buffer
# Saves relative position data
class predict_subscriber():
	MAX_LEN = 10
	Audio_Buffer = collections.deque(maxlen=MAX_LEN)
	RelPos_Buffer = collections.deque(maxlen=MAX_LEN)
	static_ar = []
	dynamic_ar = []

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
		elif data.msg._type == 'visualization_msgs/Marker':
			if data.msg.id == 0: #id 0 = static
				# print np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]).shape
				static_ar = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
			elif data.msg.id == 9: #id 9 = dynamic
				dynamic_ar = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

			if static_ar is not None and dynamic_ar is not None:
				RelPos_Buffer.append(np.array(static_ar - dynamic_ar), dtype=np.float64)
				static_ar, dynamic_ar = [], []

	def run(self):
		rospy.init_node('listener', anonymous=True)
		rospy.Subscriber('rosbag_node', rosbag_message, callback)
		rospy.spin()

def main():
	p_subs = predict_subscriber()
	p_subs.run()

if __name__ == '__main__':
	main()    
