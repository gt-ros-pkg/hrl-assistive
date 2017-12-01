# One Core per thread
# publisher nodes, predict_subscriber, predictor, visualizer all have its own core

import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from matplotlib import style

import config as cf
from std_msgs.msg import String, Float64, Float64MultiArray, MultiArrayLayout
from hrl_multimodal_prediction.msg import audio, pub_relpos, pub_mfcc
from visualization_msgs.msg import Marker


#Subscribes to scaled-back Audio MFCC, and relpos data and plots in realtime 
# Will Receive Predicted and Audio data 
class visualizer():
	relpos = None
	mfcc = None
	# Above will be combined in one message and published from the predictor
	# Can use the same callback in that case

	xs_1, ys_1, pxs_1, pys_1 = [], [], [], []	
	xs_2, ys_2, pxs_2, pys_2 = [], [], [], []
	xs_3, ys_3, pxs_3, pys_3 = [], [], [], []
	xs_4, ys_4, pxs_4, pys_4 = [], [], [], []
	xs_5, ys_5, pxs_5, pys_5 = [], [], [], []
	xs_6, ys_6, pxs_6, pys_6 = [], [], [], []

	def __init__(self):
		#initialize plotters
		# style.use('fivethirtyeight')
		self.fig = plt.figure()
		self.ax1 = self.fig.add_subplot(3,2,1)
		self.ax2 = self.fig.add_subplot(3,2,2)
		self.ax3 = self.fig.add_subplot(3,2,3)
		self.ax4 = self.fig.add_subplot(3,2,4)
		self.ax5 = self.fig.add_subplot(3,2,5)
		self.ax6 = self.fig.add_subplot(3,2,6)		

	def animate(self,i):
		if self.relpos is not None and self.mfcc is not None:
			# 1,3,5 relpos
			stamp = self.relpos.header.stamp
			time = stamp.secs + stamp.nsecs * 1e-9
			self.xs_1.append(time)
			self.ys_1.append(self.relpos.relpos[0]) #x
			self.xs_3.append(time)
			self.ys_3.append(self.relpos.relpos[1]) #y
			self.xs_5.append(time)
			self.ys_5.append(self.relpos.relpos[2]) #z
			# More for Predicted

			# 2,4,6, mfcc
			stamp = self.mfcc.header.stamp
			time = stamp.secs + stamp.nsecs * 1e-9
			self.xs_2.append(time)
			self.ys_2.append(self.mfcc.mfcc[6]) 
			self.xs_4.append(time)
			self.ys_4.append(self.mfcc.mfcc[7]) 
			self.xs_6.append(time)
			self.ys_6.append(self.mfcc.mfcc[8]) 
			# More for Predicted

			# ax1,3,5 = relative position
			self.ax1.clear()
			self.ax1.set_title('Reletive Position')
			self.ax1.grid(True)
			self.ax1.set_xlabel("t")
			self.ax1.set_ylabel("position x")
			self.ax1.plot(self.xs_1, self.ys_1, color='blue')	#original data
			# self.ax1.plot(self.pxs_1, self.pys_1, color='red') #predicted data
			self.ax3.grid(True)
			self.ax3.set_xlabel("t")
			self.ax3.set_ylabel("position y")
			self.ax3.plot(self.xs_3, self.ys_3, color='blue')	#original data
			# self.ax3.plot(self.pxs_3, self.pys_3, color='red') #predicted data
			self.ax5.grid(True)
			self.ax5.set_xlabel("t")
			self.ax5.set_ylabel("position z")
			self.ax5.plot(self.xs_5, self.ys_5, color='blue')	#original data
			# self.ax5.plot(self.pxs_5, self.pys_5, color='red') #predicted data

			# ax2,4,6 = mfcc
			self.ax2.clear()
			self.ax2.set_title('MFCC')
			self.ax2.grid(True)
			self.ax2.set_xlabel("t")
			self.ax2.set_ylabel("energy for freq range 1")
			self.ax2.plot(self.xs_2, self.ys_2, color='blue')	#original data
			# self.ax2.plot(self.pxs_1, self.pys_1, color='red') #predicted data
			self.ax4.grid(True)
			self.ax4.set_xlabel("t")
			self.ax4.set_ylabel("energy for freq range 2")
			self.ax4.plot(self.xs_4, self.ys_4, color='blue')	#original data
			# self.ax4.plot(self.pxs_3, self.pys_3, color='red') #predicted data
			self.ax6.grid(True)
			self.ax6.set_xlabel("t")
			self.ax6.set_ylabel("energy for freq range 3")
			self.ax6.plot(self.xs_6, self.ys_6, color='blue')	#original data
			# self.ax6.plot(self.pxs_5, self.pys_5, color='red') #predicted data


	def callback(self, data):
		self.relpos = data

	def callback2(self, data):
		self.mfcc = data

	def run(self):
		rospy.Subscriber('preprocessed_relpos', pub_relpos, self.callback)
		rospy.Subscriber('preprocessed_audio', pub_mfcc, self.callback2)
		ani = animation.FuncAnimation(self.fig, self.animate, interval=1000)
		plt.show()
		rospy.spin()
		

def main():
	rospy.init_node('visualizer', anonymous=True)
	viz = visualizer()
	viz.run()

if __name__ == '__main__':
	main()    

