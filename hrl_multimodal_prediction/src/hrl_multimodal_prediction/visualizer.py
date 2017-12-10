# One Core per thread
# publisher nodes, predict_subscriber, predictor, visualizer all have its own core

import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from matplotlib import style

import config as cf
from std_msgs.msg import String, Float64, Float64MultiArray, MultiArrayLayout
from hrl_multimodal_prediction.msg import audio, pub_relpos, pub_mfcc, plot_pub
from visualization_msgs.msg import Marker


#Subscribes to scaled-back Audio MFCC, and relpos data and plots in realtime 
# Will Receive Predicted and Audio data 
class visualizer():
	orig_pred = None

	xs_1, ys_1, pxs_1, pys_1 = [], [], [], []	
	xs_2, ys_2, pxs_2, pys_2 = [], [], [], []
	xs_3, ys_3, pxs_3, pys_3 = [], [], [], []
	xs_4, ys_4, pxs_4, pys_4 = [], [], [], []
	xs_5, ys_5, pxs_5, pys_5 = [], [], [], []
	xs_6, ys_6, pxs_6, pys_6 = [], [], [], []

	init_time = None
	init_flag = True
	def __init__(self):
		#initialize plotters
		# style.use('fivethirtyeight')
		self.fig = plt.figure()
		# self.ax1 = self.fig.add_subplot(2,1,1)
		self.ax2 = self.fig.add_subplot(2,1,2)
		# self.ax3 = self.fig.add_subplot(2,1,1)
		# self.ax4 = self.fig.add_subplot(3,2,4)
		self.ax5 = self.fig.add_subplot(2,1,1)
		# self.ax6 = self.fig.add_subplot(3,2,6)		

	def animate(self,i):
		if self.orig_pred is not None:
			# Get universal time for all plotting data
			stamp = self.orig_pred.header.stamp
			time = stamp.secs + stamp.nsecs * 1e-9
			if self.init_flag:
				self.init_flag = False
				self.init_time = time
			time = time - self.init_time
			# time = time - 1.51250859 *1e9
			# print time, time+1

			# 1,3,5 relpos
			relpos_pred = np.array(self.orig_pred.pred_relpos).reshape(cf.TIMESTEP_OUT,cf.IMAGE_DIM)
			# self.xs_1.append(time+0.2)
			# self.ys_1.append(self.orig_pred.orig_relpos[0]) #x
			# self.pxs_1.append(time)
			# self.pys_1.append(relpos_pred[9,0]) #x
			# time_var = time
			# for i in range(cf.TIMESTEP_OUT):
			# 	time_var += 0.2
			# 	self.pxs_1.append(time_var)
			# 	self.pys_1.append(relpos_pred[i,0]) #x

			# self.xs_3.append(time+1)
			# self.ys_3.append(self.orig_pred.orig_relpos[1]) #y
			# self.pxs_3.append(time)
			# self.pys_3.append(relpos_pred[9,1]) #y
			# # time_var = time
			# # for i in range(cf.TIMESTEP_OUT):
			# # 	time_var += 0.2
			# # 	self.pxs_3.append(time_var)
			# # 	self.pys_3.append(relpos_pred[i,1]) #y
			
			self.xs_5.append(time+0.2)
			self.ys_5.append(self.orig_pred.orig_relpos[2]) #z
			self.pxs_5.append(time)
			self.pys_5.append(relpos_pred[9,2]) #z
			# time_var = time
			# for i in range(cf.TIMESTEP_OUT):
			# 	time_var += 0.2
			# 	self.pxs_5.append(time_var)
			# 	self.pys_5.append(relpos_pred[i,2]) #z

			# 2,4,6, mfcc
			mfcc_pred = np.array(self.orig_pred.pred_mfcc).reshape(cf.TIMESTEP_OUT,cf.MFCC_DIM)
			self.xs_2.append(time+0.2)
			self.ys_2.append(self.orig_pred.orig_mfcc[0]) 
			self.pxs_2.append(time)
			self.pys_2.append(mfcc_pred[9,0]) 
			# time_var = time
			# for i in range(cf.TIMESTEP_OUT):
			# 	time_var += 0.2
			# 	self.pxs_2.append(time_var)
			# 	self.pys_2.append(mfcc_pred[i,0]) 
			
			# self.xs_4.append(time)
			# self.ys_4.append(self.orig_pred.orig_mfcc[1]) 
			# self.pxs_4.append(time+1)
			# self.pys_4.append(mfcc_pred[9,1]) 
			# # time_var = time
			# for i in range(cf.TIMESTEP_OUT):
			# 	time_var += 0.2
			# 	self.pxs_4.append(time_var)
			# 	self.pys_4.append(mfcc_pred[i,1]) 

			# self.xs_6.append(time)
			# self.ys_6.append(self.orig_pred.orig_mfcc[2]) 
			# self.pxs_6.append(time+1)
			# self.pys_6.append(mfcc_pred[9,2]) 
			# # time_var = time
			# for i in range(cf.TIMESTEP_OUT):
			# 	time_var += 0.2
			# 	self.pxs_6.append(time_var)
			# 	self.pys_6.append(mfcc_pred[i,2]) 

			# ax1,3,5 = relative position
			# self.ax1.clear()
			# self.ax1.set_title('Distance between two AR tags')
			# self.ax1.grid(True)
			# self.ax1.set_xlabel("Time [s]", fontweight='bold', fontsize=14)
			# self.ax1.set_ylabel("Distance Between Two AR Tags", fontweight='bold', fontsize=14)
			# self.ax1.plot(self.xs_1, self.ys_1, color='blue', label='current')	#original data
			# self.ax1.plot(self.pxs_1, self.pys_1, color='red', label='predicted') #predicted data
			# self.ax1.set_ylim([-0.4, 0.4])
			# self.ax1.legend()
			# self.ax3.grid(True)
			# self.ax3.set_xlabel("t")
			# self.ax3.set_ylabel("position y")
			# self.ax3.plot(self.xs_3, self.ys_3, color='blue')	#original data
			# self.ax3.plot(self.pxs_3, self.pys_3, color='red') #predicted data
			self.ax5.clear()
			self.ax5.grid(True)
			self.ax5.set_xlabel("Time [s]", fontweight='bold', fontsize=22)
			self.ax5.set_ylabel("Distance Between Two AR Tags", fontweight='bold', fontsize=22)
			self.ax5.plot(self.xs_5, self.ys_5, color='blue', linewidth=7, label='current')	#original data
			self.ax5.plot(self.pxs_5, self.pys_5, color='red', linewidth=7, label='predicted') #predicted data
			self.ax5.set_ylim([0, 0.25])
			self.ax5.legend(prop={'size': 22, 'weight': 'bold'})

			# ax2,4,6 = mfcc
			self.ax2.clear()
			# self.ax2.set_title('Sound')
			self.ax2.grid(True)
			self.ax2.set_xlabel("Time [s]", fontweight='bold', fontsize=22)
			self.ax2.set_ylabel("Sound Energy", fontweight='bold', fontsize=22)
			self.ax2.plot(self.xs_2, self.ys_2, color='blue', linewidth=7, label='current')	#original data
			self.ax2.plot(self.pxs_2, self.pys_2, color='red', linewidth=7, label='predicted') #predicted data
			self.ax2.set_ylim([300, 1100])
			self.ax2.legend(prop={'size': 22, 'weight': 'bold'})
			# self.ax4.grid(True)
			# self.ax4.set_xlabel("t")
			# self.ax4.set_ylabel("energy for freq range 2")
			# self.ax4.plot(self.xs_4, self.ys_4, color='blue')	#original data
			# self.ax4.plot(self.pxs_4, self.pys_4, color='red') #predicted data
			# self.ax6.grid(True)
			# self.ax6.set_xlabel("t")
			# self.ax6.set_ylabel("energy for freq range 3")
			# self.ax6.plot(self.xs_6, self.ys_6, color='blue')	#original data
			# self.ax6.plot(self.pxs_6, self.pys_6, color='red') #predicted data


	def callback(self, data):
		self.orig_pred = data

	def run(self):
		rospy.Subscriber('orig_pred_plotData', plot_pub, self.callback)
		ani = animation.FuncAnimation(self.fig, self.animate, interval=1000)
		plt.show()
		rospy.spin()
		

def main():
	rospy.init_node('visualizer', anonymous=True)
	viz = visualizer()
	viz.run()

if __name__ == '__main__':
	main()    

