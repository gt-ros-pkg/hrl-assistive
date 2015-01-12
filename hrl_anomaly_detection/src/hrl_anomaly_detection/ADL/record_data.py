#!/usr/bin/env python

# System
import numpy as np
import time, sys, threading
import cPickle as pkl


# ROS
import roslib
roslib.load_manifest('hrl_anomaly_detection')
roslib.load_manifest('geometry_msgs')
roslib.load_manifest('hrl_lib')
import rospy, optparse, math, time
import tf
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import TransformStamped, WrenchStamped
from std_msgs.msg import Bool

# HRL
from hrl_srvs.srv import None_Bool, None_BoolResponse
from hrl_msgs.msg import FloatArray

# Private
#import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad
#import hrl_anomaly_detection.advait.arm_trajectories as at

def log_parse():
	parser = optparse.OptionParser('Input the Pose node name and the ft sensor node name')

	parser.add_option("-t", "--tracker", action="store", type="string",\
		dest="tracker_name", default="adl2")
	parser.add_option("-f", "--force" , action="store", type="string",\
		dest="ft_sensor_name",default="/netft_data")

	(options, args) = parser.parse_args()

	return options.tracker_name, options.ft_sensor_name 


class tool_ft():
	def __init__(self,ft_sensor_node_name):
		self.init_time = 0.
		self.counter = 0
		self.counter_prev = 0
		self.force = np.matrix([0.,0.,0.]).T
		self.force_raw = np.matrix([0.,0.,0.]).T
		self.torque = np.matrix([0.,0.,0.]).T
		self.torque_raw = np.matrix([0.,0.,0.]).T
		self.torque_bias = np.matrix([0.,0.,0.]).T

		self.time_data = []
		self.force_data = []
		self.force_raw_data = []
		self.torque_data = []
		self.torque_raw_data = []

		#capture the force on the tool tip	
		## self.force_sub = rospy.Subscriber(ft_sensor_node_name,\
		## 	WrenchStamped, self.force_cb)
		#raw ft values from the NetFT
		self.force_raw_sub = rospy.Subscriber(ft_sensor_node_name,\
			WrenchStamped, self.force_raw_cb)
		## self.force_zero = rospy.Publisher('/tool_netft_zeroer/rezero_wrench', Bool)
		rospy.logout('Done subscribing to '+ft_sensor_node_name+' topic')


	def force_cb(self, msg):
		self.time = msg.header.stamp.to_time()
		self.force = np.matrix([msg.wrench.force.x, 
					msg.wrench.force.y,
					msg.wrench.force.z]).T
		self.torque = np.matrix([msg.wrench.torque.x, 
					msg.wrench.torque.y,
					msg.wrench.torque.z]).T
		self.counter += 1


	def force_raw_cb(self, msg):
		self.force_raw = np.matrix([msg.wrench.force.x, 
					msg.wrench.force.y,
					msg.wrench.force.z]).T
		self.torque_raw = np.matrix([msg.wrench.torque.x, 
					msg.wrench.torque.y,
					msg.wrench.torque.z]).T


	def reset(self):
		self.force_zero.publish(Bool(True))
	

	def log(self, log_file):
		if self.counter > self.counter_prev:
			self.counter_prev = self.counter
			time_int = self.time-self.init_time
			print >> log_file, time_int, self.counter,\
				## self.force[0,0],self.force[1,0],self.force[2,0],\
				self.force_raw[0,0],self.force_raw[1,0],self.force_raw[2,0],\
				## self.torque[0,0],self.torque[1,0],self.torque[2,0],\
				self.torque_raw[0,0],self.torque_raw[1,0],self.torque_raw[2,0]

			## self.force_data.append(self.force)
			self.force_raw_data.append(self.force_raw)
			## self.torque_data.append(self.torque)
			self.torque_raw_data.append(self.torque_raw)
			self.time_data.append(self.time)


	def static_bias(self):
		print '!!!!!!!!!!!!!!!!!!!!'
		print 'BIASING FT'
		print '!!!!!!!!!!!!!!!!!!!!'
		f_list = []
		t_list = []
		for i in range(20):
			f_list.append(self.force)
			t_list.append(self.torque)
			rospy.sleep(2/100.)
		if f_list[0] != None and t_list[0] !=None:
			self.force_bias = np.mean(np.column_stack(f_list),1)
			self.torque_bias = np.mean(np.column_stack(t_list),1)
			print self.gravity
			print '!!!!!!!!!!!!!!!!!!!!'
			print 'DONE Biasing ft'
			print '!!!!!!!!!!!!!!!!!!!!'
		else:
			print 'Biasing Failed!'


class ADL_log():
	def __init__(self, robot=False):
		self.init_time = 0.
		rospy.init_node('ADLs_log', anonymous = True)

        if robot:
            self.tflistener = tf.TransformListener()
		tool_tracker_name, self.ft_sensor_node_name = log_parse()
		rospy.logout('ADLs_log node subscribing..')

        #subscribe to the rigid body nodes		
        if robot:
            self.tool_tracker = tracker_pose(tool_tracker_name)
            self.head_tracker = tracker_pose('head')


	def task_cmd_input(self):
		confirm = False
		while not confirm:
			valid = True
			self.sub_name=raw_input("Enter subject's name: ")
			num=raw_input("Enter the number for the choice of task:"+\
					"\n1) cup \n2) door \n3) wipe"+\
					"\n4) spoon\n5) tooth brush\n6) comb\n: ")
			if num == '1':
				self.task_name = 'cup'
			elif num == '2':
				self.task_name = 'door'
			else:
				print '\n!!!!!Invalid choice of task!!!!!\n'
				valid = False

			if valid:
				num=raw_input("Select actor:\n1) human \n2) robot\n: ")
				if num == '1':
					self.actor = 'human'
				elif num == '2':
					self.actor = 'robot'
				else:
					print '\n!!!!!Invalid choice of actor!!!!!\n'
					valid = False
			if valid:
				self.trial_name=raw_input("Enter trial's name (e.g. arm1, arm2): ")
				self.file_name = self.sub_name+'_'+self.task_name+'_'+self.actor+'_'+self.trial_name			
				ans=raw_input("Enter y to confirm that log file is:  "+self.file_name+"\n: ")
				if ans == 'y':
					confirm = True


	def init_log_file(self):	
		self.task_cmd_input()
		## self.tool_tip = tool_pose(self.tool_name,self.tflistener)
		self.ft = tool_ft(self.ft_sensor_node_name)
		self.ft_log_file = open(self.file_name+'_ft.log','w')
		## self.tool_tracker_log_file = open(self.file_name+'_tool_tracker.log','w')
		## self.tooltip_log_file = open(self.file_name+'_tool_tip.log','w')
		## self.head_tracker_log_file = open(self.file_name+'_head.log','w')
		## self.gen_log_file = open(self.file_name+'_gen.log','w')
		self.pkl = open(self.file_name+'.pkl','w')

		## raw_input('press Enter to set origin')
		## self.tool_tracker.set_origin()
		## self.tool_tip.set_origin()
		## self.head_tracker.set_origin()
		## self.ft.reset()
			
		raw_input('press Enter to begin the test')
		self.init_time = rospy.get_time()
		## self.head_tracker.init_time = self.init_time
		## self.tool_tracker.init_time = self.init_time
		## self.tool_tip.init_time = self.init_time
		self.ft.init_time = self.init_time

		## print >> self.gen_log_file,'Begin_time',self.init_time,\
		## 	'\nTime X Y Z Rotx Roty Rotz',\
		## 	'\ntool_tracker_init_pos', self.tool_tracker.init_pos[0,0],\
		## 			self.tool_tracker.init_pos[1,0],\
		## 			self.tool_tracker.init_pos[2,0],\
		## 	'\ntool_tracker_init_rot', self.tool_tracker.init_rot[0,0],\
		## 			self.tool_tracker.init_rot[1,0],\
		## 			self.tool_tracker.init_rot[2,0],\
		## 	'\ntool_tip_init_pos', self.tool_tip.init_pos[0,0],\
		## 			self.tool_tip.init_pos[1,0],\
		## 			self.tool_tip.init_pos[2,0],\
		## 	'\ntool_tip_init_rot', self.tool_tip.init_rot[0,0],\
		## 			self.tool_tip.init_rot[1,0],\
		## 			self.tool_tip.init_rot[2,0],\
		## 	'\nhead_init_pos', self.head_tracker.init_pos[0,0],\
		## 			self.head_tracker.init_pos[1,0],\
		## 			self.head_tracker.init_pos[2,0],\
		## 	'\nhead_init_rot', self.head_tracker.init_rot[0,0],\
		## 			self.head_tracker.init_rot[1,0],\
		## 			self.head_tracker.init_rot[2,0],\
		## 	'\nTime Fx Fy Fz Fx_raw Fy_raw Fz_raw \
		## 		Tx Ty Tz Tx_raw Ty_raw Tz_raw'


	def log_state(self, bias=True):
		## self.head_tracker.log(self.head_tracker_log_file, log_delta_rot=True)
		## self.tool_tracker.log(self.tool_tracker_log_file)
		## self.tool_tip.log(self.tooltip_log_file)
		self.ft.log(self.ft_log_file)
		## print '\nTool_Pos\t\tForce:\t\t\tHead_rot',\
		## 	'\nX: ', self.tool_tracker.delta_pos[0,0],'\t',\
		## 		self.ft.force[0,0],'\t',\
		## 		math.degrees(self.head_tracker.delta_rot[0,0]),\
		## 	'\nY: ', self.tool_tracker.delta_pos[1,0],'\t',\
		## 		self.ft.force[1,0],'\t',\
		## 		math.degrees(self.head_tracker.delta_rot[1,0]),\
		## 	'\nZ: ', self.tool_tracker.delta_pos[2,0],'\t',\
		## 		self.ft.force[2,0],'\t',\
		## 		math.degrees(self.head_tracker.delta_rot[2,0])


	def close_log_file(self):
		dict = {}
		dict['init_time'] = self.init_time
		## dict['init_pos'] = self.tool_tracker.init_pos
		## dict['pos'] = self.tool_tracker.pos_data
		## dict['quat'] = self.tool_tracker.quat_data
		## dict['rot_data'] = self.tool_tracker.rot_data
		## dict['ptime'] = self.tool_tracker.time_data
		
		## dict['h_init_pos'] = self.head_tracker.init_pos
		## dict['h_init_rot'] = self.head_tracker.init_rot
		## dict['h_pos'] = self.head_tracker.pos_data
		## dict['h_quat'] = self.head_tracker.quat_data
		## dict['h_rot_data'] = self.head_tracker.rot_data
		## dict['htime'] = self.head_tracker.time_data

		## dict['tip_init_pos'] = self.tool_tip.init_pos
		## dict['tip_init_rot'] = self.tool_tip.init_rot
		## dict['tip_pos'] = self.tool_tip.pos_data
		## dict['tip_quat'] = self.tool_tip.quat_data
		## dict['tip_rot_data'] = self.tool_tip.rot_data
		## dict['ttime'] = self.tool_tip.time_data

		## dict['force'] = self.ft.force_data
		dict['force_raw'] = self.ft.force_raw_data
		## dict['torque'] = self.ft.torque_data
		dict['torque_raw'] = self.ft.torque_raw_data
		dict['ftime'] = self.ft.time_data
		pkl.dump(dict, self.pkl)
		self.pkl.close()

		self.ft_log_file.close()
		## self.tool_tracker_log_file.close()
		## self.tooltip_log_file.close()
		## self.head_tracker_log_file.close()
		## self.gen_log_file.close()
		print 'Closing..  log files have saved..'

                

if __name__ == '__main__':
	log = ADL_log()
	log.init_log_file()
	
	while not rospy.is_shutdown():
		log.log_state()
		rospy.sleep(1/1000.)

	log.close_log_file()
    
    ## ar = adl_recording()   
    ## ar.start()

    






## class adl_recording():
##     def __init__(self, obj_id_list, netft_flag_list):
##         self.ftc_list = []                                                                                       
##         for oid, netft in zip(obj_id_list, netft_flag_list):                                                     
##             self.ftc_list.append(ftc.FTClient(oid, netft))                                                       
##         self.oid_list = copy.copy(obj_id_list)
        
##         ## self.initComms()
##         pass

        
##     def initComms(self):
        
##         # service
##         #rospy.Service('door_opening/mech_analyse_enable', None_Bool, self.mech_anal_en_cb)
        
##         # Subscriber
##         rospy.Subscriber('/netft_rdt', Wrench, self.ft_sensor_cb)                    

        
##     # returns a dict of <object id: 3x1 np matrix of force>
##     def get_forces(self, bias = True):
##         f_list = []
##         for i, ft_client in enumerate(self.ftc_list):
##             f = ft_client.read(without_bias = not bias)
##             f = f[0:3, :]

##             ## trans, quat = self.tf_lstnr.lookupTransform('/torso_lift_link',
##             ##                                             self.oid_list[i],
##             ##                                             rospy.Time(0))
##             ## rot = tr.quaternion_to_matrix(quat)
##             ## f = rot * f
##             f_list.append(-f)

##         return dict(zip(self.oid_list, f_list))


##     def bias_fts(self):
##         for ftcl in self.ftc_list:
##             ftcl.bias()
        
        
##     # TODO
##     def ft_sensor_cb(self, msg):

##         with self.ft_lock:
##             self.ft_data = [msg.force.x, msg.force.y, msg.force.z] # tool frame

##             # need to convert into torso frame?


##     def start(self, bManual=False):
##         rospy.loginfo("ADL Online Recording Start")

##         ar.bias_fts()
##         rospy.loginfo("FT bias complete")

        
##         rate = rospy.Rate(2) # 25Hz, nominally.
##         rospy.loginfo("Beginning publishing waypoints")
##         while not rospy.is_shutdown():         
##             f = self.get_forces()[0]
##             print f
##             #print rospy.Time()
##             rate.sleep()        
            
    
