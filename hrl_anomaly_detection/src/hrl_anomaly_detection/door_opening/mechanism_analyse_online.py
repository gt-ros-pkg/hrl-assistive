#!/usr/bin/env python

# System
import numpy as np, math
import time
import sys
import threading

# ROS
import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy
import tf

# HRL
from hrl_srvs.srv import None_Bool, None_BoolResponse
from hrl_msgs.msg import FloatArray

# Private
import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad
import hrl_anomaly_detection.advait.arm_trajectories as at


class mech_analyse():
    def __init__(self):

        self.robot_path = '/pr2'
        self.mech_anal_flag = False
        self.radius = 1.1

        self.aPts = None
        self.ftan_list = []
        self.config_list = []        
        self.ft_data = [1.0,1.0,1.0]

        # define lock
        self.ft_lock = threading.RLock() ## ft sensing lock
        
        #
        rospy.init_node("mech_online")        
        self.getParams()        
        self.initComms()
        pass

    def getParams(self):

        self.torso_frame = rospy.get_param('haptic_mpc'+self.robot_path+'/torso_frame' )
        self.ee_frame = rospy.get_param('haptic_mpc'+self.robot_path+'/end_effector_frame' )

        
    def initComms(self):
        
        # service
        rospy.Service('door_opening/mech_analyse_enable', None_Bool, self.mech_anal_en_cb)

        # Publisher
        self.mech_data_pub = rospy.Publisher("door_opening/mech_data", FloatArray)        
        
        # Subscriber
        ## rospy.Subscriber('haptic_mpc/ft_sensor', PoseArray, self.ft_cb)                    

        try:
            self.tf_lstnr = tf.TransformListener()
        except rospy.ServiceException, e:
            rospy.loginfo("ServiceException caught while instantiating a TF listener. Seems to be normal")
            pass                  

        
    def mech_anal_en_cb(self, req):

        # Run analyse tasks
        if req.data is True:
            self.mech_anal_flag = True
        else:
            self.mech_anal_flag = False
                
        return None_BoolResponse(True)


    # TODO
    def ft_sensor_cb(self, req):

        with self.ft_lock:
            self.ft_data = [1.0,1.0,1.0]

        
    # Pose
    def getEndeffectorPose(self):

        self.tf_lstnr.waitForTransform(self.torso_frame, self.ee_frame, rospy.Time(0), rospy.Duration(5.0))
        [self.end_effector_pos, self.end_effector_orient_quat] = self.tf_lstnr.lookupTransform(self.torso_frame, self.ee_frame, rospy.Time(0))
        self.end_effector_pos = np.matrix(self.end_effector_pos).T
            
        return [self.end_effector_pos, self.end_effector_orient_quat]
        
        
    def generateMechData(self):
        # Estimate door angle and tangential force.

        mech_data         = None ## add something
        [ee_pos, ee_quat] = self.getEndeffectorPose()

        if self.aPts is None:
            self.aPts = ee_pos
            pts_2d = self.aPts[:2,:]
        else:
            self.aPts = np.hstack([self.aPts, ee_pos])
            pts_2d = self.aPts[:2,:]

        x_guess = self.aPts[0,0]
        y_guess = self.aPts[1,0] - self.radius
        rad_guess = self.radius

        rad, cx, cy = at.fit_circle(rad_guess,x_guess,y_guess,pts_2d,
                                    method='fmin_bfgs',verbose=False,
                                    rad_fix=True)
        
        ## print 'rad, cx, cy:', rad, cx, cy

        p0 = self.aPts[:,0] # 3x1
        rad_vec_init = np.matrix((p0[0,0]-cx, p0[1,0]-cy)).T
        rad_vec_init = rad_vec_init / np.linalg.norm(rad_vec_init)

        rad_vec = np.array([ee_pos[0,0]-cx,ee_pos[1,0]-cy])
        rad_vec = rad_vec/np.linalg.norm(rad_vec)
        
        ang = np.arccos((rad_vec.T*rad_vec_init)[0,0])
        tan_vec = (np.matrix([[0,-1],[1,0]]) * np.matrix(rad_vec).T).A1
        f_vec = np.array([self.ft_data[0],self.ft_data[1]])

        f_tan_mag = abs(np.dot(f_vec, tan_vec))

        msg = FloatArray()
        msg.data = [f_tan_mag, ang]
        self.mech_data_pub.publish(msg)
        

    def start(self, bManual=False):
        rospy.loginfo("Mech Analyse Online start")
        
        rate = rospy.Rate(2) # 25Hz, nominally.
        rospy.loginfo("Beginning publishing waypoints")
        while not rospy.is_shutdown():         
            self.generateMechData()
            #print rospy.Time()
            rate.sleep()        
           

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--manual_start', '--man', action='store_true', dest='bStart',
                 default=False, help='Start this program manually.')
    
    opt, args = p.parse_args()
    
    ma = mech_analyse()
    ma.start(opt.bStart)
