#!/usr/bin/env python

# System
import numpy as np, math
import time

# ROS
import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy
import tf

# HRL
from hrl_srvs.srv import None_Bool, None_BoolResponse
from hrl_msgs.msg import FloatArray

# Private
import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad


class mech_analyse():
    def __init__(self):

        self.robot_path = '/pr2'
        self.mech_anal_flag = False
        self.radius = 1.1
                
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


    def ft_sensor_cb(self, req):

        return
        

    # Pose
    def getEndeffectorPose(self):

        self.tf_lstnr.waitForTransform(self.torso_frame, self.ee_frame, rospy.Time(0), rospy.Duration(5.0))
        [self.end_effector_pos, self.end_effector_orient_quat] = self.tf_lstnr.lookupTransform(self.torso_frame, self.ee_frame, rospy.Time(0))
        self.end_effector_pos = np.matrix(self.end_effector_pos).T
            
        return [self.end_effector_pos, self.end_effector_orient_quat]
        
        
    def generateMechData(self):

        mech_data         = None ## add something
        [ee_pos, ee_quat] = self.getEndeffectorPose()

        p_list = None
        f_list = None

        config_list, ftan_list = mad.online_force_estimation(p_list, f_list, radius=self.radius)
        
        msg = FloatArray()
        ## msg.data = mech_data.tolist()         
        msg.data = [1.0,1.0]
        self.mech_data_pub.publish(msg)
        

    def start(self, bManual=False):
        rospy.loginfo("Mech Analyse Online start")
        
        rate = rospy.Rate(25) # 25Hz, nominally.
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
