#!/usr/bin/env python
import numpy as np
import roslib; roslib.load_manifest('hrl_msgs'); roslib.load_manifest('tf')
import rospy
import tf
from hrl_msgs.msg import FloatArrayBare
from sensor_msgs.msg import JointState
from math import *
import operator
from scipy.signal import remez
from scipy.signal import lfilter

from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped, Point, Pose, PoseStamped 
from std_msgs.msg import ColorRGBA
from autobed_occupied_client import autobed_occupied_status_client
from tf.transformations import quaternion_from_euler


MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 


class AutobedStatePublisherNode(object):
    def __init__(self):
        self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=100)

        # self.autobed_occupied_state_client = autobed_occupied_status_client()
        # self.pressure_grid_pub = rospy.Publisher('pressure_grid', Marker)
        self.sendToRviz=tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

        init_centered = JointState()
        init_centered.header.stamp = rospy.Time.now()
        init_centered.name = [None]*(1)
        init_centered.position = [None]*(1)
        init_centered.name[0] = "head_bed_leftright_joint"
        init_centered.position[0] = 0
        self.joint_pub.publish(init_centered)

        rospy.Subscriber("/abdout0", FloatArrayBare, self.bed_pose_cb)

        #rospy.Subscriber("/camera_o/pose", TransformStamped, 
        #        self.camera_pose_cb)
        # rospy.Subscriber("/fsascan", FloatArrayBare,
        #         self.pressure_map_cb)

        #Initialize camera pose to standard position of Kinect in the test
        #chamber
        # self.camera_p = (-0.1093, 1.108, 2.86)
        #self.camera_q = (0.27, -0.011, -0.958, 0.0975)
        # self.camera_q = tuple(quaternion_from_euler(1.57, 1.57, 0.0))
        #Low pass filter design
        self.bed_height = 0
        self.bin_numbers = 5
        self.bin_numbers_for_leg_filter = 21
        self.collated_head_angle = np.zeros((self.bin_numbers, 1))
        self.collated_leg_angle = np.zeros((
            self.bin_numbers_for_leg_filter, 1))
        self.lpf = remez(self.bin_numbers, [0, 0.1, 0.25, 0.5], [1.0, 0.0])
        self.lpf_for_legs = remez(self.bin_numbers_for_leg_filter, 
                [0, 0.0005, 0.1, 0.5], [1.0, 0.0])
        # self.pressuremap_flat = np.zeros((1, NUMOFTAXELS_X*NUMOFTAXELS_Y))
        #Publisher for Markers (can send them all as one marker message instead of an array because they're all spheres of the same size
        # self.marker_pub=rospy.Publisher('visualization_marker', Marker)

        print 'Autobed robot state publisher is ready!'



    #callback for the pose messages from the autobed
    def bed_pose_cb(self, data): 
        poses=np.asarray(data.data);
        
        self.bed_height = ((poses[1]/100) - 0.09) if (((poses[1]/100) - 0.09) 
                > 0) else 0
        head_angle = (poses[0]*pi/180)
        leg_angle = (poses[2]*pi/180 - 0.1)
        self.collated_head_angle = np.delete(self.collated_head_angle, 0)
        self.collated_head_angle = np.append(self.collated_head_angle,
                [head_angle])
        self.collated_leg_angle = np.delete(self.collated_leg_angle, 0)
        self.collated_leg_angle = np.append(self.collated_leg_angle,
                [leg_angle])
 

    # def pressure_map_cb(self, data):
    #     '''This callback accepts incoming pressure map from
    #     the Vista Medical Pressure Mat and sends it out.
    #     Remember, this array needs to be binarized to be used'''
    #     self.pressuremap_flat = [data.data]

    def filter_data(self):
        '''Creates a low pass filter to filter out high frequency noise'''
        self.leg_filt_data = self.truncate(np.dot(self.lpf_for_legs,
                    self.collated_leg_angle))
        self.head_filt_data = np.dot(self.lpf, self.collated_head_angle)
        return

    def truncate(self, f):
        '''Truncates/pads a float f to 1 decimal place without rounding'''
        fl_as_str = "%.2f" %f
        return float(fl_as_str)

    def run(self):
        rate = rospy.Rate(30) #30 Hz

        joint_state = JointState()

        dict_of_links = ({'/head_rest_link':0.762659,
                          '/leg_rest_upper_link':1.04266,
                          '/leg_rest_lower_link':1.41236})
        list_of_links = dict_of_links.keys()
        while not rospy.is_shutdown():
            joint_state.header.stamp = rospy.Time.now()
            #Resize the pressure map data
            # p_map = np.reshape(self.pressuremap_flat, (NUMOFTAXELS_X,
            #     NUMOFTAXELS_Y))
            #Clear pressure map grid
            #Filter data
            self.filter_data()
            joint_state.name = [None]*(9)
            joint_state.position = [None]*(9)
            joint_state.name[0] = "tele_leg_joint"
            joint_state.name[1] = "head_rest_hinge"
            joint_state.name[2] = "leg_rest_upper_joint"
            joint_state.name[3] = "leg_rest_upper_lower_joint"
            joint_state.position[0] = self.bed_height
            joint_state.position[1] = self.head_filt_data
            joint_state.position[2] = self.leg_filt_data
            joint_state.position[3] = -(1+(4.0/9.0))*self.leg_filt_data
            self.joint_pub.publish(joint_state)

            self.set_autobed_user_configuration(self.head_filt_data, autobed_occupied_status_client().state)

        return

    def set_autobed_user_configuration(self, headrest_th, occupied_state):

        human_joint_state = JointState()
        human_joint_state.header.stamp = rospy.Time.now()

        human_joint_state.name = [None]*(17)
        human_joint_state.position = [None]*(17)
        human_joint_state.name[0] = "neck_body_joint"
        human_joint_state.name[1] = "upper_mid_body_joint"
        human_joint_state.name[2] = "mid_lower_body_joint"
        human_joint_state.name[3] = "body_quad_left_joint"
        human_joint_state.name[4] = "body_quad_right_joint"
        human_joint_state.name[5] = "quad_calf_left_joint"
        human_joint_state.name[6] = "quad_calf_right_joint"
        human_joint_state.name[7] = "calf_foot_left_joint"
        human_joint_state.name[8] = "calf_foot_right_joint"
        human_joint_state.name[9] = "body_arm_left_joint"
        human_joint_state.name[10] = "body_arm_right_joint"
        human_joint_state.name[11] = "arm_forearm_left_joint"
        human_joint_state.name[12] = "arm_forearm_right_joint"
        human_joint_state.name[13] = "forearm_hand_left_joint"
        human_joint_state.name[14] = "forearm_hand_right_joint"
        human_joint_state.name[15] = "head_neck_joint1"
        human_joint_state.name[16] = "head_neck_joint2"

        print bth
        # bth = m.degrees(headrest_th)
        bth = headrest_th
        # 0 degrees, 0 height
        if (bth >= 0) and (bth <= 40):  # between 0 and 40 degrees
            human_joint_state.position[0] = (bth/40)*(-.2-(-.1))+(-.1)
            human_joint_state.position[1] = (bth/40)*(-.17-.4)+.4
            human_joint_state.position[2] = (bth/40)*(-.76-(-.72))+(-.72)
            human_joint_state.position[3] = -0.4
            human_joint_state.position[4] = -0.4
            human_joint_state.position[5] = 0.1
            human_joint_state.position[6] = 0.1
            human_joint_state.position[7] = (bth/40)*(-.05-.02)+.02
            human_joint_state.position[8] = (bth/40)*(-.05-.02)+.0
            human_joint_state.position[9] = (bth/40)*(-.06-(-.12))+(-.12)
            human_joint_state.position[10] = (bth/40)*(-.06-(-.12))+(-.12)
            human_joint_state.position[11] = (bth/40)*(.58-0.05)+.05
            human_joint_state.position[12] = (bth/40)*(.58-0.05)+.05
            human_joint_state.position[13] = -0.1
            human_joint_state.position[14] = -0.1
        elif (bth > 40) and (bth <= 80):  # between 0 and 40 degrees
            human_joint_state.position[0] = ((bth-40)/40)*(-.55-(-.2))+(-.2)
            human_joint_state.position[1] = ((bth-40)/40)*(-.51-(-.17))+(-.17)
            human_joint_state.position[2] = ((bth-40)/40)*(-.78-(-.76))+(-.76)
            human_joint_state.position[3] = -0.4
            human_joint_state.position[4] = -0.4
            human_joint_state.position[5] = 0.1
            human_joint_state.position[6] = 0.1
            human_joint_state.position[7] = ((bth-40)/40)*(-0.1-(-.05))+(-.05)
            human_joint_state.position[8] = ((bth-40)/40)*(-0.1-(-.05))+(-.05)
            human_joint_state.position[9] = ((bth-40)/40)*(-.01-(-.06))+(-.06)
            human_joint_state.position[10] = ((bth-40)/40)*(-.01-(-.06))+(-.06)
            human_joint_state.position[11] = ((bth-40)/40)*(.88-0.58)+.58
            human_joint_state.position[12] = ((bth-40)/40)*(.88-0.58)+.58
            human_joint_state.position[13] = -0.1
            human_joint_state.position[14] = -0.1
        else:
            print 'Error: Bed angle out of range (should be 0 - 80 degrees)'
        human_joint_state.position[15] = 0.
        human_joint_state.position[16] = 0.
        self.joint_pub.publish(human_joint_state)
        unoccupied_shift = JointState()
        unoccupied_shift.header.stamp = rospy.Time.now()
        unoccupied_shift.name = [None]*(1)
        unoccupied_shift.position = [None]*(1)
        unoccupied_shift.name[0] = "head_bed_updown_joint"
        if not occupied_state:
            unoccupied_shift.position[0] = 15
        else:
            unoccupied_shift.position[0] = 0.
        self.joint_pub.publish((unoccupied_shift))


if __name__ == "__main__":
    rospy.init_node('autobed_state_publisher_node', anonymous = False)
    a = AutobedStatePublisherNode()
    a.run()
