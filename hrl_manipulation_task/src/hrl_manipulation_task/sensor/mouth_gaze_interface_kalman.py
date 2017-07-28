#! /usr/bin/env python

# Will take in "kinect" or "wrist" as command line argument to specify camera.
# No argument will default to wrist.
#
# Text box will be different colors depending on condition.
#   Orange: Conditions not met
#   Green: Conditions met
#
# Detection of whether mouth is open considers both the distance between lips and
# the area of the mouth relative to the face area.
#
# Gaze detection splits the right eye into 3 areas and
# compares the darkness of the three areas.
# Only performed when head is not rotated.
# Right eye is checked first, and if result is 'unknown', left eye is checked.
#
# Head rotation is detected by comparing the position of the
# corners of mouth with the edge of the cheek.


import sys
import dlib
import numpy as np

import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from hrl_msgs.msg import StringArray

import cv2
from cv_bridge import CvBridge, CvBridgeError

import time
import math

import argparse

from pykalman import KalmanFilter

from sound_play.libsoundplay import SoundClient

# dlib colors
yellow = dlib.rgb_pixel(255, 255, 0)
red = dlib.rgb_pixel(255, 0, 0)
green = dlib.rgb_pixel(0, 255, 0)
blue = dlib.rgb_pixel(0, 0, 255)
orange = dlib.rgb_pixel(255, 157, 20)

# cv2 colors (rgb)
cv2_white = (255, 255, 255)
cv2_black = (0, 0, 0)
cv2_yellow = (255, 255, 0)
cv2_blue = (0, 0, 255)
cv2_green = (0, 255, 0)
cv2_orange = (255, 157, 20)

QUEUE_SIZE = 10

class DlibFaceLandmarkDetector:

    def __init__(self, img_topic='/SR300/rgb/image_raw'):
        # Constants
        self.feed_seconds = 2.0
        self.scoop_seconds = 2.0
        self.stop_seconds = 2.0

        # Make window.
        self.win = dlib.image_window()

        # Subscribe to image topic.
        self.img_topic = img_topic
        self.image_sub = rospy.Subscriber(self.img_topic, Image, self.callback)

        # Bridge
        self.bridge = CvBridge()
        
        # Load detector.
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./detector_tools/shape_predictor_68_face_landmarks.dat')

        # Use when detecting faces every 4 frames.
        self.count = 0
        self.dets = None
        if self.img_topic == '/SR300/rgb/image_raw':
            self.frame_num = 2
        else:
            self.frame_num = 4

        # Use to print number of detected faces only when number detected changes.
        self.prev = 0

        # True when both mouth and eye conditions are met.
        self.conditions_met = False
        self.outliers = 0

        # Timer
        self.timer_started = False
        self.start_time = None

        # Scoop/stop tools
        self.stop_condition = False
        self.stop_timer = None
        self.stop_outliers = 0
        self.scoop_condition = False
        self.scoop_timer = None
        self.scoop_outliers = 0

        # Publisher/subscriber
        self.imagePub = rospy.Publisher("/hrl_manipulation_task/mouth_gaze_detector", Image, queue_size=10)
        self.statusPub = rospy.Publisher("/manipulation_task/status", String, queue_size=1)
        self.guiStatusPub = rospy.Publisher("/manipulation_task/gui_status", String, queue_size=1, latch=True)
        self.availablePub = rospy.Publisher("/manipulation_task/available", String, queue_size=QUEUE_SIZE)
        self.emergencyPub = rospy.Publisher("/manipulation_task/emergency", String, queue_size=QUEUE_SIZE)
        self.userInputPub = rospy.Publisher("/manipulation_task/user_input", String, queue_size=QUEUE_SIZE)
        self.feed_message_published = False
        self.gui_status = ''
        rospy.Subscriber("/manipulation_task/gui_status", String, self.guiCallback, queue_size=1)

        # Sound
        self.sound_handle = SoundClient()

        # Kalman filter
        init_state = np.array([1, 0])
        init_covariance = 1.0e-3*np.eye(2)
        transition_cov = 1.0e-4*np.eye(2)
        observation_cov = 1.0e-1*np.eye(2)
        transition_mat = np.array([[1, 1], [0, 1]])
        observation_mat = np.eye(2)

        self.kf2 = KalmanFilter(transition_matrices = transition_mat,
                            observation_matrices = observation_mat,
                            initial_state_mean = init_state,
                            initial_state_covariance = init_covariance,
                            transition_covariance = transition_cov, 
                            observation_covariance = observation_cov)
        self.prev_state = init_state
        self.prev_ratio = 1
        self.prev_covariance = init_covariance
        self.prev_vel = 0

        # Wiping 
        self.wiping_start = True
        self.wiping_outlier = 0

    def guiCallback(self, msg):
        self.gui_status = msg.data


    def callback(self, data):
        nodded = False
        shaken = False
        # TODO
        corrected = False
        # Convert sensor_msgs/Image to rgb.
        img = self.bridge.imgmsg_to_cv2(data, 'rgb8')

        # Flip image
        #if self.img_topic == '/SR300/rgb/image_raw':
        #    img = cv2.flip(img, 0)
        #    img = cv2.flip(img, 1)

        # Make image half the size of original (speedup).
        h, w, c = img.shape
        big_img = img
        img = cv2.resize(img, (w/2, h/2))  # 320x240 for Kinect
        
        # Grayscale (speedup)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Text string describing status.
        status = ''
        console_status = ''

        # Face detection ----------------------------------------------------------------------------------------------------------
        # Detect face every 4 (or 2) frames (speedup).
        if (self.count%self.frame_num) == 0:
            self.dets = self.detector(gray_img, 1)
            self.count += 1
        # Do not detect face (for speedup).
        else:
            self.count += 1
        # -------------------------------------------------------------------------------------------------------------------------

        
        # First iteration, only show image?
        if self.dets == None:
            self.win.set_image(img)
            self.imagePub.publish(self.bridge.cv2_to_imgmsg(img, "rgb8"))

        # Landmark detection ------------------------------------------------------------------------------------------------------------
        # At least one face has been detected.
        if len(self.dets)>=1:
            if len(self.dets) != self.prev:
                print '---------------DETECTED ' + str(len(self.dets)) + ' FACE(S)---------------'
                self.prev = len(self.dets)
            # Find largest face.
            largest_area = 0
            largest_box = None
            for k, d in enumerate(self.dets):
                if d.area() > largest_area:
                    largest_area = d.area()
                    largest_box = d
            # Landmarks for only largest face.
            shape = self.predictor(gray_img, largest_box)
            # Correction for when face is "squished"
            if d.bottom() > shape.part(8).y-5:
                largest_box = dlib.rectangle(largest_box.left(), largest_box.top()+10, largest_box.right(), largest_box.bottom()+10)
                shape = self.predictor(gray_img, largest_box)
                corrected = True

            # Mouth open check.
            mouth_open = self.lips_open(shape, largest_area)
            if mouth_open:
                status += 'OPEN'
                console_status += 'Mouth open'
            else:
                status = 'CLOSED'
                console_status += 'Mouth closed'

            # Head rotated check.
            head_rotated, head_status, ratio = self.is_head_rotated(shape)
            status += ', {}'.format(head_status)

            ratio_observed = ratio
            velocity_observed = ratio - self.prev_state[0]
            (next_state, next_covariance) = self.kf2.filter_update(self.prev_state, self.prev_covariance, np.array([ratio_observed, velocity_observed]))
            ratio_kalman = next_state[0]
            self.prev_state = next_state
            self.prev_covariance = next_covariance

            # Scoop condition check
            # TODO
            if ratio_kalman < 0.4 and mouth_open:
                self.scoop_outliers = 0
                if self.scoop_condition:
                    elapsed = time.time() - self.scoop_timer
                    if elapsed > self.scoop_seconds:
                        print 'scooping condition met, gui status: {}'.format(self.gui_status)
                        if (self.gui_status == 'select task') or (self.gui_status == 'stopped'):
                            self.statusPub.publish('Scooping')
                            self.availablePub.publish('true')
                            self.userInputPub.publish('Start')
                            print 'scooping command published'
                            self.sound_handle.say('Scoop ing')
                            print 'said scooping'
                        elif self.gui_status == 'wait start':
                            self.userInputPub.publish('Start')
                else:
                    print 'scoop timer started'
                    self.scoop_timer = time.time()
                    self.scoop_condition = True
            elif self.scoop_outliers <= 5:
                status = 'OPEN, LEFT'
                print 'scoop outlier #{}'.format(self.scoop_outliers)
                self.scoop_outliers += 1
            else:
                self.scoop_condition = False

            # Wiping condition check
            if not self.wiping_start:
                if mouth_open and not head_rotated:
                    self.wiping_start = True
                    self.wiping_outlier = 0
            else:
                if (not mouth_open) or (next_state[1] > 0):
                    if self.wiping_outlier < 3: self.wiping_outlier += 1
                    else: self.wiping_start = False
                elif ratio_kalman < 0.4:
                    self.publish_wipe = True
                    print 'wiping condition met, gui status: {}'.format(self.gui_status)
                    if (self.gui_status == 'select task') or (self.gui_status == 'stopped'):
                        self.statusPub.publish('Clean')
                        self.availablePub.publish('true')
                        self.userInputPub.publish('Start')
                        print 'wiping command published'
                        self.sound_handle.say('Cleaning spoon')
                    elif self.gui_status == 'wait start':
                        self.userInputPub.publish('Start')

            # -----------------------------------------------------------------
            # Update conditions_met bool.
            # Allow 4 consecutive outliers.
            if mouth_open and not head_rotated:
                self.conditions_met = True
                self.outliers = 0
            elif self.conditions_met and mouth_open and (self.outliers < 5):
                self.outliers += 1
                print 'feeding outlier #{}'.format(self.outliers)
                status = 'OPEN, STRAIGHT'
            else:
                self.conditions_met = False
                self.outliers = 0

            # Face box coordinates
            left = largest_box.left()
            top = largest_box.top()
            right = largest_box.right()
            bottom = largest_box.bottom()

            # Check if three seconds have passed while conditions were continuously met.
            color = None
            if (self.conditions_met) and (not self.timer_started):  # conditions_met: False -> True.
                print 'feeding timer started'
                self.start_time = time.time()
                self.timer_started = True
                color = cv2_green
            elif (self.conditions_met) and (self.timer_started):  # conditions_met: True -> True, >= 3 secs
                if (time.time() - self.start_time) >= self.feed_seconds:
                    cv2.putText(img, '3 seconds passed!', (2, 230), cv2.FONT_HERSHEY_PLAIN, 2, cv2_green, 2)
                    color = cv2_green
                    print 'feeding conditions met, gui status: {}'.format(self.gui_status)
                    if (self.gui_status == 'select task') or (self.gui_status == 'stopped'):
                        self.statusPub.publish('Feeding')
                        self.availablePub.publish('true')
                        self.userInputPub.publish('Start')
                        print 'Feeding command published'
                        self.sound_handle.say('feed ing')
                        print 'said feeding'
                    elif self.gui_status == 'wait start':
                        self.userInputPub.publish('Start')
                else:  # conditions_met: True -> True, < 3 secs
                    color = cv2_green
            else:  # conditions_met: False
                color =cv2_orange
                self.timer_started = False
            cv2.rectangle(img, (left-20, top-30), (left+132, top-15), color, -1)
            cv2.rectangle(img, (left, top), (right, bottom), color, 1)
            cv2.putText(img, status, (left-18, top-17), cv2.FONT_HERSHEY_PLAIN, 1, cv2_black)
            self.win.set_image(img)
            self.win.clear_overlay()
            self.win.add_overlay(shape, orange)
            self.imagePub.publish(self.bridge.cv2_to_imgmsg(img, "rgb8"))

            # Stop condition check
            if ratio_kalman > 2.5:
                self.stop_outliers = 0
                if self.stop_condition:
                    elapsed = time.time() - self.stop_timer
                    if elapsed > self.stop_seconds:
                        print 'stop condition met, gui status: {}'.format(self.gui_status)
                        if (self.gui_status == 'in motion') or (self.gui_status == 'wait start'):
                            self.emergencyPub.publish('STOP')
                            print 'stopping command published'
                            self.sound_handle.say('Stop ing')
                            print 'said stopping'
                else:
                    print 'stop timer started'
                    self.stop_timer = time.time()
                    self.stop_condition = True
            elif self.stop_outliers <= 5:
                print 'stop outlier #{}'.format(self.stop_outliers)
                self.stop_outliers += 1
            else:
                self.stop_condition = False     
    
        # No faces have been detected.
        else:
            if len(self.dets) != self.prev:
                print '----------------NO FACE DETECTED----------------'
                self.prev = len(self.dets)
                self.win.clear_overlay()
            # Head ratio detector with Kalman filter.
            ratio_observed = self.prev_state[0]
            velocity_observed = self.prev_state[1] * 0.5
            (next_state, next_covariance) = self.kf2.filter_update(self.prev_state, self.prev_covariance, np.array([ratio_observed, velocity_observed]))
            ratio_kalman = next_state[0]
            self.prev_state = next_state
            self.prev_covariance = next_covariance
      
            # Stop condition check
            if ratio_kalman > 2.5:
                self.stop_outliers = 0
                if self.stop_condition:
                    elapsed = time.time() - self.stop_timer
                    if elapsed > 3.0:
                        if (self.gui_status == 'in motion') or (self.gui_status == 'wait start'):
                            self.emergencyPub.publish('STOP')
                            print 'stopping command published'
                            self.sound_handle.say('Stop ing')
                            print 'said stopping'
                else:
                    print 'stop timer started'
                    self.stop_timer = time.time()
                    self.stop_condition = True
            elif self.stop_outliers <= 5:
                self.stop_outliers += 1
            else:
                self.stop_condition = False  
            self.win.set_image(img)
            self.imagePub.publish(self.bridge.cv2_to_imgmsg(img, "rgb8"))

            # Scoop condition check
            if self.scoop_condition:
                elapsed = time.time() - self.scoop_timer
                if elapsed >= 2.0:
                    print 'scooping condition met, gui status: {}'.format(self.gui_status)
                    if (self.gui_status == 'select task') or (self.gui_status == 'stopped'):
                        self.statusPub.publish('Scooping')
                        self.availablePub.publish('true')
                        self.userInputPub.publish('Start')
                        print 'scooping command published'
                        self.sound_handle.say('Scoop ing')
                        print 'said scooping'
                    elif self.gui_status == 'wait start':
                        self.userInputPub.publish('Start')
                               

            # Uncomment to add rectangle around face.
            #self.win.add_overlay(self.dets)

        #else:
        #    self.win.set_image(img)


    def lips_open(self, shape, face_area):
        """Determine whether mouth is open."""
        # r_diff is the height of opening of right side of mouth
        r_diff = shape.part(67).y - shape.part(61).y
        c_diff = shape.part(66).y - shape.part(62).y
        l_diff = shape.part(65).y - shape.part(63).y
        mouth_height = ((c_diff >= 5) and (r_diff >= 3) and (l_diff >= 3))
        mouth_area = self.mouth_area_ratio(shape, face_area) > 1.3
        # Consider both distance between inner lips and area of mouth.
        # This helps detect open mouth when the face is small
        if mouth_height or mouth_area:
            return True
        else:
            return False


    def mouth_shape_ratio(self, shape):
        """Return percentage of height of mouth w.r.t. width of mouth."""
        height = np.absolute(float(shape.part(66).y)-float(shape.part(62).y))
        width = np.absolute(float(shape.part(64).x)-float(shape.part(60).x))
        # Prevent div by 0 errors.
        if width > 0.0:
            return (height/width)*100
        else:
            return 0


    def mouth_area_ratio(self, shape, face_area):
        """Return percentage of mouth area w.r.t face area."""
        mouth = self.mouth_area(shape)
        if face_area > 0:
            return (mouth/face_area)*100
        else:
            return 0

    def mouth_area(self, shape):
        """Calculate area of the mouth between inner lips."""
        # Make two vectors for each triangle.
        v1 = [shape.part(61).x-shape.part(60).x, shape.part(61).y-shape.part(60).y]
        v2 = [shape.part(67).x-shape.part(60).x, shape.part(67).y-shape.part(60).y]

        v3 = [shape.part(63).x-shape.part(61).x, shape.part(63).y-shape.part(61).y]
        v4 = [shape.part(67).x-shape.part(61).x, shape.part(67).y-shape.part(61).y]

        v5 = [shape.part(63).x-shape.part(65).x, shape.part(63).y-shape.part(65).y]
        v6 = [shape.part(67).x-shape.part(65).x, shape.part(67).y-shape.part(65).y]

        v7 = [shape.part(63).x-shape.part(64).x, shape.part(63).y-shape.part(64).y]
        v8 = [shape.part(65).x-shape.part(64).x, shape.part(65).y-shape.part(64).y]

        # Calculate area using cross product.
        tri1 = np.absolute(np.cross(v1, v2))*0.5
        tri2 = np.absolute(np.cross(v3, v4))*0.5
        tri3 = np.absolute(np.cross(v5, v6))*0.5
        tri4 = np.absolute(np.cross(v7, v8))*0.5

        return tri1+tri2+tri3+tri4


    def get_dist(self, pt1, pt2):
        """Return distance between two 'point's."""
        return math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)


    def is_head_rotated(self, shape):
        """Determine whether head is rotated."""
        # Compares distance between corner of mouth and edge of cheek.
        dist1 = self.get_dist(shape.part(48), shape.part(4))
        dist2 = self.get_dist(shape.part(54), shape.part(12))
        if dist2 > 0.0:
            ratio = dist1/dist2
        else: ratio = 0
        if ratio > 1.6: #previously: 1.6
            return True, 'RIGHT', ratio
        elif ratio < 0.625: #previously: 0.625
            return True, 'LEFT', ratio
        else:
            return False, 'STRAIGHT', ratio


def main(args):
    # Can specify to use kinect camera or wrist mounted (SR300) camera.
    if len(sys.argv) == 2:
        if sys.argv[1] == 'kinect':
            thing = DlibFaceLandmarkDetector(img_topic='/camera/rgb/image_raw')
        elif sys.argv[1] == 'wrist':
            thing = DlibFaceLandmarkDetector()
        else:
            print 'Invalid arugment. Please specify "kinect" or "wrist". Default is wrist.'
            sys.exit()       
    elif len(sys.argv) == 1:
        thing = DlibFaceLandmarkDetector()
    else:
        print 'Invalid argument. Please specify "kinect" or "wrist". Default is wrist.'
        sys.exit()

    rospy.init_node('mouth_gaze_interface', anonymous=True)
    try:
        rospy.spin()
        thing.debug_file.close()
    except:
        pass

if __name__ == '__main__':
    main(sys.argv)
