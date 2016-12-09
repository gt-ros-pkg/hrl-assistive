#!/usr/bin/env python

import os
import rospy
import dlib
import message_filters
import cv2
import time
import tf
import tf.transformations as tft
import numpy as np
import math
import random
import hrl_lib.util as ut
import hrl_lib.quaternion as qt
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point32, PolygonStamped, Vector3, Quaternion
from std_msgs.msg import Float32 as rosFloat
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skimage.transform import radon, rescale
from cv_bridge import CvBridge, CvBridgeError

class MouthPoseDetector:
    def __init__(self, camera_link, rgb_image, depth_image, rgb_info, depth_info, depth_scale, offset,
                 display_2d=True, display_3d=True, flipped=False, rgb_mode="bgr8", save_loc=None, load_loc=None):
        #display registration
        self.display_2d = display_2d
        self.display_3d = display_3d
        if display_2d:
            self.win = dlib.image_window()
            self.win2 = dlib.image_window()
        #for image processings
        self.bridge = CvBridge()
        self.previous_face = []
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.expanduser('~') + '/Desktop/shape_predictor_68_face_landmarks.dat')
        self.wrong_coor = (-100, -100, -100)

        #for tf processing
        if display_3d:
            self.br = tf.TransformBroadcaster()      
        self.tf_listnr = tf.TransformListener()
        self.gripper_to_sensor = None

        #camera informateions
        self.camera_link  = camera_link
        self.frame_ready  = False
        self.depth_scale  = depth_scale
        self.flipped      = flipped
        self.rgb_mode     = rgb_mode
        self.offset       = offset
        gripper           = "/right/haptic_mpc/gripper_pose"
        
        #for initializing frontal face data
        self.save_loc          = save_loc
        if self.save_loc is not None:
            self.save_loc = os.path.expanduser('~') + save_loc
        self.first                 = True
        self.relation              = None
        self.dist                  = []
        self.reverse_dist          = []
        self.half_dist             = []
        self.point_set             = []
        self.current_positions     = []
        self.object_points         = []
        self.point_set_index       = []
        self.sizes                 = []
        self.lm_coor               = []
        self.registered_eye_vertical   = []
        self.registered_eye_horizontal = []
        self.registered_faces      = 0
        self.min_w                 = 0
        self.min_h                 = 0
        self.previous_position     = (0.0, 0.0, 0.0)
        self.previous_orientation  = (0.0, 0.0, 0.0, 0.0)

        if load_loc is not None:
            self.load_loc = os.path.expanduser('~') + load_loc
            self.load(self.load_loc)

        """
        rate = rospy.Rate(10) # 25Hz, nominally.    
        while not rospy.is_shutdown():
            try:
                self.tf_listnr.waitForTransform("/r_gripper_tool_frame", self.camera_link, rospy.Time.now(), rospy.Duration(0))
                self.gripper_to_sensor = self.tf_listnr.lookupTransform("/r_gripper_tool_frame", self.camera_link, rospy.Time.now())#
            except:
                print "Waiting tf"

            if self.gripper_to_sensor is not None: 
                print self.gripper_to_sensor
                break
            rate.sleep()
        """
        #self.gripper_to_sensor =  [[-0.072,-0.011,-0.068], [0.479, -0.514, 0.464, -0.540]]
        self.gripper_to_sensor =  [[-0.044,0.006,-0.058], [-0.488, 0.512, -0.471, 0.527]]
        #if self.gripper_to_sensor is None:
        #    self.gripper_to_sensor = np.array([[ 0.04152687,  0.00870336,  0.99909948,  -0.072        ],\
        #                                       [-0.99300273,  0.11099984,  0.04030652,  -0.011        ],\
        #                                       [-0.11054908, -0.99378231,  0.01325194,  -0.068        ],\
        #                                       [ 0.        ,  0.        ,  0.        ,  1.        ]])
        #self.gripper_to_sensor = np.array([[ 0.04152687,  0.00870336,  0.99909948,  -0.015        ],\
        #[-0.99300273,  0.11099984,  0.04030652,  -0.066        ],\
        #[-0.11054908, -0.99378231,  0.01325194,  0.073        ],\
        #[ 0.        ,  0.        ,  0.        ,  1.        ]])



        #subscribers
        self.image_sub      = message_filters.Subscriber(rgb_image, Image, queue_size=1)
        self.depth_sub      = message_filters.Subscriber(depth_image, Image, queue_size=1)
        self.gripper_sub    = message_filters.Subscriber(gripper, PoseStamped, queue_size=1)
        self.rgb_info_sub   = message_filters.Subscriber(rgb_info, CameraInfo, queue_size=10)
        self.depth_info_sub = message_filters.Subscriber(depth_info, CameraInfo, queue_size=10)
        self.info_ts        = message_filters.ApproximateTimeSynchronizer([self.rgb_info_sub,  self.depth_info_sub], 10, 100)
        self.info_ts.registerCallback(self.initialize_frames)
        self.ts             = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.gripper_sub], 10, 100)
        self.ts.registerCallback(self.callback)
        
        self.subscribed_success          = False
        
        
        #publishers
        self.mouth_pub = rospy.Publisher('/hrl_manipulation_task/mouth_pnp_pose', PoseStamped, queue_size=10)
        #self.mouth_calc_pub = rospy.Publisher('/hrl_manipulation_task/mouth_pose_backpack', PoseStamped, queue_size=10)
        self.mouth_calc_pub = rospy.Publisher('/hrl_manipulation_task/mouth_pose_backpack_unfiltered', PoseStamped, queue_size=10)
        self.quat_pub = rospy.Publisher('/hrl_manipulation_task/mouth_pose_backpack_quat', Quaternion, queue_size=10)
        
        #displays
        if display_3d:
            self.poly_pub = []
            for i in xrange(200):
                self.poly_pub.append(rospy.Publisher('/poly_pub' + str(i), PolygonStamped, queue_size=10))        

        timeout_cnt = 0
        while not self.subscribed_success or not self.frame_ready:
            timeout_cnt += 1
            if not self.frame_ready:
                print "frame data was not registered"
            if not self.subscribed_success:
                print "data is either not published or not synchronized"
                time.sleep(1)
            if timeout_cnt > 10:
                print "failed to initialize"
                return
        print "successfully initialized"

    def load(self, load_loc):
        if os.path.isfile(load_loc) == False:
            return
        d = ut.load_pickle(load_loc)
        self.first             = d['first']
        self.relation          = d['relation'] 
        self.dist              = d['dist']
        self.reverse_dist      = d['reverse_dist']
        self.half_dist         = d['half_dist']
        self.point_set         = d['point_set']
        self.current_positions = d['current_positions']
        self.object_points     = d['object_points']
        self.point_set_index   = d['point_set_index']
        self.sizes             = d['sizes']
        self.lm_coor           = d['lm_coor']
        self.gripper_to_sensor = d['gripper_to_sensor']
        self.wrong_coor        = d['wrong_coor']
        self.previous_position = (0.0, 0.0, 0.0)
        self.previous_orientation = (0.0, 0.0, 0.0, 0.0)
        

    def save(self, save_loc):
        d = {}
        d['first']             = self.first
        d['relation']          = self.relation
        d['dist']              = self.dist
        d['reverse_dist']      = self.reverse_dist
        d['half_dist']         = self.half_dist
        d['point_set']         = self.point_set
        d['current_positions'] = self.current_positions
        d['object_points']     = self.object_points
        d['point_set_index']   = self.point_set_index
        d['sizes']             = self.sizes
        d['lm_coor']           = self.lm_coor
        d['gripper_to_sensor'] = self.gripper_to_sensor
        d['wrong_coor']        = self.wrong_coor
        d['previous_position'] = self.previous_position
        d['previous_orientation'] = self.previous_orientation
        ut.save_pickle(d, save_loc)

    def callback(self, data, depth_data, gripper_pose):
        #if data is not recent enough, reject
        time1= time.time()
        self.subscribed_success = True
        if data.header.stamp.to_sec() - rospy.get_time() < -.1 or not self.frame_ready:# or abs(data.header.stamp.to_sec() - depth_data.header.stamp.to_sec()) > 0.1 or abs(depth_data.header.stamp.to_sec() - gripper_pose.header.stamp.to_sec()) > 0.1:
            return
            
        
        #print "hello world"
        #print "difference in time on camera images", data.header.stamp.to_sec() - depth_data.header.stamp.to_sec()
        #print "difference in image and tf", data.header.stamp.to_sec() - gripper_pose.header.stamp.to_sec()
        #print "rgb time", data.header.stamp
        #print "depth time", depth_data.header.stamp
        #gripper_pose = PoseStamped()
        position, orientation = self.pose_to_tuple(gripper_pose)
        base_to_gripper = tft.quaternion_matrix(orientation)
        for i in xrange(3):
            base_to_gripper[i][3] = position[i]

        #get rgb and depth image
        img = self.bridge.imgmsg_to_cv2(data, self.rgb_mode)
        if self.flipped:
            img = cv2.flip(img, 1)
            img = cv2.flip(img, 0)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = img.copy()

        #radoned  = radon(gray_img, theta=0)
        if self.display_2d:
            self.win.clear_overlay()
            self.win.set_image(img)
        depth = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")
        best_pose_points = []
        if len(self.current_positions) > 1:
            if type(self.current_positions[0]) is tuple:
                for position in self.current_positions:
                    best_pose_points.append(Point(position))
            else:
                best_pose_points = self.current_positions

        #detect face in 2d, if not found assume face is in previous location
        faces = self.detector(img)
        if len(faces) < 1:
            if self.first:
                return
            faces = dlib.dlib.rectangles()
            if self.registered_faces <45:
                faces.append(dlib.dlib.rectangle(self.previous_face[0].left() - 15, self.previous_face[0].top(), self.previous_face[0].right()- 15, self.previous_face[0].bottom()))
                faces.append(dlib.dlib.rectangle(self.previous_face[0].left() + 15, self.previous_face[0].top(), self.previous_face[0].right()+ 15, self.previous_face[0].bottom()))
            else:
                for i in xrange(3):
                    faces.append(dlib.dlib.rectangle(self.previous_face[0].left() - 10 * (i+1), self.previous_face[0].top(), self.previous_face[0].right()- 10 * (i + 1), self.previous_face[0].bottom()))
                    faces.append(dlib.dlib.rectangle(self.previous_face[0].left() + 10 * (i+1), self.previous_face[0].top(), self.previous_face[0].right()+ 10 * (i + 1), self.previous_face[0].bottom()))
            #faces.append(self.previous_face[0])
            faces.append(self.previous_face[0])
            #faces = self.previous_face
            self.face_detected = False
            best = 9999
        else:
            self.previous_face = faces
            self.face_detected = True
            best = 0
        #print img.shape
        best_pose = PoseStamped()
        best_point_set = None
        best_rect = self.previous_face[0]
        for d in faces:
            x = d.left()
            y = d.top()
            w = d.right() - d.left()
            h = d.bottom() - d.top()
            if x + (w/2) < img.shape[1] and y + (h/2) < img.shape[0]:
                #find landmarks
                shape = self.predictor(img, d)
                landmarks = shape.parts()
                if self.flipped:
                    new_landmarks = []
                    for mark in landmarks:
                        new_landmarks.append(Point((img.shape[1] - 1 - mark.x, img.shape[0] - 1 - mark.y, 0)))
                    landmarks = new_landmarks

                face_img = gray_img[y:y+h, x:x+w]#.astype('uint8')
                face_img=face_img.astype('uint8')
                eye_processing_time = time.time()
                theta = [0.0]
                vertical_integral = radon(face_img, theta=theta).tolist()
                eye_x, eye_y, eye_w, eye_h = self.find_boundary(landmarks[36:48], img.shape, size_limit=False)
                l_eye_thetha = self.find_angle(landmarks[36], landmarks[45])
                if self.first:
                    self.min_w = 24#int(eye_w)
                    self.min_h = 30#int(eye_h)
                    #print self.min_w
                    #print self.min_h
                if eye_y < 0 or eye_x < 0 or eye_y+(2 * eye_h) >= gray_img.shape[0] or eye_x+(2 * eye_w) >= gray_img.shape[1]:
                    continue
                else:
                    eye_img = gray_img[eye_y:eye_y+(2 * eye_h), eye_x:eye_x+(2 * eye_w)].astype('uint8')
                eye_w = self.min_w
                eye_h = self.min_h
                eye_img = cv2.resize(eye_img, (eye_w * 2, eye_h * 2))
                img_rotation_matrix = cv2.getRotationMatrix2D((eye_img.shape[0] / 2, eye_img.shape[1] / 2), l_eye_thetha, 1)
                #print img_rotation_matrix, eye_img.shape
                eye_img = cv2.warpAffine(eye_img, img_rotation_matrix, (eye_img.shape[0], eye_img.shape[1]))
                eye_img = eye_img[int( 5 * eye_h / 6):int(7 * eye_h / 6), int( eye_w/ 3):int(5 * eye_w / 3)].astype('uint8')
                if self.display_2d:
                    self.win2.set_image(eye_img)

                theta = [90.0]
                left_horizontal_integral = radon(eye_img, theta=theta).tolist()
                theta = [0.0]
                left_vertical_integral = radon(eye_img, theta=theta).tolist()
                if self.registered_faces < 45 and self.face_detected:
                    #self.registered_vertical.append(vertical_integral)
                    self.registered_eye_horizontal.append(left_horizontal_integral)
                    #self.registered_eye_horizontal.append(right_horizontal_integral)
                    self.registered_eye_vertical.append(left_vertical_integral)
                    #self.registered_eye_vertical.append(right_vertical_integral)
                    self.registered_faces += 1
                    if self.registered_faces is 45:
                        #self.face_vertical_model  = self.find_mean_var(self.registered_vertical)
                        self.eye_horizontal_model = self.find_mean_var(self.registered_eye_horizontal)
                        self.eye_vertical_model = self.find_mean_var(self.registered_eye_vertical)
                        #print self.eye_horizontal_model
                        #print "finished registering!"
                        #print self.registered_vertical
                if self.display_2d:
                    self.win.add_overlay(faces)
                    self.win.add_overlay(shape)
                if self.first:
                    mouth = self.get_3d_pixel(landmarks[62].x, landmarks[62].y, depth)
                    if not self.flipped:
                        second_point = self.get_3d_pixel(int(x + (3 * w/ 4)), int (y + (4 * h / 9)), depth)
                        third_point  = self.get_3d_pixel(int(x + (w / 4)), int (y + (4 * h / 9)), depth)
                    else:
                        second_point = self.get_3d_pixel(int(img.shape[1]- x - 1 - (3 * w/ 4)), int (img.shape[0]- y - 1 - (4 * h / 9)), depth)
                        third_point  = self.get_3d_pixel(int(img.shape[1]- x - 1 - (w / 4)), int (img.shape[0]- y - 1 - (4 * h / 9)), depth)
                    special_points = [mouth, second_point, third_point]
                    valid_registration = True
                    for point in special_points:
                        if np.allclose(point, (0.0, 0.0, 0.0)):
                            valid_registration = False
                    if not valid_registration:
                        continue
                    self.point_set_index=self.find_best_set(landmarks, depth)
                point_set, points_ordered = self.retrieve_points(landmarks, depth)
                #print points_ordered

                #register values for PnPRansac for comparison
                """
                if self.first:
                    temp_mouth = points_ordered[62]
                    for object_point in points_ordered[17:]:
                        if not np.allclose(object_point, (0.0, 0.0, 0.0)):
                            self.object_points.append([(object_point[1]-temp_mouth[1])*-1.0, (object_point[2]-temp_mouth[2]), (object_point[0]-temp_mouth[0]) * -1.0])
                        else:
                            self.object_points.append([0.0, 0.0, 0.0])
                    self.object_points = np.asarray(self.object_points)
                    self.object_points = self.object_points.astype('float32')
                pnp_trans = self.use_pnp_ransac(landmarks, self.cam_matrix)
                """
                if self.first:
                    ## try:
                    ##     self.tf_listnr.waitForTransform("/r_gripper_tool_frame", self.camera_link, rospy.Time(), rospy.Duration(5.0))
                    ## except:
                    ##     self.tf_listnr.waitForTransform("/r_gripper_tool_frame", self.camera_link, rospy.Time(), rospy.Duration(5.0))
                    ## self.gripper_to_sensor = self.tf_listnr.lookupTransform("/r_gripper_tool_frame", self.camera_link, rospy.Time())#
                    ## print self.gripper_to_sensor
                    matrix_form = tft.quaternion_matrix(self.gripper_to_sensor[1])
                    for i in xrange(3):
                        matrix_form[i][3] = self.gripper_to_sensor[0][i]
                    self.gripper_to_sensor = matrix_form #
                    #register values for frontal face
                    #find special points in 3D

                    #initialize variables to hold informations
                    for i in xrange(len(special_points)):
                        self.dist.append([])
                        self.half_dist.append([])
                    #find relation to retrieve special points from 3 point sets, and their expected size (of 3d triangle)
                    for points in point_set:
                        for i in xrange(len(special_points)):
                            points_and_point = points + [Point(special_points[i])]
                            print "1"
                            self.dist[i].append(self.find_dist(points_and_point))
                            print "2"
                            self.half_dist[i].append(self.find_half_dist(points_and_point, depth))
                            print "3"
                    #retrieve points for checking
                    for i in xrange(len(special_points)):
                        self.current_positions.append((0.0, 0.0, 0.0))
                    pose, pose_points = self.retrieve_special_pose(point_set, depth)
                    #find inverse relation to landmark points in case 2d landmark failure
                    for i in xrange(len(landmarks)):
                        reverse_point_set = pose_points + [Point(points_ordered[i])]
                        self.reverse_dist.append([self.find_dist(reverse_point_set), self.find_half_dist(reverse_point_set, depth)])
                    retrieved_points = self.retrieve_points_from_pose(pose_points, depth) 
                    for i in xrange(len(landmarks)):
                        prev_point = points_ordered[i]
                        curr_point = (retrieved_points[i].pose.position.x,  retrieved_points[i].pose.position.y,  retrieved_points[i].pose.position.z)

                    orientation = self.pose_to_tuple(pose)[1]
                    self.relation = tft.unit_vector(self.get_quaternion_relation([orientation, tuple(tft.quaternion_from_euler(0, 0, np.math.pi/2))]))
                    orientation = tft.quaternion_multiply(orientation, self.relation)
                    orientation = tft.unit_vector(orientation)
                    transform_matrix = tft.quaternion_matrix(orientation)
                    for i, offset in enumerate(self.pose_to_tuple(pose)[0]):
                        transform_matrix[i][3] = offset
                    transform_matrix = tft.inverse_matrix(transform_matrix)
                    #print transform_matrix
                    for point in points_ordered:
                        lm_point = tft.translation_matrix(point)
                        lm_point = np.array(np.matrix(transform_matrix)*np.matrix(lm_point))
                        lm_point = tft.translation_from_matrix(lm_point)
                        self.lm_coor.append(lm_point)
                        if np.allclose(point, (0.0, 0.0, 0.0)):
                            #print lm_point
                            self.wrong_coor = lm_point
                    self.first = False 
                    if self.save_loc is not None:
                        self.save(self.save_loc)
                else:
                    if self.registered_faces >= 45 and not self.face_detected:
                        time_to_align = time.time()
                        left_horizontal_integral  = self.align_integral(left_horizontal_integral, self.eye_horizontal_model)
                        #right_horizontal_integral = self.align_integral(right_horizontal_integral, self.eye_horizontal_model)
                        left_vertical_integral  = self.align_integral(left_vertical_integral, self.eye_vertical_model)
                        #right_vertical_integral = self.align_integral(right_vertical_integral, self.eye_vertical_model)

                        #print "time to align: ", time.time() - time_to_align
                        #count = left_horizontal_integral[1]
                        count = (left_horizontal_integral[1] / self.min_w + left_vertical_integral[1]) / (self.min_h / 3)
                        count = count / 2
                        #count  = (left_horizontal_integral[1]) + right_horizontal_integral[1]) / 4
                        #count += (left_vertical_integral[1]) + right_vertical_integral[1]) / 4
                        count = count
                        if best > count:
                            best_rect = d
                            best = count
                            best_point_set = point_set
                            if self.display_2d:
                                self.win2.set_image(eye_img)
                        #time.sleep(2)
                        continue
                    #retrieve points and make pose
                    pose, pose_points = self.retrieve_special_pose(point_set, depth)

                    #check validity
                    valid_pose = not np.allclose(self.pose_to_tuple(pose)[0], (0.0, 0.0, 0.0)) and not np.allclose(self.pose_to_tuple(pose)[1], (0.0, 0.0, 0.0, 0.0))
                    valid_pose = valid_pose and not np.isnan(self.pose_to_tuple(pose)[0][0]) and not np.isnan(self.pose_to_tuple(pose)[1][0])
                    if not valid_pose:
                        continue
                        
                    time2 = time.time()
                    retrieved_points = self.retrieve_points_from_orientation(pose)
                    retrieved_points_tuple = []
                    for ret_point in retrieved_points:
                        retrieved_points_tuple.append(self.pose_to_tuple(ret_point)[0])
                    retrieved_landmarks = dlib.dlib.points()
                    for i in xrange(len(landmarks)):
                        prev_point = points_ordered[i]
                        curr_point = (retrieved_points[i].pose.position.x,  retrieved_points[i].pose.position.y,  retrieved_points[i].pose.position.z)
                        curr_point_2d = self.get_2d_pixel(curr_point)
                        retrieved_landmarks.append(dlib.dlib.point(int(curr_point_2d[0]), int(curr_point_2d[1])))
                    new_point_set, new_points_ordered = self.retrieve_points(retrieved_landmarks, depth, use_points=False)#point_set, points_ordered#
                    #print self.pose_to_tuple(pose)
                    transformation_matrix = tft.quaternion_matrix(self.pose_to_tuple(pose)[1])
                    for i in xrange(3):
                        transformation_matrix[i][3] = self.pose_to_tuple(pose)[0][i]
                    transformation_matrix = tft.inverse_matrix(transformation_matrix)
                    current_lm = []
                    for coor in new_points_ordered:
                        lm_point = tft.translation_matrix(coor)
                        lm_point = np.array(np.matrix(transformation_matrix)*np.matrix(lm_point))
                        lm_point = tft.translation_from_matrix(lm_point)
                        if not np.allclose(coor, (0.0, 0.0, 0.0)):
                            current_lm.append(lm_point)
                        else:
                            current_lm.append((0.0, 0.0, 0.0))
                    valid_amount = []
                    for i in xrange(30,31):
                        valid_amount.append(0)
                    for i in xrange(30,31):
                        if not np.allclose(current_lm[i], (0.0, 0.0, 0.0)) and not np.allclose(self.lm_coor[i], (0.0, 0.0, 0.0)):
                            #print self.get_dist(current_lm[i], self.lm_coor[i])
                            if self.get_dist(current_lm[i], self.lm_coor[i]) < 0.05:
                                for j in xrange(len(current_lm)):
                                    if not np.allclose(current_lm[j], (0.0, 0.0, 0.0)) and not np.allclose(self.lm_coor[j], (0.0, 0.0, 0.0)):
                                        coor_vect      = tft.unit_vector(self.vector_sub(self.lm_coor[i], self.lm_coor[j]))
                                        curr_coor_vect = tft.unit_vector(self.vector_sub(current_lm[i], current_lm[j]))
                                        dist           = np.linalg.norm(self.vector_sub(coor_vect, curr_coor_vect))
                                        if dist < 0.3:
                                            valid_amount[0] += 1
                    count = valid_amount[0]
                    if best < count:
                        best = count
                        best_pose = pose
                        best_rect = d
                        best_pose_points = pose_points
        if self.first:
            return
        if best_point_set is not None:
            best_pose, best_pose_points = self.retrieve_special_pose(best_point_set, depth)
        position, orientation = self.pose_to_tuple(best_pose)
        current_positions = []
        if len(best_pose_points) > 1:
            pose_points = best_pose_points
        for pose_point in pose_points:
            current_positions.append(pose_point.get_tuple())
        no_jump = True
        if not self.face_detected:
            for i in xrange(len(current_positions)):
                if self.get_dist(current_positions[i], self.current_positions[i]) > 0.05:
                    no_jump = False
            if best > 0.5 and best_point_set is not None:
                no_jump = False
            if best < 15 and best_point_set is None:
                no_jump = False
        #print no_jump
        if no_jump:
            self.current_positions = current_positions
            self.previous_position, self.previous_orientation = position, orientation
            self.previous_face = dlib.dlib.rectangles()
            self.previous_face.append(best_rect)
        else:
            position, orientation = self.previous_position, self.previous_orientation
        #display frame
        if self.display_3d:
            self.br.sendTransform(position, orientation, rospy.Time.now(), "/mouth_position", self.camera_link)
        #publish
        temp_pose = tft.quaternion_matrix(qt.quat_normal(orientation))
        curr_angle = qt.quat_angle(orientation, tft.quaternion_from_euler(0, 0, np.math.pi/2)) / np.math.pi * 180            
        #print curr_angle
        for i in xrange(3):
            temp_pose[i][3] = position[i]
        temp_pose = np.array(np.matrix(self.gripper_to_sensor)*np.matrix(temp_pose))
        temp_pose = np.array(np.matrix(base_to_gripper)*np.matrix(temp_pose))
        temp_pose = self.make_pose(tft.translation_from_matrix(temp_pose), orientation=tft.quaternion_from_matrix(temp_pose))
        if not np.isnan(temp_pose.pose.position.x) and not np.isnan(temp_pose.pose.orientation.x) and (curr_angle < 45 or curr_angle > 135):
            self.quat_pub.publish(temp_pose.pose.orientation)
            try:
                temp_pose.header.stamp = data.header.stamp#rospy.Time.now() #
                temp_pose.header.frame_id = "torso_lift_link"
                #temp_pose = self.tf_listnr.transformPose("torso_lift_link", temp_pose)
                self.mouth_calc_pub.publish(temp_pose)
                #print "pose time", temp_pose.header.stamp.to_sec()
                #print "image times", data.header.stamp.to_sec(), depth_data.header.stamp.to_sec()
            except:
                print "failed"
        """
        else:
            print "publishing previous points"
            temp_pose = tft.quaternion_matrix(qt.quat_normal(orientation))
            for i in xrange(3):
                temp_pose[i][3] = position[i]
            temp_pose = np.array(np.matrix(self.gripper_to_sensor)*np.matrix(temp_pose))
            temp_pose = np.array(np.matrix(base_to_gripper)*np.matrix(temp_pose))
            temp_pose = self.make_pose(tft.translation_from_matrix(temp_pose), orientation=tft.quaternion_from_matrix(temp_pose))
            self.quat_pub.publish(temp_pose.pose.orientation)
            try:
                temp_pose.header.stamp = data.header.stamp#rospy.Time.now() #
                temp_pose.header.frame_id = "torso_lift_link"
                #temp_pose = self.tf_listnr.transformPose("torso_lift_link", temp_pose)
                self.mouth_calc_pub.publish(temp_pose)
                #print "pose time", temp_pose.header.stamp.to_sec()
                #print "image times", data.header.stamp.to_sec(), depth_data.header.stamp.to_sec()
            except:
                print "failed"
        """
        #self.mouth_pub.publish(best_pose)
        if best_point_set is not None:
            print "best was ", best
            #time.sleep(2)
        ## self.mouth_pub.publish(pnp_pose)
        print time.time() - time1

    def find_angle(self, p1, p2):
        if self.flipped:
            d_x = p1.x - p2.x
            d_y = p1.y - p2.y
        else:
            d_x = p2.x - p1.x
            d_y = p2.y - p1.y
        #print "d_x and d_y", d_x, d_y
        #print tft.unit_vector((d_x, d_y))
        thetha =  np.arccos(np.clip(np.dot(tft.unit_vector((d_x, d_y)), (1.0, 0.0)), -1.0, 1.0))
        thetha = thetha / np.math.pi * 180
        if d_x * d_y < 0:
            thetha = thetha * -1.0
        #print "angle was ", thetha
        return thetha
        

    def find_boundary(self, marks, shape, size_limit = False):
        min_x = 999
        min_y = 999
        max_x = 0
        max_y = 0
        c_x   = 0
        c_y   = 0
        for mark in marks:
            c_x += mark.x
            c_y += mark.y
            if mark.x < min_x:
                min_x = mark.x
            if mark.y < min_y:
                min_y = mark.y
            if mark.x > max_x:
                max_x = mark.x
            if mark.y > max_y:
                max_y = mark.y
        w  = max_x - min_x
        h  = max_y - min_y
        c_x /= len(marks)
        c_y /= len(marks)
        if self.flipped:
            min_x = shape[1] - 1 - max_x
            min_y = shape[0] - 1 - max_y
            c_x = shape[1] - 1 - c_x
            c_y = shape[0] - 1 - c_y
        if size_limit:
            w = self.min_w
            h = self.min_h
        if w > h:
            h = w
        else:
            w = h
        return int(c_x - int(w)), int(c_y - int(h)), w, h

    #assuming a (translation in value), b (shear), and c (scale in signal value) are near 0
    def align_integral(self, integral, model):
        d_min = -10.0
        d_max = 10.0
        e_min = 0.8
        e_max = 1.2
        best_set = (0, 0)
        best_score = -1
        best_euler = 0 #normalized euler distance
        best_valid = False
        for i in xrange(1):
            d_sample = [0]#np.linspace(d_min, d_max, 3)
            e_sample = [1]#np.linspace(e_min, e_max, 3)
            for d in d_sample:
                for e in e_sample:
                    dist = 0
                    euler_dist = 0
                    cnt  = 0
                    cnt2 = 0
                    #valid = 0
                    for j in xrange(len(integral)):
                        current = (j - d) / e
                        #print current
                        #print current >= 0.0 and current < len(model[0]) - 1.0 and current < len(integral) - 1.0
                        if current >= 0.0 and current < len(model[0]) - 1.0 and current < len(integral) - 1.0:
                            cnt  += 1
                            rounded  = int(current)
                            remainder = current - current
                            mean = model[0][rounded] * (1 - remainder) + model[0][rounded + 1] * remainder
                            var  = model[1][rounded] * (1 - remainder) + model[1][rounded + 1] * remainder
                            val  = integral[rounded][0] * (1 - remainder) + integral[rounded + 1][0] * remainder
                            #print mean, val, var
                            if var > 0.1:
                                cnt2 += 1
                                dist += ((mean - val) ** 2) / var
                                euler_dist += (mean - val) ** 2
                            #else:
                                #dist += ((mean - val) ** 2) / 1.0
                    if cnt2 > 0:
                        dist = dist / cnt2
                    else:
                        dist = 99999
                    #valid = float(cnt2)/float(cnt)
                    if dist < best_score or best_score is -1:
                        best_score = dist
                        best_set = (d, e)
                        best_euler = euler_dist ** 0.5
                        #best_valid = valid
            if np.allclose(best_set[0], d_min):
                d_max = d_min + (d_max - d_min) / 4.0
            elif np.allclose(best_set[0], d_max):
                d_min = d_max - (d_max - d_min) / 4.0
            else:
                temp  = d_min
                d_min = d_max - (d_max - d_min) / 4.0
                d_max = temp  + (d_max - temp ) / 4.0
            if np.allclose(best_set[1], e_min):
                e_max = e_min + (e_max - e_min) / 4.0
            elif np.allclose(best_set[1], e_max):
                e_min = e_max - (e_max - e_min) / 4.0
            else:
                temp  = e_min
                e_min = e_max - (e_max - e_min) / 4.0
                e_max = temp  + (e_max - temp ) / 4.0
        #print best_score, best_set
        return best_set, best_score

    def find_mean_var(self, datas):
        mean = []
        var  = []
        for i in xrange(len(datas[0])):
            mean.append(0)
            var.append(0)
        for data in datas:
            for i in xrange(len(data)):
                mean[i] += data[i][0]
        for i in xrange(len(datas[0])):
            mean[i] = mean[i] / len(datas)
        for data in datas:
            for i in xrange(len(data)):
                var[i] += (data[i][0] - mean[i]) ** 2
        for i in xrange(len(datas[0])):
            var[i] = (var[i] / len(datas))
        return mean, var

    def initialize_frames(self, rgb_info, depth_info):
        self.rgb_f   = (rgb_info.P[0], rgb_info.P[5])
        self.rgb_c   = (rgb_info.P[2], rgb_info.P[6])
        self.depth_f = (depth_info.P[0], depth_info.P[5])
        self.depth_c = (depth_info.P[2], depth_info.P[6])
        self.info_ts        = None
        self.depth_info_sub = None
        self.rgb_info_sub   = None
        self.cam_matrix =  np.asarray(rgb_info.K).reshape((3,3)).astype('float32')
        self.frame_ready = True

    def use_pnp_ransac(self, landmarks, cam_matrix, distortion=None, tvec=None):
        marks = []
        object_points = []
        for i, mark in enumerate(landmarks[17:]):
            if not np.allclose(self.object_points[i], (0.0, 0.0, 0.0)):
                marks.append([float(mark.x), float(mark.y)])
                object_points.append(self.object_points[i])
        object_points = np.asarray(object_points)
        object_points = object_points.astype('float32')
        marks = np.asarray(marks).astype('float32')
        rvec, tvec, inliers = cv2.solvePnPRansac(object_points, marks, cam_matrix, distortion)
        rot = cv2.Rodrigues(rvec)[0].tolist()
        tvec = tvec.tolist()
        rot[0].append(tvec[0][0])
        rot[1].append(tvec[1][0])
        rot[2].append(tvec[2][0])
        rot.append([0.0, 0.0, 0.0, 1.0])
        return rot

    def find_rect(self, landmark_points, shape):
        center = (landmark_points[30].x, landmark_points[30].y)
        half_width  = int(abs(landmark_points[17].x - center[0]))
        half_height = int(abs(landmark_points[48].y - center[1]) *1.5)
        if half_width > half_height:
            half_height = half_width
        else:
            half_width = half_height
        if self.flipped:
            #print "flipped!"
            return dlib.dlib.rectangle(shape[1]- (center[0]+half_width) - 1, shape[0] - (center[1]+half_height)-1, shape[1] - (center[0]-half_width) - 1, shape[0] - (center[1]-half_height) - 1)
        else:
            return dlib.dlib.rectangle(center[0]-half_width, center[1]-half_height, center[0]+half_width, center[1]+half_height)

    def retrieve_points(self, landmarks, depth, use_points=True):
        points = []
        curr_points = []
        for curr in self.current_positions:
            curr_points.append(Point(curr))
        #if use_points:
        #    retrieved_previous_points = self.retrieve_points_from_pose(curr_points, depth)
        #else:
        retrieved_previous_points = self.retrieve_points_from_orientation(self.make_pose(self.previous_position, self.previous_orientation))
        for i, mark in enumerate(landmarks):
            point = self.get_3d_pixel(mark.x, mark.y, depth)
            if len(self.current_positions) >= 1:
                if self.get_dist(self.pose_to_tuple(retrieved_previous_points[i])[0], point) > 0.05 and use_points:
                    points.append((0.0, 0.0, 0.0))
                else:
                    points.append(point)
            else:
                points.append(point)
        count = 0
        for i, mark in enumerate(landmarks):
            if not np.allclose(points[i], (0.0, 0.0, 0.0)) and not np.isnan(points[i][0]):
                count = count + 1
        #print "retrieved points valid ", count
        if count < 20:
            points=[]
            for mark in landmarks:
                point = self.get_3d_pixel(mark.x, mark.y, depth)
                points.append(point)
        for mark in self.point_set_index[0]:
            marks = []
            for index in mark:
                marks.append(landmarks[index])
            point = self.get_3d_center(marks, depth)
            points.append(point)
        final_result = []
        for set in self.point_set_index[1]:
            final_result.append([Point(points[set[0]]), Point(points[set[1]]), Point(points[set[2]])])
        no_poly = [Point32(x=0.0, y=0.0, z=0.0), Point32(x=0.0, y=0.0, z=0.0), Point32(x=0.0, y=0.0, z=0.0)]
        if self.display_3d:
            for j, points_index in enumerate(self.point_set_index[1]):
                if j < len(self.poly_pub):
                    poly_points = []
                    for point_index in points_index:
                        pose = Point32()
                        point = points[point_index]
                        if not np.allclose(point, (0.0, 0.0, 0.0)):
                            pose.x = point[0]
                            pose.y = point[1]
                            pose.z = point[2]
                            poly_points.append(pose)
                        else:
                            poly_points=(no_poly)
                            break
                    poly = PolygonStamped()
                    poly.header.frame_id = self.camera_link
                    poly.polygon.points = poly_points
                    self.poly_pub[j].publish(poly)
        return final_result, points

    #uses 3 points that defines mouth pose to retrieve 3d landmarks
    def retrieve_points_from_pose(self, current_pose, depth):
        points = []
        for dist in self.reverse_dist:
            points.append(self.find_using_ratios(current_pose, dist[0], dist[1], depth, 1.0, False))
        return points

    #uses pose and orientation to retrieve 3d landmarks:
    def retrieve_points_from_orientation(self, pose):
        lm_points = []
        transform = tft.quaternion_matrix(self.pose_to_tuple(pose)[1])
        for i, offset in enumerate(self.pose_to_tuple(pose)[0]):
            transform[i][3] = offset
        #transform = tft.inverse_matrix(transform)
        for coor in self.lm_coor:
            if not np.allclose(coor, self.wrong_coor):
                lm_point = tft.translation_matrix(coor)
                lm_point = np.array(np.matrix(transform)*np.matrix(lm_point))#np.dot(transform, lm_point)
                lm_point = tft.translation_from_matrix(lm_point)
                lm_points.append(self.make_pose(lm_point))
            else:
                lm_points.append(self.make_pose((0.0, 0.0, 0.0)))
        return lm_points

    def make_point(self, point):
        return Point32(x=point[0], y=point[1], z=point[2])

    # first set is set of points to combine, second set is set of points together. -# = new points, +# = landmarks from detector
    def find_best_set(self, landmarks, depth):
        points3d = []
        best_delaunay = []
        best_delaunay_score = 9999999999
        best_2d_3d = []
        invalid_indices = []#[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]
        for i, mark in enumerate(landmarks):
            point = self.get_3d_pixel(mark.x, mark.y, depth)
            points3d.append(list(point))
            if np.allclose(point, [0.0, 0.0, 0.0]):
                invalid_indices.append(i)
        num_list=[]
        for i in xrange(100):
            remove_count = 0#random.randint(0, len(landmarks) / 4)
            removed = [] + invalid_indices
            generated = []
            points2d = []
            sizes= []
            total = 0
            for j in xrange(remove_count):
                remove_ind = random.randint(0, len(landmarks)-1)
                while remove_ind in removed:
                    remove_ind = random.randint(0, len(landmarks)-1)
                removed.append(remove_ind)
            index2dto3d = []
            for j, mark in enumerate(landmarks):
                if j not in removed:
                    point = [mark.x, mark.y]
                    points2d.append(point)
                    index2dto3d.append(j)
            tri = Delaunay(np.array(points2d))
            best_delaunay = tri
            for slice in tri.simplices:
                point_set = []
                valid = True
                for point_ind in slice:
                    if np.allclose(points3d[index2dto3d[point_ind]], [0.0, 0.0, 0.0]):
                        valid = False
                    point_set.append(points3d[index2dto3d[point_ind]])
                if valid:
                    size = self.find_size(point_set)
                    total = total + size
                    sizes.append(size)
            size_var = 999
            if len(sizes) > 2:
                size_var = 0
                total = total / (len(sizes))
                for size in sizes:
                    size_var = size_var + ((size - total) ** 2)
                size_var = size_var / (len(sizes) - 1)
                size_var = size_var - total
            if best_delaunay_score > size_var:
                best_delaunay = tri
                best_delaunay_score = size_var
                best_2d_3d = index2dto3d
        points = points3d
        slices = best_delaunay.simplices
        new_slices = []
        number_array = []
        for i in range(len(landmarks)):
            number_array.append(i)
        for slice in slices:
            new_slice = [best_2d_3d[slice[0]], best_2d_3d[slice[1]], best_2d_3d[slice[2]]]
            new_slice2 = []
            for val in new_slice:
                if val in number_array[27:36]:
                    new_slice2.append(val)
            for val in new_slice:
                if val in number_array[36:48]:
                    new_slice2.append(val)
            for val in new_slice:
                if val in number_array[17:27]:
                    new_slice2.append(val)
            for val in new_slice:
                if val in number_array[48:69]:
                    new_slice2.append(val)
            for val in new_slice:
                if val in number_array[0:17]:
                    new_slice2.append(val)
            new_slices.append(new_slice2)
        slices =new_slices
        best_set = []
        best_score = -1
        for i in xrange(1):
            sets = []
            sets_indices = []
            sizes = []
            poses=[]
            selected = []
            for j in xrange(len(slices)):
                valid = False
                new_set =[]
                for index in slices[j]:
                    new_set.append(index)
                if new_set[0] is not new_set[1] and new_set[1] is not new_set[2] and new_set[0] is not new_set[2]:
                    new_set_indices = new_set
                    new_set = points[new_set[0]], points[new_set[1]], points[new_set[2]]
                    vectors=[]
                    vectors.append([new_set[1][0] - new_set[0][0], new_set[1][1] - new_set[0][1], new_set[1][2] - new_set[0][2]])
                    vectors.append([new_set[2][0] - new_set[0][0], new_set[2][1] - new_set[0][1], new_set[2][2] - new_set[0][2]])
                    quaternion = self.get_quaternion2(vectors[0], vectors[1])
                    if np.allclose(quaternion, [0.0, 0.0,0.0,0.0]) or np.allclose(new_set[0], [0.0, 0.0, 0.0]) or np.allclose(new_set[1], [0.0, 0.0, 0.0]) or np.allclose(new_set[2], [0.0, 0.0, 0.0]):
                        valid = False
                    else:
                        valid = True
                        sets.append(new_set)
                        sets_indices.append(new_set_indices)
                        sizes.append(self.find_size(new_set))
                        center = [0.0, 0.0, 0.0]
                        for point in new_set:
                            center[0] = center[0] + point[0] / 3 
                            center[1] = center[1] + point[1] / 3
                            center[2] = center[2] + point[2] / 3
                        pose = self.make_pose(center, orientation=quaternion, stamped=False)
                        selected.append(j)
            best_set = sets_indices
            best_sizes = sizes
        self.sizes = best_sizes
        return [[3, 30, 36], [14, 30, 45], [18, 25]], [[-1, -2, -3]] + best_set

    def find_size(self, points):
        length =[]
        length.append(self.get_dist(points[0], points[1]))
        length.append(self.get_dist(points[1], points[2]))
        length.append(self.get_dist(points[2], points[0]))
        total = (length[0] + length[1] + length[2]) / 2.0
        return (total * (total - length[0]) * (total - length[1]) * (total - length[2])) ** 0.5

    def get_dist(self, point1, point2, scale = 1.0):
        return ((((point1[0] - point2[0]) * scale) ** 2) + (((point1[1] - point2[1]) * scale) ** 2) + (((point1[2] - point2[2]) * scale) ** 2)) ** 0.5
        
    #gives quaternion matrix from vector[1] to vector[0]
    def get_quaternion(self, vectors):
        axis = tft.unit_vector(np.cross(vectors[0], vectors[1]))
        quaterion = [axis[0], axis[1], axis[2]]
        quaterion.append((np.dot(vectors[0], vectors[0]) ** 0.5) * (np.dot(vectors[1], vectors[1]) ** 0.5) + np.dot(vectors[0], vectors[1]))
        return quaterion

    #returns quaternions between poses. doing 2 * arccos(w) = angle of axis-angle representation
    def get_quaternion_relation(self, poses):
        quaternions = []
        for pose in poses:
            if type(pose) is tuple or type(pose) is list:
                quaternions.append(tft.unit_vector(pose))
            else:
                orientation = pose.pose.orientation
                quaternions.append(qt.quat_normal([orientation.x, orientation.y, orientation.z, orientation.w]))
        return qt.quat_normal(tft.quaternion_multiply(tft.quaternion_inverse(quaternions[0]), quaternions[1]))

    def make_pose(self, point, orientation=None, frame_id=None, stamped=True):
        if stamped:
            new_pose = PoseStamped()
            if frame_id is None:
                new_pose.header.frame_id = self.camera_link
            else:
                new_pose.header.frame_id = frame_id
            pose = new_pose.pose
        else:
            new_pose = Pose()
            pose = new_pose
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = point[2]
        if orientation is not None:
            pose.orientation.x=orientation[0]
            pose.orientation.y=orientation[1]
            pose.orientation.z=orientation[2]
            pose.orientation.w=orientation[3]
        return new_pose
        
    def pose_to_tuple(self, pose):
        if type(pose) is PoseStamped:
            position = pose.pose.position
            orientation = pose.pose.orientation
        else:
            position = pose.position
            orientation = pose.orientation
        position = (position.x, position.y, position.z)
        orientation = (orientation.x, orientation.y, orientation.z, orientation.w)
        return position, orientation

    def find_using_ratios(self, points, dist, half_dist, depth, direction, check_error):
        vector = []
        used_dist = dist[0]
        used_points = [points[0]]
        used_half = []
        for point in points:
            if np.allclose((point.x, point.y, point.z), (0.0, 0.0, 0.0)):# or np.isnan(point.x):
                pose = PoseStamped()
                pose.header.frame_id=self.camera_link
                return pose
        if check_error:
            act_dist = (self.get_dist(points[0].get_tuple(), points[1].get_tuple()),  self.get_dist(points[0].get_tuple(), points[2].get_tuple()))
            dist_diff = [(act_dist[0] - dist[1][0]) / dist[1][0], (act_dist[1] - dist[1][1]) / dist[1][1]]
            half_dist_diff = [(act_dist[0] - half_dist[2][1][0]) / half_dist[2][1][0], (act_dist[1] - half_dist[2][1][1]) / half_dist[2][1][1]]
            if dist_diff[0] > 0.3:
                used_half.append(False)
                used_points.append(self.get_better_estimate(points[0], points[1], dist[1][0], dist_diff[0], depth)[0])
            elif dist_diff[0] < -0.3:
                estimate_point, changed = self.get_better_estimate(points[0], points[1], half_dist[2][1][0], half_dist_diff[0], depth)
                used_points.append(estimate_point)
                if changed:
                    used_half.append(True)
                else:
                    used_half.append(half_dist_diff[0] < dist_diff[0])
            else:
                used_points.append(points[1])
                used_half.append(False)
            if dist_diff[1] > 0.3:
                used_half.append(False)
                used_points.append(self.get_better_estimate(points[0], points[2], dist[1][1], dist_diff[1], depth)[0])
            elif dist_diff[1] < -0.3:
                estimate_point, changed = self.get_better_estimate(points[0], points[2], half_dist[2][1][1], half_dist_diff[1], depth)
                used_points.append(estimate_point)
                if changed:
                    used_half.append(True)
                else:
                    used_half.append(half_dist_diff[1]<dist_diff[1])
            else:
                used_points.append(points[2])
                used_half.append(False)
            which_dist = 0
            if used_half[0] is True:
                which_dist = which_dist + 1
            if used_half[1] is True:
                which_dist = which_dist + 2
            if which_dist is 0:
                used_dist = dist[0]
            else:
                used_dist = half_dist[which_dist-1][0]
        else:
            used_points = points
            used_dist = dist[0]
        vector.append((used_points[1].x - used_points[0].x, used_points[1].y - used_points[0].y, used_points[1].z - used_points[0].z))
        vector.append((used_points[2].x - used_points[0].x, used_points[2].y - used_points[0].y, used_points[2].z - used_points[0].z))
        base = (used_points[0].x, used_points[0].y, used_points[0].z)
        u = []
        u.append(vector[0])
        u_temp = vector[1]
        for vect in u:
            u_temp = self.vector_sub(u_temp, self.projection(vector[1], vect))
        u.append(u_temp)
        offset = (0, 0, 0)
        offset = self.vector_add(offset,base)
        for i, vect in enumerate(u):
            offset = self.vector_add(offset, self.vector_mult_const(tft.unit_vector(u[i]), used_dist[i]))
        offset = self.vector_add(offset, self.vector_mult_const(tft.unit_vector(np.cross(u[0], u[1])), used_dist[-1]))
        orientation = tft.unit_vector(np.cross(vector[0], vector[1]))
        pose = PoseStamped()
        pose.header.frame_id = self.camera_link
        pose.pose.position.x = offset[0]
        pose.pose.position.y = offset[1]
        pose.pose.position.z = offset[2]
        return pose

    def get_better_estimate(self, fixed_point, varied_point, expected_dist, original_dist, depth):
        if np.isnan(expected_dist) or np.allclose(expected_dist, 0.0):
            return varied_point, False
        direction = (varied_point.x - fixed_point.x, varied_point.y - fixed_point.y, varied_point.z - fixed_point.z)
        if np.allclose(direction, (0.0, 0.0, 0.0)):
            return varied_point, False
        direction = tft.unit_vector(direction)
        direction = (fixed_point.x + (direction[0] * expected_dist), fixed_point.y + (direction[1] * expected_dist), fixed_point.z + (direction[2] * expected_dist))
        estimate_point = self.get_2d_pixel(direction)
        if np.isinf(estimate_point[0]) or np.isinf(estimate_point[1]):
            estimate_point = (0.0, 0.0, 0.0)
        else:
            estimate_point = self.get_3d_pixel(int(estimate_point[0]), int(estimate_point[1]), depth)
        if (abs((self.get_dist(estimate_point, fixed_point.get_tuple()) - expected_dist) / expected_dist) - abs(original_dist)) < -0.1 and abs((self.get_dist(estimate_point, fixed_point.get_tuple()) - expected_dist) / expected_dist) < 0.3:
            return Point(estimate_point), True
        else:
            return varied_point, False

    def get_quaternion2(self, vector1, vector2):
        u = []
        norm_vect = tft.unit_vector(np.cross(vector1, vector2))
        u.append(norm_vect)
        u.append(tft.unit_vector(np.cross(vector1, norm_vect)))
        u.append(tft.unit_vector(vector1))
        matrix = [[], [], []]
        for vect in u:
            matrix[0].append(vect[0])
            matrix[1].append(vect[1])
            matrix[2].append(vect[2])
        matrix[0].append(0)
        matrix[1].append(0)
        matrix[2].append(0)
        matrix.append([0, 0, 0, 1])
        return tft.quaternion_from_matrix(matrix)


    def diff_ori(self, ori1, ori2):
        a = np.dot(tft.unit_vector(ori1), tft.unit_vector(ori2)) ** 2
        q = np.dot(tft.unit_vector(ori1), tft.unit_vector(ori2))
        try:
            q = np.math.acos(q) * 2.0
            if q > np.math.pi:
                q = abs(np.math.pi * 2.0 - q)
            q = q / np.math.pi * 180
            return q
        except:
            if np.dot(tft.unit_vector(ori1), tft.unit_vector(ori2)) is 1.0:
                return 0
            return 9999
        return 1-a

    def retrieve_special_pose(self, point_set, depth, dist=None, half_dist=None):
        special_points=[]
        special_poses = []
        current_sizes = []
        if dist is None:
            dist = self.dist
        if half_dist is None:
            half_dist = self.half_dist
        for i in xrange(len(dist)):
            special_points.append([])
        for i, points in enumerate(point_set):
            for j in xrange(len(dist)):
                special_points[j].append(self.find_using_ratios(points, dist[j][i], half_dist[j][i], depth, 1.0, False))
            new_points = []
            for new_point in points:
                point_tuple = (new_point.x, new_point.y, new_point.z)
                new_points.append(point_tuple)
            current_sizes.append(self.find_size(new_points))
        #print current_sizes
        for i in xrange(len(dist)):
            special_poses.append(self.retrieve_special_point(special_points[i], current_sizes, self.current_positions[i]))
        poses = special_poses
        pose = poses[0]
        points = []
        pose_points = []
        current_positions = []
        for ind_pose in poses:
            position=self.pose_to_tuple(ind_pose)[0]
            new_point = self.make_point(position)
            points.append(new_point)
            current_positions.append(position)
            pose_points.append(Point(position))
        """
        if not np.isnan(current_positions[0][0]) and not np.allclose(current_positions[0], (0.0, 0.0, 0.0)):
            self.current_positions = current_positions
        """
        vector = []
        vector.append((points[1].x - points[0].x, points[1].y - points[0].y, points[1].z - points[0].z))
        vector.append((points[2].x - points[0].x, points[2].y - points[0].y, points[2].z - points[0].z))
        norm_vect = tft.unit_vector(np.cross(vector[0], vector[1]))
        orientation = self.get_quaternion2(vector[0], vector[1])
        orientation = tft.unit_vector(orientation)
        if not self.relation is None:
            orientation = tft.quaternion_multiply(orientation, self.relation)
            orientation = tft.unit_vector(orientation)
        position = self.pose_to_tuple(pose)[0]
        if self.display_3d:
            poly = PolygonStamped()
            poly.header.frame_id=self.camera_link
            poly.polygon.points = points
            self.poly_pub[-1].publish(poly)
        return self.make_pose(position, orientation=orientation), pose_points
        

    def retrieve_special_point(self, poses, current_sizes, current_pos, threshold=15, threshold_radius=0.2, n=-1):
        best_ori = None
        best_pos = None
        best = 0
        invalid_cnt = 0
        if n is -1:
            n = len(poses)
        valid = [False]
        for i in range(1, n):
            if n >= len(poses):
                r = i % len(poses)
            position = poses[r].pose.position
            position = [position.x, position.y, position.z]
            dist = self.get_dist(current_pos, position)
            if np.allclose(position, [0.0,0.0,0.0]) or np.isnan(position[0]) or abs(self.sizes[i-1] - current_sizes[i])/self.sizes[i-1] > .5 or dist > 0.05:
                valid.append(False)
            else:
                valid.append(True)
        at_least_one = False
        for val in valid:
            if val:
                at_least_one = True
        if not at_least_one:
            new_valid = []
            for val in valid:
                new_valid.append(True)
            valid=new_valid
        poses_arr =[]
        best_pos = (0, 0, 0)
        for i in range(0, n):
            if n >= len(poses):
                r = i % len(poses)
            new_pose = Pose()
            new_pose.position = poses[r].pose.position
            poses_arr.append(new_pose)
            position = poses[r].pose.position
            position = [position.x, position.y, position.z]
            if not valid[i] or np.isnan(position[0]) or np.allclose(position, (0.0, 0.0, 0.0)):
                invalid_cnt = invalid_cnt + 1
            else:
                orientation = 0
                best = 1
                best_pos = (best_pos[0] + position[0], best_pos[1] + position[1], best_pos[2] + position[2])
        pose_arr_stamped = PoseArray()
        pose_arr_stamped.header.frame_id = self.camera_link
        pose_arr_stamped.poses = poses_arr
        pose = PoseStamped()
        pose.header.frame_id = self.camera_link
        if best > 0:
            pose.pose.position.x = best_pos[0] / (len(poses) - invalid_cnt)
            pose.pose.position.y = best_pos[1] / (len(poses) - invalid_cnt)
            pose.pose.position.z = best_pos[2] / (len(poses) - invalid_cnt)
            if np.allclose(best_pos, (0.0, 0.0, 0.0)) or np.isnan(pose.pose.position.x):
                pose.pose.position.x = current_pos[0]
                pose.pose.position.y = current_pos[1]
                pose.pose.position.z = current_pos[2]
        return pose

    def find_dist(self, points):
        vector = []
        act_dist = (self.get_dist(points[0].get_tuple(), points[1].get_tuple()), self.get_dist(points[0].get_tuple(), points[2].get_tuple()))
        vector.append((points[1].x - points[0].x, points[1].y - points[0].y, points[1].z - points[0].z))
        vector.append((points[2].x - points[0].x, points[2].y - points[0].y, points[2].z - points[0].z))
        vector.append((points[3].x - points[0].x, points[3].y - points[0].y, points[3].z - points[0].z))
        u = []
        u.append(vector[0])
        dist = []
        for j in xrange(len(vector)-1):
            u_temp = vector[j+1]
            for vect in u:
                proj = self.projection(vector[j+1], vect)
                u_temp = self.vector_sub(u_temp, proj)
                if vector[j + 1] is vector[-1]:
                    dist.append(self.vector_divide(proj, tft.unit_vector(vect)))
            u.append(u_temp)
        dist.append(self.vector_divide(u[-1], tft.unit_vector(np.cross(u[0], u[1]))))
        return tuple(dist), act_dist

    def find_half_dist(self, points, depth):
        half_point = ((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2, (points[0].z + points[1].z) / 2 )
        half_point = self.get_2d_pixel(half_point)
        half_point = self.get_3d_pixel(int(half_point[0]), int(half_point[1]), depth)
        half_point2 = ((points[0].x + points[2].x) / 2, (points[0].y + points[2].y) / 2, (points[0].z + points[2].z) / 2 )
        half_point2 = self.get_2d_pixel(half_point2)
        half_point2 = self.get_3d_pixel(int(half_point2[0]), int(half_point2[1]), depth)
        half_point = Point(half_point)
        half_point2 = Point(half_point2)
        half_points = [points[0], half_point, points[2], points[3]]
        half_points2 = [points[0], points[1], half_point2, points[3]]
        half_points3 = [points[0], half_point, half_point2, points[3]]
        return (self.find_dist(half_points), self.find_dist(half_points2), self.find_dist(half_points3))

    def get_2d_pixel(self, point):
        if type(point) is Point:
            new_point = (point.x, point.y, point.z)
        else:
            new_point = point
        if not np.allclose(point, (0.0, 0.0, 0.0)):
            x = self.rgb_c[0] + (self.rgb_f[0] * point[0] / point[2]) 
            y = self.rgb_c[1] + (self.rgb_f[1] * point[1] / point[2])
        else:
            x = 0
            y = 0
        if np.isnan(x) or np.isnan(y):
            x = 0
            y = 0
        return (x, y)

    def vector_divide(self, vect1, vect2):
        a = (np.dot(vect1,vect1)/ np.dot(vect2, vect2))**0.5
        if np.sign(vect1[0]) != np.sign(vect2[0]):
            a = a * -1.0
        return a

    def vector_add(self, vect1, vect2):
        temp = []
        for i, val in enumerate(vect1):
            temp.append(vect1[i] + vect2[i])
        return tuple(temp)

    def vector_sub(self, vect1, vect2):
        temp = []
        for i, val in enumerate(vect1):
            temp.append(vect1[i] - vect2[i])
        return tuple(temp)

    def vector_mult_const(self, vector, c):
        temp = []
        for val in vector:
            temp.append(val * c)
        return tuple(temp)
   
    def projection(self, vector, base):
        temp = np.dot(base, base)# ** .5
        if temp == 0:
            temp = 1
        temp = np.dot(vector, base) / temp
        vect = []
        for i, val in enumerate(base):
            vect.append(val * temp)
        return tuple(vect)

    def get_3d_center(self, points, depth):
        x = 0
        y = 0
        count = 0
        for point in points:
            x = point.x + x
            y = point.y + y
            count = count + 1
        x = int(x / count)
        y = int(y / count)
        return self.get_3d_pixel(x, y, depth)

    # cx, cy, fx, fy are from Camera_Info of rgb. 
    # assumes no-very little distortions from lens or is taken into account in calibration
    # assumes rgbd is already alligned. (or depth_f ~= rgb_f and offset between two frame is close to 0)
    def get_3d_pixel(self, rgb_x, rgb_y, depth):
        best = []
        best_val = 1000.00
        scale   = self.depth_scale
        rgb_f   = self.rgb_f
        rgb_c   = self.rgb_c
        depth_f = self.depth_f
        depth_c = self.depth_c
        shape = depth.shape
        if not (rgb_x < 0 or rgb_x >=shape[1] or rgb_y < 0 or rgb_y >= shape[0]):
            if not np.isnan(depth[rgb_y][rgb_x]) and not np.allclose(depth[rgb_y][rgb_x], 0.0):
                z_metric = depth[rgb_y][rgb_x] * scale + self.offset[2]
                app_rgb_x = z_metric * (rgb_x - rgb_c[0]) * (1.0 / rgb_f[0])
                app_rgb_y = z_metric * (rgb_y - rgb_c[1]) * (1.0 / rgb_f[1])
                return (app_rgb_x, app_rgb_y, z_metric)
        for i in [rgb_x -1, rgb_x, rgb_x+1]:
            for j in [rgb_y-1, rgb_y, rgb_y+1]:
                if not (i < 0 or i >= shape[1] or j < 0 or j >= shape[0]):
                    if not np.isnan(depth[j][i]) and not np.allclose(depth[j][i], 0.0):
                        z_metric = depth[j][i] * scale + self.offset[2]
                        approx_x = z_metric * (i - depth_c[0]) * (1.0 / depth_f[0])
                        approx_y = z_metric * (j - depth_c[1]) * (1.0 / depth_f[1])
                        translated_x = rgb_c[0] + ((approx_x) * rgb_f[0]) / z_metric
                        translated_y = rgb_c[1] + ((approx_y) * rgb_f[1]) / z_metric
                        app_rgb_x = z_metric * (rgb_x - rgb_c[0]) * (1.0 / rgb_f[0])
                        app_rgb_y = z_metric * (rgb_y - rgb_c[1]) * (1.0 / rgb_f[1])
                        projection = self.projection((approx_x, approx_y, z_metric), (app_rgb_x, app_rgb_y, z_metric))
                        norm = self.vector_sub((approx_x, approx_y, z_metric), projection)
                        distance = np.dot(norm, norm)
                        if best_val > distance:
                            best_val = distance
                            best = (app_rgb_x, app_rgb_y, z_metric)
        if len(best) > 1:
            return tuple(best)
        else:
            return tuple((0,0,0))

class Point:
    x=0
    y=0
    z=0
    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]
        self.z = data[2]

    def get_tuple(self):
        return (self.x, self.y, self.z)




if __name__ == '__main__':
    rospy.init_node('mouth_detector')

    #grab parameters
    import optparse
    p = optparse.OptionParser()
    p.add_option("-l", "--link", dest="rgb_camera_link", 
                 default="/camera_rgb_optical_frame", help="rgb optical frame link")
    p.add_option("-c", "--camera", dest="camera_type", default="kinect",
                 help="camera type to be used")
    p.add_option("-r", "--rgb", dest="rgb_image", default="/camera/rgb/image_color",
                 help="rgb image to subscribe to")
    p.add_option("-d", "--depth", dest="depth_image", 
                 default="/camera/depth_registered/image",
                 help="depth image to subscribe to")
    p.add_option("-R", "--rgb_info", dest="rgb_info",
                 default="/camera/rgb/camera_info",
                 help="rgb camera info to get calibrations")
    p.add_option("-D", "--depth_info", dest="depth_info",
                 default="/camera/depth_registered/camera_info",
                 help="rgb camera info to get calibrations")
    p.add_option("-s", "--scale", dest="depth_scale", type="float",
                 default=1.0, help="scale depth to meters")
    p.add_option("-o", "--offset", dest="offset", type="float", nargs=3, default=(0.0, 0.0, 0.0))
    p.add_option("--rgb_mode", dest="rgb_mode", default="rgb8", help="rgb format")
    p.add_option("--flip", dest="flipped", action="store_true", default=False)
    p.add_option("--display_2d", dest="display_2d", action="store_true", default=False)
    p.add_option("--display_3d", dest="display_3d", action="store_true", default=False)
    p.add_option("--save", dest="save")
    p.add_option("--load", dest="load")
    (options, args) = p.parse_args()
    #print options.offset
    detector = MouthPoseDetector(options.rgb_camera_link, options.rgb_image,
                                 options.depth_image, options.rgb_info, 
                                 options.depth_info, options.depth_scale, options.offset,
                                 options.display_2d, options.display_3d,
                                 options.flipped, options.rgb_mode, options.save, options.load)
    rospy.spin()
