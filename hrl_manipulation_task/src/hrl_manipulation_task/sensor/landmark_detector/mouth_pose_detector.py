#!/usr/bin/env python

import rospy
import os
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
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point32, PolygonStamped, Vector3
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from cv_bridge import CvBridge, CvBridgeError

class MouthPoseDetector:
    def __init__(self, camera_link, rgb_image, depth_image, rgb_info, depth_info, depth_scale, offset,
                 display_2d=True, display_3d=True, flipped=False, rgb_mode="bgr8", save_loc=None, load_loc=None):
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
        self.first             = True
        self.relation          = None
        self.dist              = []
        self.reverse_dist      = []
        self.half_dist         = []
        self.point_set         = []
        self.current_positions = []
        self.object_points     = []
        self.point_set_index   = []
        self.sizes             = []
        self.lm_coor           = []
        self.previous_position = (0.0, 0.0, 0.0)
        self.previous_orientation = (0.0, 0.0, 0.0, 0.0)

        if load_loc is not None:
            self.load(load_loc)

        #subscribers
        self.image_sub      = message_filters.Subscriber(rgb_image, Image, queue_size=10)
        self.depth_sub      = message_filters.Subscriber(depth_image, Image, queue_size=10)
        self.gripper_sub    = message_filters.Subscriber(gripper, PoseStamped, queue_size=10)
        self.rgb_info_sub   = message_filters.Subscriber(rgb_info, CameraInfo, queue_size=10)
        self.depth_info_sub = message_filters.Subscriber(depth_info, CameraInfo, queue_size=10)
        self.info_ts        = message_filters.ApproximateTimeSynchronizer([self.rgb_info_sub,  self.depth_info_sub], 10, 100)
        self.info_ts.registerCallback(self.initialize_frames)
        self.ts             = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.gripper_sub], 10, 100)
        self.ts.registerCallback(self.callback)
        
        #for image processings
        self.bridge = CvBridge()
        self.previous_face = []
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.expanduser('~') + '/Desktop/shape_predictor_68_face_landmarks.dat')
        
        #publishers
        self.mouth_pub = rospy.Publisher('/hrl_manipulation_task/mouth_pnp_pose', PoseStamped, queue_size=10)
        self.mouth_calc_pub = rospy.Publisher('/hrl_manipulation_task/mouth_pose_backpack', PoseStamped, queue_size=10)

        """
        self.failed_mouth_pub = []
        for i in xrange(4):
            self.failed_mouth_pub.append(rospy.Publisher('/mouth_pub' + str(i), PoseStamped, queue_size=10))
        """

        #displays
        self.display_2d = display_2d
        self.display_3d = display_3d
        if display_2d:
            self.win = dlib.image_window()
        if display_3d:
            self.poly_pub = []
            for i in xrange(200):
                self.poly_pub.append(rospy.Publisher('/poly_pub' + str(i), PolygonStamped, queue_size=10))        

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
        d['previous_position'] = self.previous_position
        d['previous_orientation'] = self.previous_orientation
        ut.save_pickle(d, save_loc)

    def callback(self, data, depth_data, gripper_pose):
        #if data is not recent enough, reject
        time1= time.time()
        if data.header.stamp.to_sec() - rospy.get_time() < -.1 or not self.frame_ready:
            return
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
        if self.display_2d:
            self.win.clear_overlay()
            self.win.set_image(img)
        depth = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")
        
        #detect face in 2d, if not found assume face is in previous location
        faces = self.detector(img)
        if len(faces) < 1:
            faces = dlib.dlib.rectangles()
            """
            faces.append(dlib.dlib.rectangle(self.previous_face[0].left() - 10, self.previous_face[0].top(), self.previous_face[0].right()- 10, self.previous_face[0].bottom()))
            faces.append(dlib.dlib.rectangle(self.previous_face[0].left() + 10, self.previous_face[0].top(), self.previous_face[0].right()+ 10, self.previous_face[0].bottom()))
            faces.append(self.previous_face[0])
            """
            faces = self.previous_face
            self.face_detected = False
        else:
            self.previous_face = faces
            self.face_detected = True
        #print img.shape
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
                if self.display_2d:
                    self.win.add_overlay(faces)
                    self.win.add_overlay(shape)
                try:
                    if self.first:
                        self.point_set_index=self.find_best_set(landmarks, depth)
                    point_set, points_ordered = self.retrieve_points(landmarks, depth)
                    
                    #register values for PnPRansac for comparison
                    if self.first:
                        temp_mouth = points_ordered[62]
                        for object_point in points_ordered[17:]:
                            if not np.allclose(object_point, (0.0, 0.0, 0.0)):
                                self.object_points.append([(object_point[1]-temp_mouth[1])*-1.0, (object_point[2]-temp_mouth[2]), (object_point[0]-temp_mouth[0]) * -1.0])
                            else:
                                self.object_points.append([0.0, 0.0, 0.0])
                        self.object_points = np.asarray(self.object_points)
                        self.object_points = self.object_points.astype('float32')
                    #pnp_trans = self.use_pnp_ransac(landmarks, self.cam_matrix)
                    if self.first:
                        self.tf_listnr.waitForTransform("/r_gripper_tool_frame", self.camera_link, rospy.Time(), rospy.Duration(4.0))
                        self.gripper_to_sensor = self.tf_listnr.lookupTransform("/r_gripper_tool_frame", self.camera_link, rospy.Time())#
                        print self.gripper_to_sensor
                        matrix_form = tft.quaternion_matrix(self.gripper_to_sensor[1])
                        for i in xrange(3):
                            matrix_form[i][3] = self.gripper_to_sensor[0][i]
                        self.gripper_to_sensor = matrix_form #
                        #register values for frontal face
                        #find special points in 3D
                        mouth = self.get_3d_pixel(landmarks[62].x, landmarks[62].y, depth)
                        if not self.flipped:
                            second_point = self.get_3d_pixel(int(x + (3 * w/ 4)), int (y + (4 * h / 9)), depth)
                            third_point  = self.get_3d_pixel(int(x + (w / 4)), int (y + (4 * h / 9)), depth)
                        else:
                            second_point = self.get_3d_pixel(int(img.shape[1]- x - 1 - (3 * w/ 4)), int (img.shape[0]- y - 1 - (4 * h / 9)), depth)
                            third_point  = self.get_3d_pixel(int(img.shape[1]- x - 1 - (w / 4)), int (img.shape[0]- y - 1 - (4 * h / 9)), depth)
                        fourth_point = self.get_3d_pixel(landmarks[36].x, landmarks[36].y, depth)
                        fifth_point = self.get_3d_pixel(landmarks[44].x, landmarks[44].y, depth)
                        special_points = [mouth, second_point, third_point]

                        #initialize variables to hold informations
                        for i in xrange(len(special_points)):
                            self.dist.append([])
                            self.half_dist.append([])

                        #find relation to retrieve special points from 3 point sets, and their expected size (of 3d triangle)
                        for points in point_set:
                            for i in xrange(len(special_points)):
                                points_and_point = points + [Point(special_points[i])]
                                self.dist[i].append(self.find_dist(points_and_point))
                                self.half_dist[i].append(self.find_half_dist(points_and_point, depth))

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
                            #print self.get_dist(prev_point, curr_point)

                        orientation = self.pose_to_tuple(pose)[1]
                        self.relation = tft.unit_vector(self.get_quaternion_relation([orientation, tuple(tft.quaternion_from_euler(0, 0, np.math.pi/2))]))
                        orientation = tft.quaternion_multiply(orientation, self.relation)
                        orientation = tft.unit_vector(orientation)
                        transform_matrix = tft.quaternion_matrix(orientation)
                        for i, offset in enumerate(self.pose_to_tuple(pose)[0]):
                            transform_matrix[i][3] = offset
                        transform_matrix = tft.inverse_matrix(transform_matrix)
                        for point in points_ordered:
                            lm_point = tft.translation_matrix(point)
                            lm_point = np.array(np.matrix(transform_matrix)*np.matrix(lm_point))
                            lm_point = tft.translation_from_matrix(lm_point)
                            self.lm_coor.append(lm_point)
                        self.first = False 
                        if self.save_loc is not None:
                            self.save(self.save_loc)
                    else:
                        #retrieve points and make pose
                        pose, pose_points = self.retrieve_special_pose(point_set, depth)
                        time2 = time.time()
                        retrieved_points = self.retrieve_points_from_orientation(pose)
                        #retrieved_points = self.retrieve_points_from_pose(pose_points, depth) 

                        retrieved_landmarks = dlib.dlib.points()
                        for i in xrange(len(landmarks)):
                            prev_point = points_ordered[i]
                            curr_point = (retrieved_points[i].pose.position.x,  retrieved_points[i].pose.position.y,  retrieved_points[i].pose.position.z)                     
                            curr_point_2d = self.get_2d_pixel(curr_point)
                            retrieved_landmarks.append(dlib.dlib.point(int(curr_point_2d[0]), int(curr_point_2d[1])))
                        new_point_set, new_points_ordered = self.retrieve_points(retrieved_landmarks, depth, use_points=False)
                        temp_sets = []
                        """
                        for i, point_set in enumerate(new_point_set):
                            if i is 0:
                                continue
                            temp_set = []
                            for point in point_set:
                                temp_set.append(point.get_tuple())
                            temp_sets.append((self.find_size(temp_set) - self.sizes[i -1]) / self.sizes[i-1])
                        """
                        count = 0
                        for i in xrange(len(landmarks)):
                            curr_point = (retrieved_points[i].pose.position.x,  retrieved_points[i].pose.position.y,  retrieved_points[i].pose.position.z)
                            if self.get_dist(new_points_ordered[i], curr_point) < 0.01:
                                """
                                avg = 0
                                cnt = 0
                                for j, set in enumerate(self.point_set_index[1]):
                                    if i in set:
                                        avg = avg + temp_sets[j-1]
                                        cnt = cnt + 1
                                if cnt is not 0:
                                    avg = avg / cnt
                                else:
                                    print "huh?"
                                if avg < 0.3:
                                    count = count + 1
                                """
                                count = count + 1
                        print count
                        best = count
                        best_pose = pose
                        try:
                            temp_pose = self.make_pose(self.previous_position, orientation=self.previous_orientation)
                            #print "hello"
                        except:
                            temp_pose = pose
                        temp_pose2 = pose
                        pose = temp_pose2
                        position, orientation = self.pose_to_tuple(pose)
                        current_positions = []
                        for pose_point in pose_points:
                            current_positions.append(pose_point.get_tuple())
                        no_jump = True
                        if not self.face_detected:
                            for i in xrange(len(current_positions)):
                                if self.get_dist(current_positions[i], self.current_positions[i]) > 0.05:
                                    no_jump = False
                            if count < 20:
                                no_jump = False
                        if no_jump:
                            self.current_positions = current_positions
                            self.previous_position, self.previous_orientation = position, orientation
                        else:
                            position, orientation = self.previous_position, self.previous_orientation
                        #display frame
                        if self.display_3d:
                            self.br.sendTransform(position, orientation, rospy.Time.now(), "/mouth_position", self.camera_link)
                        
                        #publish
                        #temp_pose = self.make_pose(position, orientation=orientation)
                        temp_pose = tft.quaternion_matrix(tft.unit_vector(orientation))
                        for i in xrange(3):
                            temp_pose[i][3] = position[i]
                        temp_pose = np.array(np.matrix(self.gripper_to_sensor)*np.matrix(temp_pose))
                        ## temp_pose = np.multiply(self.gripper_to_sensor, temp_pose)
                        temp_pose = np.array(np.matrix(base_to_gripper)*np.matrix(temp_pose))
                        ## temp_pose = tft.concatanate_matrices(base_to_gripper, temp_pose)
                        temp_pose = self.make_pose(tft.translation_from_matrix(temp_pose), orientation=tft.quaternion_from_matrix(temp_pose))
                        ## orientation = tft.quaternion_from_matrix(pnp_trans)
                        ## pnp_position = (pnp_trans[0][3], pnp_trans[1][3], pnp_trans[2][3])
                        ## pnp_pose = self.make_pose(pnp_position, orientation=orientation, frame_id=self.camera_link)
                        ## if self.display_3d:
                        ##     self.br.sendTransform(pnp_position, orientation, rospy.Time.now(), "/mouth_position2", self.camera_link)
                        if not np.isnan(temp_pose.pose.position.x) and not np.isnan(temp_pose.pose.orientation.x):
                            try:
                                temp_pose.header.stamp = rospy.Time.now() #gripper_pose.header.stamp
                                temp_pose.header.frame_id = "torso_lift_link"
                                #temp_pose = self.tf_listnr.transformPose("torso_lift_link", temp_pose)
                                self.mouth_calc_pub.publish(temp_pose)
                                
                            except:
                                print "failed"
                        self.mouth_pub.publish(best_pose)
                        ## self.mouth_pub.publish(pnp_pose)
                    print time.time() - time1
                except rospy.ServiceException as exc:
                    print ("serv caused an error " + str(exc))

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

    def retrieve_points(self, landmarks, depth, use_points=True):
        points = []
        curr_points = []
        for curr in self.current_positions:
            curr_points.append(Point(curr))
        if use_points:
            retrieved_previous_points = self.retrieve_points_from_pose(curr_points, depth)
        else:
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
            lm_point = tft.translation_matrix(coor)
            lm_point = np.array(np.matrix(transform)*np.matrix(lm_point))#np.dot(transform, lm_point)
            lm_point = tft.translation_from_matrix(lm_point)
            lm_points.append(self.make_pose(lm_point))
        return lm_points

    def make_point(self, point):
        return Point32(x=point[0], y=point[1], z=point[2])

    # first set is set of points to combine, second set is set of points together. -# = new points, +# = landmarks from detector
    def find_best_set(self, landmarks, depth):
        points3d = []
        best_delaunay = []
        best_delaunay_score = 9999999999
        best_2d_3d = []
        invalid_indices =[]
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
                quaternions.append(tft.unit_vector([orientation.x, orientation.y, orientation.z, orientation.w]))
        return tft.unit_vector(tft.quaternion_multiply(tft.quaternion_inverse(quaternions[0]), quaternions[1]))

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
        if not np.allclose(point, (0.0, 0.0, 0.0)) and not np.isnan(point[0]):
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

    # cx, cy, fx, fy are from Camera_Info of rgb. currently outputs (z, x, y) due to how axis is oriented.
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
    p.add_option("--rgb_mode", dest="rgb_mode", default="bgr8", help="rgb format")
    p.add_option("--flip", dest="flipped", action="store_true", default=False)
    p.add_option("--display_2d", dest="display_2d", action="store_true", default=False)
    p.add_option("--display_3d", dest="display_3d", action="store_true", default=False)
    p.add_option("--save", dest="save")
    p.add_option("--load", dest="load")
    (options, args) = p.parse_args()
    print options.offset
    detector = MouthPoseDetector(options.rgb_camera_link, options.rgb_image,
                                 options.depth_image, options.rgb_info, 
                                 options.depth_info, options.depth_scale, options.offset,
                                 options.display_2d, options.display_3d,
                                 options.flipped, options.rgb_mode, options.save, options.load)
    rospy.spin()
