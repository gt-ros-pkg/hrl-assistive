#!/usr/local/bin/python

import sys
import os
import dlib
import threading
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations as tft

import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from geometry_msgs.msg import PoseStamped

from pykalman import KalmanFilter

class img_preprocessor:
    def __init__(self, img_topic='/camera/rgb/image_color', rgb_info_topic='/camera/rgb/camera_info',
                 depth_topic='/camera/depth_registered/image_raw', depth_info_topic='/camera/depth_registered/camera_info', use_default_model=False):
        #Threading locks
        self.img_lock   = threading.RLock()
        self.depth_lock = threading.RLock()
        self.info_lock  = threading.RLock()

        #data holder
        self.img   = None
        self.depth = None
        self.info  = [None, None]
        self.time  = 0
        self.old_time = 0

        #depth scale
        self.scale =0.01

        #landmarks to use for pnp
        self.landmark_indices = xrange(17,60)#[33, 8, 36, 45, 48, 54]
        self.model = None

        # Kalman filter
        init_state = np.array([1, 0])
        init_covariance = 1.0e-3*np.eye(2)
        transition_cov = 1.0e-4*np.eye(2)
        observation_cov = 1.0e-1*np.eye(2)
        transition_mat = np.array([[1, 1], [0, 1]])
        observation_mat = np.eye(2)
        self.kf = KalmanFilter(transition_matrices = transition_mat,
                            observation_matrices = observation_mat,
                            initial_state_mean = init_state,
                            initial_state_covariance = init_covariance,
                            transition_covariance = transition_cov, 
                            observation_covariance = observation_cov)
        self.prev_state = init_state
        self.prev_ratio = 1
        self.prev_covariance = init_covariance
        self.prev_vel = 0

        #image tools
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.expanduser('~') + '/Desktop/shape_predictor_68_face_landmarks.dat')

        #Bridge
        self.bridge = CvBridge()

        #Subscriber
        self.img_sub      = rospy.Subscriber(img_topic, Image, self.img_cb, queue_size=1)
        self.depth_sub    = rospy.Subscriber(depth_topic, Image, self.depth_cb, queue_size=1)
        self.cam_info_sub = []
        self.cam_info_sub.append(rospy.Subscriber(rgb_info_topic, CameraInfo, self.info_cb, 0))
        self.cam_info_sub.append(rospy.Subscriber(depth_info_topic, CameraInfo, self.info_cb, 1))

        #Publisher
        self.pose_pub   = rospy.Publisher("/gesture_control/head_pose", PoseStamped, queue_size=10)
        self.features_pub = rospy.Publisher("/gesture_control/features", String, queue_size=10)

        self.initialized = False
        self.initialize_frames()
        self.initialize(use_default_model)

        self.run()


    def img_cb(self, data):
        with self.img_lock:
            self.img = self.bridge.imgmsg_to_cv2(data, 'rgb8')
            self.time = data.header.stamp
            diff = rospy.Time.now() - data.header.stamp
            #print diff.secs, ".", diff.nsecs

    def depth_cb(self, data):
        with self.img_lock:
            self.depth = self.bridge.imgmsg_to_cv2(data, 'passthrough')

    def info_cb(self, data, n):
        with self.info_lock:
            self.info[n] = data

    def initialize_frames(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            info = None
            with self.info_lock:
                if None not in self.info:
                    self.cam_info_sub[0].unregister()
                    self.cam_info_sub[1].unregister()
                    info = self.info
            if info is not None:
                self.rgb_f   = (info[0].P[0], info[0].P[5])
                self.rgb_c   = (info[0].P[2], info[0].P[6])
                self.depth_f = (info[1].P[0], info[1].P[5])
                self.depth_c = (info[1].P[2], info[1].P[6])
                self.cam_matrix =  np.asarray(info[0].K).reshape((3,3)).astype('float32')
                return
            print "initializing frames"
            rate.sleep()

    def initialize(self, use_default_model, n=10):
        rate = rospy.Rate(20)
        cnt = 0
        if use_default_model:
            self.model = np.array([(0.0, 0.0, 0.0), 
                                   (0.0, -330.0, -65.0),
                                   (-225.0, 170.0, -135.0),
                                   (225.0, 170.0, -135.0),
                                   (-150.0, -150.0, -125.0),
                                   (150.0, -150.0, -125.0)])
        else:
            while (not rospy.is_shutdown()):
                model = []
                for i in xrange(len(self.landmark_indices)):
                    model.append((0.0, 0.0,0.0))
                self.model = np.array(model)
                
                with self.img_lock:
                    if self.old_time != self.time:
                        img = self.img.astype('uint8')
                        self.old_time = self.time
                        time = self.old_time
                    else:
                        img = None
                with self.depth_lock:
                    depth = self.depth.astype('float32')
                    #print depth[100]
                if img is not None and depth is not None:
                    faces = self.detector(img)
                    if len(faces) == 1:
                        shape = self.predictor(img, faces[0])
                        landmarks = shape.parts()
                        points = self.get_landmarks_points(landmarks, depth)
                        if points is not None:
                            self.model = self.model + np.array(points)
                            cnt = cnt + 1
                        print "making model ", cnt, "/", n
                    if cnt == n:
                        self.model = self.model / float(n)
                        print self.model
                        temp = self.model[0].copy().copy()
                        for i in xrange(len(self.model)):
                            print self.model[i], temp, self.model[i]- temp
                            self.model[i] = self.model[i]-temp
                        print self.model
                        self.initialized = True
                        return
                rate.sleep()

    def get_landmarks_points(self, landmarks, depth):
        points = []
        for index in self.landmark_indices:
            point = self.convert_2d_3d(landmarks[index], depth)
            if np.allclose(point, (0.0, 0.0, 0.0)):
                break
            else:
                points.append(point)
        if len(points) == len(self.landmark_indices):
            return points
        return None
        

    def convert_2d_3d(self, landmark, depth):
        x, y = landmark.x, landmark.y
        if depth[y][x] is None:
            return (0.0, 0.0, 0.0)
        z_metric = depth[y][x] * self.scale
        x_metric = z_metric * (x - self.rgb_c[0]) * (1.0 / self.rgb_f[0])
        y_metric = z_metric * (y - self.rgb_c[1]) * (1.0 / self.rgb_f[1])
        return (x_metric, y_metric, z_metric)

    def run(self):
        dist_coeff= np.zeros((4,1))
        rate = rospy.Rate(10)
        if not self.initialized:
            rate.sleep()
        while not rospy.is_shutdown():
            with self.img_lock:
                if self.old_time != self.time:
                    img = self.img.astype('uint8')
                    self.old_time = self.time
                    time = self.old_time
                else:
                    img = None
            if img is not None:
                faces = self.detector(img)
                if len(faces) == 1:
                    shape = self.predictor(img, faces[0])
                    landmarks = shape.parts()
                    marks = []
                    for index in self.landmark_indices:
                        marks.append([float(landmarks[index].x),float(landmarks[index].y)]) 
                    r_diff = shape.part(67).y - shape.part(61).y
                    c_diff = shape.part(66).y - shape.part(62).y
                    l_diff = shape.part(65).y - shape.part(63).y
                    total = r_diff + c_diff + l_diff
                    marks = np.array(marks).astype('float32')
                    pose, p, q = self.pnp_ransac(marks,dist_coeff)
                    euler = tft.euler_from_quaternion(q)
                    rot_observed = euler[1] # ratio observed from image info
                    velocity_observed = euler[1] - self.prev_state[0] # observed curr ratio - filtered prev ratio
                    (next_state, next_covariance) = self.kf.filter_update(self.prev_state, self.prev_covariance, np.array([rot_observed, velocity_observed]))
                    rot_kalman = next_state[0] # filtered ratio
                    self.prev_state = next_state # update previous state for next iteration
                    self.prev_covariance = next_covariance
                    total = total * p[2]
                    features = str([time.to_time(), rot_kalman, total])
                    print features, euler[1]
                    self.features_pub.publish(features)
                    self.pose_pub.publish(pose)
                    
            rate.sleep()

    def pnp_ransac(self, marks, dist_coeff):
        rvec, tvec, inliers = cv2.solvePnPRansac(np.array(self.model).astype('float32'),
                                                 marks, self.cam_matrix, dist_coeff)
        rot = cv2.Rodrigues(rvec)[0].tolist()
        tvec = tvec.tolist()
        rot[0].append(tvec[0][0])
        rot[1].append(tvec[1][0])
        rot[2].append(tvec[2][0])
        rot.append([0.0, 0.0, 0.0, 1.0])
        q = tft.quaternion_from_matrix(np.asarray(rot))
        p = tft.translation_from_matrix(np.asarray(rot))
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = '/camera_rgb_optical_frame'
        pose.pose.position.x = p[0]
        pose.pose.position.y = p[1]
        pose.pose.position.z = p[2]
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose, p, q

def main():
    rospy.init_node('face_img_preprocess')
    img_preprocessor()

if __name__ == '__main__':
    main()
