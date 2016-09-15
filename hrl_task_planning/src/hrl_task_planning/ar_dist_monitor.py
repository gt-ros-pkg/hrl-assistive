#!/usr/bin/env python

import sys
import argparse
import threading
import numpy as np

import rospy
import tf, math
from std_msgs.msg import Bool, Float32
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import PoseArray, PoseStamped, Pose, Point, Quaternion
from helper_functions import createBMatrix, Bmat_to_pos_quat

from hrl_task_planning import pddl_utils as pddl
from hrl_msgs.msg import FloatArrayBare
from hrl_task_planning.msg import PDDLState

class BedDistanceTracker(object):
    def __init__(self, domain):
        self.domain = domain
        self.frame_lock = threading.RLock()
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        self.move_back_zone_pose = rospy.Publisher('/move_back_safe_zone/points', PoseArray, queue_size=10, latch=True)
        self.move_back_zone_width = rospy.Publisher('/move_back_safe_zone/width', Float32, queue_size=10, latch=True)
        self.move_back_zone_length = rospy.Publisher('/move_back_safe_zone/length', Float32, queue_size=10, latch=True)
        self.model = None
        self.too_close = False
        self.tfl = tf.TransformListener()

    def line_intersection_check(self, line1, line2):
        #Check if the two lines in R2 intersect
        u0 = line1[0]
        v0 = (line1[1] - line1[0])
        u1 = line2[0]
        v1 = (line2[1] - line2[0])
        A = np.array([[-v0[0], v1[0]],[-v0[1], v1[1]]])
        b = np.array([(u0[0] - u1[0]),(u0[1] - u1[1])])
        if abs(np.linalg.det(A)) <= 1e-10:
            return False
        x = np.linalg.solve(A, b)
        if any(i > 1.0 for i in x) or any(j < 0.0 for j in x): 
            return False
        else: 
            return True

    def shift_to_nearest_edge(self, final_pos):
        #shift point to nearest edge of the model
        dist_top = (self.top_right[0] - final_pos[0])
        dist_bottom = abs(final_pos[0] - self.bottom_right[0])
        dist_left = (self.top_left[1] - final_pos[1])
        dist_right = abs(self.top_right[1] - final_pos[1])
        dist_array = [dist_top, dist_bottom, dist_left, dist_right]
        edge_array = [self.top_right[0], self.bottom_right[0], self.top_left[1], self.top_right[1]]
        #index of the closest edge
        min_ind = dist_array.index(min(dist_array))
        if min_ind < (len(dist_array)/2.0):
            if dist_array[min_ind] >= 0:
                return np.array([edge_array[min_ind], final_pos[1]])
            else:
                return None
        elif min_ind >= (len(dist_array)/2.0):
            if dist_array[min_ind] >= 0:
                return np.array([final_pos[0], edge_array[min_ind]])
            else:
                return None
        else:
            return None

    def ar_distance_check(self, robot_pos, final_pos):
        robot_pos_2d = np.array([robot_pos[0], robot_pos[1]])
        final_pos_2d = np.array([final_pos[0], final_pos[1]])
        model_boundaries_2d = self.generate_model_boundaries()
        traj_line_2d = [final_pos_2d, robot_pos_2d]
        preds = []
        does_intersect = False
        #Check if Base Selected by BS is inside the CSPACE of the model.
        #If so, move that point to the nearest edge and continue
        if ((final_pos[0] < self.top_left[0] and final_pos[0] > self.bottom_left[0]) or
                (final_pos[1] < self.top_left[1] and final_pos[1] > self.top_right[1])):
            final_pos_2d = self.shift_to_nearest_edge(final_pos)
            if final_pos_2d == None:
                rospy.logwarn(" [%s] BS wants final base inside model but dont know where", rospy.get_name())
                does_intersect = True
        else:
            for model_boundary in model_boundaries_2d:
                new_intersection_check = self.line_intersection_check(traj_line_2d, model_boundary)
                does_intersect = does_intersect or new_intersection_check
        if does_intersect:
            preds.append(pddl.Predicate('TOO-CLOSE', [self.model]))
        else:
            preds.append(pddl.Predicate('TOO-CLOSE', [self.model], neg=True))
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.predicates = map(str, preds)
        self.state_pub.publish(state_msg)
        if does_intersect:
            pub_trial = self.publish_better_location(final_pos_2d)

    def create_move_back_zone(self, final_pos):
        #Zone to move into
        if self.model.upper() == 'AUTOBED':
            #Left bottom point first, left top second, right top third,
            #right bottom last.
            zone_boundary = [Point(0.0, 0.8, 0.0),
                            Point(0.0, 1.5, 0.0),
                            Point(2.0, 1.5, 0.0),
                            Point(2.0, 0.8, 0.0)]
            quat = Quaternion(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
            dimensions = [0.7, 2.0] #width, length
        elif self.model.upper() == 'WHEELCHAIR':
            if final_pos[0] > 0.0 and (final_pos[1] < 0.35 and final_pos[1] > -0.35):
                zone_boundary = [Point(final_pos[0], 0.35, 0.0),
                                 Point(final_pos[0] + 2.0, 0.35, 0.0),
                                 Point(final_pos[0] + 2.0, -0.35, 0.0),
                                 Point(final_pos[0], -0.35, 0.0)]
                quat = Quaternion(0.0, 0.0, 0.0, 1.0)
                dimensions = [0.7, 2.0]
            elif (final_pos[1] > 0.35):
                zone_boundary = [Point(0.0, final_pos[1], 0.0),
                                 Point(0.0, final_pos[1]+1.5, 0.0),
                                 Point(2.0, final_pos[1]+1.5, 0.0),
                                Point(2.0, final_pos[1], 0.0)]
                quat = Quaternion(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
                dimensions = [1.5, 2.0]
            elif (final_pos[1] < -0.35):
                zone_boundary = [Point(2.0, final_pos[1], 0.0),
                                 Point(2.0, final_pos[1]-1.5, 0.0),
                                 Point(0.0, final_pos[1]-1.5, 0.0),
                                 Point(0.0, final_pos[1], 0.0)]
                quat = Quaternion(0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
                dimensions = [1.5, 2.0]
            else:
                rospy.logwarn("[%s] Are you sure base selection is not recommending a collision?", rospy.get_name())
                zone_boundary = [Point(2.5, -0.4, 0.0),
                                Point(2.5, -1.9, 0.0),
                                Point(1.5, -1.9, 0.0),
                                Point(1.5, -0.4, 0.0)]
                quat = Quaternion(0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
                dimensions = [1.5, 2.0]
        return zone_boundary, quat, dimensions


    def publish_better_location(self, final_pos):
        # Transform to new frame
        zone_boundary, quat, dimensions  = self.create_move_back_zone(final_pos)
        box_width = dimensions[0]
        box_length = dimensions[1]
        try:
            common_time = self.tfl.getLatestCommonTime(self.out_frame, 'odom_combined')
        except tf.Exception:
            rospy.logwarn("[%s] Could not find common transform time for %s and %s",
                          rospy.get_name(), '(autobed/wheelchair)model_frame', 'odom_combined')
            return False
        poses = PoseArray()
        poses.header.frame_id = 'odom_combined' 
        poses.header.stamp = rospy.Time(0)#common_time
        ps_msg = PoseStamped()
        ps_msg.header.frame_id = self.out_frame 
        ps_msg.header.stamp = rospy.Time(0) #common_time
        for pos in zone_boundary:
            ps_msg.pose.position = pos
            ps_msg.pose.orientation = quat
            #try:
            trans, rot= self.tfl.lookupTransform('/odom_combined', self.out_frame, rospy.Time(0))
            odomBout = createBMatrix(trans, rot)
            outBzonepoint = createBMatrix([pos.x, pos.y, pos.z], [quat.x, quat.y, quat.z, quat.w])
            out_pos, out_quat =Bmat_to_pos_quat(odomBout*outBzonepoint)
            #odom_msg = self.tfl.transformPose('/odom_combined', ps_msg)

            ps = PoseStamped()
            ps.header.frame_id = 'odom_combined'  # markers[i].pose.header.frame_id
            ps.header.stamp = rospy.Time.now()  # markers[i].pose.header.stamp
            ps.pose.position.x = out_pos[0]
            ps.pose.position.y = out_pos[1]
            ps.pose.position.z = out_pos[2]
            #
            ps.pose.orientation.x = out_quat[0]
            ps.pose.orientation.y = out_quat[1]
            ps.pose.orientation.z = out_quat[2]
            ps.pose.orientation.w = out_quat[3]
            #except:
            #rospy.logerr("[%s] Error transforming goal from base_selection into odom_combined", rospy.get_name())
            #return False
            poses.poses.append(Pose(ps.pose.position, ps.pose.orientation))
        self.move_back_zone_pose.publish(poses)
        self.move_back_zone_width.publish(box_width)
        self.move_back_zone_length.publish(box_length)
        return True


    def generate_model_boundaries(self):
        #Generate boundaries of the model we are using
        if self.model.upper() == 'AUTOBED':
            self.top_right = np.array([2.0, -0.35])
            self.top_left = np.array([2.0, 0.35])
            self.bottom_left = np.array([0.0, 0.35])
            self.bottom_right = np.array([0.0, -0.35])
        elif self.model.upper() == 'WHEELCHAIR':
            self.top_right = np.array([2.1, -1.0])
            self.top_left = np.array([2.1, 1.0])
            self.bottom_left = np.array([0.0, 1.0])
            self.bottom_right = np.array([0.0, -1.0])
        model_boundary = [[self.top_right, self.top_left], 
                          [self.top_left, self.bottom_left], 
                          [self.bottom_left, self.bottom_right],
                          [self.bottom_right, self.top_right]]
        return model_boundary


    def run(self):
        rate = rospy.Rate(10.0)
        trans = np.array([0.0, 1.0, 0.0])
        rot = np.array([0.0, 0.0, 0.0, 1.0])
        while not rospy.is_shutdown():
            try:
                self.model = rospy.get_param('/pddl_tasks/%s/model_name' % self.domain, 'AUTOBED')
            except KeyError:
                rospy.logwarn("[%s] Tracking AR Tag, but current model unknown! Cannot update PDDLState", rospy.get_name())
                return
            if self.model.lower() == 'autobed':
                self.out_frame = 'autobed/base_link'
            elif self.model.lower() == 'wheelchair':
                self.out_frame = 'wheelchair/base_link'
            else:
                print 'I do not know what AR tag to look for... Abort!'
                return
            try:
                (robot_trans, robot_rot) = self.tfl.lookupTransform(self.out_frame, 'base_footprint', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            try:
                final_pos = rospy.get_param('/model_B_goal')
            except:
                continue
            print "Robot is at:"
            print robot_trans[:2]
            print "Robot needs to go to:"
            print final_pos[:2]
            self.ar_distance_check(robot_trans[:2], final_pos[:2])
            rate.sleep() 
	
def main():
    rospy.init_node('ar_dist_monitor')
    parser = argparse.ArgumentParser(description="Report when bed is too-close")
    parser.add_argument('--domain', '-d', help="The domain this monitor is updating.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = BedDistanceTracker(args.domain)
    monitor.run()
