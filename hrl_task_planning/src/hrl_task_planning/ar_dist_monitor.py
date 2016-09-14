#!/usr/bin/env python

import sys
import argparse
import threading
import numpy as np

import rospy
import tf, math
from std_msgs.msg import Bool, Float
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import PoseArray, PoseStamped, Pose, Point, Quaternion

from hrl_task_planning import pddl_utils as pddl
from hrl_msgs.msg import FloatArrayBare
from hrl_task_planning.msg import PDDLState

class BedDistanceTracker(object):
    def __init__(self, domain):
        self.domain = domain
        self.frame_lock = threading.RLock()
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        self.move_back_zone_pose = rospy.Publisher('/move_back_safe_zone/points', PoseArray, queue_size=10, latch=True)
        self.move_back_zone_width = rospy.Publisher('/move_back_safe_zone/width', Float, queue_size=10, latch=True)
        self.move_back_zone_length = rospy.Publisher('/move_back_safe_zone/length', Float, queue_size=10, latch=True)
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

    def ar_distance_check(self, robot_pos, final_pos):
        robot_pos_2d = np.array([robot_pos[0], robot_pos[1]])
        final_pos_2d = np.array([final_pos[0], final_pos[1]])
        model_boundaries_2d = self.generate_model_boundaries()
        traj_line_2d = [final_pos_2d, robot_pos_2d]
        preds = []
        does_intersect = False
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
            if final_pos[0] > 1.0 and (final_pos[1] < 0.4 and final_pos[1] > -0.4):
                zone_boundary = [Point(final_pos[0], 0.4, 0.0),
                                 Point(final_pos[0] + 2.0, 0.4, 0.0),
                                 Point(final_pos[0] + 2.0, -0.4, 0.0),
                                 Point(final_pos[0], -0.4, 0.0)]
                quat = Quaternion(0.0, 0.0, 0.0, 1.0)
                dimensions = [0.8, 2.0]
            elif (final_pos[1] > 0.4):
                zone_boundary = [Point(0.0, final_pos[1], 0.0),
                                 Point(0.0, final_pos[1]+1.5, 0.0),
                                 Point(2.0, final_pos[1]+1.5, 0.0),
                                Point(2.0, final_pos[1], 0.0)]
                quat = Quaternion(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
                dimensions = [1.5, 2.0]
            elif (final_pos[1] < -0.4):
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
        poses.header.stamp = common_time
        ps_msg = PoseStamped()
        ps_msg.header.frame_id = self.out_frame 
        ps_msg.header.stamp = common_time
        for pos in zone_boundary:
            ps_msg.pose.position = pos
            ps_msg.pose.orientation = quat
            try:
                odom_msg = self.tfl.transformPose('/odom_combined', ps_msg)
            except:
                rospy.logerr("[%s] Error transforming goal from base_selection into odom_combined", rospy.get_name())
                return False
            poses.poses.append(Pose(odom_msg.pose.position, odom_msg.pose.orientation))
        self.move_back_zone_pose.publish(poses)
        self.move_back_zone_width(width)
        self.move_back_zone_length(length)
        return True


    def generate_model_boundaries(self):
        #Generate boundaries of the model we are using
        if self.model.upper() == 'AUTOBED':
            top_right = np.array([2.0, -0.35])
            top_left = np.array([2.0, 0.35])
            bottom_left = np.array([0.0, 0.35])
            bottom_right = np.array([0.0, -0.35])
        elif self.model.upper() == 'WHEELCHAIR':
            top_right = np.array([1.0, -0.4])
            top_left = np.array([1.0, 0.4])
            bottom_left = np.array([0.0, 0.4])
            bottom_right = np.array([0.0, -0.4])
        model_boundary = [[top_right, top_left], 
                          [top_left, bottom_left], 
                          [bottom_left, bottom_right],
                          [bottom_right, top_right]]
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
                base_goals = rospy.get_param('/pddl_tasks/%s/base_goals' % self.domain)
            except:
                continue
            self.ar_distance_check(robot_trans[:2], base_goals[:2])
            rate.sleep() 
	
def main():
    rospy.init_node('ar_dist_monitor')
    parser = argparse.ArgumentParser(description="Report when bed is too-close")
    parser.add_argument('--domain', '-d', help="The domain this monitor is updating.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = BedDistanceTracker(args.domain)
    monitor.run()
