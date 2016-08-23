#!/usr/bin/env python

import sys
import argparse
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import tf
from tf import transformations as tft

from hrl_task_planning.pddl_utils import Predicate
from hrl_task_planning.msg import PDDLState


class BasePositionMonitor(object):
    def __init__(self, domain, predicate, args, distance_threshold=0.1, rotation_threshold=0.15):
        self.domain = domain
        self.predicate = predicate
        self.args = args
        self.dist_thresh = distance_threshold
        self.ang_thresh = rotation_threshold
        self.params = ["/pddl_tasks/%s/%s/%s" % (self.domain, self.predicate, arg) for arg in self.args]
        self.base_pose = PoseStamped()
        self.base_pose.header.frame_id = 'base_link'
        self.base_pose.pose.position = Point(0, 0, 0)
        self.base_pose.pose.orientation = Quaternion(0, 0, 0, 1)
        self.base_goals = {}
        self.at_goals = []
        for arg in self.args:
            self.base_goals[arg] = None
        self.tfl = tf.TransformListener()
        self.pddl_update_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=6, latch=True)
        rospy.loginfo("[%s] Base Position Monitor Ready.", rospy.get_name())

    def at_goal(self, goal):
        # Check that goal exists
        if goal is None:
            return False

        # Transform base pose into goal frame
        try:
            matchtime = self.tfl.getLatestCommonTime(self.base_pose.header.frame_id, goal.header.frame_id)
        except tf.Exception:
            rospy.logwarn("[%s] Could not find common transform time for %s and %s",
                          rospy.get_name(), self.base_pose.header.frame_id, goal.header.frame_id)
            return False
        self.base_pose.header.stamp = matchtime
        base = self.tfl.transformPose(goal.header.frame_id, self.base_pose)

        # Check distance threshold
        goal_xy = np.array([goal.pose.position.x, goal.pose.position.y])
        base_xy = np.array([base.pose.position.x, base.pose.position.y])
        dist = np.linalg.norm(goal_xy - base_xy)
        if dist > self.dist_thresh:
            return False

        # Check angle threshold
        goal_quat = (goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w)
        goal_theta = tft.euler_from_quaternion(goal_quat)[2]  # Convert to Euler, get Z component
        base_quat = (base.pose.orientation.x, base.pose.orientation.y, base.pose.orientation.z, base.pose.orientation.w)
        base_theta = tft.euler_from_quaternion(base_quat)[2]  # Convert to Euler, get Z component
        if abs(base_theta - goal_theta) > self.ang_thresh:
            return False

        # Passed all tests, must be at the goal...
        return True

    def update_goals_from_params(self):
        for arg in self.args:
            param = "/pddl_tasks/%s/%s/%s" % (self.domain, self.predicate, arg)
            try:
                pose_dict = rospy.get_param(param)
                pose_msg = self._dict_to_pose_stamped(pose_dict)
                if pose_msg.header.frame_id not in '/odom_combined /map /world':
                    rospy.logwarn("[%s] Loaded Goal for %s in frame %s. \
                                  Goals should be in a world frame \
                                  (map, world, odom_combined, etc.)",
                                  rospy.get_name(), arg, pose_msg.header.frame_id)
                self.base_goals[arg] = pose_msg
            except KeyError:
                self.base_goals[arg] = None

    def check_goals(self):
        for arg, goal in self.base_goals.iteritems():
            if arg in self.at_goals:
                if not self.at_goal(goal):
                    self.at_goals.remove(arg)
                    pred = Predicate('AT', [arg.upper()], neg=True)
                    state = PDDLState()
                    state.domain = self.domain
                    state.predicates = [str(pred)]
                    self.pddl_update_pub.publish(state)
            else:
                if self.at_goal(goal):
                    self.at_goals.append(arg)
                    pred = Predicate('AT', [arg.upper()])
                    state = PDDLState()
                    state.domain = self.domain
                    state.predicates = [str(pred)]
                    self.pddl_update_pub.publish(state)

    def run(self, rate):
        r = rospy.Rate(rate)
        while not rospy.is_shutdown():
            self.update_goals_from_params()
            self.check_goals()
            r.sleep()

    @staticmethod
    def _dict_to_pose_stamped(ps_dict):
        ps = PoseStamped()
        ps.header.seq = ps_dict['header']['seq']
        ps.header.stamp.secs = ps_dict['header']['stamp']['secs']
        ps.header.stamp.nsecs = ps_dict['header']['stamp']['nsecs']
        ps.header.frame_id = ps_dict['header']['frame_id']
        ps.pose.position.x = ps_dict['pose']['position']['x']
        ps.pose.position.y = ps_dict['pose']['position']['y']
        ps.pose.position.z = ps_dict['pose']['position']['z']
        ps.pose.orientation.x = ps_dict['pose']['orientation']['x']
        ps.pose.orientation.y = ps_dict['pose']['orientation']['y']
        ps.pose.orientation.z = ps_dict['pose']['orientation']['z']
        ps.pose.orientation.w = ps_dict['pose']['orientation']['w']
        return ps


def main():
    parser = argparse.ArgumentParser(description="Monitor parameters representing changes in a pddl domain state.")
    parser.add_argument('domain', help="The domain for which the parameter will be monitored.")
    parser.add_argument('predicate', help="The name of the predicate being monitored.")
    parser.add_argument('--args', '-a', nargs="*", default=[], help="The possible arguments of the predicate to be monitored.")
    parser.add_argument('--distance_threshold', '-d', default=0.05, help="How close must the robot be to the goal to be 'AT' a location? (meters)")
    parser.add_argument('--rotation_threshold', '-r', default=0.15, help="How closely aligned must the robot be to the goal to be 'AT' a location? (radians)")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node('%s_%s_base_monitor' % (args.domain, args.predicate))
    monitor = BasePositionMonitor(args.domain, args.predicate, args.args, args.distance_threshold, args.rotation_threshold)
    monitor.run(5)
