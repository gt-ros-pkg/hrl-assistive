#!/usr/bin/env python

from collections import deque
from copy import deepcopy
import math
from threading import Lock, Timer

import rospy
from std_msgs.msg import Int32

# Imports for specific implementations of undo actions -- may move to own files
from pr2_controllers_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped


class UndoManager(object):
    """ A manager for tracking actions and coordinating undo along a history list. """
    def __init__(self, undo_topic='undo'):
        self.actions = {}
        self.command_deque = deque([])
        self.command_subs = []
        self.undo_sub = rospy.Subscriber(undo_topic, Int32, self.undo_cb)

    def register_action(self, action_description):
        """ Register an UndoAction description to monitor commands and include them in the undo deque."""
        self.actions[action_description.name] = action_description
        sub = rospy.Subscriber(action_description.command_topic,
                               action_description.command_topic_type,
                               self.command_cb,
                               action_description.name)
        self.command_subs.append(sub)

    def undo_cb(self, msg):
        """ Handles requests to undo a number of commands in the deque."""
        num = min(msg.data, len(self.command_deque))
        cmds_to_undo = [self.command_deque.pop() for i in range(num)]  # Pop the most recent actions into their own list to undo
        undo_actions = self._get_commands_by_action(cmds_to_undo)  # Grab the oldest state for each action to undo to
        undo_list = []
        for action_cmd_list in undo_actions.itervalues():
            undo_list.append(action_cmd_list[-1])
        for cmd in cmds_to_undo:
            self.undo(cmd)

    def undo(self, command):
        """ Calls the undo function of the appropriate action with the desired state."""
        self.actions[command['action']].undo(command['state'])

    def filter_repeated_commands(self, cmd, timeout):
        """ Condense multiple, rapidly sent serial commands into a single 'command.' """
        pass  # TODO: Implement as necessary

    def is_undo_command(self, msg, action_name):
        """ Check if the received command was recently sent by the undo node itself, so that it can be ignored. """
        for sent_goal in self.actions[action_name].sent_commands.itervalues():
            if msg == sent_goal:
                print "Ignoring undo command for %s" % action_name
                return True
        return False

    def command_cb(self, msg, action_name):
        """ Callback for commands received for any undoable actions which are being tracked. """
        if self.is_undo_command(msg, action_name):
            return  # If goal was sent as an undo msg, don't add it to the list of forward commands
        command = {}
        if (msg.header.stamp and msg.header.stamp != 0):
            time = msg.header.stamp
        else:
            time = rospy.Time.now()
        command['action'] = action_name  # Record which action this command involves
        command['time'] = time  # Record time that command was sent
        command['msg'] = msg  # Record command msg that was sent
        command['state'] = deepcopy(self.actions[action_name].state_msg)  # Record state at time command was sent
        self.command_deque.append(command)  # Append record to deque
        print "Recorded new command for %s" % command['action']
        # TODO: Enforce time-ordering? May or may not be necessary as commands should mostly argive in order

    def _get_commands_by_action(self, cmd_list):
        action_lists = {}
        for command in cmd_list:
            if command['action'] not in action_lists:
                action_lists[command['action']] = [command]
            else:
                action_lists[command['action']].append(command)
        return action_lists


class UndoAction(object):
    """ An consistent API for actions which can be undone through an UndoManager. """
    def __init__(self, name,
                 state_topic, state_topic_type,
                 command_topic, command_topic_type,
                 undo_command_topic=None, undo_command_topic_type=None):
        self.name = name
        self.state_topic = state_topic
        self.state_topic_type = state_topic_type
        self.state_sub = rospy.Subscriber(self.state_topic, self.state_topic_type, self.state_cb)
        self.command_topic = command_topic
        self.command_topic_type = command_topic_type
        self.undo_command_topic = undo_command_topic if undo_command_topic is not None else command_topic
        self.undo_command_topic_type = undo_command_topic_type if undo_command_topic_type is not None else command_topic_type
        self.undo_command_pub = rospy.Publisher(self.undo_command_topic, self.undo_command_topic_type)
        self.sent_commands_lock = Lock()
        self.sent_commands_count = 0
        self.sent_commands = {}
        self.state_msg = None

    def _remove_goal(self, goal_num):
        with self.sent_commands_lock:
            self.sent_commands.pop(goal_num)

    def undo(self, goal_state):
        """ Command the state to return to a given value."""
        goal = self.goal_from_state(goal_state)
        with self.sent_commands_lock:
            self.sent_commands_count += 1
            self.sent_commands[self.sent_commands_count] = goal
            Timer(0.5, self._remove_goal, self.sent_commands_count)
        self.undo_command_pub.publish(goal)

    def state_cb(self, msg):
        """ Record most recent copy of state data. """
        self.state_msg = msg

    def goal_from_state(self, state_msg):
        """ Accepts a msg from the state topic,
            returns a goal for the command topic
            which will return the component to the specified state.
        """
        raise NotImplementedError()


class UndoMoveHead(UndoAction):
    def _get_trajectory_time(self, start, end, vel=math.pi/4):
        """ Get the total duration for a transition from start to end angles with max rotational velocity vel. """
        assert len(start) == len(end), "Start and end joint angle lists are not the same length"
        total_angle = 0
        max_diff = 0
        for si, ei in zip(start, end):
            diff = abs(ei-si)
            total_angle += diff  # Add total angular distance to travel over all joints
            if diff > max_diff:
                max_diff = diff
        total_angle /= 1.414  # Normalize for 2 perpendicular rotations
        return total_angle/vel

    def goal_from_state(self, state_msg):
        traj_point = JointTrajectoryPoint()
        traj_point.positions = state_msg.actual.positions
        traj_point.velocities = [0]*len(state_msg.actual.positions)
        traj_point.accelerations = [0]*len(state_msg.actual.positions)
        dur = self._get_trajectory_time(self.state_msg.actual.positions, traj_point.positions)
        traj_point.time_from_start = rospy.Duration(dur)

        goal_msg = JointTrajectory()
        goal_msg.joint_names = state_msg.joint_names
        goal_msg.points.append(traj_point)
        return goal_msg


class UndoMoveTorso(UndoAction):
    def goal_from_state(self, state_msg):
        traj_point = JointTrajectoryPoint()
        traj_point.positions = state_msg.actual.positions
        traj_point.velocities = [0]*len(state_msg.actual.positions)
        traj_point.accelerations = [0]*len(state_msg.actual.positions)
        traj_point.time_from_start = rospy.Duration(0.1)

        goal_msg = JointTrajectory()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.joint_names = state_msg.joint_names
        goal_msg.points.append(traj_point)
        return goal_msg


class UndoMoveCartMPC(UndoAction):
    def goal_from_state(self, state_msg):
        goal_msg = deepcopy(state_msg)
        goal_msg.header.stamp = rospy.Time.now()
        return goal_msg


class UndoSkill(object):
    """ An consistent API for task-level skills which can be undone through an UndoManager. """
    def __init__(self):
        pass


def main():
    rospy.init_node("undo_manager")

    undo_move_head = UndoMoveHead('move_head',
                                  state_topic='/head_traj_controller/state',
                                  state_topic_type=JointTrajectoryControllerState,
                                  command_topic='/head_traj_controller/command',
                                  command_topic_type=JointTrajectory)
    undo_move_torso = UndoMoveTorso('move_torso',
                                    state_topic='/torso_controller/state',
                                    state_topic_type=JointTrajectoryControllerState,
                                    command_topic='/torso_controller/command',
                                    command_topic_type=JointTrajectory)
    undo_move_cart_mpc_right = UndoMoveCartMPC('right_mpc_cart',
                                               state_topic='/right_arm/haptic_mpc/gripper_pose',
                                               state_topic_type=PoseStamped,
                                               command_topic='/right_arm/haptic_mpc/command_pose',
                                               command_topic_type=PoseStamped)
    undo_move_cart_mpc_left = UndoMoveCartMPC('left_mpc_cart',
                                              state_topic='/left_arm/haptic_mpc/gripper_pose',
                                              state_topic_type=PoseStamped,
                                              command_topic='/left_arm/haptic_mpc/command_pose',
                                              command_topic_type=PoseStamped)
    manager = UndoManager()
    manager.register_action(undo_move_head)
    manager.register_action(undo_move_torso)
    manager.register_action(undo_move_cart_mpc_right)
    manager.register_action(undo_move_cart_mpc_left)
    rospy.spin()
