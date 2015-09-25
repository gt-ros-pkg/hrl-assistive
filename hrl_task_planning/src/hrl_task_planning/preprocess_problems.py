#!?usr/bin/env python

import rospy
from std_msgs.msg import Bool

from hrl_task_planning.srv import PreprocessProblem
import hrl_task_planning.pddl_utils as pddl


class PlanPreprocessor(object):
    def __init__(self, domain):
        self.domain = domain
        try:
            self.const_preds = rospy.get_param('/%s/predicates' % self.domain)
        except KeyError as e:
            rospy.logwarn("[%s] Could not find parameter: %s", rospy.get_name(), e.message)
        try:
            self.default_goal = rospy.get_param('/%s/default_goal' % self.domain)
        except KeyError as e:
            rospy.logwarn("[%s] Could not find parameter: %s", rospy.get_name(), e.message)
        self.service = rospy.Service('/preprocess_problem/%s' % self.domain, PreprocessProblem, self.problem_cb)

    def problem_cb(self, req):
#        rospy.loginfo("[%s] Received Problem to process:\n%s", rospy.get_name(), req.problem)
        req.problem.init.extend(self.const_preds)
        req.problem.goal = req.problem.goal if req.problem.goal else self.default_goal
        req = self.update_request(req)
        return req

    def update_objects(self, req):
        return req

    def update_state(self, req):
        return req


class MoveObjectPreprocessor(PlanPreprocessor):
    def __init__(self, domain):
        super(MoveObjectPreprocessor, self).__init__(domain)
        self.gripper_grasp_state = {'right-gripper': None, 'left-gripper': None}
        self.l_gripper_grasp_state_sub = rospy.Subscriber("/grasping/left_gripper", Bool, self.grasp_state_cb, "left-gripper")
        self.r_gripper_grasp_state_sub = rospy.Subscriber("/grasping/right_gripper", Bool, self.grasp_state_cb, "right-gripper")
        rospy.loginfo("[%s] MOVE_OBJECT Plan Preprocessor Ready", rospy.get_name())

    def grasp_state_cb(self, msg, gripper):
        self.gripper_grasp_state[gripper] = msg.data

    def update_request(self, req):
        # Check for state
        if None in self.gripper_grasp_state.itervalues():
            raise rospy.ServiceError("[%s] Unknown grasp state. Cannot correctly formulate plan.", rospy.get_name())
        # Update initial state predicates
        preds = []
        locations = ["start", "goal", "somewhere", "somewhere-else"]
        known_locations = [rospy.get_param("/%s/%s" % (req.problem.name, loc), None) for loc in locations]
        known_locations = [loc for loc in known_locations if loc is not None]
        preds.extend([pddl.Predicate("KNOWN", [loc]) for loc in known_locations])
        for gripper, grasping in self.gripper_grasp_state.iteritems():
            if grasping:
                preds.append(pddl.Predicate("GRASPING", [gripper, "%s-object" % gripper]))
        req.problem.init.extend(map(str, preds))

        # Update Object list
        # Check to see if objects are already established for this task
        obj_list = rospy.get_param("/%s/objects" % req.problem.name, None)
        if obj_list is None:
            # If objects not established, add them as necessary
            objs = [pddl.Object("target-object", "object)")]  # default to one target object, not in any gripper
            for gripper, grasping in self.gripper_grasp_state.iteritems():
                if grasping:
                    objs.append(pddl.Object("%s-object" % gripper, "object"))
            # Save relevant objects to param server for later calls to find
            obj_list = map(str, objs)
            rospy.set_param("/%s/objects" % req.problem.name, obj_list)
        req.problem.objects.extend(obj_list)
#        rospy.loginfo("[%s] Returning processed problem:\n%s", rospy.get_name(), req.problem)
        return req.problem


def main():
    rospy.init_node('preprocess_problem')
    move_object_preprocessor = MoveObjectPreprocessor('move_object')
    rospy.spin()
