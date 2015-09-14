import rospy
from std_msgs.msg import String, Bool

from hrl_task_planning.msg import PDDLProblem, PDDLSolution

from pddl_utils import PDDLPredicate, PDDLObject, PDDLPlanStep


class ManualMoveObjectManager(object):
    task_name = "move-object"

    def __init__(self):
        self.task_req_sub = rospy.Subscriber("task_planning/request", String, self.req_cb)
        self.task_problem_pub = rospy.Publisher("/task_planner/problem", PDDLProblem)
        self.task_state = {'empty': []}
        self.problem_count = 0
        self.task_solution_sub = rospy.Subscriber("/task_planner/solution", PDDLSolution, self.solution_cb)
        self.l_gripper_grasp_state_sub = rospy.Subscriber("/grasping/left_gripper", Bool, self.grasp_state_cb, "left-gripper")
        self.r_gripper_grasp_state_sub = rospy.Subscriber("/grasping/right_gripper", Bool, self.grasp_state_cb, "right-gripper")
        rospy.loginfo("[%s] Ready" % rospy.get_name())

    def grasp_state_cb(self, msg, gripper):
        already_empty_list = [i for i, list_ in enumerate(self.task_state['empty']) if gripper == list_[0]]
        if not msg.data:
            if not already_empty_list:  # (Hand is empty, not known to be so)
                self.task_state["empty"].append([gripper])
        else:
            if already_empty_list:  # (Hand full, known to be empty)
                for item in already_empty_list:
                    self.task_state["empty"].pop(item)

    def solution_cb(self, sol):
        self.solutions[sol.problem] = [PDDLPlanStep.from_string(act) for act in sol.actions]
        rospy.loginfo("[%s] Received Plan:\n %s" % (rospy.get_name(), '\n'.joing(map(str, self.solutions[sol.problem]))))

    def req_cb(self, req):
        if req.data != self.task_name:
            return
        problem = PDDLProblem()
        self.problem_count += 1
        problem.name = "manual-move-object-"+str(self.problem_count)
        problem.objects = [str(PDDLObject("object-to-move", "object"))]
        problem.init = []
        for act, arg_sets in self.task_state.iteritems():
            for args in arg_sets:
                problem.init.append(str(PDDLPredicate(act, args)))
        problem.goal = []
        problem.goal.append(str(PDDLPredicate('at', ["object-to-move", "goal"])))
        self.task_problem_pub.publish(problem)


def main():
    rospy.init_node("move_object_task_manager")
    manager = ManualMoveObjectManager()
    rospy.spin()
