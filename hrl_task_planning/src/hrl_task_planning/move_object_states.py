import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
import smach

# pylint: disable=W0102

SPA = ["succeeded", "preempted", "aborted"]


def get_action_state(plan_step, domain, problem):
    if plan_step.name == 'PICK':
        def outcome_cb(outcomes):
            if 'aborted' in outcomes.itervalues():
                return 'aborted'
            if 'succeeded' in outcomes.itervalues():
                return 'succeeded'
            return 'preempted'

        concurrence = smach.Concurrence(outcomes=SPA,
                                        default_outcome='aborted',
                                        output_keys=['grasp_side'],
                                        child_termination_cb=lambda so: True,
                                        outcome_cb=outcome_cb)

        grasp_state_left = WaitForGraspState("/grasping/left_gripper", side="left", outcomes=SPA, output_keys=['grasp_side'])
        grasp_state_right = WaitForGraspState("/grasping/right_gripper", side="right", outcomes=SPA, output_keys=['grasp_side'])
        with concurrence:
            concurrence.add('grasp-left', grasp_state_left)
            concurrence.add('grasp-right', grasp_state_right)
        return concurrence

    elif plan_step.name == 'PLACE':
        def outcome_cb(outcomes):
            print "Place cbs:", outcomes
            if 'succeeded' in outcomes.itervalues():
                return'succeeded'
            if all([result == 'aborted' for result in outcomes.itervalues()]):
                return 'aborted'
            return 'preempted'

        concurrence = smach.Concurrence(outcomes=SPA,
                                        default_outcome='aborted',
                                        input_keys=['grasp_side'],
                                        child_termination_cb=lambda so: False,
                                        outcome_cb=outcome_cb)

        release_state_left = WaitForReleaseState("/grasping/left_gripper", side="left", outcomes=SPA, input_keys=['grasp_side'])
        release_state_right = WaitForReleaseState("/grasping/right_gripper", side="right", outcomes=SPA, input_keys=['grasp_side'])
        with concurrence:
            concurrence.add('release-left', release_state_left)
            concurrence.add('release-right', release_state_right)
        return concurrence

    elif plan_step.name == "ID-LOCATION":
        return IDLocationState("id_location", plan_step.args[0])

    elif plan_step.name == "FORGET-LOCATION":
        return ForgetLocationState(plan_step.args[0])


class IDLocationState(smach.State):
    def __init__(self, topic, location_name, outcomes=SPA, input_keys=['problem_name'], output_keys=['location']):
        super(IDLocationState, self).__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.location_name = location_name
        self.running = False
        self.pose = None
        self.sub = rospy.Subscriber(topic, PoseStamped, self.pose_cb)

    def pose_cb(self, ps_msg):
        if self.running:
            self.pose = ps_msg

    @staticmethod
    def _pose_stamped_to_dict(ps_msg):
        return {'header':
                {'seq': ps_msg.header.seq,
                 'stamp': {'secs': ps_msg.header.stamp.secs,
                           'nsecs': ps_msg.header.stamp.nsecs},
                 'frame_id': ps_msg.header.frame_id},
                'pose':
                {'position': {'x': ps_msg.pose.position.x,
                              'y': ps_msg.pose.position.y,
                              'z': ps_msg.pose.position.z},
                 'orientation': {'x': ps_msg.pose.orientation.x,
                                 'y': ps_msg.pose.orientation.y,
                                 'z': ps_msg.pose.orientation.z,
                                 'w': ps_msg.pose.orientation.w}}}

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

    def execute(self, ud):
        print "Running ID Location"
        self.running = True

        try:
            self.pose = rospy.get_param("/%s/%s" % (ud.problem_name, self.location_name))
            ud['location'] = self.pose
            return 'succeeded'
        except KeyError:
            pass

        while not self.preempt_requested() and self.pose is None:
            rospy.sleep(0.05)
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        if self.pose:
            pose_dict = self._pose_stamped_to_dict(self.pose)
            rospy.set_param("/%s/%s" % (ud.problem_name, self.location_name), pose_dict)
            return 'succeeded'


class ForgetLocationState(smach.State):
    def __init__(self, location_name, outcomes=SPA, input_keys=['problem_name'], output_keys=[]):
        super(ForgetLocationState, self).__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.location_name = location_name

    def execute(self, ud):
        print "Running Forget Location (%s)" % self.location_name
        param = "/%s/%s" % (ud.problem_name, self.location_name)
        try:
            rospy.delete_param(param)
        except KeyError:
            print "Forget Location: Cannot delete param (%s), since it doesn't exist." % param
        # Just in case we got preempted
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        return 'succeeded'


class WaitForGraspState(smach.State):
    def __init__(self, topic, side, outcomes=SPA, input_keys=[], output_keys=['grasp_side']):
        super(WaitForGraspState, self).__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.side = side
        self.running = False
        self.grasped = False
        self.sub = rospy.Subscriber(topic, Bool, self.msg_cb)

    def msg_cb(self, msg):
        if self.running and msg.data:
            print "%s Gripper Grasped!" % self.side.capitalize()
            self.grasped = True

    def execute(self, ud):
        print "Running Wait for Grasp %s" % self.side
        self.running = True
        while not self.preempt_requested() and not self.grasped:
            rospy.sleep(0.05)
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"
        if self.grasped:
            ud['grasp_side'] = self.side
            print ud
            return "succeeded"
        return "aborted"


class WaitForReleaseState(smach.State):
    def __init__(self, topic, side, outcomes=SPA, input_keys=['grasp_side'], output_keys=[]):
        super(WaitForReleaseState, self).__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.side = side
        self.running = False
        self.released = False
        self.sub = rospy.Subscriber(topic, Bool, self.msg_cb)

    def msg_cb(self, msg):
        if self.running and not msg.data:
            print "%s Gripper Released!" % self.side
            self.released = True

    def execute(self, ud):
        print ud
        if self.side == ud.grasp_side:
            print "Running Wait for Release %s" % self.side
            self.running = True
            while not self.preempt_requested() and not self.released:
                rospy.sleep(0.05)
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"
            if self.released:
                return "succeeded"
        return "aborted"
