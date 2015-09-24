import rospy
from std_msgs.msg import Bool
import smach

# pylint: disable=W0102

SPA = ["succeeded", "preempted", "aborted"]

def get_action_state(plan_step):
    if plan_step == 'PICK':

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
                                        outcome_cb=outcome_cb
                                        )

        grasp_state_left = WaitForGraspState("/grasping/left_gripper", side="left", outcomes=SPA, output_keys=['grasp_side'])
        grasp_state_right = WaitForGraspState("/grasping/right_gripper", side="right", outcomes=SPA, output_keys=['grasp_side'])
        with concurrence:
            concurrence.add('grasp-left', grasp_state_left)
            concurrence.add('grasp-right', grasp_state_right)
        return ("pick", concurrence)

    elif plan_step == 'PLACE':
        concurrence = smach.Concurrence(outcomes=SPA,
                                        default_outcome='aborted',
                                        input_keys=['grasp_side'],
                                        outcome_map={'succeeded': {'release-right': 'succeeded'},
                                                     'succeeded': {'release-left': 'succeeded'},
                                                     'aborted': {'release-left': 'aborted', 'release-right': 'aborted'},
                                                     'preempted': {'release-left': 'preempted'},
                                                     'preempted': {'release-right': 'preempted'}
                                                     }
                                        )

        release_state_left = WaitForReleaseState("/grasping/left_gripper", side="left", outcomes=SPA, input_keys=['grasp_side'])
        release_state_right = WaitForReleaseState("/grasping/right_gripper", side="right", outcomes=SPA, input_keys=['grasp_side'])
        with concurrence:
            concurrence.add('release-left', release_state_left)
            concurrence.add('release-right', release_state_right)
        return ("place", concurrence)


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
