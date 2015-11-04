import math
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Quaternion
from tf import TransformListener
import smach

# pylint: disable=W0102

SPA = ["succeeded", "preempted", "aborted"]


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


def get_action_state(plan_step, domain, problem):
    if plan_step.name == 'GRAB':
        side = 'right' if 'RIGHT' in problem.upper() else 'left'
        return WaitForGraspState("/grasping/%s_gripper" % side, side=side)

    elif plan_step.name == 'RELEASE':
        side = 'right' if 'RIGHT' in problem.upper() else 'left'
        return WaitForReleaseState("/grasping/%s_gripper" % side, side="right")

    elif plan_step.name == "ID-LOCATION":
        return IDLocationState("id_location", plan_step.args[0], problem)

    elif plan_step.name == "FORGET-LOCATION":
        return ForgetLocationState(plan_step.args[0], problem)

    elif plan_step.name == 'MOVE-ARM':
        side = 'right' if 'RIGHT' in problem.upper() else 'left'
        return MoveArmState(plan_step.args[1], side, problem)


class MoveArmState(smach.State):
    def __init__(self, location_name, side, problem, outcomes=SPA, input_keys=[], output_keys=[]):
        super(MoveArmState, self).__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.location_name = location_name
        self.side = side
        self.problem = problem
        self.current_pose = None
        self.tfl = TransformListener()
        self.mpc_pub = rospy.Publisher("/%s_arm/haptic_mpc/goal_pose" % self.side, PoseStamped)
        self.state_sub = rospy.Subscriber("/%s_arm/haptic_mpc/gripper_pose" % self.side, PoseStamped, self.pose_cb)

    def pose_cb(self, pose_msg):
        self.current_pose = pose_msg

    @staticmethod
    def is_near(current_pose, goal_pose, threshold=0.1):
        dx = current_pose.pose.position.x - goal_pose.pose.position.x
        dy = current_pose.pose.position.y - goal_pose.pose.position.y
        dz = current_pose.pose.position.z - goal_pose.pose.position.z
        dist = math.sqrt((dx*dx) + (dy*dy) + (dz*dz))
        return dist < threshold

    def execute(self, ud):
        self.running = True
        try:
            goal_pose_dict = rospy.get_param('/%s/%s' % (self.problem, self.location_name))
            goal_pose = _dict_to_pose_stamped(goal_pose_dict)
        except KeyError:
            print "Move Arm Cannot find location %s on parameter server" % self.location_name
            return 'aborted'
        # Correct goal pose for checking
        if self.current_pose.header.frame_id != goal_pose.header.frame_id:
            try:
                now = rospy.Time.now()
                goal_pose.header.stamp = now
                self.tfl.waitForTransform(self.current_pose.header.frame_id, goal_pose.header.frame_id, now, rospy.Duration(10))
                goal_pose = self.tfl.transformPose(self.current_pose.header.frame_id, goal_pose)
            except:
                print "TF ERROR"
                return 'aborted'
        if not self.is_near(self.current_pose, goal_pose, threshold=0.1):
            goal_pose.pose.position.z += 0.1
            goal_pose.pose.orientation = Quaternion(0.0, 0.0, 0.38, 0.925)
            self.mpc_pub.publish(goal_pose)

        # Try to get into position autonomously
        # Wait to get into position one way or another...
        while not self.is_near(self.current_pose, goal_pose, threshold=0.1):
            if self.preempt_requested():
                self.service_preempt()
                return 'preempted'
            rospy.sleep(0.1)
        return 'succeeded'


class IDLocationState(smach.State):
    def __init__(self, topic, location_name, problem, outcomes=SPA, input_keys=[], output_keys=[]):
        super(IDLocationState, self).__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.location_name = location_name
        self.problem = problem
        self.running = False
        self.pose = None
        self.sub = rospy.Subscriber(topic, PoseStamped, self.pose_cb)

    def pose_cb(self, ps_msg):
        if self.running:
            self.pose = ps_msg

    def execute(self, ud):
        print "Running ID Location for %s" % self.location_name
        self.running = True
        try:
            self.pose = rospy.get_param("/%s/%s" % (self.problem, self.location_name))
            return 'succeeded'
        except KeyError:
            pass

        while not self.preempt_requested() and self.pose is None:
            rospy.sleep(0.05)
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        if self.pose:
            pose_dict = _pose_stamped_to_dict(self.pose)
            rospy.set_param("/%s/%s" % (self.problem, self.location_name), pose_dict)
            return 'succeeded'


class ForgetLocationState(smach.State):
    def __init__(self, location_name, problem, outcomes=SPA, input_keys=[], output_keys=[]):
        super(ForgetLocationState, self).__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.location_name = location_name
        self.problem = problem

    def execute(self, ud):
        print "Running Forget Location (%s)" % self.location_name
        param = "/%s/%s" % (self.problem, self.location_name)
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
    def __init__(self, topic, side, outcomes=SPA, input_keys=[], output_keys=[]):
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
            return "succeeded"
        return "aborted"


class WaitForReleaseState(smach.State):
    def __init__(self, topic, side, outcomes=SPA, output_keys=[], input_keys=[]):
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
        print "Running Wait for Release %s" % self.side
        self.running = True
        while not self.preempt_requested() and not self.released:
            rospy.sleep(0.05)
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"
        if self.released:
            return "succeeded"
