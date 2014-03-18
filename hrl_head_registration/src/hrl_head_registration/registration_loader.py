#! /usr/bin/python

import copy

import roslib; roslib.load_manifest('hrl_head_registration')
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from tf import TransformBroadcaster, TransformListener

from hrl_head_registration.srv import HeadRegistration, ConfirmRegistration

class RegistrationLoader(object):
    WORLD_FRAME = "odom_combined"
    HEAD_FRAME = "head_frame"
    def __init__(self):
        self.head_reg_pose = None
        self.head_frame_tf = None
        self.head_frame_bcast = TransformBroadcaster()
        self.tfl = TransformListener()
        self.init_reg_srv = rospy.Service("/initialize_registration", HeadRegistration, self.init_reg_cb)
        self.confirm_reg_srv = rospy.Service("/confirm_registration", ConfirmRegistration, self.confirm_reg_cb)
        self.head_registration_r = rospy.ServiceProxy("/head_registration_r", HeadRegistration)
        self.head_registration_l = rospy.ServiceProxy("/head_registration_l", HeadRegistration)
        self.feedback_pub = rospy.Publisher("/feedback", String)
        self.test_pose = rospy.Publisher("/test_head_pose", PoseStamped)

    def publish_feedback(self, msg):
        rospy.loginfo("[%s] %s" % (rospy.get_name(), msg))
        self.feedback_pub.publish(msg)

    def init_reg_cb(self, req):
        # TODO REMOVE THIS FACE SIDE MESS
        self.face_side = rospy.get_param("/face_side", 'r')
        if self.face_side == 'r':
            head_registration = self.head_registration_r
        else:
            head_registration = self.head_registration_l
        print "Head Registration: ", head_registration
        try:
            self.head_reg_pose = head_registration(req.u, req.v).reg_pose
        except rospy.ServiceException as se:
            self.publish_feedback("Registration failed: %s" %se)
            return None
        side = "right" if (self.face_side == 'r') else "left"
        self.publish_feedback("Registered head using %s cheek model, please visually confirm." %side)
        rospy.loginfo('[%s] Head PC frame registered at:\r\n %s' %(rospy.get_name(), self.head_reg_pose))
        self.test_pose.publish(self.head_reg_pose)
        return self.head_reg_pose

    def confirm_reg_cb(self, req):
        if self.head_reg_pose is None:
            raise rospy.ServiceException("Head has not been registered.");
            return False
        hp = copy.copy(self.head_reg_pose)
        now = rospy.Time.now()
        self.tfl.waitForTransform(self.WORLD_FRAME, hp.header.frame_id, now, rospy.Duration(10))
        hp.header.stamp = now
        hp_world = self.tfl.transformPose(self.WORLD_FRAME, hp)
        pos = (hp_world.pose.position.x, hp_world.pose.position.y, hp_world.pose.position.z)
        quat = (hp_world.pose.orientation.x, hp_world.pose.orientation.y,
                hp_world.pose.orientation.z, hp_world.pose.orientation.w)
        self.head_frame_tf = (pos, quat)
        rospy.loginfo("[%s] Head Registration Confirmed" % rospy.get_name())
        return True

if __name__ == "__main__":
    rospy.init_node("registration_loader")
    rl = RegistrationLoader()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if rl.head_frame_tf is not None:
            rl.head_frame_bcast.sendTransform(rl.head_frame_tf[0],
                                              rl.head_frame_tf[1],
                                              rospy.Time.now(),
                                              rl.HEAD_FRAME,
                                              rl.WORLD_FRAME)
        rate.sleep()
