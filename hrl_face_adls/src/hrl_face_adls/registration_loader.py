#! /usr/bin/python

import sys
import numpy as np

import roslib
roslib.load_manifest('hrl_face_adls')
import rospy
import rosbag
from std_msgs.msg import String
from geometry_msgs.msg import Transform, Pose

from hrl_ellipsoidal_control.msg import EllipsoidParams
from hrl_face_adls.srv import InitializeRegistration, InitializeRegistrationResponse
from hrl_face_adls.srv import RequestRegistration, RequestRegistrationResponse
from hrl_head_tracking.srv import HeadRegistration
from hrl_generic_arms.pose_converter import PoseConverter

class RegistrationLoader(object):
    def __init__(self):
        self.head_reg_tf = None
        self.init_reg_srv = rospy.Service("/initialize_registration", InitializeRegistration, 
                                          self.init_reg_cb)
        self.req_reg_srv = rospy.Service("/request_registration", RequestRegistration, 
                                          self.req_reg_cb)
        self.head_registration_r = rospy.ServiceProxy("/head_registration_r", HeadRegistration) # TODO
        self.head_registration_l = rospy.ServiceProxy("/head_registration_l", HeadRegistration) # TODO
#self.ell_params_pub = rospy.Publisher("/ellipsoid_params", EllipsoidParams, latched=True)
        self.feedback_pub = rospy.Publisher("/feedback", String)

    def publish_feedback(self, msg):
        rospy.loginfo("[registration_loader] %s" % msg)
        self.feedback_pub.publish(msg)

    def init_reg_cb(self, req):
        # TODO REMOVE THIS SHAVING SIDE MESS
        self.shaving_side = rospy.get_param("/shaving_side", 'r')
        if self.shaving_side == 'r':
            head_registration = self.head_registration_r
        else:
            head_registration = self.head_registration_l
        # TODO

        try:
            self.head_reg_tf = head_registration(req.u, req.v).tf_reg
        except:
            self.publish_feedback("Registration failed.")
            return None

        if self.shaving_side == 'r':
            self.publish_feedback("Registered head using right cheek model, please visually confirm.")
        else:
            self.publish_feedback("Registered head using left cheek model, please visually confirm.")
        return InitializeRegistrationResponse()

    def req_reg_cb(self, req):
        reg_e_params = EllipsoidParams()
        if self.head_reg_tf is None:
            rospy.logwarn("[registration_loader] Head registration not loaded yet.")
            return RequestRegistrationResponse(False, reg_e_params)
        reg_prefix = rospy.get_param("~registration_prefix", "")
        registration_files = rospy.get_param("~registration_files", None)
        if req.mode not in registration_files:
            rospy.logerr("[registration_loader] Mode not in registration_files parameters")
            return RequestRegistrationResponse(False, reg_e_params)

        try:
            bag = rosbag.Bag(reg_prefix + registration_files[req.mode][req.side], 'r')
            e_params = None
            for topic, msg, ts in bag.read_messages():
                e_params = msg
            assert e_params is not None
            bag.close()
        except:
            rospy.logerr("[registration_loader] Cannot load registration parameters from %s" %
                         registration_files[req.mode][req.side])
            return RequestRegistrationResponse(False, reg_e_params)

        head_reg_mat = PoseConverter.to_homo_mat(self.head_reg_tf)
        ell_reg = PoseConverter.to_homo_mat(Transform(e_params.e_frame.transform.translation,
                                                      e_params.e_frame.transform.rotation))
        reg_e_params.e_frame = PoseConverter.to_tf_stamped_msg(head_reg_mat**-1 * ell_reg)
        reg_e_params.e_frame.header.frame_id = self.head_reg_tf.header.frame_id
        reg_e_params.height = e_params.height
        reg_e_params.E = e_params.E
#self.ell_params_pub.publish(reg_e_params)
        return RequestRegistrationResponse(True, reg_e_params)


def main():
    rospy.init_node("registration_loader")
    rl = RegistrationLoader()
    rospy.spin()
        
if __name__ == "__main__":
    main()
