#! /usr/bin/python

import copy

import roslib; roslib.load_manifest('hrl_head_registration'); roslib.load_manifest('hrl_geom')
import rospy
import rosbag
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Transform, Point, Quaternion
from tf import TransformBroadcaster, TransformListener
import numpy as np
from hrl_lib.transforms import *

from hrl_head_registration.srv import HeadRegistration, ConfirmRegistration
from hrl_geom.pose_converter import PoseConv

class RegistrationLoader(object):
    WORLD_FRAME = "odom_combined"
    HEAD_FRAME = "head_frame"
    def __init__(self):
        self.head_pose = None
        self.head_pc_reg = None
        self.head_frame_tf = None
        self.head_frame_bcast = TransformBroadcaster()
        self.tfl = TransformListener()
        self.init_reg_srv = rospy.Service("/initialize_registration", HeadRegistration, self.init_reg_cb)
        self.confirm_reg_srv = rospy.Service("/confirm_registration", ConfirmRegistration, self.confirm_reg_cb)
        self.head_registration_r = rospy.ServiceProxy("/head_registration_r", HeadRegistration)
        self.head_registration_l = rospy.ServiceProxy("/head_registration_l", HeadRegistration)
        self.feedback_pub = rospy.Publisher("/feedback", String)
        self.test_pose = rospy.Publisher("/test_head_pose", PoseStamped)
        self.reg_dir = rospy.get_param("~registration_dir", "")
        self.subject = rospy.get_param("~subject", None)

    def publish_feedback(self, msg):
        rospy.loginfo("[%s] %s" % (rospy.get_name(), msg))
        self.feedback_pub.publish(msg)

    def init_reg_cb(self, req):
        # TODO REMOVE THIS FACE SIDE MESS
        self.publish_feedback("Performing Head Registration. Please Wait.")
        self.face_side = rospy.get_param("~face_side1", 'r')
        bag_str = self.reg_dir + '/' + '_'.join([self.subject, self.face_side, "head_transform"]) + ".bag"
        rospy.loginfo("[%s] Loading %s" %(rospy.get_name(), bag_str))
        try:
            bag = rosbag.Bag(bag_str, 'r')
            for topic, msg, ts in bag.read_messages():
                head_tf = msg
            assert (head_tf is not None), "Error reading head transform bagfile"
            bag.close()
        except Exception as e:
            self.publish_feedback("Registration failed: Error loading saved registration.")
            rospy.logerr("[%s] Cannot load registration parameters from %s:\r\n%s" %
                         (rospy.get_name(), bag_str, e))
            return (False, PoseStamped())

        if self.face_side == 'r':
            head_registration = self.head_registration_r
        else:
            head_registration = self.head_registration_l
        try:
            rospy.loginfo("[%s] Requesting head registration for %s at pixel (%d, %d)." %(rospy.get_name(),
                                                                                          self.subject,
                                                                                          req.u, req.v))
            self.head_pc_reg = head_registration(req.u, req.v).reg_pose
            if ((self.head_pc_reg.pose.position == Point(0.0, 0.0, 0.0)) and
                (self.head_pc_reg.pose.orientation == Quaternion(0.0, 0.0, 0.0, 1.0))):
               raise rospy.ServiceException("Unable to find a good match.")
               self.head_pc_reg = None
        except rospy.ServiceException as se:
            self.publish_feedback("Registration failed: %s" %se)
            return (False, PoseStamped())

        pc_reg_mat = PoseConv.to_homo_mat(self.head_pc_reg)
        head_tf_mat = PoseConv.to_homo_mat(Transform(head_tf.transform.translation,
                                                     head_tf.transform.rotation))
        self.head_pose = PoseConv.to_pose_stamped_msg(pc_reg_mat**-1 * head_tf_mat)
        self.head_pose.header.frame_id = self.head_pc_reg.header.frame_id
        self.head_pose.header.stamp = self.head_pc_reg.header.stamp

        side = "right" if (self.face_side == 'r') else "left"
        self.publish_feedback("Registered head using %s cheek model, please check and confirm." %side)
#        rospy.loginfo('[%s] Head frame registered at:\r\n %s' %(rospy.get_name(), self.head_pose))
        self.test_pose.publish(self.head_pose)
        return (True, self.head_pose)

    def confirm_reg_cb(self, req):
        if self.head_pose is None:
            raise rospy.ServiceException("Head has not been registered.");
            return False
        try:
            hp = copy.copy(self.head_pose)
            now = rospy.Time.now() + rospy.Duration(0.5)
            self.tfl.waitForTransform(self.WORLD_FRAME, hp.header.frame_id, now, rospy.Duration(10))
            hp.header.stamp = now
            hp_world = self.tfl.transformPose(self.WORLD_FRAME, hp)
            pos = (hp_world.pose.position.x, hp_world.pose.position.y, hp_world.pose.position.z)
            quat = (hp_world.pose.orientation.x, hp_world.pose.orientation.y,
                    hp_world.pose.orientation.z, hp_world.pose.orientation.w)

            #Temp: hack for feeding system
            print "Modifying head frame into upright frame"
            rot = np.matrix([[1 - 2*quat[1]*quat[1] - 2*quat[2]*quat[2],    2*quat[0]*quat[1] - 2*quat[2]*quat[3],      2*quat[0]*quat[2] + 2*quat[1]*quat[3]],
                             [2*quat[0]*quat[1] + 2*quat[2]*quat[3],         1 - 2*quat[0]*quat[0] - 2*quat[2]*quat[2],  2*quat[1]*quat[2] - 2*quat[0]*quat[3]],     
                             [2*quat[0]*quat[2] - 2*quat[1]*quat[3],         2*quat[1]*quat[2] + 2*quat[0]*quat[3],      1 - 2*quat[0]*quat[0] - 2*quat[1]*quat[1]]]) 
            rot[0,2]=rot[1,2]=0.0
            rot[2,2]=1.0
            rot[2,0]=rot[2,1]=0.0

            print rot.shape
            x_norm = np.linalg.norm(rot[:,0])
            rot[0,0] /= x_norm
            rot[1,0] /= x_norm
            y_norm = np.linalg.norm(rot[:,1])
            rot[0,1] /= y_norm
            rot[1,1] /= y_norm
            quat = matrix_to_quaternion(rot)
            print "Completed to modify head frame into upright frame"
                
            self.head_frame_tf = (pos, quat)
            self.publish_feedback("Head registration confirmed.");
            return True
        except Exception as e:
            rospy.logerr("[%s] Error: %s" %(rospy.get_name(), e))
            raise rospy.ServiceException("Error confirming head registration.")

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
