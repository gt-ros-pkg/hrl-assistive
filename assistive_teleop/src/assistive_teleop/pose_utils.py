#!/usr/bin/python
import numpy as np
import math
from copy import deepcopy

import roslib; roslib.load_manifest('tf')
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf import transformations as tft

def pose_relative_trans(pose, x=0., y=0., z=0.):
    """Return a pose translated relative to a given pose."""
    ps = deepcopy(pose)
    M_trans = tft.translation_matrix([x,y,z])
    q_ps = [ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w]
    M_rot = tft.quaternion_matrix(q_ps)
    trans = np.dot(M_rot,M_trans)
    ps.pose.position.x += trans[0][-1]
    ps.pose.position.y += trans[1][-1]
    ps.pose.position.z += trans[2][-1]
    #print ps
    return ps

def pose_relative_rot(pose, r=0., p=0., y=0., degrees=True):
    """Return a pose rotated relative to a given pose."""
    ps = deepcopy(pose) 
    if degrees:
        r = math.radians(r)
        p = math.radians(p)
        y = math.radians(y)
    des_rot_mat = tft.euler_matrix(r,p,y) 
    q_ps = [ps.pose.orientation.x, 
            ps.pose.orientation.y, 
            ps.pose.orientation.z, 
            ps.pose.orientation.w]
    state_rot_mat = tft.quaternion_matrix(q_ps) 
    final_rot_mat = np.dot(state_rot_mat, des_rot_mat) 
    ps.pose.orientation = Quaternion(
                            *tft.quaternion_from_matrix(final_rot_mat))
    return ps

def aim_frame_to(target_pt, point_dir=(1,0,0)):
    goal_dir = np.array([target_pt.x, target_pt.y, target_pt.z])
    goal_norm = np.divide(goal_dir, np.linalg.norm(goal_dir))
    point_norm = np.divide(point_dir, np.linalg.norm(point_dir))
    axis = np.cross(point_norm, goal_norm)
    angle = np.arccos(np.vdot(goal_norm, point_norm))
    return tft.quaternion_about_axis(angle, axis)

def aim_pose_to(ps, pts, point_dir=(1,0,0)):
    if not (ps.header.frame_id.lstrip('/') == pts.header.frame_id.lstrip('/')):
        rospy.logerr("[Pose_Utils.aim_pose_to]: Pose and point must be in same frame: %s, %s"
                    %(ps.header.frame_id, pt2.header.frame_id))
    target_pt = np.array((pts.point.x, pts.point.y, pts.point.z))
    base_pt = np.array((ps.pose.position.x,
                        ps.pose.position.y,
                        ps.pose.position.z)) 
    base_quat = np.array((ps.pose.orientation.x, ps.pose.orientation.y,
                          ps.pose.orientation.z, ps.pose.orientation.w))

    b_to_t_vec = np.array((target_pt[0]-base_pt[0],
                           target_pt[1]-base_pt[1],
                           target_pt[2]-base_pt[2]))
    b_to_t_norm = np.divide(b_to_t_vec, np.linalg.norm(b_to_t_vec))

    point_dir_hom = (point_dir[0], point_dir[1], point_dir[2], 1)
    base_rot_mat = tft.quaternion_matrix(base_quat)
    point_dir_hom = np.dot(point_dir_hom, base_rot_mat.T)
    point_dir = np.array((point_dir_hom[0]/point_dir_hom[3],
                         point_dir_hom[1]/point_dir_hom[3],
                         point_dir_hom[2]/point_dir_hom[3]))
    point_dir_norm = np.divide(point_dir, np.linalg.norm(point_dir))
    axis = np.cross(point_dir_norm, b_to_t_norm)
    angle = np.arccos(np.vdot(point_dir_norm, b_to_t_norm))
    quat = tft.quaternion_about_axis(angle, axis)
    new_quat = tft.quaternion_multiply(quat, base_quat)
    ps.pose.orientation = Quaternion(*new_quat)

def find_approach(pose, standoff=0., axis='x'):
    """Return a PoseStamped pointed down the z-axis of input pose."""
    ps = deepcopy(pose)
    if axis == 'x':
        ps = pose_relative_rot(ps, p=90)
        ps = pose_relative_trans(ps, -standoff)
    return ps
    
def calc_dist(ps1, ps2):
    """ Return the cartesian distance between the points of 2 poses."""
    p1 = ps1.pose.position
    p2 = ps2.pose.position
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

class PoseUtilsTest():
    def __init__(self):
        rospy.Subscriber('pose_in', PoseStamped, self.cb)
        self.recd_pub = rospy.Publisher('pose_utils_test_received', PoseStamped)
        self.trans_pub = rospy.Publisher('pose_utils_test_trans', PoseStamped)
        self.rot_pub = rospy.Publisher('pose_utils_test_rot', PoseStamped)
        self.ps = PoseStamped()
        self.ps.header.frame_id = '/torso_lift_link'
        self.ps.header.stamp = rospy.Time.now()
        self.ps.pose.position.x = 5
        self.ps.pose.position.y = 2
        self.ps.pose.position.z = 3

    def cb(self, ps):
        self.ps.pose.orientation = Quaternion(*tft.random_quaternion())
        print ps
        rospy.sleep(0.5)
        self.recd_pub.publish(ps)
        
        #trans_pose = pose_relative_trans(ps, 0.5, 0.5, 0.2)
        #self.trans_pub.publish(trans_pose)
        #rospy.loginfo("Pose Utils Test: Pose Translated: \n\r %s" %trans_pose)

        ps_rot = pose_relative_rot(ps, 90, 30 , 45)
        self.rot_pub.publish(ps_rot)
        rospy.loginfo("Pose Utils Test: Pose Rotated: \n\r %s" %ps_rot)

if __name__=='__main__':
    rospy.init_node('pose_utils_test')
    put = PoseUtilsTest()
    while not rospy.is_shutdown():
        put.cb(put.ps)
        rospy.sleep(5)
