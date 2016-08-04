import rospy
import threading

import numpy as np
import hrl_lib.circular_buffer as cb
import hrl_lib.quaternion as qt
from tf_conversions import posemath

from geometry_msgs.msg import PoseStamped, Quaternion

class MouthPoseFilter():
    def __init__(self):
        self.mouth_pos  = np.zeros((3, 1))
        self.mouth_quat = np.zeros((4, 1))
        self.pos_buf    = cb.CircularBuffer(4, (3,))
        self.quat_buf   = cb.CircularBuffer(4, (4,))

        self.lock = threading.Lock()

        try:
            self.min=rospy.get_param('/hrl_manipulation_task/mouth_pose_limits/min')
            self.max=rospy.get_param('/hrl_manipulation_task/mouth_pose_limits/max')
        except:
            print "no max or min, assuming no limit"
            self.min = [-999, -999, -999]
            self.max = [999, 999, 999]
        self.sub=rospy.Subscriber('/hrl_manipulation_task/mouth_pose_backpack_unfiltered', PoseStamped, self.callback, queue_size=10)
        self.pub=rospy.Publisher('/hrl_manipulation_task/mouth_pose_backpack', PoseStamped, queue_size=10)
        self.quat_pub=rospy.Publisher('/hrl_manipulation_task/mouth_pose_backpack_filtered_quat', Quaternion, queue_size=10)

    def callback(self, pose):
        self.time = pose.header.stamp
        with self.lock:
            if self.inRange(pose, self.max, self.min):
                cur_p = np.array([pose.pose.position.x, 
                                  pose.pose.position.y, 
                                  pose.pose.position.z])
                cur_q = np.array([pose.pose.orientation.x,
                                  pose.pose.orientation.y,
                                  pose.pose.orientation.z,
                                  pose.pose.orientation.w])
                if len(self.quat_buf) < 1:
                    self.pos_buf.append(cur_p)
                    self.quat_buf.append(cur_q)
                else:
                    first_p = self.pos_buf[-1]
                    first_q = self.quat_buf[-1]
                    
                    if np.dot(cur_q, first_q) < 0.0:
                        cur_q *= -1.0

                    self.pos_buf.append(cur_p)
                    self.quat_buf.append(cur_q)
            self.publish_current()

                
    def publish_current(self):
        position = self.pos_buf.get_array()
        quaternion = self.quat_buf.get_array()

        p = None
        q = None
        if False:
            p = np.mean(positions, axis=0)
            q = qt.quat_avg(quaternion)
        else:
            p = np.median(position, axis=0)
            q = np.median(quaternion, axis=0)
            q = qt.quat_normal(q)


        self.mouth_pos  = p.reshape(3, 1)
        self.mouth_quat = q.reshape(4, 1)
        pose = PoseStamped()
        pose.pose.position.x = p[0]
        pose.pose.position.y = p[1]
        pose.pose.position.z = p[2]

        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        pose.header.stamp    = self.time
        pose.header.frame_id = "torso_lift_link"

        self.pub.publish(pose)
        self.quat_pub.publish(pose.pose.orientation)

    def inRange(self, pose, max, min):
        position = pose.pose.position
        position = (position.x, position.y, position.z)
        for i in xrange(len(position)):
            if position[i] < min[i] or position[i] > max[i]:
                return False
        return True

if __name__ == '__main__':
    rospy.init_node('mouth_pose_filter')
    pose_filter = MouthPoseFilter()
    while not rospy.is_shutdown():
        rospy.spin()
