import rospy
import message_filters
import hrl_lib.quaternion as qt
import hrl_lib.circular_buffer as cb
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
import numpy as np
import threading
import PyKDL
from tf_conversions import posemath
"""
class AngleMeasurer:
    def __init__(self, measured_pose, reference_pose=None):
        self.xyz = ['pitch', 'roll', 'yaw']
        if reference_pose is not None:
            self.ref_pub = []
            for i in xrange(0, 3):
                self.ref_pub.append(rospy.Publisher('/hrl_manipulation_task/mouth/angles/' + self.xyz[i], Float64, queue_size=10))
            partialFunc = partial(self.getAngle, self, 'reference', self.ref_pub)
            self.ref_sub = rospy.Subscriber(reference_pose, PoseStamped, self.getAngle, queue_size=10)
        self.mea_pub = []
        for i in xrange(0, 3):
            self.mea_pub.append(rospy.Publisher('/hrl_manipulation_task/mouth/angles/' + self.xyz[i], Float64, queue_size=10))
        partialFunc = partial(self.getAngle, self, 'measured', self.mea_pub)
        self.sub = rospy.Subscriber(measured_pose, PoseStamped, partialFunc, queue_size=10)
        
    def getAngle(self, name, publisher, data):
        quaternion = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        angles = tft.euler_from_quaternion(
"""

class ErrorMeasurer:
    def __init__(self, measured_pose, reference_pose=None, median=None, static=None, transform=None):
        rospy.init_node('mouth_error_measuring_node')
        self.reference_pose = None
        self.sub = None
        self.reference_offset=None
        self.transform = transform
        self.median=median
        self.static=static
        self.reference_poses=[]
        if self.median is not None:
            self.reference_positions=cb.CircularBuffer(median, (3,))
            self.reference_orientations=cb.CircularBuffer(median, (4,))
        self.first_x = 0
        self.angle = 0
        self.dist  = 0
        self.angle_pub = rospy.Publisher("/hrl_manipulation_task/mouth/orientation_error", Float64, queue_size=10)
        self.dist_pub  = rospy.Publisher("/hrl_manipulation_task/mouth/position_error", Float64, queue_size=10)
        self.ref_pose_pub = rospy.Publisher("/hrl_manipulation_task/mouth/pose_error", PoseStamped, queue_size=10)
        print measured_pose
        if reference_pose is None:
            self.sub = rospy.Subscriber(measured_pose, PoseStamped, self.update_pose, queue_size=10)
        else:
            if static is None:
                measured_sub =message_filters.Subscriber(measured_pose, PoseStamped, queue_size=10)
                reference_sub =  message_filters.Subscriber(reference_pose, PoseStamped, queue_size=10)
                self.sub = message_filters.ApproximateTimeSynchronizer([measured_sub, reference_sub], 10, 0.1)
                self.sub.registerCallback(self.update_pose_relative)
            else:
                self.measured_sub=rospy.Subscriber(measured_pose, PoseStamped, self.update_pose, queue_size=10)
                self.reference_sub=rospy.Subscriber(reference_pose, PoseStamped, self.get_median_reference, queue_size=10)
        self.angles = []
        self.dists = []
                

    def update_pose(self, data):
        if self.static is None:
            self.get_median_reference(data)
        if self.reference_pose is None:
            return
        measured_ori = [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w]
        reference_pose =self.reference_pose
        reference_ori = [reference_pose.pose.orientation.x, reference_pose.pose.orientation.y, reference_pose.pose.orientation.z, reference_pose.pose.orientation.w]
        angle = qt.quat_angle(measured_ori, reference_ori) * 180 / np.math.pi
        self.dist  = (reference_pose.pose.position.x - data.pose.position.x) ** 2 +\
                     (reference_pose.pose.position.y - data.pose.position.y) ** 2 +\
                     (reference_pose.pose.position.z - data.pose.position.z) ** 2
        self.dist  = self.dist**0.5
        self.angles.append(angle)
        self.dists.append(self.dist)
        print "mean, std, min, max of angles"
        print np.mean(self.angles), np.std(self.angles), np.min(self.angles), np.max(self.angles)
        print "mean, std, min, max of distances"
        print np.mean(self.dists), np.std(self.dists), np.min(self.dists), np.max(self.dists)
        self.angle_pub.publish(angle)
        self.dist_pub.publish(self.dist * 100)

    def get_median_reference(self, data):
        if self.reference_pose is None:
            if self.median is None:
                self.reference_pose = data
            else:
                self.reference_poses.append(data)
                self.first_x += 1
                if self.first_x == self.median:
                    reference_positions = []
                    reference_orientations = []
                    for pose in self.reference_poses:
                        reference_positions.append(np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]))
                        curr_q = np.array([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
                        if len(reference_orientations) < 1:
                            reference_orientations.append(curr_q)
                        else:
                            if np.dot(curr_q, reference_orientations[0]) < 0.0:
                                curr_q = curr_q * -1.0
                            reference_orientations.append(curr_q)
                    if False:
                        avg_p = np.mean(reference_positions, axis=0)
                        avg_q = qt.quat_avg(np.array(reference_orientations))
                    else:
                        ref_positions = np.sort(reference_positions, axis=0)
                        ref_orientations = np.sort(reference_orientations, axis=0)
                        avg_p = ref_positions[self.median/2]
                        avg_q = ref_orientations[self.median/2]
                        avg_q = qt.quat_normal(avg_q)
                        avg_q = PyKDL.Rotation.Quaternion(avg_q[0], avg_q[1], avg_q[2], avg_q[3])
                        avg_q = avg_q.GetQuaternion()
                    new_reference_pose = PoseStamped()
                    new_reference_pose.pose.position.x = avg_p[0]
                    new_reference_pose.pose.position.y = avg_p[1]
                    new_reference_pose.pose.position.z = avg_p[2]

                    new_reference_pose.pose.orientation.x = avg_q[0]
                    new_reference_pose.pose.orientation.y = avg_q[1]
                    new_reference_pose.pose.orientation.z = avg_q[2]
                    new_reference_pose.pose.orientation.w = avg_q[3]
                    self.reference_pose = new_reference_pose
                else:
                    return


    def update_pose_relative(self, measured_data, reference_data):
        if self.median is not None:
            reference_positions = self.reference_positions.get_array()
            reference_orientations = self.reference_orientations.get_array()
            if len(reference_positions) < self.median:
                return
            avg_p = np.mean(reference_positions, axis=0)
            avg_q = qt.quat_avg(np.array(reference_orientations))
            new_reference_pose = PoseStamped()
            new_reference_pose.pose.position.x = avg_p[0]
            new_reference_pose.pose.position.y = avg_p[1]
            new_reference_pose.pose.position.z = avg_p[2]
            new_reference_pose.pose.orientation.x = avg_q[0]
            new_reference_pose.pose.orientation.y = avg_q[1]
            new_reference_pose.pose.orientation.z = avg_q[2]
            new_reference_pose.pose.orientation.w = avg_q[3]
            
            self.reference_pose = new_reference_pose
        else:
            if self.transform is not None:
                new_data = PoseStamped()
                ref_frame = posemath.fromMsg(reference_data.pose)
                if self.reference_offset is None:
                    tar_frame = posemath.fromMsg(measured_data.pose)
                    self.reference_offset= ref_frame.Inverse() * tar_frame
                reference_pose = ref_frame * self.reference_offset
                new_data.pose.position.x = reference_pose.p[0]
                new_data.pose.position.y = reference_pose.p[1]
                new_data.pose.position.z = reference_pose.p[2]

                new_data.pose.orientation.x = reference_pose.M.GetQuaternion()[0]
                new_data.pose.orientation.y = reference_pose.M.GetQuaternion()[1]
                new_data.pose.orientation.z = reference_pose.M.GetQuaternion()[2]
                new_data.pose.orientation.w = reference_pose.M.GetQuaternion()[3]
                self.reference_pose = new_data
                self.reference_pose.header.stamp = rospy.Time.now()
                self.reference_pose.header.frame_id = "torso_lift_link"
                self.ref_pose_pub.publish(self.reference_pose)
            else:
                self.reference_pose = reference_data
        self.update_pose(measured_data)

if __name__ == '__main__':
    measured_pose = '/hrl_manipulation_task/mouth_pose_backpack_unfiltered'
    measured_pose_filtered = '/hrl_manipulation_task/mouth_pose_backpack'
    reference_pose = '/ar_track_alvar/head_pose'
    reference_mouth_pose = '/ar_track_alvar/mouth_pose'
    #measurer = ErrorMeasurer(measured_pose_filtered)
    #measurer = ErrorMeasurer(measured_pose_filtered,reference_pose)
    #measurer = ErrorMeasurer(measured_pose, reference_pose)
    #measurer = ErrorMeasurer(measured_pose, reference_pose, transform=True)
    #measurer = ErrorMeasurer(measured_pose_filtered, reference_pose, transform=True)
    #measurer = ErrorMeasurer(measured_pose_filtered, reference_pose, median=50, static=True)
    #measurer = ErrorMeasurer(reference_pose, reference_pose, median=50, static=True)
    measurer = ErrorMeasurer(measured_pose, measured_pose, median=10, static=True)
    #measurer = ErrorMeasurer(reference_pose, median=10)
    #measurer = ErrorMeasurer(reference_mouth_pose, reference_mouth_pose, median=50, static=True)
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rate.sleep()
    
