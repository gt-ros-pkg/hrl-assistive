import rospy
import threading
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Point32, Quaternion, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String

class fPlotter:
    
    def __init__(self):
        self.ft_sub     = rospy.Subscriber('/ft/l_gripper_motor', WrenchStamped, self.ft_callback)
        """
        self.status_sub = rospy.Subscriber('/manipulation_task/proceed', String, self.status_callback) 
        self.stop_motion_pub = rospy.Publisher('/manipulation_task/InterruptAction', String, queue_size = 10)
        self.emergency_pub = rospy.Publisher('/manipulation_task/emergency', String, queue_size = 10)
        """

        self.ft_lock = threading.RLock()
        self.curr_time = None
        self.init_time = None
        self.force = None
        self.prev_f = 0
        self.prev_time = 0
        plt.ion()
        self.run()


    def ft_callback(self, data):
        with self.ft_lock:
            if self.init_time is None:
                self.init_time = data.header.stamp.to_sec()
            self.curr_time = data.header.stamp.to_sec()
            self.force = data.wrench.force
            self.torque = data.wrench.torque

    def status_callback(self, data):
        if data.data == "Set: Wiping 2, Wiping 3, Wipe":
            self.detect_stop = True
        else:
            self.detect_stop = False
        if data.data == "Set: Wiping 3, Wipe, Retract":
            self.wiping = True
        else:
            if self.wiping and self.plot_en:
                self.wipe_finished = True
            self.wiping = False

    def run(self):
        while not rospy.is_shutdown():
            with self.ft_lock:
                force = self.force
                curr_time = self.curr_time
            if curr_time is not None and force is not None and self.prev_time is not curr_time:
                plt.plot([self.prev_time, self.curr_time], [self.prev_f, force.y], 'b')
                plt.pause(0.01)
                self.prev_time = curr_time
                if self.prev_f == 0:
                    plt.close()
                self.prev_f = force.y
        

if __name__ == '__main__':
    rospy.init_node('force_plotter')
    fPlotter()
