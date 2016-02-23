#!/usr/bin/python

import roslib; roslib.load_manifest('hrl_dressing')
roslib.load_manifest('force_torque') #can try later adding to efri manifest?

import rospy
import numpy as np, math
import sys, time, os

import hrl_lib.transforms as tr
import force_torque.FTClient as ftc
from hrl_msgs.msg import FloatArray
from std_msgs.msg import Header, Bool, Empty

import pdb

#Code adapted from hrl_cody_arms arm_server.py

## 1D kalman filter update. From Advait, hrl_cody_arms arm_server.py
def kalman_update(xhat, P, Q, R, z):
    xhatminus = xhat
    Pminus = P + Q
    K = Pminus / (Pminus + R)
    xhat = xhatminus + K * (z-xhatminus)
    P = (1-K) * Pminus
    return xhat, P

class FTServer():
    def __init__(self):
        # kalman filtering force vector. (self.step and bias_wrist_ft)
        self.Q_force, self.R_force = {}, {}
        self.xhat_force, self.P_force = {}, {}

        self.Q_force['right_arm'] = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
        self.R_force['right_arm'] = [0.1**2, 0.1**2, 0.1**2, 0.05**2, 0.05**2, 0.05**2]
        self.xhat_force['right_arm'] = [0., 0., 0., 0., 0., 0.]
        self.P_force['right_arm'] = [1.0, 1.0, 1.0, 0., 0., 0.]
        self.Q_force['left_arm'] = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
        self.R_force['left_arm'] = [0.1**2, 0.1**2, 0.1**2, 0.05**2, 0.05**2, 0.05**2]
        self.xhat_force['left_arm'] = [0., 0., 0., 0., 0., 0.]
        self.P_force['left_arm'] = [1.0, 1.0, 1.0, 0., 0., 0.]

        rospy.init_node('efri_ft_server', anonymous=False)

        #self.ftclient_r = ftc.FTClient('force_torque_ft4', True)
        self.ftclient_r = ftc.FTClient('force_torque_ft6', True)
        #self.ftclient_l = ftc.FTClient('force_torque_ft6', True)
        self.ftclient_l = ftc.FTClient('force_torque_ft9', True)

        rospy.sleep(1.0)

        self.efri_force_r_pub = rospy.Publisher('/sleeve/force_fast', FloatArray, queue_size=10)
        self.efri_force_l_pub = rospy.Publisher('/arm/force_fast', FloatArray, queue_size=10)

    def step(self):
        for arm in ['left_arm', 'right_arm']:
            z = self.get_wrist_force(arm).A1 # Force vector

            for i in range(6):
                xhat, p = kalman_update(self.xhat_force[arm][i],
                                        self.P_force[arm][i],
                                        self.Q_force[arm][i],
                                        self.R_force[arm][i], z[i])
                if abs(z[i] - self.xhat_force[arm][i]) > 3.:
                    xhat = z[i] # not filtering step changes.
                self.xhat_force[arm][i] = xhat
                self.P_force[arm][i] = p

    ##3X1 numpy matrix of forces measured by the wrist FT sensor.
    #(This is the force that the environment is applying on the wrist)
    # @param arm - 'left_arm' or 'right_arm'
    # @return in SI units
    #coord frame - tool tip coord frame (parallel to the base frame in the home position)
    # 2010/2/5 Advait, Aaron King, Tiffany verified that coordinate frame
    #from Meka is the left-hand coordinate frame.
    def get_wrist_force(self,arm):
        if arm == 'right_arm' and self.ftclient_r != None:
            return self.get_wrist_force_netft('r')

        if arm == 'left_arm' and self.ftclient_l != None:
            return self.get_wrist_force_netft('l')

    def get_wrist_force_netft(self,arm):
        if arm == 'r':
            #pdb.set_trace()
            w = self.ftclient_r.read()
        elif arm == 'l':
            #pdb.set_trace()
            w = self.ftclient_l.read()

        r = tr.Rz(math.radians(30.))
        f = r * w[0:3]
        t = r * w[0:3]
        f[1,0] = f[1,0] * -1
        t[1,0] = t[1,0] * -1
        return np.row_stack((f,t))

    def run(self):
        r_arm = 'right_arm'
        l_arm = 'left_arm'

        self.step()

        f_r = self.xhat_force[r_arm]
        f_l = self.xhat_force[l_arm]

        time_stamp = rospy.Time.now()
        h = Header()
        h.stamp = time_stamp

        h.frame_id = 'should_not_be_using_this'
        self.efri_force_r_pub.publish(FloatArray(h, f_r))
        self.efri_force_l_pub.publish(FloatArray(h, f_l))


if __name__ == '__main__':

    ft = FTServer()
    while not rospy.is_shutdown():
        start_time = time.time()
        ft.run()
        rospy.sleep(0.008)
        end_time = time.time()
        #rospy.logout(1/(end_time-start_time))
    rospy.spin()


