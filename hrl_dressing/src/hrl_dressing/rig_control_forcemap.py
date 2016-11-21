import os, sys, inspect, roslib, rospy
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, parentdir)

# ROS Libraries
from geometry_msgs.msg import PoseStamped, WrenchStamped
roslib.load_manifest('zenither')
import zenither.zenither as zenither

import numpy as np
import matplotlib.pyplot as plt

import util, datapreprocess

class RigControlForcemap(object):
    def __init__(self, baselineDirectory, velocity=0.1):
        # Velocity default of 0.1 m/s
        self.velocity = velocity
        self.forceTorqueChange = False

        # Set up plotting
        self.setupPlots(baselineDirectory)

        # Set up linear actuator
        self.initialize_zenither()

    def initialize_zenither(self):
        self.z = zenither.Zenither(robot='test_rig')
        self.ft_sleeve_sub = rospy.Subscriber('/force_torque_sleeve', WrenchStamped, self.forceTorque)

        if not self.z.calibrated:
            self.zenither_calibrate()

    def zenither_calibrate(self):
        self.z.nadir()
        self.z.calibrated = True

        pos = self.z.get_position_meters()
        while True:
            rospy.sleep(0.5)
            new_pos = self.z.get_position_meters()
            if np.abs(new_pos-pos)<0.005:
                rospy.sleep(0.5)
                break
            pos = new_pos

        print 'Hit the end stop.'
        print 'Setting the origin.'
        self.z.estop()
        self.z.set_origin()
        print '-'*20
        print 'Calibration Over'
        pos = self.z.get_position_meters()
        print 'Current position is: ', pos

    def zenither_move(self, pos, vel, acc):
        acceleration = self.z.limit_acceleration(acc)
        velocity = self.z.limit_velocity(vel)
        self.z.set_pos_absolute(pos)
        self.z.set_velocity(velocity)
        self.z.set_acceleration(acceleration)
        self.z.go()

    def forceTorque(self, msg):
        self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.torceTorqueChange = True

    def performDressing(self):
        # Reset position
        self.zenither_move(0.9, 0.1, 0.1)
        pos = self.z.get_position_meters()
        print 'Current position is:', pos
        print 'Moving to initial position:', reset_pos
        start_move_time = rospy.Time.now()
        rospy.sleep(1.0)
        new_pos = self.z.get_position_meters()
        while np.abs(new_pos - pos) > 0.005 and rospy.Time.now().to_sec() - start_move_time.to_sec() < 20.0:
            pos = self.z.get_position_meters()
            rospy.sleep(0.5)
            new_pos = self.z.get_position_meters()
        self.initGripperPos = self.z.get_position_meters()
        print 'Current position is:', self.initGripperPos

        self.startTime = rospy.get_time()
        # Wait for 5 seconds to match simulation
        rospy.sleep(5.0)
        # Start end effector movement
        self.zenither_move(0.05, self.velocity, 1.0)
        # Move arm forward until 12 seconds have passed
        prevTime = rospy.get_time()
        runTime = rospy.get_time() - self.startTime
        while runTime < 12:
            if self.forceTorqueChange:
                self.forceTorqueChange = False
                timeDelta = rospy.get_time() - prevTime
                prevTime = rospy.get_time()
                runTime = rospy.get_time() - self.startTime
                self.recordData(runTime, timeDelta)
                self.updatePlot()

    def recordData(self, runTime, timeDelta):
        # Get change in gripper position and divide by change in time
        self.data['gripperVelocities'].append((self.z.get_position_meters() - self.initGripperPos) / timeDelta)
        self.data['recordedTimes'].append(runTime)
        self.data['gripperForce'].append(self.force)
        self.data['gripperTorque'].append(self.torque)

    def updatePlot(self):
        self.xforce_realtime.set_xdata(self.data['recordedTimes'])
        self.xforce_realtime.set_ydata(np.array(self.data['gripperForce'])[:, 0])
        self.yforce_realtime.set_xdata(self.data['recordedTimes'])
        self.yforce_realtime.set_ydata(np.array(self.data['gripperForce'])[:, 1])
        self.zforce_realtime.set_xdata(self.data['recordedTimes'])
        self.zforce_realtime.set_ydata(np.array(self.data['gripperForce'])[:, 2])

        self.xtorque_realtime.set_xdata(self.data['recordedTimes'])
        self.xtorque_realtime.set_ydata(np.array(self.data['gripperTorque'])[:, 0])
        self.ytorque_realtime.set_xdata(self.data['recordedTimes'])
        self.ytorque_realtime.set_ydata(np.array(self.data['gripperTorque'])[:, 1])
        self.ztorque_realtime.set_xdata(self.data['recordedTimes'])
        self.ztorque_realtime.set_ydata(np.array(self.data['gripperTorque'])[:, 2])

        self.xvelocity_realtime.set_xdata(self.data['recordedTimes'])
        self.xvelocity_realtime.set_ydata(np.array(self.data['gripperVelocities'])[:])

        plt.draw()

    def setupPlots(self, baselineDirectory):
        # Load baseline simulation data
        filename = util.getFilenames([baselineDirectory], util.rawDataDir)[0]
        data = util.loadFile(filename)
        times = np.array(data['recordedTimes'])
        forces = np.array(data['gripperForce'])
        torques = np.array(data['gripperTorque'])
        positions = np.array(data['gripperPos'])

        plt.subplot(3, 3, 1)
        self.xforce_baseline, = plt.plot(times, forces[:, 0])
        self.xforce_realtime, = plt.plot([])
        plt.xlabel('time (s)')
        plt.ylabel('x-axis force on gripper (N)')
        plt.title('X-axis Force')
        plt.subplot(3, 3, 2)
        self.yforce_baseline, = plt.plot(times, forces[:, 1])
        self.yforce_realtime, = plt.plot([])
        plt.xlabel('time (s)')
        plt.ylabel('y-axis force on gripper (N)')
        plt.title('Y-axis Force')
        plt.subplot(3, 3, 3)
        self.zforce_baseline, = plt.plot(times, forces[:, 2])
        self.zforce_realtime, = plt.plot([])
        plt.xlabel('time (s)')
        plt.ylabel('z-axis force on gripper (N)')
        plt.title('Z-axis Force')

        plt.subplot(3, 3, 4)
        self.xtorque_baseline, = plt.plot(times, torques[:, 0])
        self.xtorque_realtime, = plt.plot([])
        plt.xlabel('time (s)')
        plt.ylabel('x-axis toruqe on gripper (N)')
        plt.title('X-axis Torque')
        plt.subplot(3, 3, 5)
        self.ytorque_baseline, = plt.plot(times, torques[:, 1])
        self.ytorque_realtime, = plt.plot([])
        plt.xlabel('time (s)')
        plt.ylabel('y-axis toruqe on gripper (N)')
        plt.title('Y-axis Torque')
        plt.subplot(3, 3, 6)
        self.ztorque_baseline, = plt.plot(times, torques[:, 2])
        self.ztorque_realtime, = plt.plot([])
        plt.xlabel('time (s)')
        plt.ylabel('z-axis toruqe on gripper (N)')
        plt.title('Z-axis Torque')

        velocities = datapreprocess.gripperVelocity({'gripperPos': [positions], 'recordedTimes': [times]})[0]
        velocities = np.around(velocities, 3)
        plt.subplot(3, 3, 8)
        self.xvelocity_baseline, = plt.plot(times, velocities[:, 0])
        self.xvelocity_realtime, = plt.plot([])
        plt.xlabel('time (s)')
        plt.ylabel('x-axis velocity on gripper (N)')
        plt.title('X-axis Velocity')

        plt.ion()
        plt.show()

if __name__ == '__main__':
    rospy.init_node('rigcontrolforcemap')
    rigcontrol = RigControlForcemap('2016-09-30_arm', 0.1)
    rigcontrol.performDressing()

