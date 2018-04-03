import rospy, time, sys, os, argparse, random, tf
import numpy as np
import scipy.io
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import WrenchStamped
from datetime import datetime
import cPickle as pickle
from scipy import signal
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed

import controller

baseline = None
forceBaseline = None
torqueBaseline = None

class CapacitivePR2:
    def __init__(self, forcetorqueEnabled=False, participant=0, stage=0, baseheight=0.0, armlength=0.8, save=True):
        self.control = controller.Controller('torso_lift_link')
        # Movement update frequency
        self.hz = 10
        # Record frequency
        self.recordRate = rospy.Rate(100)
        self.forcetorqueEnabled = forcetorqueEnabled
        self.participant = participant
        self.baseheight = baseheight
        self.armlength = armlength
        self.forceThreshold = 10
        self.save = save
        self.stage = stage
        # NOTE: Show user where to place their hand before each trial using a meter stick
        # Stage 0: capacitive sensing, start 5 cm from arm
        # Stage 1: capacitive sensing, start 10 cm from arm
        # Stage 2: capacitive sensing, start 15 cm from arm
        # Stage 3: capacitive sensing, start 20 cm from arm
        # Stage 4: linear trajectory, no capacitive, start 5 cm from arm
        # Stage 5: linear trajectory, no capacitive, start 10 cm from arm
        # Stage 6: linear trajectory, no capacitive, start 15 cm from arm # catch
        # Stage 7: linear trajectory, no capacitive, start 20 cm from arm # miss
        # Demo Stage 8: participant freely moves 20 cm upward or downward tilt with capacitive sensing, start 5 cm from arm
        # Demo Stage 9: long sleeve shirt, participant freely moves 20 cm upward or downward tilt with capacitive sensing, start 5 cm from arm
        # Demo Stage 10: cardigan, participant freely moves 20 cm upward or downward tilt with capacitive sensing, start 5 cm from arm

        # [+ away from robot, + left (robot's perspective) towards center of robot, + up towards the sky]
        # height = {0: -0.2, 1: -0.15, 2: -0.05}[stage%3] if stage < 6 else -0.2
        # self.initRightPos = np.array([0.65, -0.7, 0.2 + baseheight + height])
        height = {0: 0.05, 1: 0.1, 2: 0.15, 3: 0.2}[stage%4] if stage < 8 else 0.05 # success, success, catch, miss
        self.initRightPos = np.array([0.65, -0.8, -0.05 + height + baseheight])
        self.initRightRPY = np.array([0.0, 0.0, 0.0])
        self.initLeftPos = np.array([0.5, 0.5, -0.3])
        self.initLeftRPY = np.array([0.0, 0.0, 0.0])
        self.position = np.copy(self.initRightPos)
        self.integral = 0
        self.prevError = 0

        self.baselineValues = []
        self.zeroing = False
        self.start = False
        # self.t = None
        # self.count = 0

        self.control.setJointGuesses(rightGuess=[-0.7776049413654433, 0.6857264466847096, 0.1578287014773736, -1.5132614529302735, -2.2168160661193794, -1.2178333521491904, -0.6567642801777287], leftGuess=[0.75, 0.5, 0.0, -2.0, 2.3, -1.2, 0.35])
        # self.control.setJointGuesses(rightGuess=[0.12, -0.08, -0.95, -1.31, -3.14, -1.45, 0.94], leftGuess=[0.75, 0.5, 0.0, -2.0, 2.3, -1.2, 0.35])
        rospy.sleep(0.05)
        self.control.printJointStates()

        # Set up capacitive sensor serial reading
        self.forceList = []
        self.torqueList = []
        rospy.Subscriber('/capDressing/capacitance', Float64MultiArray, self.capacitanceCallback)

        if self.forcetorqueEnabled:
            self.forceTorqueSub = rospy.Subscriber('/netft_data', WrenchStamped, self.forceTorque)
            print 'Connected to netft'

        self.data = {'force': [], 'torque': [], 'pos': [], 'posDiff': [], 'time': [], 'capacitance': [], 'proxInMeters': [], 'baseheight': baseheight}

    def forceTorque(self, msg):
        global forceBaseline, torqueBaseline
        # Collect force and torque data
        if forceBaseline is not None:
            self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]) - forceBaseline
            self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]) - torqueBaseline
        elif self.zeroing:
            # Zero out force data
            self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
            self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
            self.forceList.append(self.force)
            self.torqueList.append(self.torque)
            if len(self.forceList) > 50:
                forceBaseline = np.mean(self.forceList, axis=0)
                torqueBaseline = np.mean(self.torqueList, axis=0)
                # print 'Mean forces:', self.forceBaseline, 'Mean torques:', self.torqueBaseline

    def capacitanceCallback(self, msg):
        global baseline
        capacitance = msg.data
        if len(capacitance) == 1:
            # Use just first capacitance value
            capacitance = capacitance[0]

        # Zero out data
        if baseline is None and self.zeroing:
            self.baselineValues.append(capacitance)
            if len(self.baselineValues) > 50:
                baseline = np.mean(self.baselineValues, axis=0)
        elif baseline is not None:
            self.capacitance = -(capacitance - baseline)
            self.capacitance = max(self.capacitance, -3.5)

            # print 'Capacitance reading received:', self.capacitance
            # self.capacitanceToMoveDistance()

    def capacitanceToMoveDistance(self):
        # self.proxInMeters = 0.03986*np.exp(0.0349*self.capacitance) + 0.02656*np.exp(0.1192*self.capacitance)
        # self.proxInMeters = 0.06527*np.exp(0.03923*self.capacitance) + 0.006535*np.exp(0.00265*self.capacitance) + 0.07573*np.exp(0.2204*self.capacitance)
        # NOTE: Use for Bare Conductive board on the PR2's end effector
        # self.proxInMeters = -0.9966 / (self.capacitance - 6.714)
        # self.proxInMeters = 0.7825 / (self.capacitance + 7.513) # Old garment holder
        self.proxInMeters = 0.8438 / (self.capacitance + 4.681) # New garment holder

        # Start when participant places their hand underneath the capacitive sensor
        # if self.proxInMeters <= 0.045:
        #     self.start = True

        # print 'Proximity:', self.proxInMeters

        # moveDist = 0.05 - self.proxInMeters
        # For moving gripper in linear trajectory in the x
        # Travelling 80 cm in 16 seconds -> 5 cm per second
        xdist = 0.05
        if 4 <= self.stage <= 7:
            return np.array([0, xdist, 0])*(1.0/self.hz) # Linear trajectory
        else:
            integralMin = -0.05
            integralMax = 0.05
            Kp = 3.0
            Kd = 2.0
            Ki = 0.0
            error = 0.05 - self.proxInMeters
            # Bound integral term
            self.integral = max(min(self.integral + error, integralMax), integralMin)
            # Compute PID
            result = Kp*error + Ki*self.integral + Kd*(error - self.prevError)
            self.prevError = error

            return np.array([0, xdist, result])*(1.0/self.hz)
            # return np.array([0, xdist, moveDist*2])*(1.0/self.hz)

    def initialize(self, zero=True):
        global baseline
        raw_input('Press enter to move to starting position')

        # Move head and Kinect to point at the start location
        self.control.lookAt(self.initRightPos + np.array([0, 0.4, 0]))

        # Zero out capacitance readings
        if baseline is None and zero:
            self.control.moveGripperTo(np.array([0.65, -0.55, 0.05]) + np.array([-0.2, -0.2, 0]), None, self.initRightRPY, timeout=4.0, wait=True, rightArm=True, useInitGuess=True)
            self.zeroData()

        # Move to starting locations
        print 'Grippers moving to starting positions'
        self.control.moveGripperTo(self.position, None, self.initRightRPY, timeout=4.0, wait=False, rightArm=True, useInitGuess=True)
        self.control.moveGripperTo(self.initLeftPos, None, self.initLeftRPY, timeout=4.0, wait=True, rightArm=False, useInitGuess=True)
        rospy.sleep(1.0)
        self.control.initJoints()
        # self.control.printJointStates()

    def zeroData(self):
        global baseline, forceBaseline
        baseline = None
        forceBaseline = None
        self.baselineValues = []
        self.forceList = []
        self.torqueList = []
        self.zeroing = True
        print 'Zeroing data'
        while baseline is None or (False if not self.forcetorqueEnabled else forceBaseline is None):
            rospy.sleep(0.1)
        self.zeroing = False
        print 'Capacitance and force readings have been zeroed'
        rospy.sleep(1.0)

    def recordData(self):
        if self.forcetorqueEnabled:
            self.data['force'].append(self.force)
            self.data['torque'].append(self.torque)
        self.data['pos'].append(self.pos3d)
        self.data['posDiff'].append(self.difference3d)
        self.data['time'].append(rospy.get_time() - self.startTime)
        self.data['capacitance'].append(self.capacitance)
        self.data['proxInMeters'].append(self.proxInMeters)

    def saveData(self, isSuccess=False):
        filedir = 'data/participant_%d' % self.participant
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        successFail = '_success' if isSuccess else '_fail'
        filename = os.path.join(filedir, datetime.now().strftime('stage_' + str(self.stage) + '_%Y-%m-%d_%H-%M-%S' + successFail + '.pkl'))
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def begin(self):
        self.initialize()

        # Begin movement process
        while True:
            # dist, orient = self.capacitanceToMoveDistance()
            raw_input('Press enter to begin dressing')
            self.start = True
            if self.start:
                self.difference = 0
                self.startTime = rospy.get_time()
                self.lastMoveTime = rospy.get_time()
                while self.difference <= self.armlength and (not self.forcetorqueEnabled or np.linalg.norm(self.force) < self.forceThreshold):
                    # Determine movement amounts
                    dist = self.capacitanceToMoveDistance()
                    if rospy.get_time() - self.lastMoveTime >= 1.0/self.hz - 0.001 or self.difference == 0:
                        self.lastMoveTime = rospy.get_time()
                        x = self.control.moveGripperTo(self.position + dist*2, None, self.initRightRPY, timeout=1.0/self.hz*2, wait=False, rightArm=True)
                        if x is not None:
                            self.position += dist
                    pos = self.control.getGripperPosition(rightArm=True)[0][1]
                    self.difference = pos - self.initRightPos[1]
                    # print self.difference, np.linalg.norm(self.force)
                    self.pos3d = self.control.getGripperPosition(rightArm=True)[0]
                    self.difference3d = self.pos3d - self.initRightPos
                    self.recordRate.sleep()
                    # Record all data
                    self.recordData()

                # Save data
                if self.save:
                    if self.difference > self.armlength:
                        print 'Dressing completed'
                    elif self.forcetorqueEnabled and np.linalg.norm(self.force) >= self.forceThreshold:
                        print 'Stopped due to forces exceeding %d N' % self.forceThreshold
                    isSuccess = raw_input('Success? Is the arm and shoulder both in the sleeve of the gown? [y/n] ') == 'y'
                    if not isSuccess:
                        repeat = raw_input('Would you like to repeat the trial? [y/n] ') == 'y'
                        if repeat:
                            return False
                    self.saveData(isSuccess)
                return True

    def capacitiveToProximity(self, participantStudy=False):
        # Collect data for building a function that maps capacitance readings to estimated distance from a person's arm
        # NOTE: Move PR2 torso to top
        self.initRightPos = np.array([0.65, -0.55, 0.05])
        self.position = np.copy(self.initRightPos)

        self.initialize(zero=False)
        print 'Beginning Initialization'
        rospy.sleep(1.0)

        for iteration in xrange(6 if not participantStudy else 1):
            raw_input('Press enter to begin next round')
            print 'Stage:', iteration

            # Zero out capacitance readings
            self.control.moveGripperTo(self.initRightPos + np.array([-0.2, -0.2, 0]), None, self.initRightRPY, timeout=2.0, wait=True, rightArm=True, useInitGuess=True)
            self.zeroData()

            # Data lists
            verticalCapacitanceReadingsUp = {'capacitance': [], 'posDiff': [], 'pos': [], 'basePos': [], 'time': []}
            verticalCapacitanceReadingsDown = {'capacitance': [], 'posDiff': [], 'pos': [], 'basePos': [], 'time': []}

            # Move to starting location
            self.control.moveGripperTo(self.initRightPos, None, self.initRightRPY, timeout=4.0, wait=True, rightArm=True, useInitGuess=True)
            rospy.sleep(2.0)

            # Find zero by touching the person
            self.control.moveGripperTo(self.initRightPos + np.array([0, 0, -0.2]), None, self.initRightRPY, timeout=5.0, wait=False, rightArm=True)
            while self.capacitance < 300:
                self.recordRate.sleep()
            baseHeight = self.control.getGripperPosition(rightArm=True)[0][2]
            basePos = np.array([self.initRightPos[0], self.initRightPos[1], baseHeight])
            # relativeBasePos = np.array([self.initRightPos[0], self.initRightPos[1], baseHeight])
            self.control.moveGripperTo(basePos, None, self.initRightRPY, timeout=0.1, wait=True, rightArm=True)

            # Move upwards and collect data
            print 'Beginning to collect vertical data'
            velocity = 0.01
            posGoal = np.copy(basePos)
            pos = self.control.getGripperPosition(rightArm=True)[0]
            for j in xrange(2):
                rospy.sleep(1.0)
                lastMoveTime = rospy.get_time()
                t = time.time()
                verticalCapacitanceReadings = verticalCapacitanceReadingsUp if j == 0 else verticalCapacitanceReadingsDown
                action = np.array([0, 0, 1 if j == 0 else -1]) * velocity * (1.0/self.hz)
                while (pos[2] < basePos[2] + 0.15) if j == 0 else (pos[2] > basePos[2]):
                    if rospy.get_time() - lastMoveTime >= 1.0/self.hz - 0.001:
                        lastMoveTime = rospy.get_time()
                        x = self.control.moveGripperTo(posGoal + action*2, None, self.initRightRPY, timeout=1.0/self.hz*2, wait=False, rightArm=True)
                        if x is not None:
                            posGoal += action
                    pos = self.control.getGripperPosition(rightArm=True)[0]
                    verticalCapacitanceReadings['capacitance'].append(self.capacitance)
                    verticalCapacitanceReadings['posDiff'].append(pos - basePos)
                    verticalCapacitanceReadings['pos'].append(pos)
                    verticalCapacitanceReadings['basePos'].append(basePos)
                    verticalCapacitanceReadings['time'].append(time.time() - t)
                    self.recordRate.sleep()

            if not participantStudy:
                with open('calibration/singlesensor_armmount_newholder_%.1f_up.pkl' % (iteration*0.1), 'wb') as f:
                    pickle.dump(verticalCapacitanceReadingsUp, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open('calibration/singlesensor_armmount_newholder_%.1f_down.pkl' % (iteration*0.1), 'wb') as f:
                    pickle.dump(verticalCapacitanceReadingsDown, f, protocol=pickle.HIGHEST_PROTOCOL)
                # Save .m file for Matlab
                scipy.io.savemat('calibration/singlesensor_armmount_newholder_%.1f_up' % (iteration*0.1), verticalCapacitanceReadingsUp)
                scipy.io.savemat('calibration/singlesensor_armmount_newholder_%.1f_down' % (iteration*0.1), verticalCapacitanceReadingsDown)
                # with open('calibration/singlesensor_armmount_newholder_gown_up.pkl', 'wb') as f:
                #     pickle.dump(verticalCapacitanceReadingsUp, f, protocol=pickle.HIGHEST_PROTOCOL)
                # with open('calibration/singlesensor_armmount_newholder_gown_down.pkl', 'wb') as f:
                #     pickle.dump(verticalCapacitanceReadingsDown, f, protocol=pickle.HIGHEST_PROTOCOL)
                # # Save .m file for Matlab
                # scipy.io.savemat('calibration/singlesensor_armmount_newholder_gown_up', verticalCapacitanceReadingsUp)
                # scipy.io.savemat('calibration/singlesensor_armmount_newholder_gown_down', verticalCapacitanceReadingsDown)
            else:
                with open('calibration/participant_%d_armmount_newholder_%.1f_up.pkl' % (self.participant, iteration*0.1), 'wb') as f:
                    pickle.dump(verticalCapacitanceReadingsUp, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open('calibration/participant_%d_armmount_newholder_%.1f_down.pkl' % (self.participant, iteration*0.1), 'wb') as f:
                    pickle.dump(verticalCapacitanceReadingsDown, f, protocol=pickle.HIGHEST_PROTOCOL)
                # Save .m file for Matlab
                scipy.io.savemat('calibration/participant_%d_armmount_newholder_%.1f_up' % (self.participant, iteration*0.1), verticalCapacitanceReadingsUp)
                scipy.io.savemat('calibration/participant_%d_armmount_newholder_%.1f_down' % (self.participant, iteration*0.1), verticalCapacitanceReadingsDown)
                # Move to starting location
                self.control.moveGripperTo(self.initRightPos, None, self.initRightRPY, timeout=4.0, wait=True, rightArm=True, useInitGuess=True)

if __name__ == '__main__':
    rospy.init_node('cappr2')

    # pr2 = CapacitivePR2(forcetorqueEnabled=True, participant=21, stage=4, baseheight=0.01, armlength=0.71, save=True)
    # pr2.begin()

    pr2 = CapacitivePR2(forcetorqueEnabled=False, participant=0, stage=0, baseheight=0, armlength=0)
    pr2.capacitiveToProximity()
    exit()

    parser = argparse.ArgumentParser(description='Capacitive sensing for dressing.')
    parser.add_argument('-p', '--participant', help='Participant number', type=int, required=True)
    parser.add_argument('-bh', '--baseheight', help='Base height adjustment for horizontal human limb', type=float, default=0.0)
    parser.add_argument('-al', '--armlength', help='Length of the participants arm in meters', type=float, default=0.8)
    args = parser.parse_args()

    # NOTE: Four practice runs
    for stage in [4, 5, 6, 7]:
    # for stage in []:
        pr2 = CapacitivePR2(forcetorqueEnabled=True, participant=args.participant, stage=stage, baseheight=args.baseheight, armlength=args.armlength, save=False)
        pr2.begin()

    # NOTE: Full study
    if args.participant % 2 == 0:
        stages = [0, 4]*5 + [1, 5]*5 + [2, 6]*5 + [3, 7]*5
    else:
        stages = [4, 0]*5 + [5, 1]*5 + [6, 2]*5 + [7, 3]*5

    for stage in stages:
        success = False
        while not success:
            pr2 = CapacitivePR2(forcetorqueEnabled=True, participant=args.participant, stage=stage, baseheight=args.baseheight, armlength=args.armlength)
            success = pr2.begin()

    # NOTE: One practice run for demo
    for stage in [8]:
        pr2 = CapacitivePR2(forcetorqueEnabled=True, participant=args.participant, stage=stage, baseheight=args.baseheight, armlength=args.armlength, save=False)
        pr2.begin()

    # NOTE: Demos
    for stage in [8, 9]:
    # for stage in [10]:
        success = False
        while not success:
            pr2 = CapacitivePR2(forcetorqueEnabled=True, participant=args.participant, stage=stage, baseheight=args.baseheight, armlength=args.armlength)
            success = pr2.begin()

    for stage in [10]:
        success = False
        while not success:
            pr2 = CapacitivePR2(forcetorqueEnabled=True, participant=args.participant, stage=stage, baseheight=args.baseheight, armlength=args.armlength)
            # Zero out capacitance readings
            pr2.control.moveGripperTo(np.array([0.65, -0.55, 0.05]) + np.array([-0.2, -0.2, 0]), None, pr2.initRightRPY, timeout=4.0, wait=True, rightArm=True, useInitGuess=True)
            raw_input('Press enter to begin zeroing data')
            pr2.zeroData()
            success = pr2.begin()

    # Collect calibration data for the participant
    pr2 = CapacitivePR2(forcetorqueEnabled=True, participant=args.participant, stage=0, baseheight=args.baseheight, armlength=args.armlength, save=False)
    # Zero out capacitance readings
    pr2.control.moveGripperTo(np.array([0.65, -0.55, 0.05]) + np.array([-0.2, -0.2, 0]), None, pr2.initRightRPY, timeout=4.0, wait=True, rightArm=True, useInitGuess=True)
    pr2.zeroData()
    # Run data collection
    pr2.capacitiveToProximity(participantStudy=True)

    # python capacitivepr2.py -p 0 -bh 0.0 -al 0.75
    # python capacitivepr2.py -p 1 -bh 0.05 -al 0.76
    # python capacitivepr2.py -p 2 -bh 0.03 -al 0.69
    # python capacitivepr2.py -p 3 -bh 0.025 -al 0.69
    # python capacitivepr2.py -p 4 -bh 0.055 -al 0.73
    # python capacitivepr2.py -p 5 -bh 0.03 -al 0.72
    # python capacitivepr2.py -p 6 -bh 0.02 -al 0.67
    # python capacitivepr2.py -p 7 -bh 0.045 -al 0.75
    # python capacitivepr2.py -p 8 -bh -0.04 -al 0.68
    # python capacitivepr2.py -p 9 -bh 0.015 -al 0.65

    # python capacitivepr2.py -p 6 -bh 0.005 -al 0.74

    # Zackory: python capacitivepr2.py -p 0 -bh 0.01 -al 0.71
    # Maggie: python capacitivepr2.py -p 0 -bh 0.01 -al 0.67
    # Ari: python capacitivepr2.py -p -1 -bh 0.01 -al 0.75
    # python capacitivepr2.py -p 0 -bh 0.04 -al 0.74
    # python capacitivepr2.py -p 1 -bh 0.04 -al 0.73
    # python capacitivepr2.py -p 2 -bh -0.015 -al 0.63
    # python capacitivepr2.py -p 3 -bh 0.06 -al 0.73
    # python capacitivepr2.py -p 4 -bh 0.05 -al 0.68
    # pr2.calibrate(int(sys.argv[-1]))

