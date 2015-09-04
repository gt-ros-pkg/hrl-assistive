#!/usr/bin/env python

__author__ = 'zerickson'

import time
import ghmm
import rospy
import pyaudio
import threading
from threading import Thread
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
import hmm.icra2015Batch as onlineHMM
from audio.tool_audio_slim import tool_audio_slim
from hmm.util import *

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import vision.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, WrenchStamped, Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from roslib import message
from sensor_msgs.msg import JointState

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import image_geometry
from cv_bridge import CvBridge, CvBridgeError
from sound_play.libsoundplay import SoundClient
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures

class onlineAnomalyDetection(Thread):
    MAX_INT = 32768.0
    CHUNK   = 1024 # frame per buffer
    RATE    = 48000 # sampling rate
    UNIT_SAMPLE_TIME = 1.0 / float(RATE)
    CHANNEL = 2 # number of channels
    FORMAT  = pyaudio.paInt16

    def __init__(self, subject='s1', task='s', targetFrame=None, tfListener=None, isScooping=True, audioTool=None):
        super(onlineAnomalyDetection, self).__init__()
        self.daemon = True
        self.cancelled = False
        self.isRunning = False

        # Predefined settings
        self.downSampleSize = 100 #200
        self.cov_mult       = 5.0
        self.isScooping     = isScooping
        self.subject        = subject
        self.task           = task
        if self.isScooping:
            self.nState         = 10
            self.cutting_ratio  = [0.0, 0.9] #[0.0, 0.7]
            self.anomaly_offset = -15.0
            self.scale          = [1.0,1.0,1.0,0.7]  #10
            self.ml_thres_pkl='ml_scooping_thres.pkl'
        else:
            self.nState         = 15
            self.cutting_ratio  = [0.0, 0.7]
            self.anomaly_offset = -20
            self.scale          = [1.0,1.0,0.7,1.0]  #10
            self.ml_thres_pkl='ml_feeding_thres.pkl'

        print 'is scooping:', self.isScooping

        self.publisher = rospy.Publisher('visualization_marker', Marker)
        self.interruptPublisher = rospy.Publisher('InterruptAction', String)
        self.targetFrame = targetFrame

        # Data logging
        self.updateNumber = 0
        self.lastUpdateNumber = 0
        self.init_time = rospy.get_time()

        if tfListener is None:
            self.transformer = tf.TransformListener()
        else:
            self.transformer = tfListener

        # Gripper
        self.lGripperPosition = None
        self.lGripperRotation = None
        self.mic = None
        self.grips = []
        # Spoon
        self.spoon = None

        # FT sensor
        self.force = None
        self.torque = None

        # Audio
        if audioTool is None:
            self.audioTool = tool_audio_slim()
            self.audioTool.start()
        else:
            self.audioTool = audioTool

        # Kinematics
        self.jointAngles = None
        self.jointVelocities = None

        self.soundHandle = SoundClient()

        saveDataPath = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/hmm/batchDataFiles/%s_%d_%d_%d_%d.pkl'
        # Setup HMM to perform online anomaly detection
        self.hmm, self.minVals, self.maxVals, self.minThresholds \
        = onlineHMM.iteration(downSampleSize=self.downSampleSize,
                              scale=self.scale, nState=self.nState,
                              cov_mult=self.cov_mult, anomaly_offset=self.anomaly_offset, verbose=False,
                              isScooping=self.isScooping, use_pkl=False,
                              train_cutting_ratio=self.cutting_ratio,
                              findThresholds=True, ml_pkl=self.ml_thres_pkl,
                              savedDataFile=saveDataPath % (('scooping' if self.isScooping else 'feeding'),
                                            self.downSampleSize, self.scale[0], self.nState, int(self.cov_mult)))

        print 'Threshold:', self.minThresholds

        self.forces = []
        self.distances = []
        self.angles = []
        self.audios = []
        self.forcesRaw = []
        self.distancesRaw = []
        self.anglesRaw = []
        self.audiosRaw = []
        self.times = []
        self.likelihoods = []
        self.anomalyOccured = False

        # self.avgAngle = onlineHMM.trainData[2]

        self.forceSub = rospy.Subscriber('/netft_data', WrenchStamped, self.forceCallback)
        print 'Connected to FT sensor'

        self.objectCenter = None
        self.objectCenterSub = rospy.Subscriber('/ar_track_alvar/bowl_cen_pose' if isScooping else '/ar_track_alvar/mouth_pose', PoseStamped, self.objectCenterCallback)
        print 'Connected to center of object publisher'

        groups = rospy.get_param('/haptic_mpc/groups' )
        for group in groups:
            if group['name'] == 'left_arm_joints':
                self.joint_names_list = group['joints']

        self.jstate_lock = threading.RLock()
        self.jointSub = rospy.Subscriber("/joint_states", JointState, self.jointStatesCallback)
        print 'Connected to robot kinematics'

    def reset(self):
        self.isRunning = True
        self.forces = []
        self.distances = []
        self.angles = []
        self.audios = []
        self.forcesRaw = []
        self.distancesRaw = []
        self.anglesRaw = []
        self.audiosRaw = []
        self.times = []
        self.anomalyOccured = False
        self.updateNumber = 0
        self.lastUpdateNumber = 0
        self.init_time = rospy.get_time()
        self.lGripperPosition = None
        self.lGripperRotation = None
        self.mic = None
        self.grips = []
        self.spoon = None
        self.force = None
        self.torque = None
        self.jointAngles = None
        self.jointVelocities = None
        self.objectCenter = None
        self.audioTool.reset(self.init_time)

    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""
        # rate = rospy.Rate(1000) # 25Hz, nominally.
        while not self.cancelled:
            if self.isRunning and self.updateNumber > self.lastUpdateNumber and self.objectCenter is not None:
                self.lastUpdateNumber = self.updateNumber
                if not self.processData(): continue
                
                if not self.anomalyOccured and len(self.forces) > 15:
                    # Perform anomaly detection
                    (anomaly, error) = self.hmm.anomaly_check(self.forces, self.distances, self.angles, self.audios, self.minThresholds)
                    print 'Anomaly error:', error
                    if anomaly:
                        if self.isScooping:
                            self.interruptPublisher.publish('Interrupt')
                        else:
                            self.interruptPublisher.publish('InterruptHead')
                        self.anomalyOccured = True
                        self.soundHandle.play(2)
                        print 'AHH!! There is an anomaly at time stamp', rospy.get_time() - self.init_time, (anomaly, error)

                        fig = plt.figure()
                        for i, modality in enumerate([[self.forces] + onlineHMM.trainData[0][:13], [self.distances] + onlineHMM.trainData[1][:13], [self.angles] + onlineHMM.trainData[2][:13], [self.audios] + onlineHMM.trainData[3][:13]]):
                            ax = plt.subplot(int('41' + str(i+1)))
                            for index, (modal, times) in enumerate(zip(modality, [self.times] + onlineHMM.trainTimeList[:3])):
                                ax.plot(times, modal, label='%d' % index)
                            ax.legend()
                        fig.savefig('/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/fooboohooyou.pdf')
                        print "saved pdf file"
                        rospy.sleep(2.0)
                        # plt.show()
            # rate.sleep()
        print 'Online anomaly thread cancelled'

    def cancel(self, cancelAudio=True):
        self.isRunning = False
        if cancelAudio:
            self.audioTool.cancel()
        self.saveData()
        rospy.sleep(1.0)

    def saveData(self):
        # TODO Save data (Check with daehyung if any more data should be added)
        data = dict()
        data['forces'] = self.forces
        data['distances'] = self.distances
        data['angles'] = self.angles
        data['audios'] = self.audios
        data['forcesRaw'] = self.forcesRaw
        data['distancesRaw'] = self.distancesRaw
        data['anglesRaw'] = self.anglesRaw
        data['audioRaw'] = self.audiosRaw
        data['times'] = self.times
        data['anomalyOccured'] = self.anomalyOccured
        data['minThreshold'] = self.minThresholds
        data['likelihoods'] = self.likelihoods
        data['jointAngles'] = self.jointAngles
        data['jointVelocities'] = self.jointVelocities

        directory = os.path.join(os.path.dirname(__file__), 'onlineDataRecordings/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        fileName = os.path.join(directory, self.subject + '_' + self.task + '_' + time.strftime('%m-%d-%Y_%H-%M-%S.pkl'))
        with open(fileName, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print 'Online data saved to file.'

    def processData(self):
        # Find nearest time stamp from training data
        # timeStamp = rospy.get_time() - self.init_time
        # index = np.abs(self.times - timeStamp).argmin()

        self.transposeGripper()

        # Use magnitude of forces
        force = np.linalg.norm(self.force)

        # Determine distance between mic and center of object
        distance = np.linalg.norm(self.mic - self.objectCenter)
        # Find angle between gripper-object vector and gripper-spoon vector
        micSpoonVector = self.spoon - self.mic
        micObjectVector = self.objectCenter - self.mic
        angle = np.arccos(np.dot(micSpoonVector, micObjectVector) / (np.linalg.norm(micSpoonVector) * np.linalg.norm(micObjectVector)))

        # Process either visual or audio data depending on which we're using
        if len(self.audioTool.audio_data_raw) > 0:
            audio = self.audioTool.audio_data_raw[-1]
        else:
            return False
        if audio is None:
            print 'Audio is None'
            return False
        audio = get_rms(audio)
        # print 'Audio:', audio

        self.forcesRaw.append(force)
        self.distancesRaw.append(distance)
        self.anglesRaw.append(angle)
        self.audiosRaw.append(audio)

        # Scale data
        force = self.scaling(force, minVal=self.minVals[0], maxVal=self.maxVals[0], scale=self.scale[0])
        distance = self.scaling(distance, minVal=self.minVals[1], maxVal=self.maxVals[1], scale=self.scale[1])
        angle = self.scaling(angle, minVal=self.minVals[2], maxVal=self.maxVals[2], scale=self.scale[2])
        audio = self.scaling(audio, minVal=self.minVals[3], maxVal=self.maxVals[3], scale=self.scale[3])

        # Find nearest time stamp from training data
        timeStamp = rospy.get_time() - self.init_time
        index = np.abs(np.array(onlineHMM.trainTimeList[0]) - timeStamp).argmin()

        self.forces.append(force)
        self.distances.append(distance)
        self.angles.append(angle)
        #self.angles.append(onlineHMM.trainData[2][0][index])
        self.audios.append(audio)
        self.times.append(rospy.get_time() - self.init_time)
        if len(self.forces) > 1:
            self.likelihoods.append(self.hmm.likelihoods(self.forces, self.distances, self.angles, self.audios))

        return True
            
    @staticmethod
    def scaling(x, minVal, maxVal, scale=1.0):
        return (x - minVal) / (maxVal - minVal) * scale

    def forceCallback(self, msg):
        self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.updateNumber += 1

    def objectCenterCallback(self, msg):
        self.objectCenter = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def jointStatesCallback(self, data):
        joint_angles = []
        ## joint_efforts = []
        joint_vel = []
        jt_idx_list = [0]*len(self.joint_names_list)
        for i, jt_nm in enumerate(self.joint_names_list):
            jt_idx_list[i] = data.name.index(jt_nm)

        for i, idx in enumerate(jt_idx_list):
            if data.name[idx] != self.joint_names_list[i]:
                raise RuntimeError('joint angle name does not match.')
            joint_angles.append(data.position[idx])
            ## joint_efforts.append(data.effort[idx])
            joint_vel.append(data.velocity[idx])

        with self.jstate_lock:
            self.jointAngles  = joint_angles
            ## self.joint_efforts = joint_efforts
            self.jointVelocities = joint_vel

    def transposeGripper(self):
        # Transpose gripper position to camera frame
        self.transformer.waitForTransform(self.targetFrame, '/l_gripper_tool_frame', rospy.Time(0), rospy.Duration(5))
        try :
            self.lGripperPosition, self.lGripperRotation = self.transformer.lookupTransform(self.targetFrame, '/l_gripper_tool_frame', rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(self.lGripperPosition), tf.transformations.quaternion_matrix(self.lGripperRotation))
        except tf.ExtrapolationException:
            print 'Transpose of gripper failed!'
            return

        # Use a buffer of gripper positions
        if len(self.grips) >= 2:
            lGripperTransposeMatrix = self.grips[-2]
        else:
            lGripperTransposeMatrix = transMatrix
        self.grips.append(transMatrix)

        # Determine location of mic
        mic = [0.12, -0.02, 0]
        # print 'Mic before', mic
        self.mic = np.dot(lGripperTransposeMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
        # print 'Mic after', self.mic
        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        self.spoon = np.dot(lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]

    def find_input_device(self):
        device_index = None
        for i in range(self.p.get_device_count()):
            devinfo = self.p.get_device_info_by_index(i)
            print('Device %d: %s'%(i, devinfo['name']))

            for keyword in ['mic', 'input', 'icicle']:
                if keyword in devinfo['name'].lower():
                    print('Found an input: device %d - %s'%(i, devinfo['name']))
                    device_index = i
                    return device_index

        if device_index is None:
            print('No preferred input found; using default input device.')

        return device_index

    def publishPoints(self, name, points, size=0.01, r=0.0, g=0.0, b=0.0, a=1.0):
        marker = Marker()
        marker.header.frame_id = '/torso_lift_link'
        marker.ns = name
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.scale.x = size
        marker.scale.y = size
        marker.color.a = a
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        for point in points:
            p = Point()
            # print point
            p.x, p.y, p.z = point
            marker.points.append(p)
        self.publisher.publish(marker)


