#!/usr/bin/env python

# System
import numpy as np
import time, sys
import cPickle as pkl
from collections import deque
import pyaudio
import struct
import scipy.signal as signal
import scipy.fftpack
import operator
from threading import Thread
import glob

# ROS
import roslib
roslib.load_manifest('hrl_anomaly_detection')
roslib.load_manifest('geometry_msgs')
roslib.load_manifest('hrl_lib')
import rospy, optparse, math, time
import tf
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import TransformStamped, WrenchStamped
from std_msgs.msg import Bool, Float32

# HRL
from hrl_srvs.srv import None_Bool, None_BoolResponse
from hrl_msgs.msg import FloatArray
import hrl_lib.util as ut

# External Utils
import matplotlib.pyplot as pp
import matplotlib as mpl
from pylab import *


# Private
#import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad
#import hrl_anomaly_detection.advait.arm_trajectories as at

def log_parse():
    parser = optparse.OptionParser('Input the Pose node name and the ft sensor node name')

    parser.add_option("-t", "--tracker", action="store", type="string",\
    dest="tracker_name", default="adl2")
    parser.add_option("-f", "--force" , action="store", type="string",\
    dest="ft_sensor_name",default="/netft_data")

    (options, args) = parser.parse_args()

    return options.tracker_name, options.ft_sensor_name 


class tool_audio(Thread):
    MAX_INT = 32768.0
    CHUNK   = 1024 #frame per buffer
    RATE    = 44100 #sampling rate
    UNIT_SAMPLE_TIME = 1.0 / float(RATE)
    CHANNEL=1 #number of channels
    FORMAT=pyaudio.paInt16
    DTYPE = np.int16
    
    def __init__(self):
        super(tool_audio, self).__init__()
        self.daemon = True
        self.cancelled = False
        
        self.init_time = 0.
        self.noise_freq_l = None
        self.noise_band = 150.0
        self.noise_amp_num = 0 #20 #10
        self.noise_amp_thres = 0.0  
        self.noise_amp_mult = 2.0  
        self.noise_bias = 0.0
        
        self.audio_freq = np.fft.fftfreq(self.CHUNK, self.UNIT_SAMPLE_TIME) 
        self.audio_data = []
        self.audio_amp  = []

        self.time_data = []
        
        self.b,self.a = self.butter_bandpass(1,400, self.RATE, order=6)
        
                
        self.p=pyaudio.PyAudio()
        self.stream=self.p.open(format=self.FORMAT, channels=self.CHANNEL, rate=self.RATE, \
                                input=True, frames_per_buffer=self.CHUNK)
        rospy.logout('Done subscribing audio')


    def run(self):
        """Overloaded Thread.run, runs the update 
        method once per every xx milliseconds."""

        self.stream.start_stream()        
                
        while not self.cancelled:
            self.log()
            ## sleep(0.01)
        
    def log(self):

        data=self.stream.read(self.CHUNK)
        self.time_data.append(rospy.get_time()-self.init_time)
        
        audio_data = np.fromstring(data, self.DTYPE)
        ## audio_data = signal.lfilter(self.b, self.a, audio_data)
        

        # Exclude low rms data
        ## amp = self.get_rms(data)            
        ## if amp < self.noise_amp_thres*2.0:
        ##     audio_data = audio_data*np.exp( - self.noise_amp_mult*(self.noise_amp_thres - amp))

        ## audio_data -= self.noise_bias
        new_F = F = np.fft.fft(audio_data / float(self.MAX_INT))  #normalization & FFT          
        
        # Remove noise
        ## for noise_freq in self.noise_freq_l:
        ##     new_F = np.array([self.filter_rule(x,self.audio_freq[j], noise_freq, self.noise_band) for j, x in enumerate(new_F)])
                                
        ## frame = np.fft.ifft(new_F)*float(self.MAX_INT)
        frame = audio_data

        self.audio_amp.append(new_F)
        self.audio_data.append(frame)
                
        ## self.time_data.append(self.time)

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        rospy.sleep(1.0)

        self.stream.stop_stream()
        self.stream.close()


    def skip(self, seconds):
        samples = int(seconds * self.RATE)
        count = 0
        while count < samples:
            self.stream.read(self.CHUNK)
            count += self.CHUNK
            rospy.sleep(0.01)        
            
    ## def save_audio(self):
        
    ##     ## RECORD_SECONDS = 9.0

    ##     string_audio_data = np.array(self.audio_data, dtype=self.DTYPE).tostring() 
    ##     import wave
    ##     WAVE_OUTPUT_FILENAME = "/home/dpark/git/pyaudio/test/output.wav"
    ##     wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    ##     wf.setnchannels(self.CHANNEL)
    ##     wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
    ##     wf.setframerate(self.RATE)
    ##     wf.writeframes(b''.join(string_audio_data))
    ##     wf.close()
            

        ## pp.figure()        
        ## pp.subplot(211)
        ## pp.plot(new_frames,'b-')
        ## pp.plot(new_filt_frames,'r-')
        
        ## pp.subplot(212)
        ## pp.plot(f[:n/10],np.abs(F[:n/10]),'b')
        ## if new_F is not None:
        ##     pp.plot(f[:n/10],np.abs(new_F[:n/10]),'r')
        ## pp.stem(noise_freq_l, values, 'k-*', bottom=0)        
        ## pp.show()


    def reset(self):

        self.skip(1.0)
        rospy.sleep(1.0)
        
        # Get noise frequency        
        frames=None
        
        ## for i in range(0, int(self.RATE/self.CHUNK * RECORD_SECONDS)):
        data=self.stream.read(self.CHUNK)
        audio_data=np.fromstring(data, self.DTYPE)

        if frames is None: frames = audio_data
        else: frames = np.hstack([frames, audio_data])
        self.noise_amp_thres = self.get_rms(data)            
        
        ## self.noise_bias = np.mean(audio_data)
        ## audio_data -= self.noise_bias

        F = np.fft.fft(audio_data / float(self.MAX_INT))  #normalization & FFT       
        f  = np.fft.fftfreq(len(F), self.UNIT_SAMPLE_TIME) 
        n=len(f)
                
        import heapq
        values = heapq.nlargest(self.noise_amp_num, F[:n/2]) #amplitude

        self.noise_freq_l = []
        for value in values:
            self.noise_freq_l.append([f[j] for j, k in enumerate(F[:n/2]) if k.real == value.real])
        self.noise_freq_l = np.array(self.noise_freq_l)

        print "Amplitude threshold: ", self.noise_amp_thres
        print "Noise bias: ", self.noise_bias

        ## self.skip(1.0)
        self.stream.stop_stream()
        
        ## #temp
        ## ## F1 = F[:n/2]        
        ## for noise_freq in self.noise_freq_l:
        ##     F = np.array([self.filter_rule(x,f[j], noise_freq, self.noise_band) for j, x in enumerate(F)])
        ## ## new_F = np.hstack([F1, F1[::-1]])
        ## new_F = F
        
        ## temp_audio_data = np.fft.ifft(new_F) * float(self.MAX_INT)
        ## print len(temp_audio_data), self.noise_freq_l
        
        ## pp.figure()
        ## pp.subplot(211)
        ## pp.plot(audio_data,'r-')
        ## pp.plot(temp_audio_data,'b-')
        
        ## pp.subplot(212)
        ## pp.plot(f[:n/4],np.abs(F[:n/4]),'b')
        ## pp.stem(self.noise_freq_l, values, 'r-*', bottom=0)
        ## pp.show()
        ## ## raw_input("Enter anything to start: ")
                        

    def filter_rule(self, x, freq, noise_freq, noise_band):
        if np.abs(freq) > noise_freq+noise_band or np.abs(freq) < noise_freq-noise_band:
            return x
        else:
            return 0
                  

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        '''
        fs: sampling frequency
        '''
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

        
    def get_rms(self, block):
        # Copy from http://stackoverflow.com/questions/4160175/detect-tap-with-pyaudio-from-live-mic
        
        # RMS amplitude is defined as the square root of the 
        # mean over time of the square of the amplitude.
        # so we need to convert this string of bytes into 
        # a string of 16-bit samples...

        # we will get one short out for each 
        # two chars in the string.
        count = len(block)/2
        format = "%dh"%(count)
        shorts = struct.unpack( format, block )

        # iterate over the block.
        sum_squares = 0.0
        for sample in shorts:
        # sample is a signed short in +/- 32768. 
        # normalize it to 1.0
            n = sample / self.MAX_INT
            sum_squares += n*n

        return math.sqrt( sum_squares / count )        

class tool_ft(Thread):
    def __init__(self,ft_sensor_topic_name):
        super(tool_ft, self).__init__()
        self.daemon = True
        self.cancelled = False

        self.init_time = 0.
        self.counter = 0
        self.counter_prev = 0
        self.force = np.matrix([0.,0.,0.]).T
        self.force_raw = np.matrix([0.,0.,0.]).T
        self.torque = np.matrix([0.,0.,0.]).T
        self.torque_raw = np.matrix([0.,0.,0.]).T
        self.torque_bias = np.matrix([0.,0.,0.]).T

        self.time_data = []
        self.force_data = []
        self.force_raw_data = []
        self.torque_data = []
        self.torque_raw_data = []

        #capture the force on the tool tip	
        ## self.force_sub = rospy.Subscriber(ft_sensor_topic_name,\
        ## 	WrenchStamped, self.force_cb)
        #raw ft values from the NetFT
        self.force_raw_sub = rospy.Subscriber(ft_sensor_topic_name,\
        WrenchStamped, self.force_raw_cb)
        ## self.force_zero = rospy.Publisher('/tool_netft_zeroer/rezero_wrench', Bool)
        rospy.logout('Done subscribing to '+ft_sensor_topic_name+' topic')


    def force_cb(self, msg):
        self.force = np.matrix([msg.wrench.force.x, 
        msg.wrench.force.y,
        msg.wrench.force.z]).T
        self.torque = np.matrix([msg.wrench.torque.x, 
        msg.wrench.torque.y,
        msg.wrench.torque.z]).T


    def force_raw_cb(self, msg):
        self.time = msg.header.stamp.to_time()
        self.force_raw = np.matrix([msg.wrench.force.x, 
        msg.wrench.force.y,
        msg.wrench.force.z]).T
        self.torque_raw = np.matrix([msg.wrench.torque.x, 
        msg.wrench.torque.y,
        msg.wrench.torque.z]).T
        self.counter += 1


    def reset(self):
        ## self.force_zero.publish(Bool(True))
        pass
        
    def run(self):
        """Overloaded Thread.run, runs the update 
        method once per every xx milliseconds."""

        rate = rospy.Rate(1000) # 25Hz, nominally.            
        while not self.cancelled:
            self.log()
            rospy.sleep(1/1000.)
    

    def log(self):
        if self.counter > self.counter_prev:
            self.counter_prev = self.counter
            self.force_raw[0,0],self.force_raw[1,0],self.force_raw[2,0],\
            self.torque_raw[0,0],self.torque_raw[1,0],self.torque_raw[2,0]
            ## self.force[0,0],self.force[1,0],self.force[2,0],\
            ## self.torque[0,0],self.torque[1,0],self.torque[2,0],\

            ## self.force_data.append(self.force)
            self.force_raw_data.append(self.force_raw)
            ## self.torque_data.append(self.torque)
            self.torque_raw_data.append(self.torque_raw)
            self.time_data.append(rospy.get_time()-self.init_time)
            


    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        rospy.sleep(1.0)
            
            
    ## def static_bias(self):
    ##     print '!!!!!!!!!!!!!!!!!!!!'
    ##     print 'BIASING FT'
    ##     print '!!!!!!!!!!!!!!!!!!!!'
    ##     f_list = []
    ##     t_list = []
    ##     for i in range(20):
    ##         f_list.append(self.force)
    ##         t_list.append(self.torque)
    ##         rospy.sleep(2/100.)
    ##     if f_list[0] != None and t_list[0] !=None:
    ##         self.force_bias = np.mean(np.column_stack(f_list),1)
    ##         self.torque_bias = np.mean(np.column_stack(t_list),1)
    ##         print self.gravity
    ##         print '!!!!!!!!!!!!!!!!!!!!'
    ##         print 'DONE Biasing ft'
    ##         print '!!!!!!!!!!!!!!!!!!!!'
    ##     else:
    ##         print 'Biasing Failed!'


class ADL_log():
    def __init__(self, ft=True, audio=False, manip=False, test_mode=False):
        rospy.init_node('ADLs_log', anonymous = True)

        self.ft = ft
        self.audio = audio
        self.manip = manip
        self.test_mode = test_mode

        self.init_time = 0.
        self.file_name = 'test'
        self.tool_tracker_name, self.ft_sensor_topic_name = log_parse()        
        rospy.logout('ADLs_log node subscribing..')

        if self.manip:
            rospy.wait_for_service("/adl/arm_reach_enable")
            self.armReachAction = rospy.ServiceProxy("/adl/arm_reach_enable", None_Bool)


    def task_cmd_input(self, subject=None, task=None, actor=None, trial_name=None):
        confirm = False
        while not confirm:
            valid = True

            if subject is not None: self.sub_name = subject
            else: self.sub_name=raw_input("Enter subject's name: ")

            if task is not None: num = task
            else:
                num=raw_input("Enter the number for the choice of task:")
                ## "\n1) cup \n2) door \n3) drawer"+\
                ## "\n4) staple\n5) microwave_black\n6) dishwasher\n7) wallsw\n: ")
                
            if num == '1':
                self.task_name = 'microwave_black'
            elif num == '2':
                self.task_name = 'microwave_white'
            elif num == '3':
                self.task_name = 'microwave_kitchen'
            elif num == '4':
                self.task_name = 'door_room'
            elif num == '5':
                self.task_name = 'door_storage'
            elif num == '6':
                self.task_name = 'door_reception'
            elif num == '7':
                self.task_name = 'drawer_white'
            elif num == '8':
                self.task_name = 'drawer_desk'
            elif num == '9':
                self.task_name = 'drawer_reception',
            elif num == '10':
                self.task_name = 'wallsw_nidrr_room'

            elif num == '11':
                self.task_name = 'cup'
            elif num == '12':
                self.task_name = 'staple'
            elif num == '13':
                self.task_name = 'dishwasher'
            else:
                print '\n!!!!!Invalid choice of task!!!!!\n'
                valid = False
                sys.exit()

            if valid:
                if actor is not None: num = actor
                else: num=raw_input("Select actor:\n1) human \n2) robot\n: ")
                if num == '1':
                    self.actor = 'human'
                elif num == '2':
                    self.actor = 'robot'
                else:
                    print '\n!!!!!Invalid choice of actor!!!!!\n'
                    valid = False
                    sys.exit()
                    
            if valid:
                if trial_name is not None: self.trial_name = trial_name
                else:
                    self.trial_name=raw_input("Enter trial's name (e.g. success, failure_reason): ")
                self.file_name = self.sub_name+'_'+self.task_name+'_'+self.actor+'_'+self.trial_name			
                ## ans=raw_input("Enter y to confirm that log file is:  "+self.file_name+"\n: ")
                ## if ans == 'y':
                confirm = True
                    
    def init_log_file(self, subject=None, task=None, actor=None, trial_name=None):

        if self.test_mode is False: 
            self.task_cmd_input(subject, task, actor, trial_name)

        if self.ft: 
            self.ft = tool_ft(self.ft_sensor_topic_name)
            ## self.ft_log_file = open(self.file_name+'_ft.log','w')
            
        if self.audio: 
            self.audio = tool_audio()
            ## self.audio_log_file = open(self.file_name+'_audio.log','w')        


        pkl_list = glob.glob('*.pkl')
        max_num = 0
        for pkl in pkl_list:
            if pkl.find(self.file_name)>=0:
                num = pkl.split('_')[-1].split('.')[0] 
                if max_num < num:
                    max_num = num
        max_num = int(max_num)+1
        self.pkl = self.file_name+'_'+str(max_num)+'.pkl'

        print "File name: ", self.pkl

        raw_input('press Enter to reset')
        if self.ft: self.ft.reset()
        ## if self.audio: self.audio.reset()
        

    def log_start(self):
        
        raw_input('press Enter to begin the test')
        self.init_time = rospy.get_time()
        if self.ft: 
            self.ft.init_time = self.init_time
            self.ft.start()
        if self.audio: 
            self.audio.init_time = self.init_time
            self.audio.start()

        if self.manip:
            rospy.sleep(1.0)
            ret = self.armReachAction()
            print ret
                    

                            
        
    def close_log_file(self):
        # Finish data collection
        if self.ft: self.ft.cancel()
        if self.audio: self.audio.cancel()
        
        
        d = {}
        d['init_time'] = self.init_time

        if self.ft:
            ## dict['force'] = self.ft.force_data
            ## dict['torque'] = self.ft.torque_data
            d['ft_force_raw']  = self.ft.force_raw_data
            d['ft_torque_raw'] = self.ft.torque_raw_data
            d['ft_time']       = self.ft.time_data

        if self.audio:
            d['audio_data']  = self.audio.audio_data
            d['audio_amp']   = self.audio.audio_amp
            d['audio_freq']  = self.audio.audio_freq
            d['audio_chunk'] = self.audio.CHUNK
            d['audio_time']  = self.audio.time_data
        
        ut.save_pickle(d, self.pkl)

        ## self.tool_tracker_log_file.close()
        ## self.tooltip_log_file.close()
        ## self.head_tracker_log_file.close()
        ## self.gen_log_file.close()
        print 'Closing..  log files have saved..'

                

if __name__ == '__main__':

    subject = 'dh'
    task = '1'
    actor = '1'
    trial_name = 'success'
    
    log = ADL_log(audio=True, ft=True, manip=False, test_mode=False)
    log.init_log_file(subject, task, actor, trial_name)

    log.log_start()
    
    rate = rospy.Rate(1000) # 25Hz, nominally.    
    while not rospy.is_shutdown():
        ## log.log_state()
        rate.sleep()
        ## rospy.sleep(1/1000.)

    log.close_log_file()
    

    






## class adl_recording():
##     def __init__(self, obj_id_list, netft_flag_list):
##         self.ftc_list = []                                                                                       
##         for oid, netft in zip(obj_id_list, netft_flag_list):                                                     
##             self.ftc_list.append(ftc.FTClient(oid, netft))                                                       
##         self.oid_list = copy.copy(obj_id_list)
        
##         ## self.initComms()
##         pass

        
##     def initComms(self):
        
##         # service
##         #rospy.Service('door_opening/mech_analyse_enable', None_Bool, self.mech_anal_en_cb)
        
##         # Subscriber
##         rospy.Subscriber('/netft_rdt', Wrench, self.ft_sensor_cb)                    

        
##     # returns a dict of <object id: 3x1 np matrix of force>
##     def get_forces(self, bias = True):
##         f_list = []
##         for i, ft_client in enumerate(self.ftc_list):
##             f = ft_client.read(without_bias = not bias)
##             f = f[0:3, :]

##             ## trans, quat = self.tf_lstnr.lookupTransform('/torso_lift_link',
##             ##                                             self.oid_list[i],
##             ##                                             rospy.Time(0))
##             ## rot = tr.quaternion_to_matrix(quat)
##             ## f = rot * f
##             f_list.append(-f)

##         return dict(zip(self.oid_list, f_list))


##     def bias_fts(self):
##         for ftcl in self.ftc_list:
##             ftcl.bias()
        
        
##     # TODO
##     def ft_sensor_cb(self, msg):

##         with self.ft_lock:
##             self.ft_data = [msg.force.x, msg.force.y, msg.force.z] # tool frame

##             # need to convert into torso frame?


##     def start(self, bManual=False):
##         rospy.loginfo("ADL Online Recording Start")

##         ar.bias_fts()
##         rospy.loginfo("FT bias complete")

        
##         rate = rospy.Rate(2) # 25Hz, nominally.
##         rospy.loginfo("Beginning publishing waypoints")
##         while not rospy.is_shutdown():         
##             f = self.get_forces()[0]
##             print f
##             #print rospy.Time()
##             rate.sleep()        
            
    
