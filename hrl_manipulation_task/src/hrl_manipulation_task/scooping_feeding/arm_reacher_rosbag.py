#!/usr/bin/env python
# system library

# ROS library
import rospy, rospkg
from std_msgs.msg import String
from hrl_msgs.msg import StringArray
from hrl_manipulation_task.record_data import feedback_to_label

import subprocess
import os, signal, sys
import time
import threading
QUEUE_SIZE = 10


class rosbagStarter:

    def __init__(self, data_path=None, file_path=None):
        '''Initialize GUI'''

        #variables
        self.raw_path = os.path.expanduser('~')
        self.end_data_path = data_path
        self.file_path = file_path
        self.actionStatus = 'Init'
        self.rosbagStatus = False
        self.rosbagPid = None
        self.rosbagName = None
        self.rosbagTime = None
        self.guiStatus = None
        self.rosbagStopped = False
        self.rosbagLock = threading.RLock()
        self.scoopSubID = self.findSubID('scooping')
        self.feedSubID  = self.findSubID('feeding')
        self.initComms()
        self.run()


    def findSubID(self, task):
        curr_max = 0
        for file_name in os.listdir(self.end_data_path):
            if task in file_name and '.bag' in file_name:
                splitted = file_name.split('_')
                if len(splitted) > 1:
                    if splitted[1].isdigit():
                        if (int(splitted[1]) + 1) > curr_max:
                            curr_max = int(splitted[1]) + 1
        return curr_max

    def initComms(self):
        #Publisher:
        self.rosbaggerPub = rospy.Publisher("/manipulation_task/rosbag_status", String, queue_size=QUEUE_SIZE)

        #subscriber:
        rospy.Subscriber("/manipulation_task/status", String, self.statusCallback)
        rospy.Subscriber("/manipulation_task/gui_status", String, self.guiCallback, queue_size=1)
        #rospy.Subscriber("/manipulation_task/emergency", String, self.emergencyCallback, queue_size=10)
        rospy.Subscriber("/manipulation_task/user_feedback", StringArray, self.feedbackCallback, queue_size=10)
        rospy.Subscriber("/manipulation_task/rosbag_request", String, self.rosbagCallback)
        rospy.Subscriber("/manipulation_task/rosbag_rename", StringArray, self.renameCallback)

    def statusCallback(self, msg):
        self.actionStatus = msg.data
        #Change the status, depending on the button pressed.
        return


    def guiCallback(self, msg):
        self.guiStatus = msg.data
        if self.guiStatus == 'in motion' or self.guiStatus == 'standby':
            print self.actionStatus
            ## if self.actionStatus.lower() == 'scooping' or self.actionStatus.lower() == 'feeding':
            if self.actionStatus.lower() == 'feeding':
                self.rosbagStopped = False
                self.startRosbag()
        elif self.guiStatus == 'request feedback' or self.guiStatus == 'stopped':
            #if actionStatus == 'feeding':
            self.stopRosbag()

    def feedbackCallback(self, msg):
        rosbag_result = feedback_to_label(msg.data)
        print self.rosbagStopped
        with self.rosbagLock:
            if rosbag_result.lower() == 'success':
                if self.actionStatus.lower() == 'scooping':
                    self.rename(['0_' + str(self.scoopSubID) + '_success_scooping'])
                    self.scoopSubID = self.scoopSubID + 1
                elif self.actionStatus.lower() == 'feeding':
                    self.rename(['0_' + str(self.feedSubID) + '_success_feeding'])
                    self.feedSubID = self.feedSubID + 1
            elif rosbag_result.lower() == 'failure':
                subType = '14_'
                if len(msg.data) > 3:
                    subType = msg.data[3] + '_'
                if self.actionStatus.lower() == 'scooping':
                    self.rename([subType + str(self.scoopSubID) + '_failure_scooping'])
                    self.scoopSubID = self.scoopSubID + 1
                elif self.actionStatus.lower() == 'feeding':
                    self.rename([subType + str(self.feedSubID) + '_failure_feeding'])
                    self.feedSubID = self.feedSubID + 1
        return

    def rosbagCallback(self, msg):
        req = msg.data
        if req == 'stop':
            with self.rosbagLock:
                self.stopRosbag()
        elif req == 'start':
            with self.rosbagLock:
                self.startRosbag()

    def stopRosbag(self):
        with self.rosbagLock:
            if self.rosbagPid is not None:
                try:
                    os.kill(self.rosbagPid, signal.SIGINT)
                    self.rosbagPid = None
                except:
                    print "failed to kill"
                    return
                if self.rosbagTime is not None:
                    if len(self.rosbagName) >= len(self.rosbagTime):
                        print self.rosbagName[-len(self.rosbagTime):]
                        if self.rosbagTime == self.rosbagName[-len(self.rosbagTime):]:
                            name = self.rosbagName.split('/')[-1][:-len(self.rosbagTime)]
                            if name[-1] == '_':
                                name = name[:-1]
                            print name
                            while os.path.isfile(self.rosbagName + '.active'):
                                time.sleep(0.1)
                            self.rename([self.end_data_path, name])
                self.rosbagStopped = True
                        
            else:
                print "no rosbag to kill"

    def startRosbag(self):
        with self.rosbagLock:
            p = subprocess.Popen(['python', self.file_path], stdout=subprocess.PIPE)
            out, err = p.communicate()
            pid = -1
            try:
                pid = int(out.strip())
            except:
                print "pid couldn't be extracted"
            if pid == -1 or pid is None:
                print "rosbag wasn't started correctly"
                return False
            else:
                self.rosbagPid = pid
                self.rosbagStatus = True
                found = False
                while not found:
                    for rosfile in os.listdir(self.raw_path):
                        if '.bag.active' in rosfile:
                            self.rosbagName = os.path.join(self.raw_path, rosfile)
                            self.rosbagTime = self.rosbagName.split('_')[-1][:-7]
                            self.rosbagName = self.rosbagName[:-7]
                            found = True
                            break
                print self.rosbagName, self.rosbagTime, os.path.isfile(self.rosbagName)
                return True 

    def renameCallback(self, msg):
        self.rename(msg.data)

    def rename(self, data):
        print "renaming to"
        print data
        if self.rosbagName == None:
            print "do not have latest rosbag, either it was removed or never recorded"
            return
        else:
            if not os.path.isfile(self.rosbagName):
                print "do not have access to latest rosbag"
                print self.rosbagName
                return
        if len(data) == 1:
            if data[0] == 'rm':
                os.remove(self.rosbagName)
                self.rosbagName = None
            else:
                newName = None
                if self.rosbagTime == None:
                    newName = os.path.join(self.end_data_path, data[0]+ '.bag')
                else:
                    newName = os.path.join(self.end_data_path, data[0] + '_' + self.rosbagTime)
                os.rename(self.rosbagName, newName)
                self.rosbagName = newName
        else:
            newName = ''
            for path in data:
                newName = os.path.join(newName, path)
            if self.rosbagTime is not None:
                newName = newName + '_' + self.rosbagTime
            else:
                newName = newName + '.bag'
            os.rename(self.rosbagName, newName)
            self.rosbagName = newName

    # --------------------------------------------------------------------------
    def run(self):
        rospy.loginfo("Continous run function called")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.rosbaggerPub.publish(self.rosbagStatus)
            rate.sleep()

        # failsafe
        self.stopRosbag()

 
if __name__ == '__main__':
    
    import optparse
    recordDataPath = '/home/hkim/rosbag_test' # where you want to save rosbags
    p = optparse.OptionParser()
    p.add_option('--data_path', action='store', dest='sRecordDataPath',
                 default=recordDataPath,
                 #default='/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017', \
                 help='Enter a record data path')
    rospack = rospkg.RosPack()
    try:
        path = rospack.get_path('hrl_manipulation_task')
    except:
        print "path to hrl_manipulation_task not found"
    path = os.path.join(path, 'src/hrl_manipulation_task/scooping_feeding/rosbag_creater.py')
    if not os.path.isfile(path):
        print "invalid path"
        print path
        sys.exit(0)
    print path
                        
    print os.path.abspath(os.path.curdir)
    #sys.exit(0)
    opt, args = p.parse_args()
    rospy.init_node('rosbag_starter_client')

    rosbagger = rosbagStarter(data_path=opt.sRecordDataPath, file_path=path)
    rospy.spin()
