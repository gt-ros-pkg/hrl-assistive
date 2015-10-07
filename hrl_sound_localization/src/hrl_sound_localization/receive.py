#!/usr/bin/env python

import roslib
import rospy

import socket
import sys,re

from std_msgs.msg import String

class juliusReceiver():
    
    def __init__(self):        
        rospy.init_node('recog_receiver')

        host = "localhost"
        port = 10500

        print "Define socket -------------------------------"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print "Connect socket ------------------------------"
        self.sock.connect((host, port))
        print "Connection to Julius succeeded"
        sys.stdout.flush()
        
        self.initComms()
        

    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        print "Initialize communication funcitons!"
        self.cmd_pub = rospy.Publisher('julius_recog_cmd', String, queue_size=3)

    def run(self):

        print "Start run-function"
        word = {}
        angle = {}
        parsed = []
        currentID = None

        line = ""
        while not rospy.is_shutdown():
            # build a valid line
            tmpline = self.sock.recv(1024).replace(".\n", "").rstrip()
            line += tmpline

            if len(tmpline) == 0: continue
            if not tmpline[-1] == ">":
                continue

            # parse it.
            print line
            res = re.search(
                '<SOURCEINFO SOURCEID="([^"]*)".*AZIMUTH="([^"]*)".*', line)
            if not res is None:
                angle[int(res.group(1))] = float(res.group(2))

            res = re.search(
                '<RECOG(OUT|FAIL) SOURCEID="([0-9]*)".*', line)
            if not res is None:
                currentID = int(res.group(2))

            if not currentID is None and "<WHYPO" in line:
                res = re.search('<WHYPO.*WORD="([^<][^"]+)".*', line)
                if not res is None:
                    word[currentID] = res.group(1)
                    parsed.append((currentID, angle[currentID], res.group(1)))

                    out = "%d, %f, %s\n" % (currentID, angle[currentID], res.group(1))
                    #handle = open("julius_out.txt", "a")
                    #handle.write(out)
                    #handle.close()
                    #print out
                    print 'source_id = %d, azimuth = %f' % (currentID, angle[currentID])
                    print 'sentence1: <s> %s </s>' % res.group(1)
                    sys.stdout.flush()

                    self.cmd_pub.publish(res.group(1))
                    
            line = ""


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()

    ds = juliusReceiver()
    ds.run()
    

