#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)
import harkpython.harkbasenode as harkbasenode
import exceptions
import numpy as np, math

# ROS message
from std_msgs.msg import Bool, Empty, Int32, Int64, Float32, Float64

# HRL library

import speech_recognition as sr


class HarkNode(harkbasenode.HarkBaseNode):
    def __init__(self):
        print "Initialization"
        self.outputNames = ("OUTPUT", )
        self.outputTypes = ("prim_int",)

        self.r = sr.Recognizer()
        ## rospy.init_node('google_speech')
        ## self.initComms()
               
    ## def initComms(self):
    ##     '''
    ##     Initialize pusblishers and subscribers
    ##     '''
    ##     print "Initialize communication funcitons!"
    ##     self.cmd_pub = rospy.Publisher('google_recog_cmd', String, queue_size=3)
        

    def calculate(self):
        ''' Run this code per each input '''

        if self.SOURCES.keys() is not []:            
            for key in self.SOURCES.keys():
                print self.SOURCES[key]
                print self.SOURCES[key]
                audio = self.r.listen(self.SOURCES[key])
                ## print self.count, key, self.SOURCES[key]

                
                try:
                    print("Google Speech Recognition thinks you said " + self.r.recognize_google(audio))
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                
        ##     print src
            ## if src.has_key("id"):
            ##     print "source id: ",src["id"]
            
        ## for w in self.WAV:

        ##     if w.has_key("id"):
        ##         print w.keys()
        ##         ## print self.w["id"]

        self.outputValues["OUTPUT"] = 0


    def __del__(self):
        print "delete..."
