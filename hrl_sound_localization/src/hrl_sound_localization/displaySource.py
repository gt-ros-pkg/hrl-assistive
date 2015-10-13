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

import numpy as np, math
import roslib
import rospy
import threading
import pylab

# ROS message
import tf
from std_msgs.msg import Bool, Empty, Int32, Int64, Float32, Float64, String
from hark_msgs.msg import HarkSource, HarkSrcFFT, HarkSrcFeature

# HRL library
from hrl_common_code_darpa_m3.visualization import draw_scene as ds


def RMS(z, nFFT):

    return np.sqrt(np.sum(np.dot(z,z))) / np.sqrt(2.0*nFFT)


class displaySource():

    FRAME_SIZE = 512
    
    def __init__(self, enable_info_all, enable_info_cen, enable_fft_cen, enable_feature_cen, \
                 enable_recog, viz=False):
        rospy.init_node('display_sources')

        self.enable_info_all = enable_info_all
        self.enable_info_cen = enable_info_cen
        self.enable_fft_cen  = enable_fft_cen
        self.enable_feature_cen = enable_feature_cen
        self.enable_recog    = enable_recog

        # Source info from all direction -----------------
        self.count_info_all = 0
        self.exist_info_num_all = 0
        self.src_info_all   = 0
        self.src_info_all_lock = threading.RLock()

        # Source info from front direction -----------------
        self.count_info_cen = 0
        self.exist_info_num_cen = 0
        self.src_info_cen   = 0
        self.src_info_cen_lock = threading.RLock()

        # Source FFT from front direction -----------------
        self.count_fft_cen     = 0
        self.exist_fft_num_cen = 0
        self.src_fft_cen       = 0
        self.src_fft_cen_lock = threading.RLock()
        self.showFFTData = {}

        # Source FEATURE from front direction -----------------
        self.count_feature_cen     = 0
        self.exist_feature_num_cen = 0
        self.src_feature_cen       = 0
        self.src_feature_cen_lock = threading.RLock()
        self.showFeatureData = {}
        self.plotFeatureFlag = False
        
        # Recognition -----------------
        self.recog_cmd   = ""
        self.recog_cmd_lock = threading.RLock()
        
        # -----------------------------------------------------
        # plot
        self.viz         = viz
        self.displen     = 300
        self.max_sources = 3       
        self.plotFlag = False

        # Color
        self.src_color_all = [0.0,0,0.7,0.7]
        self.src_color_cen = [0.7,0,0,0.7]
        self.text_color    = [0.7,0.7,0.7,0.7]

        # -----------------------------------------------------
        # tf
        self.torso_frame = 'torso_lift_link'
        self.head_frame  = 'head_mount_link'

        self.initComms()
        ## self.initParams()


    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        print "Initialize pusblishers and subscribers"
        if self.enable_info_all: rospy.Subscriber('HarkSource/all', HarkSource, self.harkSrcInfoAllCallback)
        if self.enable_info_cen: rospy.Subscriber('HarkSource/center', HarkSource, self.harkSrcInfoCenterCallback)
        if self.enable_fft_cen:  rospy.Subscriber('HarkSrcFFT/center', HarkSrcFFT, self.harkSrcFFTCenterCallback)
        if self.enable_feature_cen: rospy.Subscriber('HarkSrcFeature/all', HarkSrcFeature, \
                                                     self.harkSrcFeatureCenterCallback)
        if self.enable_recog:    rospy.Subscriber('julius_recog_cmd', String, self.harkCmdCallback)

        # drawing
        self.source_viz_all = ds.SceneDraw("hark/source_viz_all", "/world")
        self.source_viz_cen = ds.SceneDraw("hark/source_viz_cen", "/world")
        self.source_viz_cmd = ds.SceneDraw("hark/source_viz_cmd", "/world")

        # tf
        ## try:
        ##     self.tf_lstnr = tf.TransformListener()
        ## except rospy.ServiceException, e:
        ##     rospy.loginfo("ServiceException caught while instantiating a TF listener. Seems to be normal")
        ##     pass
               

    def initParams(self):
        '''
        Initialize parameters
        '''
        try:
            self.tf_lstnr.waitForTransform(self.torso_frame, self.head_frame, rospy.Time(0), \
                                           rospy.Duration(5.0))
        except:
            self.tf_lstnr.waitForTransform(self.torso_frame, self.head_frame, rospy.Time(0), \
                                           rospy.Duration(5.0))
                                           
        [self.head_pos, self.head_orient_quat] = \
          self.tf_lstnr.lookupTransform(self.torso_frame, self.head_frame, rospy.Time(0))  

        
    def harkSrcInfoAllCallback(self, msg):
        '''
        Get all the source locations from hark. 
        '''
        with self.src_info_all_lock:        
            self.count_info_all     = msg.count
            self.exist_info_num_all = msg.exist_src_num
            self.src_info_all       = msg.src
            
    def harkSrcInfoCenterCallback(self, msg):
        '''
        Get all the source locations from hark. 
        '''
        with self.src_info_cen_lock:
            self.count_info_cen     = msg.count
            self.exist_info_num_cen = msg.exist_src_num
            self.src_info_cen       = msg.src

    def harkSrcFFTCenterCallback(self, msg):
        '''
        Get all the source locations from hark. 
        '''
        with self.src_fft_cen_lock:
            self.count_fft_cen     = msg.count
            self.exist_fft_num_cen = msg.exist_src_num
            self.src_fft_cen       = msg.src

            if self.viz and self.exist_fft_num_cen > 0:


                ## for i, key in enumerate( self.showFFTData.keys() ):

                ##     exist_key=False
                ##     for j in xrange(len(self.src_fft_cen)):
                ##         if key == self.src_fft_cen[j].id:
                ##             exist_key = True
                ##     if exist_key == False:
                ##         del self.showFFTData[key]

                for i in xrange(self.exist_fft_num_cen):
                    src_id = self.src_fft_cen[i].id

                    # Force to use single source id
                    src_id = 0
                    
                    ## if len(self.showFFTData.keys()) > 0:
                    ##     self.src_fft_cen[i]

                    real_fft = self.src_fft_cen[i].fftdata_real

                    if src_id not in self.showFFTData.keys():
                        self.showFFTData[src_id] = np.array([ RMS(real_fft, self.FRAME_SIZE) ])
                    else:
                        self.showFFTData[src_id] = np.hstack([ self.showFFTData[src_id], \
                                                            np.array([ RMS(real_fft, self.FRAME_SIZE) ]) ])

            elif self.viz and self.exist_fft_num_cen == 0:
                if len(self.showFFTData.keys()) == 0: return
                if len(self.showFFTData[0]) < 1 : return

                src_id = 0
                self.showFFTData[src_id] = np.hstack([ self.showFFTData[src_id], \
                                                    np.array([ self.showFFTData[src_id][-1] ]) ])


    def harkSrcFeatureCenterCallback(self, msg):
        '''
        Get MFCC features from hark. 
        '''
        with self.src_feature_cen_lock:
            self.count_feature_cen     = msg.count
            self.exist_feature_num_cen = msg.exist_src_num
            self.src_feature_cen       = msg.src

            if self.viz and self.exist_feature_num_cen > 0:

                # get head frame

                # save data
                for i in xrange(self.exist_feature_num_cen):
                    src_id = self.src_feature_cen[i].id

                    # Force to use single source id
                    src_id = 0
                    
                    ## if len(self.showFeatureData.keys()) > 0:
                    ##     self.src_feature_cen[i]

                    power   = self.src_feature_cen[i].power #float32
                    azimuth = self.src_feature_cen[i].azimuth #float32
                    length  = self.src_feature_cen[i].length
                    feature = self.src_feature_cen[i].featuredata #float32 list

                    if len(feature) < 1: continue

                    if src_id not in self.showFeatureData.keys():
                        self.showFeatureData[src_id] = {}
                        self.showFeatureData[src_id]['power'] = np.array([ power ])
                        self.showFeatureData[src_id]['azimuth'] = np.array([ azimuth ])
                        self.showFeatureData[src_id]['feature'] = np.array([ feature ]).T
                    else:
                        self.showFeatureData[src_id]['power'] = np.hstack([\
                            self.showFeatureData[src_id]['power'], np.array([ power ]) ])
                        self.showFeatureData[src_id]['azimuth'] = np.hstack([\
                            self.showFeatureData[src_id]['azimuth'], np.array([ azimuth ]) ])
                        self.showFeatureData[src_id]['feature'] = np.hstack([\
                            self.showFeatureData[src_id]['feature'], \
                            np.array([ feature ]).T ])


            elif self.viz and self.exist_feature_num_cen == 0:
                if len(self.showFeatureData.keys()) == 0: return
                if len(self.showFeatureData[0]) < 1 : return

                src_id = 0
                self.showFeatureData[src_id]['power']\
                = np.hstack([ self.showFeatureData[src_id]['power'], \
                              np.array([ self.showFeatureData[src_id]['power'][-1] ]) ])

                self.showFeatureData[src_id]['azimuth']\
                = np.hstack([ self.showFeatureData[src_id]['azimuth'], \
                              np.array([ self.showFeatureData[src_id]['azimuth'][-1] ]) ])

                self.showFeatureData[src_id]['feature']\
                = np.hstack([ self.showFeatureData[src_id]['feature'], \
                              self.showFeatureData[src_id]['feature'][:,-1:] ])

            
    def harkCmdCallback(self, msg):
        '''
        Get recognized cmd
        '''
        with self.recog_cmd_lock:
            self.recog_cmd = msg.data
        
    def draw_sources_all(self, start_id=0):
        '''
        Draw id and location of sounds
        '''
        
        for i in xrange(4):        
            self.source_viz_all.pub_body([0, 0, 0],
                                         [0,0,0,1],
                                         [0.1, 0.1, 0.1],
                                         self.src_color_all, 
                                         start_id+i,
                                         self.source_viz_all.Marker.SPHERE,
                                         action=2)
            
        self.src_info_all_lock.acquire()
        for i in xrange(self.exist_info_num_all):
            src_id    = self.src_info_all[i].id
            power     = self.src_info_all[i].power 
            pos_x     = self.src_info_all[i].x
            pos_y     = self.src_info_all[i].y
            pos_z     = self.src_info_all[i].z
            azimuth   = self.src_info_all[i].azimuth
            elevation = self.src_info_all[i].elevation
            
            self.source_viz_all.pub_body([pos_x, pos_y, pos_z],
                                     [0,0,0,1],
                                     [0.1, 0.1, 0.1],
                                     self.src_color_all, 
                                     start_id+i,
                                     self.source_viz_all.Marker.SPHERE)
            ## print src_id, power
        self.src_info_all_lock.release()

    def draw_sources_cen(self, start_id=0):
        '''
        Draw id and location of sounds
        '''

        for i in xrange(4):        
            self.source_viz_cen.pub_body([0, 0, 0],
                                         [0,0,0,1],
                                         [0.1, 0.1, 0.1],
                                         self.src_color_cen, 
                                         start_id+i,
                                         self.source_viz_cen.Marker.SPHERE,
                                         action=2)
        
        self.src_info_cen_lock.acquire()
        for i in xrange(self.exist_info_num_cen):
            src_id    = self.src_info_cen[i].id
            power     = self.src_info_cen[i].power 
            pos_x     = self.src_info_cen[i].x
            pos_y     = self.src_info_cen[i].y
            pos_z     = self.src_info_cen[i].z
            azimuth   = self.src_info_cen[i].azimuth
            elevation = self.src_info_cen[i].elevation

            if power-31.0 > 0.0:
                self.src_color_cen[3] = (power-31.0)/2.0 + 0.4
                if self.src_color_cen[3] > 1.0:
                    self.src_color_cen[3] = 1.0
            else:
                self.src_color_cen[3] = 0.4
            
            self.source_viz_cen.pub_body([pos_x, pos_y, pos_z],
                                         [0,0,0,1],
                                         [0.1, 0.1, 0.1],
                                         self.src_color_cen, 
                                         start_id+i,
                                         self.source_viz_cen.Marker.SPHERE)
            ## print src_id, power
        self.src_info_cen_lock.release()



    def draw_sources_cmd(self, start_id=0):
        '''
        Draw id and location of sounds
        '''
        if self.recog_cmd == "" or self.recog_cmd==None: return
        
        self.recog_cmd_lock.acquire()
        self.source_viz_cmd.pub_text([0,0,0.3],
                                     [0,0,0,1],
                                     0.2,
                                     self.text_color, 
                                     start_id,
                                     text=self.recog_cmd)
            ## print src_id, power
        self.recog_cmd_lock.release()


    ## def plotPower(self):

    ##     ## if self.count_info_cen % 10 == 0:
    ##     for i, key in enumerate(self.showData.keys()):
    ##         if len(self.showData[key]) > self.powerlen:
    ##             self.showData[key] = self.showData[key][len(self.showData)-self.powerlen:]

    ##         if len(self.showData[key]) < 1: continue

    ##         ## pylab.subplot(self.max_powers, 1, i+1)
    ##         pylab.plot(self.showData[key], label=str(key))
    ##         pylab.xlim([0, self.powerlen])
    ##         pylab.xticks(range(0, self.powerlen, 5000),
    ##                      range(self.count_info_cen, self.count_info_cen + self.powerlen, 5000))
    ##         pylab.ylabel("Power of ID ")
    ##         ## mx = max(abs(self.showData[key]))
    ##         ## pylab.ylim([-mx, mx])
    ##         pylab.ylim([20.0, 50.0])

    ##     if self.count_info_cen == 0:
    ##         pylab.xlabel("Time [frame]")
    ##     pylab.legend()
    ##     pylab.draw()
            

    def plotFFTData(self):

        if self.plotFFTFlag == False:
            self.fft_fig = pylab.figure(1)
            pylab.ion()
            pylab.hold(False)
            self.plotFFTFlag = True
            self.showFFTData[0]=[]
        else:
            pylab.figure(1)
            
        with self.src_fft_cen_lock:

            if len(self.showFFTData.keys()) > 1:
                print "Failure: too many sources!! : ", self.showFFTData.keys()
                return
            else:
            
                key = self.showFFTData.keys()[0]
                if len(self.showFFTData[0]) > self.displen:
                    self.showFFTData[0] = self.showFFTData[0][len(self.showFFTData)-self.displen:]

                if len(self.showFFTData[0]) < 1: return

                ## pylab.subplot(self.max_powers, 1, i+1)
                pylab.plot(self.showFFTData[0], label=str(key))

                
        pylab.xlim([0, self.displen])
        pylab.xticks(range(0, self.displen, 5000),
                     range(self.count_fft_cen, self.count_fft_cen + self.displen, 5000))
                
        pylab.ylabel("RMS of ID ")
        ## mx = max(abs(self.showFFTData[key]))
        ## pylab.ylim([-mx, mx])
        ## pylab.ylim([20.0, 50.0])
        pylab.xlabel("Time [frame]")
        pylab.legend()
        pylab.draw()
        

    def plotFeatureData(self):

        if self.plotFeatureFlag == False:
            self.feature_fig = pylab.figure(2)
            pylab.ion()
            pylab.hold(False)
            self.plotFeatureFlag = True
            ## self.showFeatureData = {}
            self.im = None
        else:
            pylab.figure(2)
                    
        with self.src_feature_cen_lock:

            if len(self.showFeatureData.keys()) > 1:
                print "Failure: too many sources!! : ", self.showFeatureData.keys()
                return
            elif len(self.showFeatureData.keys()) == 0:
                print "Failure: no sources!! ", self.showFeatureData
                return
            else:
            
                key = self.showFeatureData.keys()[0]

                if len(self.showFeatureData[0]['power']) > self.displen:
                    self.showFeatureData[0]['power']   = self.showFeatureData[0]['power']\
                      [len(self.showFeatureData)-self.displen:]
                    self.showFeatureData[0]['azimuth'] = self.showFeatureData[0]['azimuth']\
                      [len(self.showFeatureData)-self.displen:]
                    self.showFeatureData[0]['feature'] = self.showFeatureData[0]['feature']\
                      [:, len(self.showFeatureData)-self.displen:]

                if len(self.showFeatureData[0]['power']) < 1: return

                pylab.subplot(3, 1, 1)
                pylab.plot(self.showFeatureData[0]['power'])
                pylab.ylim([20, 35])

                pylab.subplot(3, 1, 2)
                pylab.plot(self.showFeatureData[0]['azimuth'])
                pylab.ylim([-90.0, 90.0])

                pylab.subplot(3, 1, 3)
                max_feature = 12
                if self.im is None:
                    self.im = pylab.imshow(self.showFeatureData[0]['feature'][:max_feature,:])
                    ## self.cb = pylab.colorbar()
                    pylab.xlim([0, self.displen])
                    pylab.gca().invert_yaxis()
                else:
                    self.im.set_array( self.showFeatureData[0]['feature'][:max_feature,:] )
                    ext = self.im.get_extent()
                    self.im.set_extent((ext[0], self.showFeatureData[0]['feature'].shape[1], ext[2], ext[3]))
                    ## self.cb.set_clim(vmin=self.showFeatureData[0]['feature'].min(), \
                    ##                  vmax=self.showFeatureData[0]['feature'].max())
                                                            
                ## pylab.plot(self.showFeatureData[0], label=str(key))
                ## pylab.imshow(self.showFeatureData[0]['feature'])
                ## pylab.ylim([0, 24])
                pylab.axis('tight')

        pylab.xlim([0, self.displen])
        ## pylab.xticks(range(0, self.displen, 50),
        ##              range(self.count_feature_cen - self.displen, self.count_feature_cen, 50))
        pylab.xlabel("Time [frame]")
                
        ## pylab.ylabel("RMS of ID ")
        ## mx = max(abs(self.showFeatureData[key]))
        ## pylab.ylim([-mx, mx])
        ## pylab.ylim([20.0, 50.0])
        ## pylab.legend()
        pylab.draw()

        

    def tfBroadcaster(self):
        br = tf.TransformBroadcaster()
        br.sendTransform((0, 0, 0),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         "torso_lift_link",
                         "world")
        

    def run(self):
        '''
        Update the visualization data and publish it.
        '''
        rt = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.tfBroadcaster()
            if self.enable_info_all: self.draw_sources_all(10)
            if self.enable_info_cen: self.draw_sources_cen(20)
            if self.enable_recog:    self.draw_sources_cmd(30)
            if self.enable_fft_cen and self.viz: self.plotFFTData()
            if self.enable_feature_cen and self.viz: self.plotFeatureData()
            rt.sleep()
            

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--visualize', '--viz', action='store_true', dest='bViz',
                 default=False, help='Visualize powers.')    
    opt, args = p.parse_args()


    enable_info_all = False
    enable_info_cen = False
    enable_fft_cen  = False
    enable_feature_cen  = True
    enable_recog    = True

    ds = displaySource( enable_info_all, enable_info_cen, enable_fft_cen, enable_feature_cen, enable_recog,\
                        viz=opt.bViz)
    ds.run()
    

