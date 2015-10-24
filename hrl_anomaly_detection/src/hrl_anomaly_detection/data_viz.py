#!/usr/bin/env python
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

# system
import rospy
import roslib
roslib.load_manifest('hrl_anomaly_detection')
import os, sys, copy

# util
import numpy as np
import hrl_lib.util as ut
import util
import PyKDL
import hrl_lib.quaternion as qt

# visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec


class data_viz:
    azimuth_max = 90.0
    azimuth_min = -90.0
    
    def __init__(self, subject=None, task=None, verbose=False):
        rospy.loginfo('log data visualization..')

        self.subject = subject
        self.task    = task
        self.verbose = verbose
        
        self.initParams()

    def initParams(self):
        '''
        # load parameters
        '''        
        self.record_root_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016'
        self.folderName = os.path.join(self.record_root_path, self.subject + '_' + self.task)

        
    def getAngularSpatialRF(self, cur_pos, dist_margin ):

        dist = np.linalg.norm(cur_pos)
        ang_margin = np.arcsin(dist_margin/dist)
        
        cur_pos /= np.linalg.norm(cur_pos)
        ang_cur  = np.arccos(cur_pos[1]) - np.pi/2.0
        
        ang_margin = 10.0 * np.pi/180.0

        ang_max = ang_cur + ang_margin
        ang_min = ang_cur - ang_margin

        return ang_max, ang_min

    
    def extractLocalFeature(self):

        success_list, failure_list = util.getSubjectFileList(self.record_root_path, [self.subject], self.task)

        # Divide it into training and test set
        # -------------------------------------------------------------
        
        # -------------------------------------------------------------
        # loading and time-sync
        d = util.loadData(success_list)

        force_array = None
        for idx in xrange(len(d['timesList'])):
            if force_array is None:
                force_array = d['ftForceList'][idx]
            else:
                force_array = np.hstack([force_array, d['ftForceList'][idx] ])

        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        res = pca.fit_transform( force_array.T )

        # -------------------------------------------------------------        
        # loading and time-sync
        d = util.loadData(failure_list)

        # extract local features
        r = 0.25


        for idx in xrange(len(d['timesList'])):

            timeList     = d['timesList'][idx]
            audioAzimuth = d['audioAzimuthList'][idx]
            audioPower   = d['audioPowerList'][idx]
            kinEEPos     = d['kinEEPosList'][idx]
            kinEEQuat    = d['kinEEQuatList'][idx]
            
            kinEEPos     = d['kinEEPosList'][idx]
            kinEEQuat    = d['kinEEQuatList'][idx]
            
            ftForce      = d['ftForceList'][idx]

            kinTargetPos  = d['kinTargetPosList'][idx]
            kinTargetQuat = d['kinTargetQuatList'][idx]

            
            # Unimoda feature - Audio --------------------------------------------
            unimodal_audioPower = []
            for time_idx in xrange(len(timeList)):
                ang_max, ang_min = self.getAngularSpatialRF(kinEEPos[:,time_idx], r)
                
                if audioAzimuth[time_idx] > ang_min and audioAzimuth[time_idx] < ang_max:
                    unimodal_audioPower.append(audioPower[time_idx])
                else:
                    unimodal_audioPower.append(power_min) # or append white noise?

            ## power_max   = np.amax(d['audioPowerList'])
            ## power_min   = np.amin(d['audioPowerList'])
            ## self.audio_disp(timeList, audioAzimuth, audioPower, audioPowerLocal, \
            ##                 power_min=power_min, power_max=power_max)
                    
            # Unimodal feature - Kinematics --------------------------------------
            unimodal_kinVel = []
            
            # Unimodal feature - Force -------------------------------------------
            # ftForceLocal = np.linalg.norm(ftForce, axis=0) #* np.sign(ftForce[2])
            unimodal_ftForce = pca.transform(ftForce.T).T
            ## self.ft_disp(timeList, ftForce, ftForceLocal)
            
            # Crossmodal feature - relative dist, angle --------------------------
            crossmodal_relativeDist = np.linalg.norm(kinTargetPos - kinEEPos, axis=0)
            crossmodal_relativeAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = kinTargetQuat[:,time_idx]
                
                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_relativeAng.append( abs(diff_ang) )
            
            ## self.relativeFeature_disp(timeList, crossmodal_relativeDist, crossmodal_relativeAng)
            
        ## return [forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList], timesList
                    
            
    def audio_disp(self, timeList, audioAzimuth, audioPower, audioPowerLocal, \
                   power_min=None, power_max=None):

        if power_min is None: power_min = np.amin(audioPower)
        if power_max is None: power_max = np.amax(audioPower)
        
        # visualization
        azimuth_list    = np.arange(self.azimuth_min, self.azimuth_max, 1.0)
        audioImage      = np.zeros( (len(timeList), len(azimuth_list)) )
        audioImageLocal = np.zeros( (len(timeList), len(azimuth_list)) )
        audioImage[0,0] = 1.0
        audioImageLocal[0,0] = 1.0

        for time_idx in xrange(len(timeList)):

            azimuth_idx = min(range(len(azimuth_list)), key=lambda i: \
                              abs(azimuth_list[i]-audioAzimuth[time_idx]))

            p = audioPower[time_idx]
            audioImage[time_idx][azimuth_idx] = (p - power_min)/(power_max - power_min)

            p = audioPowerLocal[time_idx]
            audioImageLocal[time_idx][azimuth_idx] = (p - power_min)/(power_max - power_min)



        fig = plt.figure()            
        # --------------------------------------------------
        ax1 = fig.add_subplot(311)
        ax1.imshow(audioImage.T)
        ax1.set_aspect('auto')
        ax1.set_ylabel('azimuth angle', fontsize=18)

        y     = np.arange(0.0, len(azimuth_list), 30.0)
        new_y = np.arange(self.azimuth_min, self.azimuth_max, 30.0)
        plt.yticks(y,new_y)

        # --------------------------------------------------
        ax2 = fig.add_subplot(312)
        ax2.imshow(audioImageLocal.T)
        ax2.set_aspect('auto')
        ax2.set_ylabel('azimuth angle', fontsize=18)

        y     = np.arange(0.0, len(azimuth_list), 30.0)
        new_y = np.arange(self.azimuth_min, self.azimuth_max, 30.0)
        plt.yticks(y,new_y)

        # --------------------------------------------------
        ax3 = fig.add_subplot(313)
        ax3.plot(timeList, audioPowerLocal)

        plt.show()
        

    def ft_disp(self, timeList, ftForce, ftForceLocal):

        fig = plt.figure()            
        gs = gridspec.GridSpec(4, 2)
        # --------------------------------------------------
        ax1 = fig.add_subplot(gs[0,0])
        ax1.plot(timeList, ftForce[0,:])        

        ax2 = fig.add_subplot(gs[1,0])
        ax2.plot(timeList, ftForce[1,:])        
        
        ax3 = fig.add_subplot(gs[2,0])
        ax3.plot(timeList, ftForce[2,:])        

        ax4 = fig.add_subplot(gs[3,0])
        ax4.plot(timeList, np.linalg.norm(ftForce, axis=0) ) #*np.sign(ftForce[2]) )        

        # --------------------------------------------------
        ax5 = fig.add_subplot(gs[0,1])        
        ax5.plot(timeList, ftForceLocal[0,:])

        ## ax6 = fig.add_subplot(gs[1,1])        
        ## ax6.plot(timeList, ftForceLocal[1,:])
        
        plt.show()


    def relativeFeature_disp(self, timeList, relativeDist, relativeAng):

        fig = plt.figure()            
        ax1 = fig.add_subplot(211)
        ax1.plot(timeList, relativeDist)        
        ax2 = fig.add_subplot(212)
        ax2.plot(timeList, relativeAng)
        plt.show()        
        
        
        
    def audio_test(self):
        
        success_list, failure_list = util.getSubjectFileList(self.record_root_path, [self.subject], self.task)

        for fileName in failure_list:
            d = ut.load_pickle(fileName)
            print d.keys()

            time_max = np.amax(d['audio_time'])
            time_min = np.amin(d['audio_time'])

            self.azimuth_max = 90.0
            self.azimuth_min = -90.0

            power_max = np.amax(d['audio_power'])
            power_min = np.amin(d['audio_power'])

            time_list    = d['audio_time']
            azimuth_list = np.arange(self.azimuth_min, self.azimuth_max, 1.0)
            
            audio_image = np.zeros( (len(time_list), len(azimuth_list)) )

            print "image size ", audio_image.shape

            for idx, p in enumerate(d['audio_power']):

                azimuth_idx = min(range(len(azimuth_list)), key=lambda i: \
                                  abs(azimuth_list[i]-d['audio_azimuth'][idx]))
                
                audio_image[idx][azimuth_idx] = (p - power_min)/(power_max - power_min)

                
            fig = plt.figure()            
            ax1 = fig.add_subplot(211)
            ax1.imshow(audio_image.T)
            ax1.set_aspect('auto')
            ax1.set_ylabel('azimuth angle', fontsize=18)

            y     = np.arange(0.0, len(azimuth_list), 30.0)
            new_y = np.arange(self.azimuth_min, self.azimuth_max, 30.0)
            plt.yticks(y,new_y)
            #------------------------------------------------------------

            n,m = np.shape(d['audio_feature'])

            last_feature = np.hstack([ np.zeros((n,1)), d['audio_feature'][:,:-1] ])            
            feature_delta = d['audio_feature'] - last_feature
            
            ax2 = fig.add_subplot(212)
            ax2.imshow( feature_delta[:n/2] )
            ax2.set_aspect('auto')
            ax2.set_xlabel('time', fontsize=18)
            ax2.set_ylabel('MFCC derivative', fontsize=18)

            #------------------------------------------------------------
            plt.suptitle('Auditory features', fontsize=18)            
            plt.show()
            

    def ft_test(self):
        
        success_list, failure_list = util.getSubjectFileList(self.record_root_path, [self.subject], self.task)

        for fileName in failure_list:
            d = ut.load_pickle(fileName)
            print d.keys()


            fig = plt.figure()            


    def kinematics_test(self):
        
        success_list, failure_list = util.getSubjectFileList(self.record_root_path, [self.subject], self.task)

        for fileName in failure_list:
            d = ut.load_pickle(fileName)
            print d.keys()


            time_max = np.amax(d['kinematics_time'])
            time_min = np.amin(d['kinematics_time'])

            ee_pos   = d['kinematics_ee_pos']
            x_max = np.amax(ee_pos[0,:])
            x_min = np.amin(ee_pos[0,:])

            y_max = np.amax(ee_pos[1,:])
            y_min = np.amin(ee_pos[1,:])

            z_max = np.amax(ee_pos[2,:])
            z_min = np.amin(ee_pos[2,:])
            
            fig = plt.figure()            
            ax  = fig.add_subplot(111, projection='3d')
            ax.plot(ee_pos[0,:], ee_pos[1,:], ee_pos[2,:])
            
            plt.show()

        # ------------------------------------------------------------
        ## from sklearn.decomposition import PCA
        ## pca = PCA(n_components=2)
        ## res = pca.fit_transform(ee_pos.T)    
        
        ## fig = plt.figure()            
        ## plt.plot(res[:,0], res[:,1])
        ## plt.show()


        
    def reduce_cart(self):

        x_range       = np.arange(0.4, 0.9, 0.1)
        z_range       = np.arange(-0.4, 0.1, 0.1)
        azimuth_range = np.arange(-90., 90., 2.) * np.pi / 180.0

        cart_range = None
        for ang in azimuth_range:
            for x in x_range:
                M     = PyKDL.Rotation.RPY(0,0,ang)
                x_pos = PyKDL.Vector(x, 0., 0.)
                new_x_pos = M*x_pos
                new_x = np.array([[new_x_pos[0], new_x_pos[1], new_x_pos[2] ]]).T

                if cart_range is None:
                    cart_range = new_x
                else:
                    cart_range = np.hstack([cart_range, new_x])

    
        for h in z_range:
            if h == 0.0: continue
            cart_range = np.hstack([cart_range, cart_range+np.array([[0,0,h]]).T ])

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_cart = pca.fit_transform(cart_range.T)
        
        traj1 = np.vstack([ np.arange(0.4, 0.9, 0.1), np.zeros((5)), np.zeros((5)) ]).T
        reduced_traj1 = pca.transform(traj1)
        traj2 = np.vstack([ np.arange(0.4, 0.9, 0.1)+0.11, np.zeros((5)), np.zeros((5))-0.1 ]).T
        reduced_traj2 = pca.transform(traj2)
        
        
        fig = plt.figure()            
        ax1 = fig.add_subplot(211, projection='3d')
        ax1.scatter(cart_range[0,:], cart_range[1,:], cart_range[2,:])

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        print traj1.shape, reduced_traj1.shape 
        
        ax2 = fig.add_subplot(212, projection='3d')
        ## ax2.plot(reduced_traj1[:,0],reduced_traj1[:,1],'r')
        ax2.plot(reduced_traj2[:,0],reduced_traj2[:,1],'b')
        ## ax2.plot(reduced_cart,'b')
        
        plt.show()
            

if __name__ == '__main__':


    subject = 'gatsbii'
    task    = 'scooping'
    verbose = True
    

    l = data_viz(subject, task, verbose=verbose)
    ## l.audio_test()
    ## l.kinematics_test()
    ## l.reduce_cart()

    # set RF over EE
    l.extractLocalFeature()
    
