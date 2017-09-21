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

# system & utils
import os, sys, copy, random
import numpy as np
import scipy
import hrl_lib.util as ut

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_execution_monitor import util as autil
from hrl_execution_monitor import preprocess as pp

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv
import hrl_anomaly_detection.IROS17_isolation.isolation_util as iutil

import rosbag
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# visualization
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib import animation
from matplotlib import rc
import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 


class rosbagExtractor():
    # Must have __init__(self) function for a class, similar to a C++ class constructor.
    def __init__(self, save_dir, filename):

        t0 = None
        # Open bag file.
        with rosbag.Bag(filename, 'r') as bag:
            for topic, msg, t in bag.read_messages():
                if t0 is None: t0 = t

                if topic == "/SR300/rgb/image_raw" :
                    # Use a CvBridge to convert ROS images to OpenCV images so they can be saved.
                    self.bridge = CvBridge()
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    except CvBridgeError, e:
                        print e
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    image_name = str(save_dir)+"/image_"+timestr+".jpg"
                    cv2.imwrite(image_name, cv_image)

                ## if topic == "/manipulation_task/hmm_input0":                    
                ##     timestr = "%.6f" % msg.header.stamp.to_sec()
                ##     print np.shape(msg.data)

                ## if topic == "/feeding/manipulation_task/ad_sensitivity_state":
                ##     print t-t0, msg.data

                if topic == "/hrl_manipulation_task/raw_data":                    
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    print msg

                    
def extract_rosbag_subfolders(subject_path):

    if not(os.path.isdir(subject_path)): return
    list_dir = os.listdir(subject_path)
    for l in list_dir:
        print l
        extract_rosbag(os.path.join(subject_path,l))
        extract_rosbag_subfolders(os.path.join(subject_path,l))

    return

                        
def extract_rosbag(subject_path):

    if not(os.path.isdir(subject_path)): return

    # get rosbags path
    bag_files = os.listdir(subject_path)
        
    # For loop
    for idx, f in enumerate(bag_files):
        if os.path.isdir(f): continue
        if not(f.find('.bag')>=0): continue
        print idx, "/", len(bag_files), " : ", f

        folder_name = f.split('.')[0]
        ## if not(folder_name.find('success')>=0 or folder_name.find('failure')>=0):
        ##     continue
        
        # Remove time stamps on the folder name.
        if (folder_name.find('_feeding')>=0):
            folder_name = folder_name.split('_feeding')[0]

        # create save folder
        save_dir = os.path.join(subject_path, folder_name)
        if not os.path.isdir(save_dir): os.makedirs(save_dir)

        # save image file
        bag_file = os.path.join(subject_path,f)
        rosbagExtractor(save_dir, bag_file)

    return 


def animate_features(opt, param_dict, bSave=False):
    ''' Generate animation for IROS2017 video (failure data collection)'''

    success_viz = False #True
    failure_viz = False #True
    subjects = ['mikako']

    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/IROS2017/'        
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_failure_collection_video/'

    param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS', \
                                                'unimodal_kinVel',\
                                                'unimodal_kinJntEff_1', \
                                                'unimodal_ftForce_integ', \
                                                'unimodal_ftForce_zero', \
                                                'unimodal_kinDesEEChange', \
                                                'crossmodal_landmarkEEDist', \
                                                ]

    data_dict = dm.getDataLOPO(subjects, opt.task, raw_data_path, save_data_path,
                               param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                               downSampleSize=param_dict['data_param']['downSampleSize'], \
                               success_viz=success_viz, failure_viz=failure_viz,\
                               cut_data=param_dict['data_param']['cut_data'],\
                               save_pdf=opt.bSavePdf, solid_color=True,\
                               handFeatures=param_dict['data_param']['handFeatures'], \
                               data_renew=opt.bDataRenew, \
                               max_time=param_dict['data_param']['max_time']) #, target_class=target_class)

    # --------------------------------------------------------------------------------------
    nFeatures = len(param_dict['data_param']['handFeatures'])
    AddFeature_names = np.array(data_dict['param_dict'].get('feature_names',
                                                            param_dict['data_param']['handFeatures']))
    X_time = data_dict['param_dict']['timeList']
    
    p,n,m,k = np.shape(data_dict['successDataList'])
    successDataList = data_dict['successDataList']
    successDataList = np.swapaxes(successDataList, 0,1)
    
    failureData = data_dict['failureDataList'][0]
    failureData = np.swapaxes(failureData, 0,1)[0] # select single sample

    scale = np.array(data_dict['param_dict']['feature_max'])-np.array(data_dict['param_dict']['feature_min'])
    
    ## matplotlib.rcParams['animation.bitrate'] = 2000
    plt.rc('text', usetex=True)    
    fig = plt.figure(figsize=(10,10))
    #fig = plt.figure(1)
    gs = gridspec.GridSpec(nFeatures, 1) #, height_ratios=[1,1,2])
    
    #-------------------------- 1 ------------------------------------
    ax_list = []
    line_list = []
    line2_list = []
    mean_list = []
    std_list  = []
    for i in xrange(nFeatures):
        ax = fig.add_subplot(gs[i,0])
        ax.set_xlim([0, np.max(X_time)*1.0])
        ax.set_ylim([data_dict['param_dict']['feature_min'][i],
                     data_dict['param_dict']['feature_max'][i]*2.])


        mean_list.append( np.mean( successDataList[i].reshape((p*m,k))*scale[i] +
                                   data_dict['param_dict']['feature_min'][i], axis=0 ) )
        std_list.append( np.std( successDataList[i].reshape((p*m,k))*scale[i] +
                                 data_dict['param_dict']['feature_min'][i], axis=0 ) )

        if AddFeature_names[i] == 'kinVel':
            ax.set_ylabel('Spoon speed'+'\n'+'(m/s)', rotation='horizontal', verticalalignment='center',
                          horizontalalignment='center', fontsize=18 )
            ax.yaxis.set_label_coords(-0.17,0.5)
        elif AddFeature_names[i] == 'kinJntEff_1':
            ax.set_ylabel('1st Joint'+'\n'+'Torque(Nm)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=18)
            ax.yaxis.set_label_coords(-0.17,0.5)
        elif AddFeature_names[i] == 'ftForce_mag_integ':
            ax.set_ylabel('Accumulated'+'\n'+'Force'+'\n'+'on Spoon(N)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=18)
            ax.yaxis.set_label_coords(-0.17,0.5)
        elif AddFeature_names[i] == 'landmarkEEDist':
            ax.set_ylabel('Spoon-Mouth'+'\n'+'Distance(m)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=18)
            ax.yaxis.set_label_coords(-0.17,0.5)
        elif AddFeature_names[i] == 'DesEEChange':
            ax.set_ylabel('Desired Spoon'+'\n'+'Displacement'+'\n'+'(m)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=18)
            ax.yaxis.set_label_coords(-0.17,0.5)
        elif AddFeature_names[i].find('force')>=0 or AddFeature_names[i].find('Force')>=0:
            ax.set_ylabel('Force on'+'\n'+'Spoon(N)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=18)
            ax.yaxis.set_label_coords(-0.17,0.5)
        elif AddFeature_names[i].find('dist')>=0 or AddFeature_names[i].find('change')>=0\
           or AddFeature_names[i].find('Change')>=0 or AddFeature_names[i].find('Dist')>=0:
            ax.set_ylabel('Distance'+'\n'+'(m)', fontsize=18)
            ax.yaxis.set_label_coords(-0.17,0.5)
        elif AddFeature_names[i] == 'audioWristRMS':
            ax.set_ylabel('Sound'+'\n'+'Energy', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=18)
            ax.yaxis.set_label_coords(-0.17,0.5)
        else: ax.set_ylabel(AddFeature_names[i], fontsize=18)

        ax.locator_params(axis='y', nbins=3)
        if i < nFeatures-1: ax.tick_params(axis='x', bottom='off', labelbottom='off')
            
        line, = ax.plot([], [], lw=4, c='r')
        ax.fill_between(X_time, mean_list[i]-2.*std_list[i],
                        mean_list[i]+2.*std_list[i], facecolor='blue', alpha=0.5, linewidth=0.0)
        ax.plot(X_time, mean_list[i], lw=2, c='b', alpha=0.5)
        ax.legend()

        ax_list.append( ax )
        line_list.append( line )

    leg_1, = ax_list[0].plot([], [], color='#FF0000', lw=6, label='Current execution')
    leg_2, = ax_list[0].plot([], [], color='#0000FF', lw=6, label=r'Successful executions (Mean $\pm$ Std)')
    legend = ax_list[0].legend(handles=[leg_1, leg_2], loc=3, fancybox=True, shadow=True,
                               ncol=1, borderaxespad=0.,bbox_to_anchor=(0.0, 1.3), \
                               prop={'size':24})


    ax_list[-1].set_xlabel('Time [s]', fontsize=18)
    fig.subplots_adjust(left=0.23, top=0.85) 


    #-------------------------- 2 ------------------------------------
    legend = None

    def init():
        for i in xrange(nFeatures):
            line_list[i].set_data([],[])
            ## mean_list[i].set_data([],[])
        return line_list, mean_list

    def animate(i):

        x = X_time[:i]

        try:
            for j in xrange(nFeatures):
                line_list[j].set_data(x, failureData[j,:i]*scale[j] +
                                      data_dict['param_dict']['feature_min'][j])
                ## mean_list[j].set_data(x)
                ## ax_list.legend(handles=[line_list[j]], loc=2,prop={'size':12})
        except:
            print "Length error for line_",j

        return line_list, mean_list

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(failureData[0]), interval=100,
                                   repeat=False) #, blit=True

    if bSave == True:
        print "Start to save"
        ## FFMpegWriter = animation.writers['ffmpeg']
        ## writer = FFMpegWriter(fps=23, bitrate=-1)
        ## anim.save('ani_test.mp4', writer=writer)
        #anim.save('ani_test.mp4', fps=30)
        anim.save('ani_test.mp4', fps=23, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'], bitrate=-1)
        print "File saved??"
    else:
        plt.show()


    

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    ## p.add_option('--eval_isol', '--ei', action='store_true', dest='evaluation_isolation',
    ##              default=False, help='Evaluate anomaly isolation with double detectors.')    
    opt, args = p.parse_args()

    from hrl_anomaly_detection.journal_isolation.isolation_param import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)

    if opt.bFeaturePlot:
        animate_features(opt, param_dict, bSave=True)
          
    else:                                                                  
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/JOURNAL_ISOL/'+opt.task+'_1'
        save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RAW_DATA/CORL2017/'

        rospy.init_node("export_data")
        rospy.sleep(1)

        # extract data
        extract_rosbag_subfolders(save_data_path)
        #extract_rosbag(save_data_path)
        ## extract_rosbag(sys.argv[1])
