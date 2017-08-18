
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
from joblib import Parallel, delayed

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_anomaly_detection.vae import util as vutil
from hrl_anomaly_detection.vae import detector as dt 

# Private learners
from hrl_anomaly_detection.vae import keras_models as km

# visualization
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
random.seed(3334)
np.random.seed(3334)

sys.path.append('/home/dpark/git/STORN-keras')


def lstm_test(subject_names, task_name, raw_data_path, processed_data_path, param_dict, plot=False,
              fine_tuning=False, dyn_ths=False):
    ## Parameters
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    ae_renew   = param_dict['HMM']['renew']
    method     = param_dict['ROC']['methods'][0]
    
    if ae_renew: clf_renew = True
    else: clf_renew  = param_dict['SVM']['renew']
    
    #------------------------------------------
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)         
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=data_dict['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])

        d['successData'], d['failureData'], d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        ut.save_pickle(d, crossVal_pkl)

    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    d['successData']   = d['successData'][feature_list]
    d['failureData']   = d['failureData'][feature_list]
    d['param_dict']['feature_max']   = np.array(d['param_dict']['feature_max'])[feature_list]   
    d['param_dict']['feature_min']   = np.array(d['param_dict']['feature_min'])[feature_list]
    #d['param_dict']['feature_names'] = d['param_dict']['feature_names'][feature_list]

    
    # Parameters
    nDim = len(d['successData'])
    batch_size  = 1 #64
    scale = 1.8
    ths_l = np.logspace(-1.0,2.2,40) -0.1
    add_data = False

    if fine_tuning is False and add_data:
        td1, td2, td3 = vutil.get_ext_feeding_data(task_name, save_data_path, param_dict, d,
                                                   raw_feature=False)
                  

    tp_ll = [[] for i in xrange(len(ths_l))]
    fp_ll = [[] for i in xrange(len(ths_l))]
    tn_ll = [[] for i in xrange(len(ths_l))]
    fn_ll = [[] for i in xrange(len(ths_l))]
    roc_l = []

    # split data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(d['kFoldList']):
        #if idx != 0: continue
        #if not(idx == 0 or idx == 7): continue
        print "==================== ", idx, " ========================"

        # dim x sample x length
        normalTrainData   = d['successData'][:, normalTrainIdx, :]
        abnormalTrainData = d['failureData'][:, abnormalTrainIdx, :]
        normalTestData    = d['successData'][:, normalTestIdx, :]
        abnormalTestData  = d['failureData'][:, abnormalTestIdx, :]
        if fine_tuning is False and add_data:
            normalTrainData   = np.hstack([normalTrainData,
                                           copy.deepcopy(td1['successData']),
                                           copy.deepcopy(td2['successData']),
                                           copy.deepcopy(td3['successData'])])
            abnormalTrainData = np.hstack([abnormalTrainData,
                                           copy.deepcopy(td1['failureData']),
                                           copy.deepcopy(td2['failureData']),
                                           copy.deepcopy(td3['failureData'])])

        # shuffle
        np.random.seed(3334+idx)
        idx_list = range(len(normalTrainData[0]))
        np.random.shuffle(idx_list)
        normalTrainData = normalTrainData[:,idx_list]            

        normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, scaler =\
          vutil.get_scaled_data(normalTrainData, abnormalTrainData,
                                normalTestData, abnormalTestData, aligned=False, scale=scale)

        testData  = [normalTestData, [0]*len(normalTestData)]

        # scaling info to reconstruct the original scale of data
        scaler_dict = {'scaler': scaler, 'scale': scale, 'param_dict': d['param_dict']}

        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------        
        method      = 'storn'
        weights_path = os.path.join(save_data_path,'model_weights_'+method+'_'+str(idx)+'.h5')
        vae_mean   = None
        vae_logvar = None
        enc_z_mean = enc_z_std = None
        generator  = None
        x_std_div   = None
        x_std_offset= None
        z_std      = None
        dyn_ths    = False
        
        window_size = 1
        batch_size  = 256
        fixed_batch_size = True
        noise_mag   = 0.05
        sam_epoch   = 10
        patience    = 4
        h1_dim      = nDim
        phase       = 1.0

        x_std_div   = 4.0 #4
        x_std_offset= 0.05
        z_std       = 0.5
        stateful = True
        ad_method   = 'lower_bound'
            
        ths_l = np.logspace(-1.0,2.4,40) #-0.1


        sys.path.append('/home/dpark/git/STORN-keras/greenarm/anomaly_detection')
        sys.path.append('/home/dpark/git/STORN-keras/greenarm/')
        sys.path.append('/home/dpark/git/STORN-keras/greenarm/models')
        #import simple_predictive as rd
        #clf = rd.TimeSeriesPredictor()
        #clf.fit(normalTrainData, normalTrainData, validation_split=0.3)

        import STORN as rd  
        clf = rd.STORNModel(activation='sigmoid', data_dim=len(normalTrainData[0][0])  )
        #clf.fit(normalTrainData, normalTrainData, validation_split=0.3) 
        clf.fit([normalTrainData, normalTrainData], normalTrainData, validation_split=0.3)
        #clf.save(str(idx))
        



        
                

        roc_l.append(roc)

        for i in xrange(len(ths_l)):
            tp_ll[i] += tp_l[i]
            fp_ll[i] += fp_l[i]
            tn_ll[i] += tn_l[i]
            fn_ll[i] += fn_l[i]

    print "roc list ", roc_l

    d = {}
    d['tp_ll'] = tp_ll
    d['fp_ll'] = fp_ll
    d['tn_ll'] = tn_ll
    d['fn_ll'] = fn_ll
    #roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')
    #ut.save_pickle(d, roc_pkl)

    tpr_l = []
    fpr_l = []
    for i in xrange(len(ths_l)):
        tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
        fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )  

    print roc_l
    print "------------------------------------------------------"
    print tpr_l
    print fpr_l


    from sklearn import metrics
    print "roc: ", metrics.auc(fpr_l, tpr_l, True)  
    fig = plt.figure(figsize=(6,6))
    fig.add_subplot(1,1,1)
    plt.plot(fpr_l, tpr_l, '-*b', ms=5, mec='b')
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.show()







if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    
    p.add_option('--fint_tuning', '--ftn', action='store_true', dest='bFineTune',
                 default=False, help='Run fine tuning.')
    p.add_option('--dyn_ths', '--dt', action='store_true', dest='bDynThs',
                 default=False, help='Run dynamic threshold.')

    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center   = 'kinEEPos'        
    scale       = 1.0
    local_range = 10.0
    nPoints     = 40 #None
    opt.bHMMRenew = opt.bAERenew

    from hrl_anomaly_detection.vae.vae_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']

    if os.uname()[1] == 'monty1':
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm'
        #save_data_path = os.path.expanduser('~')+\
        #  '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm_pretrain'
    else:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/TCDS2017/'+opt.task+'_data_adaptation2'


    param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                               'unimodal_kinJntEff_1',\
                                               'unimodal_ftForce_integ',\
                                               'crossmodal_landmarkEEDist']
    
    lstm_test(subjects, opt.task, raw_data_path, save_data_path, param_dict,
              fine_tuning=opt.bFineTune, dyn_ths=opt.bDynThs)

