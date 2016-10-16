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
import os, sys, copy
import random

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
# util
import numpy as np
import scipy
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection.AURO2016_params import *
from hrl_anomaly_detection.optimizeParam import *
from hrl_anomaly_detection import util as util
import hrl_lib.circular_buffer as cb
import hrl_anomaly_detection.data_viz as dv

# External ml
from mvpa2.datasets.base import Dataset
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.grid_search import ParameterGrid
from scipy import stats

# private learner
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf

import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
random.seed(3334)
np.random.seed(3334)

def evaluation_test(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                    data_renew=False, save_pdf=False, verbose=False, debug=False,\
                    no_plot=False, delay_plot=True, find_param=False, data_gen=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict    = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    add_logp_d = HMM_dict.get('add_logp_d', False)
    # SVM
    SVM_dict   = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']


    #------------------------------------------
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')

    if os.path.isfile(crossVal_pkl) and data_renew is False and data_gen is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        kFold_list  = d['kFoldList']
        successData = d['successData']
        failureData = d['failureData']        
        success_files = d['success_files']
        failure_files = d['failure_files']
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''
        # Get a data set with a leave-one-person-out
        print "Extract data using getDataLOPO"
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           handFeatures=data_dict['handFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])
        successData, failureData, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        d['successData']   = successData
        d['failureData']   = failureData
        d['success_files'] = success_files
        d['failure_files'] = failure_files
        d['kFoldList']     = kFold_list
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()
        
    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']
    window_size = [10,10]

    param_dict2  = d['param_dict']
    if 'timeList' in param_dict2.keys():
        timeList    = param_dict2['timeList'][startIdx:]
    else: timeList = None
    handFeatureParams = d['param_dict']
    normalTrainData   = d['successData'] * HMM_dict['scale']


    #-----------------------------------------------------------------------------------------
    # HMM-induced vector with LOPO
    dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                              task_name, processed_data_path,\
                              HMM_dict, data_renew, startIdx, nState, cov, scale, \
                              success_files=success_files, failure_files=failure_files,\
                              add_logp_d=add_logp_d, verbose=verbose)



    ## dist_buff1 = cb.CircularBuffer(8, (1,))
    ## dist_buff2 = cb.CircularBuffer(8, (1,))
    ## dist_buff3 = cb.CircularBuffer(8, (1,))

    classes = []
    classes.append([52,53,54,55])
    classes.append([33,34,35,36,37,41,42,43,44,45,46,47,48,49,50,51,58,59,57,11])
    classes.append([31,32,60,61,39,40,38])

    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"
        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
        ## if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew): continue

        dd = ut.load_pickle(modeling_pkl)
        nEmissionDim = dd['nEmissionDim']
        ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
        ml.set_hmm_object(dd['A'],dd['B'],dd['pi'])

        # dim x sample x length
        normalTrainData   = successData[:, normalTrainIdx, :]
        abnormalTrainData = failureData[:, abnormalTrainIdx, :]
        normalTestData    = successData[:, normalTestIdx, :] 
        abnormalTestData  = failureData[:, abnormalTestIdx, :]

        abnormal_test_files = np.array(failure_files)[abnormalTestIdx].tolist()

        #-----------------------------------------------------------------------------------------
        # Classifier test data
        #-----------------------------------------------------------------------------------------
        nOrder    = 2
        testDataX = abnormalTestData*HMM_dict['scale']

        testDataY = []
        normalIdxList  = []
        normalFileList = []
        abnormalIdxList  = []
        abnormalFileList = []
        for i, f in enumerate(abnormal_test_files):
            if f.find("success")>=0:
                testDataY.append(-1)
                normalIdxList.append(i)
                normalFileList.append(f.split('/')[-1])
            elif f.find("failure")>=0:
                testDataY.append(1)
                abnormalIdxList.append(i)
                abnormalFileList.append(f.split('/')[-1])

        
        ## fileList = util.getSubjectFileList(raw_data_path, subject_names, \
        ##                                    task_name, no_split=True)                
                                           
        ## testDataX,testDataDict = dm.getDataList(fileList, data_dict['rf_center'], data_dict['local_range'],\
        ##                                         handFeatureParams,\
        ##                                         downSampleSize = data_dict['downSampleSize'], \
        ##                                         cut_data       = data_dict['cut_data'],\
        ##                                         handFeatures   = data_dict['handFeatures'])

        ## testDataY = []
        ## normalIdxList  = []
        ## normalFileList = []
        ## abnormalIdxList  = []
        ## abnormalFileList = []
        ## for i, f in enumerate(fileList):
        ##     if f.find("success")>=0:
        ##         testDataY.append(-1)
        ##         normalIdxList.append(i)
        ##         normalFileList.append(f.split('/')[-1])
        ##     elif f.find("failure")>=0:
        ##         testDataY.append(1)
        ##         abnormalIdxList.append(i)
        ##         abnormalFileList.append(f.split('/')[-1])

        ## #temp
        ## ## abnormalIdxList = normalIdxList
        ## ## abnormalFileList = normalFileList

        ## ## # reduce data
        ## ## fileList  = np.array(fileList)[abnormalIdxList]
        ## ## testDataX = np.array(testDataX)[:,abnormalIdxList,:]
        ## ## testDataY = np.array(testDataY)[abnormalIdxList]

        ## ## # Expected raw trajectory
        ## ## nOrder = 2
        ## ## exp_traj  = []
        ## ## exp_interp_traj = []
        ## ## for i in xrange(len(abnormalIdxList)):
        ## ##     kinEEPos  = testDataDict['kinDesEEPosList'][abnormalIdxList[i]]
        ## ##     targetPos = testDataDict['visionLandmarkPosList'][abnormalIdxList[i]]

        ## ##     dist = np.linalg.norm(targetPos[:,0:1]-kinEEPos, axis=0)
        ## ##     dist -= np.mean(dist[:4])
        ## ##     exp_traj.append(dist)

        ## ##     new_dist = []
        ## ##     for j in xrange(len(dist_buff1)):
        ## ##         dist_buff1[j] = np.mean(dist[:4])
        ## ##     for j in xrange(len(dist_buff2)):
        ## ##         dist_buff2[j] = np.mean(dist[:4])
        ## ##     for j in xrange(len(dist_buff3)):
        ## ##         dist_buff3[j] = np.mean(dist[:4])
        ## ##     for j in xrange(len(dist)):
        ## ##         dist_buff1.append(dist[j])
        ## ##         dist_buff2.append(dist_buff1[0])
        ## ##         dist_buff3.append(dist_buff2[0])
        ## ##         new_dist.append( np.mean(dist_buff3.get_array()) )                
        ## ##     exp_interp_traj.append(new_dist)

        ## ## # scaling
        ## ## exp_traj = ( np.array(exp_traj) - handFeatureParams['feature_min'][nOrder] )\
        ## ##   /( handFeatureParams['feature_max'][nOrder] - handFeatureParams['feature_min'][nOrder])
        ## ## exp_interp_traj = ( np.array(exp_interp_traj) - handFeatureParams['feature_min'][nOrder] )\
        ## ##   /( handFeatureParams['feature_max'][nOrder] - handFeatureParams['feature_min'][nOrder])

        ## ## refData = np.mean(normalTrainData[nOrder,:,:startIdx])
        ## ## for i in xrange(len(exp_traj)):
        ## ##     offset = refData - np.mean(exp_traj[i][:startIdx])
        ## ##     exp_traj[i] += offset
        ## ## for i in xrange(len(exp_interp_traj)):
        ## ##     offset = refData - np.mean(exp_interp_traj[i][:startIdx])
        ## ##     exp_interp_traj[i] += offset

        ## ## exp_traj = dm.applying_offset(exp_traj*HMM_dict['scale'], \
        ## ##                               normalTrainData*HMM_dict['scale'], startIdx, nEmissionDim)
        ## ## exp_interp_traj = dm.applying_offset(exp_interp_traj*HMM_dict['scale'], \
        ## ##                                      normalTrainData*HMM_dict['scale'], startIdx, nEmissionDim)
        ## ## exp_traj /= HMM_dict['scale']
        ## ## exp_interp_traj /= HMM_dict['scale']

        ## #temp
        ## ## fileList  = fileList[:5]
        ## ## testDataX = testDataX[:,:5,:]
        ## ## testDataY = testDataY[:5]

        ## # scaling and applying offset            
        ## testDataX = dm.applying_offset(testDataX*HMM_dict['scale'], \
        ##                                normalTrainData*HMM_dict['scale'], startIdx, nEmissionDim)



        # anomaly detection
        detection_idx_list = anomaly_detection(testDataX/HMM_dict['scale'], testDataY, \
                                               task_name, save_data_path, param_dict,\
                                               logp_viz=False, verbose=False)
        
        normalMean = []
        normalStd  = []
        for i in xrange(nEmissionDim):
            normalMean.append(np.mean(normalTestData[i],axis=0))
            normalStd.append(np.std(normalTestData[i],axis=0))

        x = range(len(normalMean[0]))

        targetDataX = testDataX #abnormalTestData
        targetDataY = testDataY
        print "Target Data: ", np.shape(targetDataX), np.shape(targetDataY)

        abnormal_windows = []
        abnormal_class   = []
        for i in xrange(len(targetDataX[0])):

            ## if targetDataY[i] < 0: continue

            # Expected output (prediction from partial observation)
            mu       = []
            for j in xrange(len(x)):
                if j < startIdx:
                    mu.append(ml.B[0][0])
                else:
                    ## x_pred = ml.predict_from_single_seq(exp_interp_traj[i][:j]*HMM_dict['scale'], \
                    ##                                     nOrder=2)
                    x_pred = ml.predict_from_single_seq(targetDataX[nOrder,i,:j], nOrder=2)
                    mu.append(x_pred)
            mu  = np.array(mu)

            # Estimated progress of execution
            _,_,l_logp, l_post = hmm.computeLikelihoods(0, ml.A, ml.B, ml.pi, ml.F,\
                                                        [ targetDataX[j][i] for j in xrange(nEmissionDim)],\
                                                        nEmissionDim, nState,\
                                                        startIdx=startIdx,\
                                                        bPosterior=True)
            max_idx = np.argmax(l_post, axis=1)
            max_idx = [max_idx[0]]*startIdx+max_idx.tolist()

            # Anomaly point
            anomaly_idx = detection_idx_list[i]
            if anomaly_idx is not None:

                # mean, range, slope
                abnormal_window = []
                for k in xrange(nEmissionDim):
                    single_data   = (targetDataX[k,i]-mu[:,k])/HMM_dict['scale']
                    single_window = single_data[anomaly_idx-window_size[0]:anomaly_idx-window_size[1]+1]
                    ## slope,_,_,_,_ = stats.linregress(range(len(single_window)), single_window)
                    abnormal_window += [np.mean(single_window),\
                                        np.amax(single_window)-np.amin(single_window)] 
                abnormal_windows.append(abnormal_window)
                tid = int(abnormalFileList[i].split('_')[0])
                abnormal_class.append(tid)
                ## for kk in xrange(len(classes)):
                ##     if tid in classes[kk]:
                ##         abnormal_class.append(kk)
                ##         break
            
            lim_list = []
                    
            fig = plt.figure(1)
            for k in xrange(nEmissionDim):
                lim_list.append([ np.amin(normalMean[k]-1.0*normalStd[k])*0.75, np.amax(normalMean[k]+1.0*normalStd[k])*1.5 ])
                
                ax = fig.add_subplot(nEmissionDim*100+20+k*2+1)
                ax.fill_between(x, normalMean[k]-1.0*normalStd[k], \
                                normalMean[k]+1.0*normalStd[k], \
                                facecolor='green', alpha=0.3)
                ax.plot(x, targetDataX[k,i]/HMM_dict['scale'], 'r-')
                ax.plot(x, mu[:,k]/HMM_dict['scale'], 'b-')
                if anomaly_idx is not None:
                    ax.plot([anomaly_idx, anomaly_idx], [0.0,2.0], 'm-')
                
                if k == 0:
                    for l in xrange(len(x)):
                        if l%5==0:
                            ax.text(x[l], 0.4, str(max_idx[l]+1) )

                n      = len(x)
                xx     = [0, n/2, n-1]
                labels = [int(timeList[0]), int(timeList[n/2]), int(timeList[-1])]
                ax.set_xticks(xx)
                ax.set_xticklabels(labels)
                            

                ax = fig.add_subplot(nEmissionDim*100+20+k*2+2)
                ax.plot(x, (targetDataX[k,i]-mu[:,k])/HMM_dict['scale'],'r-')
                if anomaly_idx is not None:
                    ax.plot([anomaly_idx, anomaly_idx], [-2.0,2.0], 'm-')

                    min_idx = anomaly_idx-window_size[0]
                    max_idx = anomaly_idx+window_size[1]
                    if min_idx <0 : min_idx = 0
                    ax.plot([anomaly_idx-window_size[0], anomaly_idx-window_size[1]], [-2.0,2.0], 'm-')
                    ax.plot([anomaly_idx+window_size[0], anomaly_idx+window_size[1]], [-2.0,2.0], 'm-')


            for k in xrange(nEmissionDim):
                ax = fig.add_subplot(nEmissionDim*100+20+k*2+1)
                ax.set_ylim(lim_list[k])
                ## if k==0: ax.set_ylim([-0.1,0.4])
                ## elif k==1: ax.set_ylim([0.15,0.4])
                ## elif k==2: ax.set_ylim([0.1,0.75])
                ## else: ax.set_ylim([0.1,0.7])
                ax = fig.add_subplot(nEmissionDim*100+20+k*2+2)
                ax.set_ylim([-0.2,0.2])

            plt.suptitle(abnormalFileList[i], fontsize=18)
            plt.show()

        ## print np.shape(abnormal_windows)
        ## print abnormal_windows
        
        ## pca_gamma=5.0
        ## from sklearn.decomposition import KernelPCA
        ## ml = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=False, \
        ##                gamma=pca_gamma)
        ## X_scaled = ml.fit_transform(abnormal_windows)
        ## fig = plt.figure(2)
        ## for kk in xrange(len(classes)):
        ##     color = colors.next()
        ##     shape = shapes.next()
            
        ##     idx_list = [nn for nn, c in enumerate(abnormal_class) if c == kk ]
        ##     xy_data = X_scaled[idx_list]
        ##     print np.shape(xy_data), color, shape
        ##     print idx_list
        ##     plt.scatter(xy_data[:,0], xy_data[:,1], c=color, marker=shape, label=str(kk))
        ## plt.legend(loc='lower left', prop={'size':12})            
        ## plt.show()

            
        break


def anomaly_detection(X, Y, task_name, processed_data_path, param_dict, logp_viz=False, verbose=False):
    ''' Anomaly detector that return anomalous point on each data.
    '''
    HMM_dict = param_dict['HMM']
    SVM_dict = param_dict['SVM']
    ROC_dict = param_dict['ROC']
    
    # set parameters
    method  = 'hmmgp' #'progress'
    ## weights = ROC_dict[method+'_param_range']
    weight  = -5.0 # weights[10] # need to select weight!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! sensitivity - is less sensitive
    nMaxData   = 20 # The maximun number of executions to train GP
    nSubSample = 40 # The number of sub-samples from each execution to train GP

    # Load a generative model
    idx = 0
    modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

    if verbose: print "start to load hmm data, ", modeling_pkl
    d            = ut.load_pickle(modeling_pkl)
    ## Load local variables: nState, nEmissionDim, ll_classifier_train_?, ll_classifier_test_?, nLength    
    for k, v in d.iteritems():
        # Ignore predefined test data in the hmm object
        if not(k.find('test')>=0):
            exec '%s = v' % k

    ml = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
    ml.set_hmm_object(A,B,pi)
            
    # 1) Convert training data
    if method == 'hmmgp':
        import random
        random.seed(3334)

        idx_list = range(len(ll_classifier_train_X))
        random.shuffle(idx_list)
        ll_classifier_train_X = np.array(ll_classifier_train_X)[idx_list[:nMaxData]].tolist()
        ll_classifier_train_Y = np.array(ll_classifier_train_Y)[idx_list[:nMaxData]].tolist()
        ll_classifier_train_idx = np.array(ll_classifier_train_idx)[idx_list[:nMaxData]].tolist()

        new_X = []
        new_Y = []
        new_idx = []
        for i in xrange(len(ll_classifier_train_X)):
            idx_list = range(len(ll_classifier_train_X[i]))
            random.shuffle(idx_list)
            new_X.append( np.array(ll_classifier_train_X)[i,idx_list[:nSubSample]].tolist() )
            new_Y.append( np.array(ll_classifier_train_Y)[i,idx_list[:nSubSample]].tolist() )
            new_idx.append( np.array(ll_classifier_train_idx)[i,idx_list[:nSubSample]].tolist() )

        ll_classifier_train_X = new_X
        ll_classifier_train_Y = new_Y
        ll_classifier_train_idx = new_idx

        if len(ll_classifier_train_X)*len(ll_classifier_train_X[0]) > 1000:
            print "Too many input data for GP"
            sys.exit()

    X_train, Y_train, idx_train = dm.flattenSample(ll_classifier_train_X, \
                                                   ll_classifier_train_Y, \
                                                   ll_classifier_train_idx,\
                                                   remove_fp=False)
    if verbose: print method, " : Before classification : ", np.shape(X_train), np.shape(Y_train)

    # 2) Convert test data
    startIdx   = 4
    ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx = \
      hmm.getHMMinducedFeaturesFromRawCombinedFeatures(ml, X * HMM_dict['scale'], Y, startIdx)

    if logp_viz:
        ll_logp_neg = np.array(ll_classifier_train_X)[:,:,0]
        ll_logp_pos = np.array(ll_classifier_test_X)[:,:,0]
        dv.vizLikelihood(ll_logp_neg, ll_logp_pos)
        sys.exit()

    # Create anomaly classifier
    dtc = cf.classifier( method=method, nPosteriors=nState, nLength=nLength )
    dtc.set_params( class_weight=weight )
    dtc.set_params( ths_mult = weight )    
    ret = dtc.fit(X_train, Y_train, idx_train, parallel=False)

    # anomaly detection
    detection_idx = [None for i in xrange(len(ll_classifier_test_X))]
    for ii in xrange(len(ll_classifier_test_X)):
        if len(ll_classifier_test_Y[ii])==0: continue

        est_y    = dtc.predict(ll_classifier_test_X[ii], y=ll_classifier_test_Y[ii])

        for jj in xrange(len(est_y)):
            if est_y[jj] > 0.0:                
                if ll_classifier_test_Y[ii][0] > 0:
                    detection_idx[ii] = ll_classifier_test_idx[ii][jj]
                    ## if ll_classifier_test_idx[ii][jj] ==4:
                    ##     print "Current likelihood: ", ll_classifier_test_X[ii][jj][0] 
                break

    return detection_idx





if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--AERenew', '--ar', action='store_true', dest='bAERenew',
                 default=False, help='Renew AE data.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--cfRenew', '--cr', action='store_true', dest='bClassifierRenew',
                 default=False, help='Renew Classifiers.')

    p.add_option('--task', action='store', dest='task', type='string', default='feeding',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=4,
                 help='type the desired dimension')
    p.add_option('--aeswtch', '--aesw', action='store_true', dest='bAESwitch',
                 default=False, help='Enable AE data.')

    p.add_option('--rawplot', '--rp', action='store_true', dest='bRawDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--interplot', '--ip', action='store_true', dest='bInterpDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--feature', '--ft', action='store_true', dest='bFeaturePlot',
                 default=False, help='Plot features.')
    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
    p.add_option('--viz', action='store_true', dest='bViz',
                 default=False, help='temp.')
    p.add_option('--dataselect', '--ds', action='store_true', dest='bDataSelection',
                 default=False, help='Plot data and select it.')
    
    p.add_option('--hmm_param', action='store_true', dest='HMM_param_search',
                 default=False, help='Search hmm parameters.')    
    p.add_option('--data_generation', action='store_true', dest='bDataGen',
                 default=False, help='Data generation before evaluation.')
    p.add_option('--find_param', action='store_true', dest='bFindParam',
                 default=False, help='Find hmm parameter.')
    p.add_option('--cparam', action='store_true', dest='bCustomParam',
                 default=False, help='')
                 

    p.add_option('--no_partial_fit', '--npf', action='store_true', dest='bNoPartialFit',
                 default=False, help='HMM partial fit')

    p.add_option('--debug', '--dg', action='store_true', dest='bDebug',
                 default=False, help='Set debug mode.')
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    p.add_option('--noplot', '--np', action='store_true', dest='bNoPlot',
                 default=False, help='No Plot.')    
    p.add_option('--noupdate', '--nu', action='store_true', dest='bNoUpdate',
                 default=False, help='No update.')    
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out.')

    
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    # Dectection TEST 
    local_range    = 10.0    

    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bClassifierRenew, opt.dim,\
                                                          rf_center, local_range)
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation/'+\
      str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
                                                          
    #---------------------------------------------------------------------------
    if opt.task == 'scooping':
        subjects = ['park', 'test'] #'Henry', 
    #---------------------------------------------------------------------------
    elif opt.task == 'feeding':
        subjects = ['s1', 'test']
    elif opt.task == 'pushing':
        subjects = ['microblack', 'microwhite']        
    else:
        print "Selected task name is not available."
        sys.exit()
                                                          
    #---------------------------------------------------------------------------           
    if opt.bRawDataPlot or opt.bInterpDataPlot:
        '''
        Before localization: Raw data plot
        After localization: Raw or interpolated data plot
        '''
        successData = True
        failureData = True
        modality_list   = ['kinematics', 'audio', 'ft', 'vision_landmark'] # raw plot

        dv.data_plot(subjects, opt.task, raw_data_path, save_data_path,\
                  downSampleSize=param_dict['data_param']['downSampleSize'], \
                  local_range=local_range, rf_center=rf_center, global_data=True,\
                  raw_viz=opt.bRawDataPlot, interp_viz=opt.bInterpDataPlot, save_pdf=opt.bSavePdf,\
                  successData=successData, failureData=failureData,\
                  modality_list=modality_list, data_renew=opt.bDataRenew, verbose=opt.bVerbose)

    elif opt.bDataSelection:
        '''
        Manually select and filter bad data out
        '''
        ## modality_list   = ['kinematics', 'audioWrist','audio', 'fabric', 'ft', \
        ##                    'vision_artag', 'vision_change', 'pps']
        modality_list   = ['kinematics', 'ft']
        success_viz = True
        failure_viz = True

        data_selection(subjects, opt.task, raw_data_path, save_data_path,\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       local_range=local_range, rf_center=rf_center, \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       raw_viz=opt.bRawDataPlot, save_pdf=opt.bSavePdf,\
                       modality_list=modality_list, data_renew=opt.bDataRenew, \
                       max_time=param_dict['data_param']['max_time'], verbose=opt.bVerbose)        

    elif opt.bFeaturePlot:
        success_viz = True
        failure_viz = False
        
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+opt.task+'_data_online/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        dm.getDataLOPO(subjects, opt.task, raw_data_path, save_data_path,
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], scale=scale, \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       ae_data=False,\
                       cut_data=param_dict['data_param']['cut_data'],\
                       save_pdf=opt.bSavePdf, solid_color=True,\
                       handFeatures=param_dict['data_param']['handFeatures'], data_renew=opt.bDataRenew, \
                       max_time=param_dict['data_param']['max_time'])

    elif opt.HMM_param_search:

        from hrl_anomaly_detection.hmm import run_hmm_cpy as hmm_opt
        parameters = {'nState': [20, 25], 'scale': np.linspace(3.0,15.0,10), \
                      'cov': np.linspace(0.5,10.0,5) }
        max_check_fold = 1 #None
        no_cov = False
        
        hmm_opt.tune_hmm(parameters, d, param_dict, save_data_path, verbose=True, n_jobs=opt.n_jobs, \
                         bSave=opt.bSave, method=opt.method, max_check_fold=max_check_fold, no_cov=no_cov)

    elif opt.CLF_param_search:

        clf_opt.tune_clf(save_data_path, opt.task, )
                         
    else:
        if opt.bHMMRenew: param_dict['ROC']['methods'] = ['fixed', 'progress'] 
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
        ## unexp_subjects = ['bang']
        ## unexp_subjects = ['park']
                    
        evaluation_test(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                        save_pdf=opt.bSavePdf, \
                        verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                        find_param=False, data_gen=opt.bDataGen)
