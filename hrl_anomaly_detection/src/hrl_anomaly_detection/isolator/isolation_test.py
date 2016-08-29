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


# system
import os, sys, copy
import random
import itertools
import numpy as np
import scipy

# System learning util
from mvpa2.datasets.base import Dataset
from joblib import Parallel, delayed
from sklearn import metrics
import svmutil as svm
from sklearn import preprocessing

# private learner
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv
from hrl_anomaly_detection.hmm import learning_hmm as hmm

# util
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection.optimizeParam import *
from hrl_anomaly_detection import util as util

# Task param
from hrl_anomaly_detection.isolator.isolator_params import *

# visualization
import matplotlib
#matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import matplotlib.pyplot as plt
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 


def isolation_test(subject_names, task_name, raw_data_path, processed_data_path, \
                     param_dict_hmm, param_dict_isolator, verbose=False, data_renew=False):
    '''
    processed_data_path: please, use this path to save your data
    '''

    ## Parameters
    # data
    param_dict = param_dict_hmm
    data_dict  = param_dict['data_param']
    #return
    #-----------------------------------------------------------------------------------------


    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    

    # train Anomaly Detector
    # HMM training data extraction and handover scaling parameter into isolation data extractor
    if True:
        dim = 4
        hmm_subjects = ['hyun', 'jina', 'sai', 'linda']
        hmm_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+task_name+'_data/'+\
          str(data_dict['downSampleSize'])+'_'+str(dim)
        
        hmm_d = dm.getDataSet(hmm_subjects, task_name, raw_data_path, \
                              hmm_data_path, data_dict['rf_center'], data_dict['local_range'],\
                              downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                              handFeatures=['unimodal_audioWristRMS', 'unimodal_ftForceZ', 'crossmodal_landmarkEEDist', 'crossmodal_landmarkEEAng'],\
                              data_renew=False, max_time=data_dict['max_time']) 
    else:
        dim = 4
        hmm_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+task+'_data/'+\
          str(data_param_dict['downSampleSize'])+'_'+str(dim)
        hmm_d = {'param_dict': None}

    param_dict = param_dict_isolator
    data_dict  = param_dict['data_param']
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        kFold_list = d['kFoldList'] 
    else:
        '''
        '''

        type_sets = []
        type_sets.append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
        type_sets.append([22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,48,71,72,73,75])
        type_sets.append([41,42,43,44,45,46,77,78,47])
        type_sets.append([79,80,81,82,83,84,85,89,90,91,93,94,95,86,87,88])
        """
        type_sets.append([42,43,44,45,46,47,48])
        type_sets.append([21,22,23,24,25,26,30,31,32])
        type_sets.append([18,19,20,33,34,35])
        type_sets.append([0,1,2,3,4,5,6,7,8,9,11,12,13,14])
        """

        d = dm.getDataSet(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           handFeatures=data_dict['handFeatures'],\
                           #cut_data=data_dict['cut_data'],\
                           init_param_dict=hmm_d['param_dict'],\
                           data_renew=data_renew, max_time=data_dict['max_time'])
                           
        order = [[]] * (2 * len(d['failureFiles']))
        new_type_sets = []
        for i in xrange(0, len(type_sets)):
            new_type_sets.append([])
        index_d = {}
        reverse_index_d = {}
        
        new_failureData = []
        for i in xrange(0, len(d['failureData'])):
            new_failureData.append([])
        for i, file_name in enumerate(d['failureFiles']):
            file_name_split = file_name.split("iteration_")
            file_name_split = file_name_split[1].split("_failure")[0]
            idx = int(file_name_split)
            order[idx] = i
        count = 0
        for i, sampleIdx in enumerate(order):
            curr_type = formatY(i, type_sets)
            if curr_type is not 0:
                index_d[str(sampleIdx)]     = count
                reverse_index_d[str(count)] = sampleIdx
                new_type_sets[curr_type-1].append(count)
                count = count + 1
                for dimIdx in xrange(0, len(d['failureData'])):
                    new_failureData[dimIdx].append(d['failureData'][dimIdx,sampleIdx,:])
            else:
                print "rejected or missing, ", sampleIdx
        ## print np.asarray(new_failureData).shape
        ## return
        d['failureData']=np.asarray(new_failureData)
        
        # Task-oriented hand-crafted features        
        kFold_list = dm.kFold_data_index2(len(d['successData'][0]), len(d['failureData'][0]), \
                                          data_dict['nNormalFold'], data_dict['nAbnormalFold'] )
        d['kFoldList']   = kFold_list

        d['typeSets']        = new_type_sets
        d['index_d']         = index_d
        d['reverse_index_d'] =reverse_index_d
            
        ut.save_pickle(d, crossVal_pkl)

    successData = d['successData']
    failureData = d['failureData']
    successFiles = d['successFiles']
    failureFiles = d['failureFiles']
    type_sets    = d['typeSets']
    index_d      = d['index_d']
    reverse_index_d = d['reverse_index_d']

    # Training svm, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        # dim x sample x length
        normalTrainData   = successData[:, normalTrainIdx, :] 
        abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
        normalTestData    = successData[:, normalTestIdx, :] 
        abnormalTestData  = failureData[:, abnormalTestIdx, :]

        # anomaly detector --------------------------------------------
        print "Start to find anomalous point" 
        testDataX = abnormalTrainData#[[0,1,3,4], :, :]
        testDataY = np.ones(len(abnormalTestData[0]))
        detection_train_idx_list = anomaly_detection(testDataX, testDataY, \
                                               task_name, hmm_data_path, param_dict,\
                                               verbose=True)
        testDataX = abnormalTestData#[[0,1,3,4], :, :]
        testDataY = np.ones(len(abnormalTestData[0]))
        detection_test_idx_list = anomaly_detection(testDataX, testDataY, \
                                               task_name, hmm_data_path, param_dict,\
                                               verbose=True)
        print detection_train_idx_list
        ## sys.exit()
        # -------------------------------------------------------------
        

        
        # code here
        ## test_classifier=cf.classifier()
        X = []
        y = []        
        for trainIdx in xrange(0, abnormalTrainData.shape[1]):
            curr_type = formatY(abnormalTrainIdx[trainIdx], type_sets)
            if curr_type is 0:
                print "hmm"
                continue
            print trainIdx, len(detection_train_idx_list), abnormalTrainData.shape
            limit = getRange(detection_train_idx_list[trainIdx])
            formated_X = formatX(abnormalTrainData[:, trainIdx, :], limit=limit)
            X.extend(formated_X)
            for windowIdx in xrange(0, len(formated_X)):
                y.append(curr_type)
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        mean = np.mean(X, axis=0)
        var = np.std(X, axis=0)
        commands = ''
        class_count = [0.0]*10
        for trainIdx in abnormalTrainIdx:
            class_count[formatY(trainIdx,type_sets)] += 1
            class_count[-1] += 1
        print class_count
        for i in xrange(0, len(type_sets)):
            if not np.allclose(class_count[i+1], 0):
                #if i is not 3 and i is not 2:
                commands = commands + '-w' + str(i+1) + ' ' + str((1 / (class_count[i+1]))) + ' '
                #else:
                #    commands = commands + '-w' + str(i+1) + ' ' + str((2 / (class_count[i+1]))) + ' '
        c = np.logspace(-3.0, -1.0, 4)
        models=[]
        #for i in xrange(0, 4):
            #model = svm.svm_train(y, X_scaled.tolist(), ' -b 1 ' + commands + '-g ' + g[i])
        models.append(svm.svm_train(y, X_scaled.tolist(), ' -b 1 ' + commands))# + '-g ' + str(c[i])))
        for testIdx in xrange(0, abnormalTestData.shape[1]):
            X = []
            y = []
            curr_type = formatY(abnormalTestIdx[testIdx], type_sets)
            if curr_type is 0:
                continue
            limit = getRange(detection_test_idx_list[testIdx])
            formated_X = formatX(abnormalTestData[:, testIdx, :], scaler=scaler)#, limit=limit)
            scaled_data = np.asarray(scaled_inputs(abnormalTestData[:, testIdx, :], scaler))
            X.extend(formated_X)
            for windowIdx in xrange(0, len(formated_X)):
                y.append(curr_type)
            plt.figure(1,figsize=(100,100)).suptitle(reverse_index_d[str(abnormalTestIdx[testIdx])])
            color_list = ['', 'b', 'g', 'r', 'k', 'y']
            title_list = ['', 'sound', 'force', 'face','distance', 'orientation']
            for i in xrange(0, failureData.shape[0]):
                color = color_list[i+1]#int(model.get_labels()[i])]
                title = title_list[i+1]#int(model.get_labels()[i])]
                plt.subplot2grid((failureData.shape[0],2),(int(i), 0)).set_title(title)
                #plt.axis((0, 200, 0, 1))
                plt.axis((0, 200, -2, 2))
                #plt.plot(probability[:,i], color)
                #plt.plot(abnormalTestData[i, testIdx, :], color)
                plt.plot(np.asarray(scaled_data)[i, :], color)
            for j, model in enumerate(models):
                print j
                plt.subplot2grid((failureData.shape[0],2),(j,1), rowspan=failureData.shape[0]).set_title(c[j])
                plt.axis((0, 200, 0, 1))
                prediction =  svm.svm_predict(y, X, model, '-b 1')
                probability = np.asarray(prediction[2])
                for i in xrange(0,len(type_sets)):
                    color = color_list[int(model.get_labels()[i])]
                    if model.get_labels()[i] is curr_type:
                        plt.plot(probability[:,i], color + '>')
                    else:
                        plt.plot(probability[:,i], color + '-')
            print model.get_labels()
            fig_manager = plt.get_current_fig_manager()
            fig_manager.full_screen_toggle()
            plt.show()
            plt.clf()
    
def formatX(data, time_window=20, scaler=None, limit=None):
    X = []
    if limit is not None:
        idxrange = xrange(limit[0], limit[1]-time_window +1)
    else:
        idxrange = xrange(0, data.shape[1]-time_window + 1)
    for idx in idxrange:
        x = []
        for dimIdx in xrange(0, data.shape[0]):
            temp_x = data[dimIdx][idx:idx+time_window][:]
            x.extend(temp_x)
        X.append(x)
    if scaler is not None:
        X_scaled = scaler.transform(X, copy=True)
        X= X_scaled.tolist()
    return X

def formatY(index, type_sets):
    for setId, type_set in enumerate(type_sets):
        if index in type_set:
            return setId + 1
    return 0

def getRange(index):
    if index < 0 or index > 200:
        return "error"
    if index < 30:
        return [0, 40]
    elif index > 190:
        return [160, 200]
    else:
        return [index-30, index+10]

def scaled_inputs(data, scaler):
    #data, of form 4, 200
    new_data = []
    for i in xrange(0, data.shape[0]):
        new_data.append([])
    formated_X = formatX(data, scaler=scaler) #sample data scaled, 200-time_window + 1 by time_window*4
    time_window = np.asarray(formated_X).shape[1] / data.shape[0]
    for idx in xrange(0, len(formated_X)):
        for dimIdx in xrange(0, np.asarray(formated_X).shape[1] / time_window):
            new_data[dimIdx].append(formated_X[idx][dimIdx*time_window])
    return new_data


def anomaly_detection(X, Y, task_name, processed_data_path, param_dict, verbose=False):
    ''' Anomaly detector that return anomalous point on each data.
    '''
    HMM_dict = param_dict['HMM']
    SVM_dict = param_dict['SVM']
    ROC_dict = param_dict['ROC']
    
    # set parameters
    method  = 'hmmgp'
    ## weights = ROC_dict[method+'_param_range']
    weight  = -5.0 # weights[10] # need to select weight!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
      hmm.getHMMinducedFeaturesFromRawCombinedFeatures(ml, X*HMM_dict['scale'], Y, startIdx)

    # Create anomaly classifier
    dtc = cf.classifier( method=method, nPosteriors=nState, nLength=nLength )
    dtc.set_params( class_weight=weight )
    ret = dtc.fit(X_train, Y_train, idx_train, parallel=False)

    # anomaly detection
    detection_idx = []
    for ii in xrange(len(ll_classifier_test_X)):
        if len(ll_classifier_test_Y[ii])==0: continue

        est_y    = dtc.predict(ll_classifier_test_X[ii], y=ll_classifier_test_Y[ii])

        for jj in xrange(len(est_y)):
            if est_y[jj] > 0.0:                
                if ll_classifier_test_Y[ii][0] > 0:
                    detection_idx.append(ll_classifier_test_idx[ii][jj])
                else:
                    detection_idx.append(None)
                break

    return detection_idx


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')

    p.add_option('--task', action='store', dest='task', type='string', default='feeding',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=4,
                 help='type the desired dimension')

    p.add_option('--rawplot', '--rp', action='store_true', dest='bRawDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--interplot', '--ip', action='store_true', dest='bInterpDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--feature', '--ft', action='store_true', dest='bFeaturePlot',
                 default=False, help='Plot features.')
    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
    p.add_option('--anomaly_detection', '--ad', action='store_true', dest='bAD',
                 default=False, help='Plot the change of likelihood.')
                 
    
    p.add_option('--debug', '--dg', action='store_true', dest='bDebug',
                 default=False, help='Set debug mode.')
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    p.add_option('--noplot', '--np', action='store_true', dest='bNoPlot',
                 default=False, help='No Plot.')    
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out.')

    
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    local_range    = 10.0    

    #---------------------------------------------------------------------------
    if opt.task == 'scooping':
        subjects = ['park', 'test'] #'Henry', 
    #---------------------------------------------------------------------------
    elif opt.task == 'feeding':
        #subjects = [ 'unexpected1', 'unexpected2' ]
        subjects = [ 'unexpected3' ]
    else:
        print "Selected task name is not available."
        sys.exit()

    raw_data_path, save_data_path, param_dict_hmm = getParams(opt.task, opt.bDataRenew, \
                                                          False, False, 4,\
                                                          rf_center, local_range)
    _, _, param_dict_isolator = getParams(opt.task, opt.bDataRenew, \
                                                          False, False, 4,\
                                                          rf_center, local_range)



    #---------------------------------------------------------------------------
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+opt.task+'_data_isolation/'+\
      str(param_dict_hmm['data_param']['downSampleSize'])+'_'+str(opt.dim)

    isolation_test(subjects, opt.task, raw_data_path, save_data_path, \
                   param_dict_hmm, param_dict_isolator, verbose=opt.bVerbose, data_renew=opt.bDataRenew)
