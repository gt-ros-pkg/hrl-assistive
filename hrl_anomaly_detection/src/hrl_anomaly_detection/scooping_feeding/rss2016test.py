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
import random

# util
import numpy as np
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
import PyKDL

# learning
from hrl_multimodal_anomaly_detection.hmm import learning_hmm_multi_n as hmm

# visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec



def preprocessData(subject_names, task_name, raw_data_path, processed_data_path, nSet=1, \
                   folding_ratio=0.8, downSampleSize=200,\
                   renew=False, verbose=False):

    # Check if there is already scaled data
    for i in xrange(nSet):        
        target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(i) )                    
        if os.path.isfile(target_file) is not True: renew=True
            
    if renew == False: return        

    success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

    nTrain = int(len(success_list) * folding_ratio)
    nTest  = len(success_list) - nTrain    

    if len(failure_list) < nTest: 
        print "Not enough failure data"
        sys.exit()

    # loading and time-sync
    data_dict = loadData(success_list, isTrainingData=False, downSampleSize=downSampleSize)
    
    data_min = {}
    data_max = {}
    for key in data_dict.keys():
        if 'time' in key: continue
        if data_dict[key] == []: continue
        data_min[key] = np.min(data_dict[key])        
        data_max[key] = np.max(data_dict[key])
        
    for i in xrange(nSet):

        # index selection
        success_idx  = range(len(success_list))
        failure_idx  = range(len(failure_list))
        train_idx    = random.sample(success_idx, nTrain)

        if nTest == 0: 
            success_test_idx = []
            failure_test_idx = []
        else: 
            success_test_idx = [x for x in success_idx if not x in train_idx]
            failure_test_idx = random.sample(failure_idx, nTest)

        # get training data
        trainFileList = [success_list[x] for x in train_idx]
        trainData = loadData(trainFileList, isTrainingData=True, \
                             downSampleSize=downSampleSize)

        # get test data
        if nTest != 0:        
            normalTestFileList = [success_list[x] for x in success_test_idx]
            normalTestData = loadData([success_list[x] for x in success_test_idx], 
                                                          isTrainingData=False, downSampleSize=downSampleSize)
            abnormalTestFileList = [failure_list[x] for x in failure_test_idx]
            abnormalTestData = loadData([failure_list[x] for x in failure_test_idx], \
                                        isTrainingData=False, downSampleSize=downSampleSize)

        # scaling data
        ## trainData_scaled = scaleData(trainData, scale=scale, data_min=data_min, 
        ##                              data_max=data_max, verbose=verbose)
        ## normalTestData_scaled = scaleData(normalTestData, scale=scale, data_min=data_min, 
        ##                                   data_max=data_max, verbose=verbose)
        ## abnormalTestData_scaled = scaleData(abnormalTestData, scale=scale, data_min=data_min, 
        ##                                     data_max=data_max, verbose=verbose)

        # cutting data (only traing and thresTest data)
        ## start_idx = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[0])
        ## end_idx   = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[1])

        ## for j in xrange(len(trainData_scaled)):
        ##     for k in xrange(len(trainData_scaled[j])):
        ##         trainData_scaled[j][k] = trainData_scaled[j][k][start_idx:end_idx]
                
        ## for j in xrange(len(normalTestData_scaled)):
        ##     for k in xrange(len(normalTestData_scaled[j])):                
        ##         normalTestData_scaled[j][k] = normalTestData_scaled[j][k][start_idx:end_idx]
                
        ## for j in xrange(len(abnormalTestData_scaled)):
        ##     for k in xrange(len(abnormalTestData_scaled[j])):                
        ##         abnormalTestData_scaled[j][k] = abnormalTestData_scaled[j][k][start_idx:end_idx]

        # Save data using dictionary
        d = {}
        d['trainData']        = trainData
        d['normalTestData']   = normalTestData
        d['abnormalTestData'] = abnormalTestData

        d['trainFileList']        = trainFileList
        d['normalTestFileList']   = normalTestFileList
        d['abnormalTestFileList'] = abnormalTestFileList        
        
        # Save data using dictionary
        target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(i) )

        try:
            ut.save_pickle(d, target_file)        
        except:
            print "There is already target file: "
        
        

        


def updateMinMax(param_dict, feature_name, feature_array):

    if feature_name in param_dict.keys():
        maxVal = np.max(feature_array)
        minVal = np.min(feature_array)
        if param_dict[feature_name+'_max'] < maxVal:
            param_dict[feature_name+'_max'] = maxVal
        if param_dict[feature_name+'_min'] > minVal:
            param_dict[feature_name+'_min'] = minVal
    else:
        param_dict[feature_name+'_max'] = -100000000000
        param_dict[feature_name+'_min'] =  100000000000
        
    

def likelihoodOfSequences(processed_data_path, task_name, feature_list, local_range, \
                          nSet=0, nState=10, threshold=-1.0, \
                          useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          renew=False, save_pdf=False, show_plot=True):

    target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(nSet) )                    
    if os.path.isfile(target_file) is not True: 
        print "There is no saved data"
        sys.exit()

    data_dict = ut.load_pickle(target_file)

    # training set
    trainingData, param_dict = extractLocalFeature(data_dict['trainData'], feature_list, local_range)

    # test set
    normalTestData, _ = extractLocalFeature(data_dict['normalTestData'], feature_list, local_range, \
                                            param_dict=param_dict)        
    abnormalTestData, _ = extractLocalFeature(data_dict['abnormalTestData'], feature_list, local_range, \
                                              param_dict=param_dict)

    print "======================================"
    print "Training data: ", np.shape(trainingData)
    print "Normal test data: ", np.shape(normalTestData)
    print "Abnormal test data: ", np.shape(abnormalTestData)
    print "======================================"

    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')

    ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, verbose=False)
    ret = ml.fit(trainingData, ml_pkl=detection_param_pkl, use_pkl=not(renew))
    ths = threshold
    
    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)
    
    if show_plot: fig = plt.figure()
    min_logp = 0.0
    max_logp = 0.0
        
    # training data
    if useTrain:

        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(trainingData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(2, len(trainingData[0][i])):

                X = [x[i,:j] for x in trainingData]                
                X_test = ml.convert_sequence(X)
                try:
                    logp = ml.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)
                
            # disp
            if useTrain_color:
                plt.plot(log_ll[i], label=str(i))
                print i, " : ", trainFileList[i], log_ll[i][-1]                
            else:
                plt.plot(log_ll[i], 'b-')

        if useTrain_color: 
            plt.legend(loc=3,prop={'size':16})
            
        ## plt.plot(exp_log_ll[i], 'r-')            
                                             
    # normal test data
    if useNormalTest:

        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(normalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])

            for j in range(2, len(normalTestData[0][i])):
                X = [x[i,:j] for x in normalTestData]                
                X_test = ml.convert_sequence(X)
                try:
                    logp = ml.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                ## exp_logp = ml.expLikelihoods(X_test, ths)
                exp_logp = ml.expLikelihoods(X, ths)
                exp_log_ll[i].append(exp_logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)

            # disp 
            if useNormalTest_color:
                print i, " : ", normalTestFileList[i]                
                plt.plot(log_ll[i], label=str(i))
            else:
                plt.plot(log_ll[i], 'g-')

            plt.plot(exp_log_ll[i], 'r*-')


        if useNormalTest_color: 
            plt.legend(loc=3,prop={'size':16})

    # abnormal test data
    if useAbnormalTest:
        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(abnormalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])

            for j in range(2, len(abnormalTestData[0][i])):
                X = [x[i,:j] for x in abnormalTestData]                
                X_test = ml.convert_sequence(X)
                try:
                    logp = ml.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

            # disp 
            plt.plot(log_ll[i], 'r-')
            ## plt.plot(exp_log_ll[i], 'r*-')


    plt.ylim([min_logp, max_logp])
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        if show_plot: plt.show()        

    return

        

def test(processed_data_path, task_name, nSet, feature_list, local_range, \
         nState=10,\
         renew=False, viz=False):

    for i in xrange(nSet):        
        target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(i) )                    
        if os.path.isfile(target_file) is not True: 
            print "There is no saved data"
            sys.exit()

        data_dict = ut.load_pickle(target_file)
        ## if viz: visualization_raw_data(data_dict)

        # training set
        trainingData, param_dict = extractLocalFeature(data_dict['trainData'], feature_list, local_range)
        
        # test set
        normalTestData, _ = extractLocalFeature(data_dict['normalTestData'], feature_list, local_range, \
                                                param_dict=param_dict)        
        abnormalTestData, _ = extractLocalFeature(data_dict['abnormalTestData'], feature_list, local_range, \
                                                param_dict=param_dict)

        print "======================================"
        print "Training data: ", np.shape(trainingData)
        print "Normal test data: ", np.shape(normalTestData)
        print "Abnormal test data: ", np.shape(abnormalTestData)
        print "======================================"


        if viz: visualization_hmm_data(feature_list, trainingData=trainingData, \
                                       normalTestData=normalTestData,\
                                       abnormalTestData=abnormalTestData)        
        
        # training hmm
        nEmissionDim = len(trainingData)
        detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
                
        ml = hmm.learning_hmm_multi_n(nState, nEmissionDim, verbose=False)

        print "Start to fit hmm"
        ret = ml.fit(trainingData, ml_pkl=detection_param_pkl, use_pkl=renew)

        if ret == 'Failure': 
            print "-------------------------"
            print "HMM returned failure!!   "
            print "-------------------------"
            return (-1,-1,-1,-1)


        tp_l = []
        fn_l = []
        fp_l = []
        tn_l = []
        ths_l = []
        
        # evaluation
        ## threshold_list = -(np.logspace(-1.0, 1.5, nThres, endpoint=True)-1.0 )        
        threshold_list = [-5.0]
        for ths in threshold_list:        
            tp, fn, tn, fp = onlineEvaluation(ml, normalTestData, abnormalTestData, c=ths, 
                                              verbose=True)
            if tp == -1:
                tp_l.append(0)
                fn_l.append(0)
                fp_l.append(0)
                tn_l.append(0)
                ths_l.append(ths)
            else:                       
                tp_l.append(tp)
                fn_l.append(fn)
                fp_l.append(fp)
                tn_l.append(tn)
                ths_l.append(ths)

        dd = {}
        dd['fn_l']    = fn_l
        dd['tn_l']    = tn_l
        dd['tp_l']    = tp_l
        dd['fp_l']    = fp_l
        dd['ths_l']   = ths_l
        print dd        

        
def onlineEvaluation(hmm, normalTestData, abnormalTestData, c=-5, verbose=False):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    # positive is anomaly
    # negative is non-anomaly
    if verbose: print '\nBeginning anomaly testing for test set\n'

    # for normal test data
    if normalTestData != []:    
        for i in xrange(len(normalTestData[0])):
            if verbose: print 'Anomaly Error for test set ', i

            for j in range(20, len(normalTestData[0][i])):
                try:    
                    anomaly, error = hmm.anomaly_check(normalTestData[:][i][:j], c)
                except:
                    print "anomaly_check failed: ", i, j
                    ## return (-1,-1,-1,-1)
                    falsePos += 1
                    break

                if np.isnan(error):
                    print "anomaly check returned nan"
                    falsePos += 1
                    break
                    ## return (-1,-1,-1,-1)

                if verbose: print anomaly, error

                # This is a successful nonanomalous attempt
                if anomaly:
                    falsePos += 1
                    if verbose: print 'Success Test', i,',',j, ' in ',len(normalTestData[0][i]), ' |', anomaly, 
                    error
                    break
                elif j == len(normalTestData[0][i]) - 1:
                    trueNeg += 1
                    break


    # for abnormal test data
    for i in xrange(len(abnormalTestData[0])):
        if verbose: print 'Anomaly Error for test set ', i

        for j in range(20, len(abnormalTestData[0][i])):
            try:                    
                anomaly, error = hmm.anomaly_check(abnormalTestData[:][i][:j], c)
            except:
                truePos += 1
                break

            if verbose: print anomaly, error
                
            if anomaly:
                truePos += 1
                break
            elif j == len(abnormalTestData[0][i]) - 1:
                falseNeg += 1
                if verbose: print 'Failure Test', i,',',j, ' in ',len(abnormalTestData[0][i]), ' |', anomaly, error
                break

    return truePos, falseNeg, trueNeg, falsePos

        

def visualization_hmm_data(feature_list, trainingData=None, normalTestData=None, abnormalTestData=None):

    if trainingData is not None:
        nDimension = len(trainingData)
    elif normalTestData is not None:
        nDimension = len(normalTestData)
    elif abnormalTestData is not None:
        nDimension = len(abnormalTestData)
    else:
        print "no data"
        sys.exit()
        
    fig = plt.figure()            
    # --------------------------------------------------
    for i in xrange(nDimension):
        ax = fig.add_subplot(100*nDimension+10+(i+1))
        if trainingData is not None:
            ax.plot(np.array(trainingData[i]).T, 'b')
        elif normalTestData is not None:
            ax.plot(np.array(normalTestData[i]).T, 'k')
        ## elif abnormalTestData is not None:
        ##     ax.plot(abnormalTestData[i], 'r')

        ax.set_title(feature_list[i])

    fig.savefig('test.pdf')
    fig.savefig('test.png')
    os.system('cp test.p* ~/Dropbox/HRL/')        
    ## plt.show()
    sys.exit()


def visualization_raw_data(data_dict, modality='ft'):

    ## dataList = data_dict['trainData']['ftForceList']
    ## dataList = data_dict['trainData']['kinTargetPosList']
    dataList = data_dict['trainData']['kinEEPosList']
    dataList = data_dict['trainData']['kinEEQuatList']

    fileList = data_dict['trainFileList']
    
    
    ## # Converting data structure
    ## nSample      = len(dataList)
    ## nEmissionDim = len(dataList[0])
    ## features     = []
    ## for i in xrange(nEmissionDim):
    ##     feature  = []

    ##     for j in xrange(nSample):
    ##         feature.append(dataList[j][i,:])

    ##     features.append( feature )

    count = 0
    d_list = []
    f_list = []
    
    for idx, data in enumerate(dataList):

        d_list.append( np.mean(data, axis=0) )
        f_list.append( fileList[idx].split('/')[-1] )

        if idx%10 == 9:
                
            fig = plt.figure()            
            ax1 = fig.add_subplot(111)

            for j, d in enumerate(d_list):
                ax1.plot(d_list[j], label=f_list[j])
                
            plt.legend(loc=3,prop={'size':8})

            fig.savefig('test'+str(count)+'.pdf')
            fig.savefig('test'+str(count)+'.png')
            os.system('cp test'+str(count)+'.p* ~/Dropbox/HRL/')        
            d_list = []
            f_list = []
            count += 1
            

    fig = plt.figure()            
    ax1 = fig.add_subplot(111)

    for j, d in enumerate(d_list):
        ax1.plot(d_list[j], label=f_list[j])

    plt.legend(loc=3,prop={'size':8})

    fig.savefig('test'+str(count)+'.pdf')
    fig.savefig('test'+str(count)+'.png')
    os.system('cp test'+str(count)+'.p* ~/Dropbox/HRL/')        
            
    ## plt.show()
    sys.exit()

    

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()

    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
    
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    

    opt, args = p.parse_args()

    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016'
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    subject = 'gatsbii'
    task    = 'scooping'    
    feature_list = ['unimodal_ftForce', 'crossmodal_targetRelativeDist', \
                    'crossmodal_targetRelativeAng']

    ## subject = 'gatsbii'
    ## task    = 'feeding' 
    ## feature_list = ['unimodal_audioPower', 'unimodal_ftForce', 'crossmodal_artagRelativeDist', \
    ##                 'crossmodal_artagRelativeAng']
    
    preprocessData([subject], task, raw_data_path, save_data_path, renew=opt.bRenew)

    # Dectection TEST 
    nSet         = 1
    local_range  = 0.25    
    viz          = False
    renew        = False

    if opt.bLikelihoodPlot:
        nState    = 10
        threshold = 0.0
        likelihoodOfSequences(save_data_path, task, feature_list, local_range, \
                              nState=nState, threshold=threshold,\
                              useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                              useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                              renew=renew, save_pdf=opt.bSavePdf)
    else:
        test(save_data_path, task, nSet, feature_list, local_range, renew=renew, viz=viz)
