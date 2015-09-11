#!/usr/bin/python

import sys, os, copy
import numpy as np, math
import glob
import socket
import time
import random 

import roslib; roslib.load_manifest('hrl_multimodal_anomaly_detection')
import rospy

# Util
import hrl_lib.util as ut
import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product

#
from util import *
from learning_hmm_multi_4d import *

def distributionOfSequences(task_name, target_path, setID=0, scale=1.0,\
                            useTrain=True, useThsTest=False, useNormalTest=False, useAbnormalTest=False, \
                            useTrain_color=True, useThsTest_color=False, useNormalTest_color=False,\
                            save_pdf=False, show_plot=True, verbose=False):

    # get data
    trainData, thresTestData, normalTestData, abnormalTestData, \
      trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList, \
      trainFileList, thsTestFileList, normalTestFileList, abnormalTestFileList \
    = getData(task_name, target_path, setID)

    if show_plot: fig = plt.figure()
    ax1 = plt.subplot(412)
    ax1.set_ylabel('Force\nMagnitude (N)', fontsize=16)
    ax1.set_xticks(np.arange(0, 25, 5))
    ax1.set_ylim([-scale*0.1, scale*1.1])
    
    ax2 = plt.subplot(411)
    ax2.set_ylabel('Kinematic\nDistance (m)', fontsize=16)
    ax2.set_xticks(np.arange(0, 25, 5))
    ax2.set_ylim([-scale*0.1, scale*1.1])

    ax3 = plt.subplot(414)
    ax3.set_ylabel('Kinematic\nAngle (rad)', fontsize=16)
    ax3.set_xlabel('Time (sec)', fontsize=16)
    ax3.set_xticks(np.arange(0, 25, 5))
    ax3.set_ylim([-scale*0.1, scale*1.1])

    ax4 = plt.subplot(413)
    ax4.set_ylabel('Audio\nMagnitude (dec)', fontsize=16)
    ax4.set_xticks(np.arange(0, 25, 5))
    ax4.set_ylim([-scale*0.1, scale*1.1])

    # training data
    if useTrain:
    
        count = 0
        for i in xrange(len(trainData[0])):
            
            #if count > 5: continue
            if useTrain_color:
                ax1.plot(trainTimeList[i], trainData[0][i])
                ax2.plot(trainTimeList[i], trainData[1][i])
                ax3.plot(trainTimeList[i], trainData[2][i])
                ax4.plot(trainTimeList[i], trainData[3][i], label=str(count))            
            else:
                ax1.plot(trainTimeList[i], trainData[0][i], 'b')
                ax2.plot(trainTimeList[i], trainData[1][i], 'b')
                ax3.plot(trainTimeList[i], trainData[2][i], 'b')
                ax4.plot(trainTimeList[i], trainData[3][i], 'b')            
            count = count + 1
            if verbose: print i, trainFileList[i]

        if useTrain_color: ax4.legend(loc=3,prop={'size':16})

    # threshold-test data
    if useThsTest:
    
        count = 0
        for i in xrange(len(thresTestData[0])):
            
            if useThsTest_color:
                ax1.plot(thresTestTimeList[i], thresTestData[0][i])
                ax2.plot(thresTestTimeList[i], thresTestData[1][i])
                ax3.plot(thresTestTimeList[i], thresTestData[2][i])
                ax4.plot(thresTestTimeList[i], thresTestData[3][i], label=str(count))            
            else:
                ax1.plot(thresTestTimeList[i], thresTestData[0][i], 'k--')
                ax2.plot(thresTestTimeList[i], thresTestData[1][i], 'k--')
                ax3.plot(thresTestTimeList[i], thresTestData[2][i], 'k--')
                ax4.plot(thresTestTimeList[i], thresTestData[3][i], 'k--')            
                
            count = count + 1
            if count > 8: break
            if verbose: print i, trainFileList[i]
        if useThsTest_color: ax4.legend(loc=3,prop={'size':16})

    # normal test data
    if useNormalTest:
    
        count = 0
        for i in xrange(len(normalTestData[0])):
            ax1.plot(normalTestTimeList[i], normalTestData[0][i], 'g--')
            ax2.plot(normalTestTimeList[i], normalTestData[1][i], 'g--')
            ax3.plot(normalTestTimeList[i], normalTestData[2][i], 'g--')
            ax4.plot(normalTestTimeList[i], normalTestData[3][i], 'g--')
            count = count + 1

    # normal test data
    if useAbnormalTest:
    
        count = 0
        for i in xrange(len(abnormalTestData[0])):
            ax1.plot(abnormalTestTimeList[i], abnormalTestData[0][i],'m')
            ax2.plot(abnormalTestTimeList[i], abnormalTestData[1][i],'m')
            ax3.plot(abnormalTestTimeList[i], abnormalTestData[2][i],'m')
            ax4.plot(abnormalTestTimeList[i], abnormalTestData[3][i],'m')
            count = count + 1
                           
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
        ##os.system('scp test.p* dpark@brain:~/Dropbox/HRL/')
    else:
        if show_plot: plt.show()        


def plotTestSequences(test_subject_names, task_name, data_root_path, data_target_path, setID=0, \
                      scale=1.0, downSampleSize=200,\
                      useTrain=True, useThsTest=False, useNormalTest=False, useAbnormalTest=False, \
                      useTrain_color=False,
                      save_pdf=False, verbose=False):

    fig = plt.figure()
    distributionOfSequences(task_name, data_target_path, setID=0, scale=scale, \
                            useTrain=useTrain, useThsTest=useThsTest, useNormalTest=useNormalTest, \
                            useAbnormalTest=useAbnormalTest, useTrain_color=useTrain_color,\
                            save_pdf=False, show_plot=False, verbose=verbose)        

    # Check if there is already scaled data
    target_file = os.path.join(data_target_path, task_name+'_dataSet_'+str(setID) )        
    if os.path.isfile(target_file) is not True: 
        print "Missing data: ", setID
        return

    print "file: ", target_file
    d = ut.load_pickle(target_file)
    minVals = d['minVals'] 
    maxVals = d['maxVals'] 

    ax1 = plt.subplot(412)
    ax2 = plt.subplot(411)
    ax3 = plt.subplot(414)
    ax4 = plt.subplot(413)
 
    success_list, failure_list = getSubjectFileList(data_root_path, test_subject_names, task_name)

    ## success_list = []
    ## fileName = "/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/onlineDataRecordings/ash_b_09-04-2015_09-45-31.pkl"
    ## failure_list = [fileName] 
   
    if len(success_list)>0:
        testData, testTimeList = loadData(success_list, isTrainingData=False, 
                                                downSampleSize=downSampleSize)
        testData_scaled,_ ,_  = scaleData(testData, scale=scale, minVals=minVals, 
                                          maxVals=maxVals, verbose=verbose)

        for i in xrange(len(testData[0])):
            ax1.plot(testTimeList[i], testData_scaled[0][i],'m')
            ax2.plot(testTimeList[i], testData_scaled[1][i],'m')
            ax3.plot(testTimeList[i], testData_scaled[2][i],'m')
            ax4.plot(testTimeList[i], testData_scaled[3][i],'m')            
        
    if len(failure_list)>0 and True:
        testData, testTimeList = loadData(failure_list, isTrainingData=False, 
                                          downSampleSize=downSampleSize)

        ## d = ut.load_pickle(failure_list[0])
        ## testData=[[d['forcesRaw'], d['distancesRaw'], d['anglesRaw'], d['audioRaw']]]
        ## testTimeList = [d['times']]
                
        testData_scaled,_ ,_  = scaleData(testData, scale=scale, minVals=minVals, 
                                          maxVals=maxVals, verbose=verbose)

        ## ax1.plot(testTimeList[0], testData_scaled[0][0],'m')
        ## ax2.plot(testTimeList[0], testData_scaled[0][1],'m')
        ## ax3.plot(testTimeList[0], testData_scaled[0][2],'m')
        ## ax4.plot(testTimeList[0], testData_scaled[0][3],'m')            
        
        for i in xrange(len(testData[0])):
            ax1.plot(testTimeList[i], testData_scaled[0][i],'m')
            ax2.plot(testTimeList[i], testData_scaled[1][i],'m')
            ax3.plot(testTimeList[i], testData_scaled[2][i],'m')
            ax4.plot(testTimeList[i], testData_scaled[3][i],'m')            

    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        
    
        
def evaluation(task_name, target_path, nSet=1, nState=20, cov_mult=5.0, anomaly_offset=0.0, \
               crossEvalID=None, check_method='progress', hmm_renew=False, verbose=False):

    tot_truePos = 0
    tot_falseNeg = 0
    tot_trueNeg = 0 
    tot_falsePos = 0
    
    # Check if there is already scaled data
    for i in xrange(nSet):        

        trainData, thresTestData, normalTestData, abnormalTestData, \
          trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList, \
          trainFileList, thsTestFileList, normalTestFileList, abnormalTestFileList \
          = getData(task_name, target_path, i, crossEvalID)


        if crossEvalID is None:
            dynamic_thres_pkl = os.path.join(target_path, "ml_"+task_name+"_"+str(i)+".pkl")
        else:
            dynamic_thres_pkl = os.path.join(target_path, "ml_"+task_name+"_"+str(i)+'_eval_'+str(crossEvalID)+\
                                             ".pkl")

        ## print dynamic_thres_pkl
        nDimension = len(trainData)

        # Create and train multivariate HMM
        hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=nDimension, anomaly_offset=anomaly_offset, \
                                    check_method=check_method, verbose=False)
        ret = hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],\
                      ml_pkl=dynamic_thres_pkl, use_pkl=(not hmm_renew), cov_mult=[cov_mult]*16)

        if ret == 'Failure': 
            print "-------------------------"
            print "HMM returned failure!!   "
            print "-------------------------"
            return (-1,-1,-1,-1)
                      

        minThresholds = None                  
        if hmm_renew:
            minThresholds1 = tuneSensitivityGain(hmm, trainData, method=check_method, verbose=verbose)
            minThresholds2 = tuneSensitivityGain(hmm, thresTestData, method=check_method, verbose=verbose)
            minThresholds = minThresholds2

            if type(minThresholds) == list or type(minThresholds) == np.ndarray:
                for i in xrange(len(minThresholds1)):
                    if minThresholds1[i] < minThresholds2[i]:
                        minThresholds[i] = minThresholds1[i]
            else:
                if minThresholds1 < minThresholds2:
                    minThresholds = minThresholds1

            d = ut.load_pickle(dynamic_thres_pkl)
            if d is None: d = {}
            d['minThresholds'] = minThresholds                
            ut.save_pickle(d, dynamic_thres_pkl)                
        else:
            d = ut.load_pickle(dynamic_thres_pkl)
            minThresholds = d['minThresholds']
            

        truePos, falseNeg, trueNeg, falsePos = \
        tableOfConfusionOnline(hmm, normalTestData, abnormalTestData, c=minThresholds, verbose=verbose)
        if truePos == -1: 
            print "Error with task ", task_name
            print "Error with nSet ", i
            print "Error with crossEval ID: ", crossEvalID
            return (-1,-1,-1,-1)

        tot_truePos += truePos
        tot_falseNeg += falseNeg
        tot_trueNeg += trueNeg 
        tot_falsePos += falsePos

    truePositiveRate = float(tot_truePos) / float(tot_truePos + tot_falseNeg) * 100.0
    if tot_trueNeg == 0 and tot_falsePos == 0:
        trueNegativeRate = "not available"
    else:
        trueNegativeRate = float(tot_trueNeg) / float(tot_trueNeg + tot_falsePos) * 100.0
    print "------------------------------------------------"
    print "Total set of data: ", nSet
    print "------------------------------------------------"
    print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate
    print "------------------------------------------------"

    return (tot_truePos, tot_falseNeg, tot_trueNeg, tot_falsePos)


def evaluation_all(subject_names, task_name, check_methods, data_root_path, data_target_path, nSet=1,\
                   nState=20, scale=1.0, \
                   cov_mult=5., folding_ratio=[0.6, 0.2, 0.2], downSampleSize=200, \
                   cutting_ratio=[0.0, 0.65], anomaly_offset=0.0,\
                   data_renew=False, hmm_renew=False, save_pdf=False, bPlot=False, verbose=False):

    # For parallel computing
    strMachine = socket.gethostname()+"_"+str(os.getpid())    

    count = 0
    for method in check_methods:        

        # Check the existance of workspace
        method_path = os.path.join(data_target_path, method)
        if os.path.isdir(method_path) == False:
            os.system('mkdir -p '+method_path)

        for idx, subject_name in enumerate(subject_names):

            print method, " : ", subject_name        

            ## For parallel computing
            # save file name
            res_file = task_name+'_'+subject_name+'_'+method+'.pkl'
            mutex_file_part = 'running_'+task_name+'_'+subject_name+'_'+method

            res_file = os.path.join(method_path, res_file)
            mutex_file_full = mutex_file_part+'_'+strMachine+'.txt'
            mutex_file      = os.path.join(method_path, mutex_file_full)

            if os.path.isfile(res_file): 
                count += 1            
                continue
            elif hcu.is_file(method_path, mutex_file_part): 
                continue
            ## elif os.path.isfile(mutex_file): continue
            os.system('touch '+mutex_file)
            
            ## Data pre-processing
            preprocessData([subject_name], task_name, data_root_path, data_target_path, nSet=nSet, 
                           scale=scale,\
                           folding_ratio=folding_ratio, downSampleSize=downSampleSize, \
                           train_cutting_ratio=cutting_ratio, full_abnormal_test=True,\
                           crossEvalID=idx, verbose=True)

            # Run evaluation
            (truePos, falseNeg, trueNeg, falsePos)\
              = evaluation(task_name, data_target_path, nSet=nSet, nState=nState, cov_mult=cov_mult,\
                           anomaly_offset=anomaly_offset, crossEvalID=idx, check_method=method,\
                           hmm_renew=True, verbose=True)

            truePositiveRate = float(truePos) / float(truePos + falseNeg) * 100.0
            if trueNeg == 0 and falsePos == 0:            
                trueNegativeRate = "Not available"
            else:
                trueNegativeRate = float(trueNeg) / float(trueNeg + falsePos) * 100.0
                
            print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate
                           
            if truePos!=-1 :                 
                d = {}
                d['subject'] = subject_name
                d['tp'] = truePos
                d['fn'] = falseNeg
                d['tn'] = trueNeg
                d['fp'] = falsePos
                d['nSet'] = nSet

                try:
                    ut.save_pickle(d,res_file)        
                except:
                    print "There is already the targeted pkl file"
            else:
                target_file = os.path.join(data_target_path, task_name+'_dataSet_%d_eval_'+str(idx) ) 
                for j in xrange(nSet):
                    os.system('rm '+target_file % j)
                

            os.system('rm '+mutex_file)
            print "-----------------------------------------------"

            if truePos==-1: 
                print "truePos is -1"
                sys.exit()

            
    if count == len(check_methods)*len(subject_names):
        print "#############################################################################"
        print "All file exist ", count
        print "#############################################################################"        
    else:
        return
            

    if bPlot:

        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        tableau20 = np.array(tableau20)/255.0
        width = 0.5
        methods = ('Change \n detection', 'Fixed threshold \n detection', \
                   'Fixed threshold \n & change detection', \
                   'Dynamic threshold \n detection')
        tp_mean = []
        tp_std = []

        for i, method in enumerate(check_methods):        
            
            method_path = os.path.join(data_target_path, method)
            
            tot_truePos = 0
            tot_falseNeg = 0
            tot_trueNeg = 0 
            tot_falsePos = 0

            fdr_l = []

            for j, subject_name in enumerate(subject_names):

                # save file name
                res_file = task_name+'_'+subject_name+'_'+method+'.pkl'
                res_file = os.path.join(method_path, res_file)
                d = ut.load_pickle(res_file)

                subject_name = d['subject']
                truePos  = d['tp']
                falseNeg = d['fn']
                trueNeg  = d['tn']
                falsePos = d['fp']
                nSet     = d['nSet']
                            
                # Sum up evaluatoin result
                tot_truePos += truePos
                tot_falseNeg += falseNeg
                tot_trueNeg += trueNeg
                tot_falsePos += falsePos

                fdr_l.append( float(truePos) / float(truePos + falseNeg) * 100.0 )

            truePositiveRate = float(tot_truePos) / float(tot_truePos + tot_falseNeg) * 100.0
            if trueNeg == 0 and falsePos == 0:            
                trueNegativeRate = "Not available"
            else:
                trueNegativeRate = float(tot_trueNeg) / float(tot_trueNeg + tot_falsePos) * 100.0
            print "------------------------------------------------"
            print "Method: ", method
            print "------------------------------------------------"
            print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate
            print "------------------------------------------------"

            tp_mean.append( np.mean(fdr_l) )
            tp_std.append( np.std( fdr_l ))
            

        
        fig = pp.figure()       
            
        ind = np.arange(len(check_methods))+1           
        pp.bar(ind+width/4.0, tp_mean, width, color=[tableau20[0],tableau20[2],tableau20[4],tableau20[6]], yerr=tp_std)
                
        pp.ylim([0.0, 100.0])
        pp.ylabel('Detection Rate [%]', fontsize=16)    
        pp.xticks(ind + width*3.0/4, methods )

        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')
        else:
            pp.show()
            

def getData(task_name, target_path, setID=0, crossEvalID=None):
    print "start to getting data"
    
    # Check if there is already scaled data
    if crossEvalID is None:
        target_file = os.path.join(target_path, task_name+'_dataSet_'+str(setID) )        
    else:
        target_file = os.path.join(target_path, task_name+'_dataSet_'+str(setID)+'_eval_'+str(crossEvalID) )
        
    if os.path.isfile(target_file) is not True: 
        print "Missing data: ", setID
        return

    print "file: ", target_file
    d = ut.load_pickle(target_file)
    trainData            = d['trainData']
    thresTestData        = d['thresTestData']
    normalTestData       = d['normalTestData'] 
    abnormalTestData     = d['abnormalTestData']
    trainTimeList        = d['trainTimeList'] 
    thresTestTimeList    = d['thresTestTimeList'] 
    normalTestTimeList   = d['normalTestTimeList'] 
    abnormalTestTimeList = d['abnormalTestTimeList'] 

    trainFileList        = d['trainFileList'] 
    thsTestFileList      = d['thsTestFileList'] 
    normalTestFileList   = d['normalTestFileList'] 
    abnormalTestFileList = d['abnormalTestFileList'] 
    
    print "Load complete"
    return [trainData, thresTestData, normalTestData, abnormalTestData,\
      trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList,\
      trainFileList, thsTestFileList, normalTestFileList, abnormalTestFileList]


def likelihoodOfSequences(task_name, target_path, setID=0, \
                          nState=20, cov_mult=5.0, anomaly_offset=0.0,\
                          useTrain=True, useThsTest=True, useNormalTest=True, useAbnormalTest=True,\
                          useTrain_color=False, useThsTest_color=False, useNormalTest_color=True,\
                          hmm_renew=False, save_pdf=False, show_plot=True, verbose=False):

    # get data
    trainData, thresTestData, normalTestData, abnormalTestData, \
      trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList, \
      trainFileList, thsTestFileList, normalTestFileList, abnormalTestFileList \
    = getData(task_name, target_path, setID)

    dynamic_thres_pkl = os.path.join(target_path, "ml_"+task_name+"_"+str(setID)+".pkl")

    nDimension = len(trainData)

    # Create and train multivariate HMM
    hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=nDimension, anomaly_offset=anomaly_offset, verbose=False)
    ret = hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],\
                  ml_pkl=dynamic_thres_pkl, use_pkl=(not hmm_renew), cov_mult=[cov_mult]*16)

    minThresholds = None                  
    if hmm_renew:
        minThresholds1 = tuneSensitivityGain(hmm, trainData, verbose=verbose)
        minThresholds2 = tuneSensitivityGain(hmm, thresTestData, verbose=verbose)
        minThresholds = minThresholds2
        for i in xrange(len(minThresholds1)):
            if minThresholds1[i] < minThresholds2[i]:
                minThresholds[i] = minThresholds1[i]
        d = ut.load_pickle(dynamic_thres_pkl)
        d['minThresholds'] = minThresholds                
        ut.save_pickle(d, dynamic_thres_pkl)                
    else:
        d = ut.load_pickle(dynamic_thres_pkl)
        minThresholds = d['minThresholds']
        
    min_logp = 0.0
    max_logp = 0.0
    if show_plot: fig = plt.figure()

    # training data
    if useTrain:

        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(trainData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(2, len(trainData[0][i])):
                X_test = hmm.convert_sequence(trainData[0][i][:j], trainData[1][i][:j], 
                                              trainData[2][i][:j], trainData[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                ## exp_logp = hmm.expLikelihoods(trainData[0][i][:j], trainData[1][i][:j], 
                ##                               trainData[2][i][:j], trainData[3][i][:j],
                ##                               minThresholds)
                ## exp_log_ll[i].append(exp_logp)

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
            
          
    # threshold-test data
    if useThsTest:

        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(thresTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(2, len(thresTestData[0][i])):
                X_test = hmm.convert_sequence(thresTestData[0][i][:j], thresTestData[1][i][:j], 
                                              thresTestData[2][i][:j], thresTestData[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                ## exp_logp = hmm.expLikelihoods(thresTestData[0][i][:j], thresTestData[1][i][:j], 
                ##                               thresTestData[2][i][:j], thresTestData[3][i][:j],
                ##                               minThresholds)
                ## exp_log_ll[i].append(exp_logp)


            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)
                
            # disp 
            if useThsTest_color:
                print i, " : ", thsTestFileList[i], log_ll[i][-1]
                plt.plot(log_ll[i], label=str(i))
            else:
                plt.plot(log_ll[i], 'k-')

        if useThsTest_color: 
            plt.legend(loc=3,prop={'size':16})

            ## plt.plot(log_ll[i], 'b-')
                        

    # normal test data
    if useNormalTest:

        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(normalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])

            for j in range(2, len(normalTestData[0][i])):
                X_test = hmm.convert_sequence(normalTestData[0][i][:j], normalTestData[1][i][:j], 
                                              normalTestData[2][i][:j], normalTestData[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                exp_logp = hmm.expLikelihoods(normalTestData[0][i][:j], normalTestData[1][i][:j], 
                                              normalTestData[2][i][:j], normalTestData[3][i][:j],
                                              minThresholds)
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
                X_test = hmm.convert_sequence(abnormalTestData[0][i][:j], abnormalTestData[1][i][:j], 
                                              abnormalTestData[2][i][:j], abnormalTestData[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                ## exp_logp = hmm.expLikelihoods(abnormalTestData[0][i][:j], abnormalTestData[1][i][:j], 
                ##                               abnormalTestData[2][i][:j], abnormalTestData[3][i][:j],
                ##                               minThresholds)
                ## exp_log_ll[i].append(exp_logp)
                
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

    return hmm

def plotTestLikelihoodSequences(test_subject_names, task_name, data_root_path, data_target_path, setID=0, \
                                scale=1.0, downSampleSize=200, nState=10, cov_mult=5.0, \
                                anomaly_offset= 0, hmm_renew=True,\
                                useTrain=True, useThsTest=False, useNormalTest=False, useAbnormalTest=False, \
                                save_pdf=False, verbose=False):

    fig = plt.figure()
    hmm = likelihoodOfSequences(task_name, data_target_path, setID=setID, nState=nState, cov_mult=cov_mult,\
                                anomaly_offset=anomaly_offset,\
                                useTrain=useTrain, useThsTest=useThsTest, useNormalTest=useNormalTest, \
                                useAbnormalTest=useAbnormalTest,\
                                useTrain_color=False, useThsTest_color=False, useNormalTest_color=False,\
                                hmm_renew=hmm_renew, save_pdf=False, show_plot=False, verbose=True)       

    # Check if there is already scaled data
    target_file = os.path.join(data_target_path, task_name+'_dataSet_'+str(setID) )        
    if os.path.isfile(target_file) is not True: 
        print "Missing data: ", setID
        return

    print "file: ", target_file
    d = ut.load_pickle(target_file)
    minVals = d['minVals'] 
    maxVals = d['maxVals'] 

 
    success_list, failure_list = getSubjectFileList(data_root_path, test_subject_names, task_name)
    
    if len(success_list)>0:
        testData, testTimeList = loadData(success_list, isTrainingData=False, 
                                                downSampleSize=downSampleSize)
        testData_scaled,_ ,_  = scaleData(testData, scale=scale, minVals=minVals, 
                                          maxVals=maxVals, verbose=verbose)

        log_ll = []
        for i in xrange(len(testData[0])):

            log_ll.append([])
            for j in range(2, len(testData_scaled[0][i])):
                X_test = hmm.convert_sequence(testData_scaled[0][i][:j], testData_scaled[1][i][:j], 
                                              testData_scaled[2][i][:j], testData_scaled[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

            plt.plot(log_ll[i], 'm-')
                
        
    if len(failure_list)>0 and False:
        testData, testTimeList = loadData(failure_list, isTrainingData=False, 
                                          downSampleSize=downSampleSize)
        testData_scaled,_ ,_  = scaleData(testData, scale=scale, minVals=minVals, 
                                          maxVals=maxVals, verbose=verbose)

        log_ll = []
        for i in xrange(len(testData[0])):

            log_ll.append([])
            for j in range(2, len(testData_scaled[0][i])):
                X_test = hmm.convert_sequence(testData_scaled[0][i][:j], testData_scaled[1][i][:j], 
                                              testData_scaled[2][i][:j], testData_scaled[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

            plt.plot(log_ll[i], 'm--')


        
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        
        

def preprocessData(subject_names, task_name, root_path, target_path, nSet=1, \
                   folding_ratio=[0.6, 0.2, 0.2], 
                   scale=1.0, downSampleSize=200, train_cutting_ratio=[0.0, 0.65], \
                   full_abnormal_test=False,\
                   crossEvalID=None, test_subject_name=None,\
                   renew=False, verbose=False):


    # Check if there is already scaled data
    for i in xrange(nSet):        
        if crossEvalID is None:
            target_file = os.path.join(target_path, task_name+'_dataSet_'+str(i) )        
        else:
            target_file = os.path.join(target_path, task_name+'_dataSet_'+str(i)+'_eval_'+str(crossEvalID) ) 
            
        if os.path.isfile(target_file) is not True: renew=True
            
    if renew == False: return        

    success_list, failure_list = getSubjectFileList(root_path, subject_names, task_name)
    
    # random training, threshold-test, test set selection
    nTrain   = int(len(success_list) * folding_ratio[0])

    if folding_ratio[2] == 0.0:
        nThsTest = len(success_list) - nTrain
        nTest    = 0
    else:
        nThsTest = int(len(success_list) * folding_ratio[1])
        nTest    = len(success_list) - nTrain - nThsTest

    if len(failure_list) < nTest: 
        print "Not enough failure data"
        sys.exit()

    # minimum and maximum vales for scaling
    dataList, _ = loadData(success_list, isTrainingData=False, downSampleSize=downSampleSize)
    minVals = []
    maxVals = []
    for modality in dataList:
        minVals.append(np.min(modality))
        maxVals.append(np.max(modality))
        
    for i in xrange(nSet):

        # index selection
        success_idx  = range(len(success_list))
        failure_idx  = range(len(failure_list))
        train_idx    = random.sample(success_idx, nTrain)
        ths_test_idx = random.sample([x for x in success_idx if not x in train_idx], nThsTest)

        if nTest == 0: 
            success_test_idx = []
            failure_test_idx = []
        else: 
            success_test_idx = [x for x in success_idx if not (x in train_idx or x in ths_test_idx)]
            failure_test_idx = random.sample(failure_idx, nTest)

        if full_abnormal_test: failure_test_idx = failure_idx #temp

        # get training data
        trainFileList = [success_list[x] for x in train_idx]
        trainData, trainTimeList = loadData(trainFileList, isTrainingData=True, downSampleSize=downSampleSize)

        # get threshold-test data
        thsTestFileList = [success_list[x] for x in ths_test_idx]
        thresTestData, thresTestTimeList = loadData([success_list[x] for x in ths_test_idx], 
                                                    isTrainingData=True, downSampleSize=downSampleSize)

        # get test data
        if nTest != 0:        
            normalTestFileList = [success_list[x] for x in success_test_idx]
            normalTestData, normalTestTimeList = loadData([success_list[x] for x in success_test_idx], 
                                                      isTrainingData=False, downSampleSize=downSampleSize)
            abnormalTestFileList = [failure_list[x] for x in failure_test_idx]
            abnormalTestData, abnormalTestTimeList \
            = loadData([failure_list[x] for x in failure_test_idx], \
                       isTrainingData=False, downSampleSize=downSampleSize)

        elif crossEvalID is not None:

            if False:
                normalTestFileList, abnormalTestFileList\
                   = getSubjectFileList(root_path, test_subject_name, task_name)

                normalTestData, normalTestTimeList = loadData(normalTestFileList, 
                                                              isTrainingData=False, downSampleSize=downSampleSize)
                abnormalTestData, abnormalTestTimeList \
                = loadData(abnormalTestFileList, \
                           isTrainingData=False, downSampleSize=downSampleSize)            

            else:
                normalTestFileList = []
                normalTestData =[]
                normalTestTimeList = []
                    
                abnormalTestFileList = failure_list
                abnormalTestData, abnormalTestTimeList \
                = loadData(abnormalTestFileList, \
                           isTrainingData=False, downSampleSize=downSampleSize)            

        else:
            print "no test folding ratio and cross evaluation"
            sys.exit()


            
        # scaling data
        trainData_scaled,_ ,_  = scaleData(trainData, scale=scale, minVals=minVals, 
                                                 maxVals=maxVals, verbose=verbose)
        thresTestData_scaled,_ ,_ = scaleData(thresTestData, scale=scale, minVals=minVals, maxVals=maxVals, 
                                            verbose=verbose)
        normalTestData_scaled,_ ,_ = scaleData(normalTestData, scale=scale, minVals=minVals, 
                                               maxVals=maxVals, verbose=verbose)
        abnormalTestData_scaled,_ ,_ = scaleData(abnormalTestData, scale=scale, minVals=minVals, 
                                                 maxVals=maxVals, verbose=verbose)

        # cutting data (only traing and thresTest data)
        start_idx = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[0])
        end_idx   = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[1])

        for j in xrange(len(trainData_scaled)):
            for k in xrange(len(trainData_scaled[j])):
                trainData_scaled[j][k] = trainData_scaled[j][k][start_idx:end_idx]
                trainTimeList[k]       = trainTimeList[k][start_idx:end_idx]
                
        for j in xrange(len(thresTestData_scaled)):
            for k in xrange(len(thresTestData_scaled[j])):                
                thresTestData_scaled[j][k] = thresTestData_scaled[j][k][start_idx:end_idx]
                thresTestTimeList[k]       = thresTestTimeList[k][start_idx:end_idx]

        for j in xrange(len(normalTestData_scaled)):
            for k in xrange(len(normalTestData_scaled[j])):                
                normalTestData_scaled[j][k] = normalTestData_scaled[j][k][start_idx:end_idx]
                normalTestTimeList[k]       = normalTestTimeList[k][start_idx:end_idx]
                
        for j in xrange(len(abnormalTestData_scaled)):
            for k in xrange(len(abnormalTestData_scaled[j])):                
                abnormalTestData_scaled[j][k] = abnormalTestData_scaled[j][k][start_idx:end_idx]
                abnormalTestTimeList[k]       = abnormalTestTimeList[k][start_idx:end_idx]
            
        # Save data using dictionary
        d = {}
        d['trainData'] = trainData_scaled
        d['thresTestData'] = thresTestData_scaled
        d['normalTestData'] = normalTestData_scaled
        d['abnormalTestData'] = abnormalTestData_scaled
        d['trainTimeList'] = trainTimeList
        d['thresTestTimeList'] = thresTestTimeList
        d['normalTestTimeList'] = normalTestTimeList
        d['abnormalTestTimeList'] = abnormalTestTimeList

        d['trainFileList'] = trainFileList
        d['thsTestFileList'] = thsTestFileList
        d['normalTestFileList'] = normalTestFileList
        d['abnormalTestFileList'] = abnormalTestFileList 

        d['minVals'] = minVals
        d['maxVals'] = maxVals

        if crossEvalID is None:
            target_file = os.path.join(target_path, task_name+'_dataSet_'+str(i) )
        else:
            target_file = os.path.join(target_path, task_name+'_dataSet_'+str(i)+'_eval_'+str(crossEvalID) )
        
        try:
            ut.save_pickle(d, target_file)        
        except:
            print "There is already target file: "
        
    
    return 


def scaleData(dataList, scale=10, minVals=None, maxVals=None, verbose=False):

    if dataList == []: return [], [], []
    
    # Determine max and min values
    if minVals is None:
        minVals = []
        maxVals = []
        for modality in dataList:
            minVals.append(np.min(modality))
            maxVals.append(np.max(modality))
        if verbose:
            print 'minValues', minVals
            print 'maxValues', maxVals

    nDimension = len(dataList)
    dataList_scaled = []
    for i in xrange(nDimension):
        dataList_scaled.append([])

    # Scale features
    for i in xrange(nDimension):
        if i==3: new_scale=scale #*0.2
        else: new_scale = scale
        
        for j in xrange(len(dataList[i])):
            dataList_scaled[i].append( scaling( dataList[i][j], minVals[i], maxVals[i], new_scale).tolist() )
            
    return dataList_scaled, minVals, maxVals


def tableOfConfusionOnline(hmm, normalTestData, abnormalTestData, c=-5, verbose=False):
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

            for j in range(6, len(normalTestData[0][i])):
                try:    
                    anomaly, error = hmm.anomaly_check(normalTestData[0][i][:j], 
                                                   normalTestData[1][i][:j], 
                                                   normalTestData[2][i][:j],
                                                   normalTestData[3][i][:j], c)
                except:
                    print "anomaly_check failed: ", i, j
                    return (-1,-1,-1,-1)

                if np.isnan(error):
                    print "anomaly check returned nan"
                    return (-1,-1,-1,-1)

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

        for j in range(6, len(abnormalTestData[0][i])):
            anomaly, error = hmm.anomaly_check(abnormalTestData[0][i][:j], 
                                               abnormalTestData[1][i][:j], 
                                               abnormalTestData[2][i][:j],
                                               abnormalTestData[3][i][:j], c)

            if verbose: print anomaly, error
                
            if anomaly:
                truePos += 1
                break
            elif j == len(abnormalTestData[0][i]) - 1:
                falseNeg += 1
                if verbose: print 'Failure Test', i,',',j, ' in ',len(abnormalTestData[0][i]), ' |', anomaly, error
                break

    ## try:
    ##     truePositiveRate = float(truePos) / float(truePos + falseNeg) * 100.0
    ##     trueNegativeRate = float(trueNeg) / float(trueNeg + falsePos) * 100.0
    ## except:
    ##     print np.shape(normalTestData)
    ##     print np.shape(abnormalTestData)
    ##     print truePos, falseNeg, trueNeg, falsePos
    ##     sys.exit()
        
    return truePos, falseNeg, trueNeg, falsePos
    

def crossEvaluation(subject_names, task_name, data_root_path, data_target_path, \
                    folding_ratio=[0.6, 0.2, 0.2], scale=1.0, downSampleSize=200,\
                    train_cutting_ratio=[0.0, 0.65],\
                    nSet=1, nState=20, cov_mult=5.0, anomaly_offset=0.0,\
                    data_renew=False, hmm_renew=False, verbose=False):

    # Set training and test id list (leave-one-out??)
    training_names = []
    test_names = []
    for idx, test_name in enumerate(subject_names):
        test_names.append(test_name)
        training_name = copy.deepcopy(subject_names)
        del training_name[idx]
        training_names.append(training_name)

    # over fitting
    ## for idx, test_name in enumerate(subject_names):
    ##     test_names.append([test_name])
    ##     training_names.append([test_name])
        
    tot_truePos = 0
    tot_falseNeg = 0
    tot_trueNeg = 0 
    tot_falsePos = 0
    
    # Get data
    for idx in xrange(len(test_names)):
        tr_names = training_names[idx]
        t_name   = test_names[idx]

        print idx, " : ", tr_names, t_name
        
        preprocessData(tr_names, task_name, data_root_path, data_target_path, nSet=nSet, scale=scale,\
                       folding_ratio=folding_ratio, downSampleSize=downSampleSize, \
                       train_cutting_ratio=cutting_ratio, full_abnormal_test=False,\
                       crossEvalID=idx, test_subject_name=t_name,\
                       renew=data_renew, verbose=verbose)

        # Run evaluation
        (truePos, falseNeg, trueNeg, falsePos)\
          = evaluation(task_name, data_target_path, nSet=nSet, nState=nState, cov_mult=cov_mult, \
                       anomaly_offset=anomaly_offset, \
                       crossEvalID=idx, 
                       hmm_renew=hmm_renew, verbose=verbose)

        if truePos == -1:
            print tr_names, t_name
        
        # Sum up evaluatoin result
        tot_truePos += truePos
        tot_falseNeg += falseNeg
        tot_trueNeg += trueNeg
        tot_falsePos += falsePos
        
    
    truePositiveRate = float(tot_truePos) / float(tot_truePos + tot_falseNeg) * 100.0
    trueNegativeRate = float(tot_trueNeg) / float(tot_trueNeg + tot_falsePos) * 100.0
    print "------------------------------------------------"
    print "Total set of data: ", len(test_names)
    print "------------------------------------------------"
    print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate
    print "------------------------------------------------"

    return


def fig_roc(subject_names, task_name, check_methods, data_root_path, data_target_path, nDataSet,\
            nState=20, scale=1.0, threshold_mult=[3.0],\
            cov_mult=5., downSampleSize=200, \
            cutting_ratio=[0.0, 0.65], anomaly_offset=0.0, check_dims=[4],\
            data_renew=False, hmm_renew=False, save_pdf=False, bPlot=False, bAllPlot=False, verbose=False):

    # For parallel computing
    strMachine = socket.gethostname()+"_"+str(os.getpid())    
    
    count = 0
    for subject_name in subject_names:                               
    
        threshold_list = None
        for i in xrange(nDataSet):

            # Load data
            target_file = os.path.join(data_target_path, task_name+'_'+subject_name+'dataSet_'+str(i)+
                                       '_kfold.pkl' ) 
            d = ut.load_pickle(target_file)

            true_train_data = d['trainData']
            true_test_data  = d['normalTestData']
            false_test_data = d['abnormalTestData']


            # load data
            for method in check_methods:        

                # Check the existance of workspace
                method_path = os.path.join(data_target_path, method)
                if os.path.isdir(method_path) == False:
                    os.system('mkdir -p '+method_path)

                for check_dim in check_dims:

                    ## For parallel computing
                    # save file name
                    res_file = task_name+'_'+subject_name+'_dim_'+str(check_dim)+'.pkl'
                    mutex_file_part = 'running_'+task_name+'_'+subject_name+'_dim_'+str(check_dim)

                    res_file = os.path.join(method_path, res_file)
                    mutex_file_full = mutex_file_part+'_'+strMachine+'.txt'
                    mutex_file      = os.path.join(method_path, mutex_file_full)

                    if os.path.isfile(res_file): 
                        count += 1            
                        continue
                    elif hcu.is_file(method_path, mutex_file_part): 
                        continue
                    os.system('touch '+mutex_file)


                    ## print dynamic_thres_pkl
                    nDimension = len(true_train_data)
                    if method == 'globalChange':
                        threshold_list = product(threshold_mult, threshold_mult)
                    else:
                        threshold_list = threshold_mult


                    # Create and train multivariate HMM
                    hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=nDimension, 
                                                anomaly_offset=anomaly_offset, \
                                                check_method=method, verbose=False)
                    ret = hmm.fit(xData1=true_train_data[0], xData2=true_train_data[1],\
                                  xData3=true_train_data[2], xData4=true_train_data[3],\
                                  use_pkl=False, cov_mult=[cov_mult]*16)

                    if ret == 'Failure': 
                        print "-------------------------"
                        print "HMM returned failure!!   "
                        print "-------------------------"
                        os.system('rm '+mutex_file)                    
                        return (-1,-1,-1,-1)


                    tp_l = []
                    fn_l = []
                    fp_l = []
                    tn_l = []
                    ths_l = []

                    for ths in threshold_list:

                        tp, fn, tn, fp = \
                          tableOfConfusionOnline(hmm, true_test_data, false_test_data, c=ths, 
                                                 verbose=verbose)

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


                    d = {}
                    d['fn_l']    = fn_l
                    d['tn_l']    = tn_l
                    d['tp_l']    = tp_l
                    d['fp_l']    = fp_l
                    d['ths_l']   = ths_l

                    try:
                        ut.save_pickle(d,res_file)        
                    except:
                        print "There is the targeted pkl file"

                    os.system('rm '+mutex_file)
                    print "-----------------------------------------------"

    if count == len(check_methods)*nDataSet*len(check_dims)*len(subject_names):
        print "#############################################################################"
        print "All file exist ", count
        print "#############################################################################"        
    else:
        return




    if len(check_methods) >= len(check_dims): nClass = len(check_methods)
    else: nClass = len(check_dims)
    
    if bPlot:
        for subject_name in subject_names:                               

            import itertools
            colors = itertools.cycle(['g', 'm', 'c', 'k'])
            shapes = itertools.cycle(['x','v', 'o', '+'])

            fig = pp.figure()
            
            for n in range(nClass):

                if len(check_methods) >= len(check_dims): 
                    method = check_methods[n]
                    check_dim = check_dims[0]
                else: 
                    method = check_methods[0]
                    check_dim = check_dims[n]

                # Check the existance of workspace
                method_path = os.path.join(data_target_path, method)

                if method == 'globalChange':
                    threshold_list = list(product(threshold_mult, threshold_mult))
                else:
                    threshold_list = threshold_mult

                fn_l = np.zeros(len(threshold_list))
                tp_l = np.zeros(len(threshold_list))
                tn_l = np.zeros(len(threshold_list))
                fp_l = np.zeros(len(threshold_list))

                for i in xrange(nDataSet):

                    res_file = task_name+'_'+subject_name+'_dim_'+str(check_dim)+'.pkl'
                    res_file = os.path.join(method_path, res_file)

                    d = ut.load_pickle(res_file)
                    fn_l += np.array(d['fn_l']); tp_l += np.array(d['tp_l']) 
                    tn_l += np.array(d['tn_l']); fp_l += np.array(d['fp_l'])


                tpr_l = np.zeros(len(threshold_list))
                fpr_l = np.zeros(len(threshold_list))

                for i in xrange(len(threshold_list)):
                    if tp_l[i]+fn_l[i] != 0:
                        tpr_l[i] = tp_l[i]/(tp_l[i]+fn_l[i])*100.0

                    if fp_l[i]+tn_l[i] != 0:
                        fpr_l[i] = fp_l[i]/(fp_l[i]+tn_l[i])*100.0

                sum_l = tpr_l+fpr_l 
                idx_list = sorted(range(len(sum_l)), key=lambda k: sum_l[k])
                sorted_tpr_l   = np.array([tpr_l[k] for k in idx_list])
                sorted_fpr_l   = np.array([fpr_l[k] for k in idx_list])


                color = colors.next()
                shape = shapes.next()

                if method == 'globalChange':
                    label = 'Fixed threshold & \n change detection'
                elif method == 'change':
                    label = 'Change detection'
                elif method == 'global':
                    label = 'Fixed threshold \n detection'
                elif method == 'progress':
                    label = 'Dynamic threshold \n detection'
                else:
                    label = method +"_"+str(check_dim)

                pp.plot(sorted_fpr_l, sorted_tpr_l, '-'+shape+color, label=label, mec=color, ms=8, mew=2)

            pp.xlim([-1, 101])
            pp.ylim([-1, 101])        
            pp.xlabel('False Positive Rate (Percentage)', fontsize=16)
            pp.ylabel('True Positive Rate (Percentage)', fontsize=16)    
            pp.legend(loc=4,prop={'size':16})


            if save_pdf:
                fig.savefig('test.pdf')
                fig.savefig('test.png')
                os.system('cp test.p* ~/Dropbox/HRL/')
            else:
                pp.show()


    elif bAllPlot:

        import itertools
        colors = itertools.cycle(['g', 'm', 'c', 'k'])
        shapes = itertools.cycle(['x','v', 'o', '+'])
        
        for n in range(nClass):

            if len(check_methods) >= len(check_dims): 
                method = check_methods[n]
                check_dim = check_dims[0]
            else: 
                method = check_methods[0]
                check_dim = check_dims[n]

            # Check the existance of workspace
            method_path = os.path.join(data_target_path, method)
                
            if method == 'globalChange':
                threshold_list = list(product(threshold_mult, threshold_mult))
            else:
                threshold_list = threshold_mult
                
            fn_l = np.zeros(len(threshold_list))
            tp_l = np.zeros(len(threshold_list))
            tn_l = np.zeros(len(threshold_list))
            fp_l = np.zeros(len(threshold_list))

            for subject_name in subject_names:
                for i in xrange(nDataSet):

                    res_file = task_name+'_'+subject_name+'_dim_'+str(check_dim)+'.pkl'
                    res_file = os.path.join(method_path, res_file)

                    d = ut.load_pickle(res_file)
                    fn_l += np.array(d['fn_l']); tp_l += np.array(d['tp_l']) 
                    tn_l += np.array(d['tn_l']); fp_l += np.array(d['fp_l'])


            tpr_l = np.zeros(len(threshold_list))
            fpr_l = np.zeros(len(threshold_list))

            for i in xrange(len(threshold_list)):
                if tp_l[i]+fn_l[i] != 0:
                    tpr_l[i] = tp_l[i]/(tp_l[i]+fn_l[i])*100.0

                if fp_l[i]+tn_l[i] != 0:
                    fpr_l[i] = fp_l[i]/(fp_l[i]+tn_l[i])*100.0

            sum_l = tpr_l+fpr_l 
            idx_list = sorted(range(len(sum_l)), key=lambda k: sum_l[k])
            sorted_tpr_l   = np.array([tpr_l[k] for k in idx_list])
            sorted_fpr_l   = np.array([fpr_l[k] for k in idx_list])


            color = colors.next()
            shape = shapes.next()

            if method == 'globalChange':
                label = 'Fixed threshold & \n change detection'
            elif method == 'change':
                label = 'Change detection'
            elif method == 'global':
                label = 'Fixed threshold \n detection'
            elif method == 'progress':
                label = 'Dynamic threshold \n detection'
            else:
                label = method +"_"+str(check_dim)

            pp.plot(sorted_fpr_l, sorted_tpr_l, '-'+shape+color, label=label, mec=color, ms=8, mew=2)

        pp.xlim([-1, 101])
        pp.ylim([-1, 101])        
        pp.xlabel('False Positive Rate (Percentage)', fontsize=16)
        pp.ylabel('True Positive Rate (Percentage)', fontsize=16)    
        pp.legend(loc=4,prop={'size':16})


        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')
        else:
            pp.show()

        
    ##     print "aaaaaaaaaaaaaaaaaaaaaaa"
            

def kFoldPreprocessData(subject_name, task_name, root_path, target_path, \
                        kFold=3, 
                        scale=1.0, downSampleSize=200, train_cutting_ratio=[0.0, 0.65], \
                        renew=False, verbose=False):


    # Check if there is already scaled data
    for i in xrange(kFold*kFold):        
        target_file = os.path.join(target_path, task_name+'_'+subject_name+'dataSet_'+str(i)+'_kfold.pkl' ) 
        if os.path.isfile(target_file) is not True: renew=True
            
    if renew == False: 
        return kFold*kFold       

    success_list, failure_list = getSubjectFileList(root_path, [subject_name], task_name)

    # Load all data
    true_dataList, _ = loadData(success_list, isTrainingData=True, downSampleSize=downSampleSize)
    false_dataList, _ = loadData(failure_list, isTrainingData=False, downSampleSize=downSampleSize)            


    # cutting data (only traing and thresTest data)
    start_idx = int(float(len(true_dataList[0][0]))*train_cutting_ratio[0])
    end_idx   = int(float(len(true_dataList[0][0]))*train_cutting_ratio[1])

    for j in xrange(len(true_dataList)):
        for k in xrange(len(true_dataList[j])):
            true_dataList[j][k] = true_dataList[j][k][start_idx:end_idx]
            
    for j in xrange(len(false_dataList)):
        for k in xrange(len(false_dataList[j])):
            false_dataList[j][k] = false_dataList[j][k][start_idx:end_idx]

    
    # minimum and maximum values for scaling
    minVals = []
    maxVals = []
    for modality in true_dataList:
        minVals.append(np.min(modality))
        maxVals.append(np.max(modality))

    # scaling data
    true_dataList_scaled,_ ,_  = scaleData(true_dataList, scale=scale, minVals=minVals, 
                                       maxVals=maxVals, verbose=verbose)
    false_dataList_scaled,_ ,_  = scaleData(false_dataList, scale=scale, minVals=minVals, 
                                            maxVals=maxVals, verbose=verbose)
        
    nTrueSequence = len(true_dataList[0])
    nFalseSequence = len(false_dataList[0])

    from sklearn import cross_validation
    idx_list = range(nTrueSequence)
    true_kf  = cross_validation.KFold(nTrueSequence,n_folds=kFold, shuffle=True)

    count = 0
    for true_train_index, true_test_index in true_kf:
        false_kf = cross_validation.KFold(nFalseSequence,n_folds=kFold, shuffle=True)
        for _, false_test_index in false_kf:

            true_train_data = [[],[],[],[]]
            true_train_data[0] = [true_dataList_scaled[0][x] for x in true_train_index]
            true_train_data[1] = [true_dataList_scaled[1][x] for x in true_train_index]
            true_train_data[2] = [true_dataList_scaled[2][x] for x in true_train_index]
            true_train_data[3] = [true_dataList_scaled[3][x] for x in true_train_index]

            true_test_data = [[],[],[],[]]
            true_test_data[0] = [true_dataList_scaled[0][x] for x in true_test_index]
            true_test_data[1] = [true_dataList_scaled[1][x] for x in true_test_index]
            true_test_data[2] = [true_dataList_scaled[2][x] for x in true_test_index]
            true_test_data[3] = [true_dataList_scaled[3][x] for x in true_test_index]

            false_test_data = [[],[],[],[]]
            false_test_data[0] = [false_dataList_scaled[0][x] for x in false_test_index]
            false_test_data[1] = [false_dataList_scaled[1][x] for x in false_test_index]
            false_test_data[2] = [false_dataList_scaled[2][x] for x in false_test_index]
            false_test_data[3] = [false_dataList_scaled[3][x] for x in false_test_index]
                
                
            # Save data using dictionary
            d = {}
            d['trainData'] = true_train_data
            d['thresTestData'] = []
            d['normalTestData'] = true_test_data
            d['abnormalTestData'] = false_test_data
            d['trainTimeList'] = []
            d['thresTestTimeList'] = []
            d['normalTestTimeList'] = []
            d['abnormalTestTimeList'] = []

            d['trainFileList'] = []
            d['thsTestFileList'] = []
            d['normalTestFileList'] = []
            d['abnormalTestFileList'] = []

            d['minVals'] = minVals
            d['maxVals'] = maxVals

            target_file = os.path.join(target_path, task_name+'_'+subject_name+'dataSet_'+str(count)+'_kfold.pkl' ) 
            ut.save_pickle(d, target_file)        
            count += 1
    
    return count
    


    
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print descriptions.')
    p.add_option('--plot', '--p', action='store_true', dest='bPlot',
                 default=False, help='Plot distribution of data.')
    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
    p.add_option('--plotTest', '--pt', action='store_true', dest='bPlotTest',
                 default=False, help='Plot the test data.')
    p.add_option('--plotTestLikelihood', '--ptl', action='store_true', dest='bPlotTestLikelihood',
                 default=False, help='Plot the likelihoods of test data.')
    p.add_option('--eval', '--e', action='store_true', dest='bEvaluation',
                 default=False, help='Evaluate each subject data.')
    p.add_option('--roc_online_method_check', '--ronmthd', action='store_true', \
                 dest='bRocOnlineMethodCheck',
                 default=False, help='Plot online ROC by real anomaly')    
    p.add_option('--allplot', '--all', action='store_true', dest='bAllPlot',
                 default=False, help='Plot all data.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')
    opt, args = p.parse_args()

    ## data_root_path   = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings'    
    data_root_path = '/home/dpark/svn/robot1/src/projects/anomaly/feeding'
    data_target_path = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/hmm/data'

    # Scooping
    ## subject_names  = ['pr2'] #'personal', 
    ## task_name      = 'scooping'
    ## nSet           = 1
    ## folding_ratio  = [0.5, 0.3, 0.2]
    ## downSampleSize = 100
    ## nState         = 10
    ## cov_mult       = 5.0
    ## scale          = 1.0
    ## cutting_ratio  = [0.0, 0.9]
    ## anomaly_offset = -20.0

    # Feeding
    ## subject_names  = ['s2','s3','s4'] #'personal', 's3',
    subject_names  = ['s11']
    task_name      = 'feeding' #['scooping', 'feeding']
    nSet           = 1
    folding_ratio  = [0.5, 0.3, 0.2]
    downSampleSize = 100
    nState         = 10
    cov_mult       = 5.0
    scale          = 1.0
    cutting_ratio  = [0.0, 0.7] #[0.0, 0.7]
    anomaly_offset = -20.0

    if not opt.bEvaluation:
        preprocessData(subject_names, task_name, data_root_path, data_target_path, nSet=nSet, scale=scale,\
                       folding_ratio=folding_ratio, downSampleSize=downSampleSize, \
                       train_cutting_ratio=cutting_ratio, full_abnormal_test=True,\
                       renew=opt.bDataRenew, verbose=opt.bVerbose)
                       
    # ------------------------------------------- TEST ------------------------------------------
    if opt.bPlot:
        distributionOfSequences(task_name, data_target_path, setID=0, scale=scale, \
                                useTrain=True, useThsTest=True, useNormalTest=True, useAbnormalTest=False,\
                                useTrain_color=True, useThsTest_color=False,\
                                save_pdf=opt.bSavePdf, verbose=True)        
    elif opt.bLikelihoodPlot:
        if opt.bDataRenew == True: opt.bHMMRenew=True
        likelihoodOfSequences(task_name, data_target_path, setID=0, nState=nState, cov_mult=cov_mult,\
                              anomaly_offset=anomaly_offset,\
                              useTrain=True, useThsTest=True, useNormalTest=False, useAbnormalTest=False,\
                              useTrain_color=False, useThsTest_color=True, useNormalTest_color=False,\
                              hmm_renew=opt.bHMMRenew, save_pdf=opt.bSavePdf, verbose=True)       
    elif opt.bPlotTest:
        test_subject_names=['personal4']            
        plotTestSequences(test_subject_names, task_name, data_root_path, data_target_path, \
                          scale=scale, downSampleSize=downSampleSize, \
                          useTrain=True, useThsTest=False, useNormalTest=False, useAbnormalTest=False,\
                          save_pdf=opt.bSavePdf, verbose=True)
    elif opt.bPlotTestLikelihood:
        test_subject_names=['personal4']            
        plotTestLikelihoodSequences(test_subject_names, task_name, data_root_path, data_target_path, \
                                    scale=scale, downSampleSize=downSampleSize, nState=nState, cov_mult=cov_mult,
                                    anomaly_offset=anomaly_offset, hmm_renew=opt.bHMMRenew, \
                                    useTrain=True, useThsTest=True, useNormalTest=True, \
                                    useAbnormalTest=False,\
                                    save_pdf=opt.bSavePdf, verbose=True)                
    elif opt.bEvaluation:
        subject_names  = ['s2','s4','s8','s9','s10','s11']       
        check_methods  = ['change', 'global', 'globalChange', 'progress']        
        ## subject_names  = ['s2']       
        ## check_methods  = ['change']        
        data_root_path = '/home/dpark/svn/robot1/src/projects/anomaly/feeding'
        data_target_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/ICRA2016'
        nSet = 10
        anomaly_offset = 0.0 #only for progress?
        folding_ratio  = [0.5, 0.5, 0.0]
        
        if opt.bDataRenew == True: opt.bHMMRenew=True        
        ## folding_ratio  = [0.5, 0.5, 0.0]
        ## crossEvaluation(subject_names, task_name, data_root_path, data_target_path, \
        ##                 folding_ratio=folding_ratio, downSampleSize=downSampleSize, \
        ##                 train_cutting_ratio=cutting_ratio,\
        ##                 nSet=nSet, nState=nState, cov_mult=cov_mult, anomaly_offset=anomaly_offset,
        ##                 data_renew=opt.bDataRenew, hmm_renew=opt.bHMMRenew, verbose=False)

        evaluation_all(subject_names, task_name, check_methods, data_root_path, data_target_path, \
                       nSet=nSet, nState=nState, scale=scale, \
                       cov_mult=cov_mult, folding_ratio=folding_ratio, downSampleSize=downSampleSize, \
                       cutting_ratio=cutting_ratio, anomaly_offset=anomaly_offset,\
                       data_renew = opt.bDataRenew, hmm_renew = opt.bHMMRenew, \
                       save_pdf=True, bPlot=True, verbose=False)


    elif opt.bRocOnlineMethodCheck:
        subject_names  = ['s2','s4','s8','s9','s10','s11']       
        check_methods  = ['change', 'global', 'globalChange', 'progress']        
        data_root_path = '/home/dpark/svn/robot1/src/projects/anomaly/feeding'
        data_target_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/ICRA2016'
        kFold = 3
        anomaly_offset = -50.0 #only for progress?
        threshold_mult = -1.0*(np.logspace(-1.0, 2.5, 30, endpoint=True) -2.0)
        nDataSet = None

        # data preprocessing and splitting
        for i, subject_name in enumerate(subject_names):
            nDataSet = kFoldPreprocessData(subject_name, task_name, data_root_path, data_target_path, \
                                           kFold=kFold,\
                                           scale=scale, downSampleSize=downSampleSize, \
                                           train_cutting_ratio=cutting_ratio,\
                                           verbose=False)
                                           
        print "kFoldPreprocee finished...."
        
        fig_roc(subject_names, task_name, check_methods, data_root_path, data_target_path, 
                nDataSet=nDataSet,\
                nState=nState, scale=scale, threshold_mult=threshold_mult,\
                cov_mult=cov_mult, downSampleSize=downSampleSize, \
                cutting_ratio=cutting_ratio, anomaly_offset=anomaly_offset,\
                data_renew = opt.bDataRenew, hmm_renew = opt.bHMMRenew, \
                save_pdf=False, bPlot=False, bAllPlot=opt.bAllPlot, verbose=False)

    else:            
        if opt.bDataRenew == True: opt.bHMMRenew=True
        evaluation(task_name, data_target_path, nSet=nSet, nState=nState, cov_mult=cov_mult,\
                   anomaly_offset=anomaly_offset,\
                   hmm_renew = opt.bHMMRenew, verbose=False)
