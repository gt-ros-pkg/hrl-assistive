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



def vizLikelihoods(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                   decision_boundary_viz=False, method='progress',\
                   useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                   useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                   data_renew=False, hmm_renew=False, save_pdf=False, verbose=False, dd=None,\
                   nSubSample=None):

    from hrl_anomaly_detection import data_manager as dm
    from hrl_anomaly_detection.hmm import learning_hmm as hmm

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # SVM
    
    #------------------------------------------
    if dd is None:        
        dd = dm.getDataSet(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], \
                           data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], \
                           scale=1.0,\
                           ae_data=False,\
                           handFeatures=data_dict['handFeatures'], \
                           cut_data=data_dict['cut_data'],\
                           data_renew=data_dict['renew'], max_time=data_dict.get('max_time', None))

    successData = dd['successData'] * HMM_dict['scale']
    failureData = dd['failureData'] * HMM_dict['scale']
                           

    normalTestData = None                                    
    print "======================================"
    print "Success data: ", np.shape(successData)
    ## print "Normal test data: ", np.shape(normalTestData)
    print "Failure data: ", np.shape(failureData)
    print "======================================"

    if 'kFoldList' in dd.keys():
        kFold_list = dd['kFoldList']        
    else:
        kFold_list = dm.kFold_data_index2(len(successData[0]),\
                                          len(failureData[0]),\
                                          data_dict['nNormalFold'], data_dict['nAbnormalFold'] )
    
    normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx = kFold_list[-1]
    normalTrainData   = successData[:, normalTrainIdx, :] 
    abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
    normalTestData    = successData[:, normalTestIdx, :] 
    abnormalTestData  = failureData[:, abnormalTestIdx, :] 
    
    # training hmm
    nEmissionDim = len(normalTrainData)
    ## hmm_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')    
    cov_mult = [cov]*(nEmissionDim**2)

    # generative model
    ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose)
    if data_dict['handFeatures_noise']:
        ret = ml.fit(normalTrainData+\
                     np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*HMM_dict['scale'], \
                     cov_mult=cov_mult, ml_pkl=None, use_pkl=False) # not(renew))
    else:
        ret = ml.fit(normalTrainData, cov_mult=cov_mult, ml_pkl=None, use_pkl=False) # not(renew))
    if ret == 'Failure': sys.exit()
        
    ## ths = threshold
    startIdx = 4

    if decision_boundary_viz:
        import hrl_anomaly_detection.classifiers.classifier as cf
        
        testDataX = normalTrainData
        testDataY = -np.ones(len(normalTrainData[0]))

        if not (method == 'hmmgp'):
            nSubSample = None

        ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
          hmm.getHMMinducedFeaturesFromRawCombinedFeatures(ml, testDataX, testDataY, startIdx, \
                                                           add_logp_d=False, \
                                                           cov_type='full', nSubSample=nSubSample)
    
        # flatten the data
        X_train_org, Y_train_org, idx_train_org = dm.flattenSample(ll_classifier_train_X, \
                                                                   ll_classifier_train_Y, \
                                                                   ll_classifier_train_idx)

        if method.find('svm')>=0 or method.find('sgd')>=0 or method.find('svr')>=0:
            scaler = preprocessing.StandardScaler()
            X_scaled = scaler.fit_transform(X_train_org)
        else:
            X_scaled = X_train_org
                                                                   
        # discriminative classifier
        dtc = cf.classifier( method=method, nPosteriors=nState, \
                             nLength=len(normalTestData[0,0]), ths_mult=-10.0 )
        dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=True)


    print "----------------------------------------------------------------------------"
    ## fig = plt.figure()
    min_logp = 0.0
    max_logp = 0.0
    target_idx = 1

    # training data
    if useTrain:
        ## normalTrainData = np.array(normalTrainData)[:,0:3,:]
        
        testDataY = -np.ones(len(normalTrainData[0]))
        
        ll_X, ll_Y, _ = \
          hmm.getHMMinducedFeaturesFromRawCombinedFeatures(ml, normalTrainData, testDataY, startIdx, \
                                                           add_logp_d=False, \
                                                           cov_type='full')

        exp_log_ll = []        
        for i in xrange(len(ll_X)):

            l_logp = np.array(ll_X)[i,:,0]
            l_post = np.array(ll_X)[i,:,-nState]
            
            if decision_boundary_viz: # and i==target_idx:
                fig = plt.figure()

            # disp
            if useTrain_color: plt.plot(l_logp, label=str(i))
            else: plt.plot(l_logp, 'b-', linewidth=4.0, alpha=0.6 )

            if min_logp > np.amin(l_logp): min_logp = np.amin(l_logp)
            if max_logp < np.amax(l_logp): max_logp = np.amax(l_logp)
            
            if decision_boundary_viz: # and i==target_idx:
                l_exp_logp = dtc.predict(ll_X[i]) + l_logp
                plt.plot(l_exp_logp, 'm-', lw=3.0)
                ## break
                plt.show()        

        if useTrain_color: 
            plt.legend(loc=3,prop={'size':16})
            
            
    # normal test data
    if useNormalTest:

        log_ll = []
        ## exp_log_ll = []        
        for i in xrange(len(normalTestData[0])):

            log_ll.append([])
            ## exp_log_ll.append([])
            for j in range(startIdx, len(normalTestData[0][i])):
                X = [x[i,:j] for x in normalTestData] # by dim
                logp = ml.loglikelihood(X)
                log_ll[i].append(logp)

                ## exp_logp, logp = ml.expLoglikelihood(X, ths, bLoglikelihood=True)
                ## log_ll[i].append(logp)
                ## exp_log_ll[i].append(exp_logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)

            # disp 
            if useNormalTest_color: plt.plot(log_ll[i], label=str(i))
            else: plt.plot(log_ll[i], 'k-')

            ## plt.plot(exp_log_ll[i], 'r*-')

        if useNormalTest_color: 
            plt.legend(loc=3,prop={'size':16})

    # abnormal test data
    if useAbnormalTest:
        log_ll = []
        exp_log_ll = []        
        for i in xrange(len(abnormalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])

            for j in range(startIdx, len(abnormalTestData[0][i])):
                X = [x[i,:j] for x in abnormalTestData]                
                try:
                    logp = ml.loglikelihood(X)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                ## if decision_boundary_viz and i==target_idx:
                ##     if j>=len(ll_logp[i]): continue
                ##     l_X = [ll_logp[i][j]] + ll_post[i][j].tolist()
                ##     exp_logp = dtc.predict(l_X)[0] + ll_logp[i][j]
                ##     exp_log_ll[i].append(exp_logp)


            # disp
            plt.plot(log_ll[i], 'r-')
            ## plt.plot(exp_log_ll[i], 'r*-')
        plt.plot(log_ll[target_idx], 'k-', lw=3.0)            


    ## plt.ylim([min_logp, max_logp])
    if max_logp >0: plt.ylim([0, max_logp])
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        

    return

def vizStatePath(ll_post, nState, time_list=None, single=False, save_pdf=False, step_idx=None):

    m = len(ll_post) # sample
    n = len(ll_post[0]) # length

    if time_list is None:
        time_list = range(n)
    
    path_mat  = np.zeros((nState, n))

    if single:
        for i in xrange(m):
            path_mat = np.array(ll_post[i]).T

            path_mat -= np.amin(path_mat, axis=0)
            path_mat /= np.sum(path_mat, axis=0)
            extent = [time_list[0],time_list[-1],nState,0]
            ## xticks = time_list #[time_list[0], time_list[n/2], time_list[-1]]

            fig, ax1 = plt.subplots(figsize=(10, 8))
            plt.rc('text', usetex=True)

            ## ax1 = plt.subplot(111)            
            im  = ax1.imshow(path_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper', 
                             extent=extent, aspect=0.15)

            if step_idx is not None:
                t = time_list[step_idx[i]]
                plt.plot([t,t],[0,nState], 'b-', linewidth=3.0)

            ## plt.colorbar(im, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
            ## plt.xticks(xticks, fontsize=12)
            ax1.set_xlabel("Time [sec]", fontsize=22)
            ax1.set_ylabel("Hidden State Index", fontsize=22)
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.show()

    else:
        for i in xrange(m):
            path_mat += np.array(ll_post[i]).T

        path_mat /= np.sum(path_mat, axis=0)
        extent = [0,time_list[-1],nState,1]

        fig = plt.figure()
        plt.rc('text', usetex=True)

        ax1 = plt.subplot(111)            
        im  = ax1.imshow(path_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper', 
                         extent=extent, aspect=7.0)
    
        plt.colorbar(im, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
        ax1.set_xlabel("Time [sec]", fontsize=18)
        ax1.set_ylabel("Hidden State Index", fontsize=18)
        

    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        
    return




def data_plot(subject_names, task_name, raw_data_path, processed_data_path, \
              downSampleSize=200, \
              local_range=0.3, rf_center='kinEEPos', global_data=False, \
              success_viz=True, failure_viz=False, \
              raw_viz=False, interp_viz=False, save_pdf=False, \
              successData=False, failureData=True,\
              continuousPlot=False, \
              ## trainingData=True, normalTestData=False, abnormalTestData=False,\
              modality_list=['audio'], data_renew=False, max_time=None, verbose=False):    

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    success_list, failure_list = util.getSubjectFileList(raw_data_path, subject_names, task_name)

    fig = plt.figure('all')
    time_lim    = [0.01, 0] 
    nPlot       = len(modality_list)

    for idx, file_list in enumerate([success_list, failure_list]):
        if idx == 0 and successData is not True: continue
        elif idx == 1 and failureData is not True: continue        

        ## fig = plt.figure('loadData')                        
        # loading and time-sync
        if idx == 0:
            if verbose: print "Load success data"
            data_pkl = os.path.join(processed_data_path, task_name+'_success_'+rf_center+\
                                    '_'+str(local_range))
            raw_data_dict, interp_data_dict = util.loadData(success_list, isTrainingData=True,
                                                       downSampleSize=downSampleSize,\
                                                       local_range=local_range, rf_center=rf_center,\
                                                       global_data=global_data, \
                                                       renew=data_renew, save_pkl=data_pkl, \
                                                       max_time=max_time, verbose=verbose)
        else:
            if verbose: print "Load failure data"
            data_pkl = os.path.join(processed_data_path, task_name+'_failure_'+rf_center+\
                                    '_'+str(local_range))
            raw_data_dict, interp_data_dict = util.loadData(failure_list, isTrainingData=False,
                                                       downSampleSize=downSampleSize,\
                                                       local_range=local_range, rf_center=rf_center,\
                                                       global_data=global_data,\
                                                       renew=data_renew, save_pkl=data_pkl, \
                                                       max_time=max_time, verbose=verbose)
            
        ## plt.show()
        ## sys.exit()
        if raw_viz: target_dict = raw_data_dict
        else: target_dict = interp_data_dict

        # check only training data to get time limit (TEMP)
        if idx == 0:
            for key in interp_data_dict.keys():
                if key.find('timesList')>=0:
                    time_list = interp_data_dict[key]
                    if len(time_list)==0: continue
                    for tl in time_list:
                        ## print tl[-1]
                        time_lim[-1] = max(time_lim[-1], tl[-1])
            ## continue

        # for each file in success or failure set
        for fidx in xrange(len(file_list)):
                        
            count = 0
            for modality in modality_list:
                count +=1

                if 'audioWrist' in modality:
                    time_list = target_dict['audioWristTimesList']
                    data_list = target_dict['audioWristRMSList']
                    
                elif 'audio' in modality:
                    time_list = target_dict['audioTimesList']
                    data_list = target_dict['audioPowerList']

                elif 'kinematics' in modality:
                    time_list = target_dict['kinTimesList']
                    data_list = target_dict['kinPosList']

                    # distance
                    new_data_list = []
                    for d in data_list:
                        new_data_list.append( np.linalg.norm(d, axis=0) )
                    data_list = new_data_list

                elif 'ft' in modality:
                    time_list = target_dict['ftTimesList']
                    data_list = target_dict['ftForceList']

                    # distance
                    if len(np.shape(data_list[0])) > 1:
                        new_data_list = []
                        for d in data_list:
                            new_data_list.append( np.linalg.norm(d, axis=0) )
                        data_list = new_data_list

                elif 'vision_artag' in modality:
                    time_list = target_dict['visionArtagTimesList']
                    data_list = target_dict['visionArtagPosList']

                    # distance
                    new_data_list = []
                    for d in data_list:                    
                        new_data_list.append( np.linalg.norm(d[:3], axis=0) )
                    data_list = new_data_list

                elif 'vision_landmark' in modality:
                    time_list = target_dict['visionLandmarkTimesList']
                    data_list = target_dict['visionLandmarkPosList']

                    # distance
                    new_data_list = []
                    for d in data_list:                    
                        new_data_list.append( np.linalg.norm(d[:3], axis=0) )
                    data_list = new_data_list

                elif 'vision_change' in modality:
                    time_list = target_dict['visionChangeTimesList']
                    data_list = target_dict['visionChangeMagList']

                elif 'pps' in modality:
                    time_list = target_dict['ppsTimesList']
                    data_list1 = target_dict['ppsLeftList']
                    data_list2 = target_dict['ppsRightList']

                    # magnitude
                    new_data_list = []
                    for i in xrange(len(data_list1)):
                        d1 = np.array(data_list1[i])
                        d2 = np.array(data_list2[i])
                        d = np.vstack([d1, d2])
                        new_data_list.append( np.sum(d, axis=0) )

                    data_list = new_data_list

                elif 'fabric' in modality:
                    time_list = target_dict['fabricTimesList']
                    ## data_list = target_dict['fabricValueList']
                    data_list = target_dict['fabricMagList']


                    ## for ii, d in enumerate(data_list):
                    ##     print np.max(d), target_dict['fileNameList'][ii]

                    ## # magnitude
                    ## new_data_list = []
                    ## for d in data_list:

                    ##     # d is 3xN-length in which each element has multiple float values
                    ##     sample = []
                    ##     if len(d) != 0 and len(d[0]) != 0:
                    ##         for i in xrange(len(d[0])):
                    ##             if d[0][i] == []:
                    ##                 sample.append( 0 )
                    ##             else:                                                               
                    ##                 s = np.array([d[0][i], d[1][i], d[2][i]])
                    ##                 v = np.mean(np.linalg.norm(s, axis=0)) # correct?
                    ##                 sample.append(v)
                    ##     else:
                    ##         print "WRONG data size in fabric data"

                    ##     new_data_list.append(sample)
                    ## data_list = new_data_list

                    ## fig_fabric = plt.figure('fabric')
                    ## ax_fabric = fig_fabric.add_subplot(111) #, projection='3d')
                    ## for d in data_list:
                    ##     color = colors.next()
                    ##     for i in xrange(len(d[0])):
                    ##         if d[0][i] == []: continue
                    ##         ax_fabric.scatter(d[1][i], d[0][i], c=color)
                    ##         ## ax_fabric.scatter(d[0][i], d[1][i], d[2][i])
                    ## ax_fabric.set_xlabel('x')
                    ## ax_fabric.set_ylabel('y')
                    ## ## ax_fabric.set_zlabel('z')
                    ## if save_pdf is False:
                    ##     plt.show()
                    ## else:
                    ##     fig_fabric.savefig('test_fabric.pdf')
                    ##     fig_fabric.savefig('test_fabric.png')
                    ##     os.system('mv test*.p* ~/Dropbox/HRL/')

                ax = fig.add_subplot(nPlot*100+10+count)
                if idx == 0: color = 'b'
                else: color = 'r'            

                if raw_viz:
                    combined_time_list = []
                    if data_list == []: continue

                    ## for t in time_list:
                    ##     temp = np.array(t[1:])-np.array(t[:-1])
                    ##     combined_time_list.append([ [0.0]  + list(temp)] )
                    ##     print modality, " : ", np.mean(temp), np.std(temp), np.max(temp)
                    ##     ## ax.plot(temp, label=modality)

                    for i in xrange(len(time_list)):
                        if len(time_list[i]) > len(data_list[i]):
                            ax.plot(time_list[i][:len(data_list[i])], data_list[i], c=color)
                        else:
                            ax.plot(time_list[i], data_list[i][:len(time_list[i])], c=color)

                    if continuousPlot:
                        new_color = 'm'
                        i         = fidx
                        if len(time_list[i]) > len(data_list[i]):
                            ax.plot(time_list[i][:len(data_list[i])], data_list[i], c=new_color, lw=3.0)
                        else:
                            ax.plot(time_list[i], data_list[i][:len(time_list[i])], c=new_color, lw=3.0)
                                                    
                else:
                    interp_time = np.linspace(time_lim[0], time_lim[1], num=downSampleSize)
                    for i in xrange(len(data_list)):
                        ax.plot(interp_time, data_list[i], c=color)                
                
                ax.set_xlim(time_lim)
                ax.set_title(modality)

            #------------------------------------------------------------------------------    
            if continuousPlot is False: break
            else:
                        
                print "-----------------------------------------------"
                print file_list[fidx]
                print "-----------------------------------------------"

                plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.0)

                if save_pdf is False:
                    plt.show()
                else:
                    print "Save pdf to Dropbox folder"
                    fig.savefig('test.pdf')
                    fig.savefig('test.png')
                    os.system('mv test.p* ~/Dropbox/HRL/')

                fig = plt.figure('all')

                
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.0)

    if save_pdf is False:
        plt.show()
    else:
        print "Save pdf to Dropbox folder"
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('mv test.p* ~/Dropbox/HRL/')



def ft_disp(timeList, ftForce, ftForceLocal=None):

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
    if ftForceLocal is not None: ax5.plot(timeList, ftForceLocal[0,:])

    ## ax6 = fig.add_subplot(gs[1,1])        
    ## ax6.plot(timeList, ftForceLocal[1,:])

    plt.show()


def viz(X1, normTest=None, abnormTest=None, skip=False):
    '''
    dim x sample x length
    X1: normal data
    X2: abnormal data
    '''
    import itertools

    n_cols = int(float(len(X1))/4.0)
    if n_cols == 0: n_cols = 1
    n_rows = int(float(len(X1))/float(n_cols))        
    colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])


    fig = plt.figure(figsize=(n_rows, n_cols*2))

    for i in xrange(len(X1)):

        n_col = int(i/n_rows)
        n_row = i%n_rows
        ax    = fig.add_subplot(n_rows,n_cols,i+1)

        x     = range(len(X1[i][0]))
        means = np.mean(X1[i], axis=0)
        stds  = np.std(X1[i], axis=0)

        print np.shape(x), np.shape(means)

        ax.plot(x, means, 'k-')
        ax.fill_between(x, means-stds, means+stds, facecolor='blue', alpha=0.5)
        ax.set_xlim([0,x[-1]])
        ax.set_ylim([0,1])
        plt.yticks([0,1.0])

    if normTest is not None:
        for i in xrange(len(normTest)):

            n_col = int(i/n_rows)
            n_row = i%n_rows
            ax    = fig.add_subplot(n_rows,n_cols,i+1)
            x     = range(len(normTest[i][0]))
            for j in xrange(len(normTest[i])):
                ax.plot(x, normTest[i][j], c='b')
                if j>3: break

            ax.set_xlim([0,x[-1]])
            ax.set_ylim([0,1])
            plt.yticks([0,1.0])

    if abnormTest is not None:
        for i in xrange(len(abnormTest)):

            n_col = int(i/n_rows)
            n_row = i%n_rows
            ax    = fig.add_subplot(n_rows,n_cols,i+1)
            x     = range(len(abnormTest[i][0]))
            for j in xrange(len(abnormTest[i])):
                ax.plot(x, abnormTest[i][j], c='r')
                if j>3: break

            ax.set_xlim([0,x[-1]])
            ax.set_ylim([0,1])
            plt.yticks([0,1.0])


    if skip is True: return
    plt.show()
    


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


    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='pushing_microwhite',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=3,
                 help='type the desired dimension')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    opt, args = p.parse_args()

    from hrl_anomaly_detection.params import *
    import itertools
    colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])
    shapes = itertools.cycle(['x','v', 'o', '+'])

    if opt.task == 'pushing_microwhite':
        subjects = ['gatsbii']
        ## raw_data_path, save_data_path, param_dict = getPushingMicroWhite(opt.task, opt.bDataRenew, \
        ##                                                                  opt.bAERenew, opt.bHMMRenew,\
        ##                                                                  rf_center, local_range)
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+opt.task+'_data/AE150'
        method = 'svm'
        nPoints = 20
        fig = plt.figure()

        for dim in xrange(2,5+1):
            roc_pkl = os.path.join(save_data_path+'_'+str(dim), 'roc_'+opt.task+'.pkl')
            ROC_data = ut.load_pickle(roc_pkl)

            tp_ll = ROC_data[method]['tp_l']
            fp_ll = ROC_data[method]['fp_l']
            tn_ll = ROC_data[method]['tn_l']
            fn_ll = ROC_data[method]['fn_l']
            delay_ll = ROC_data[method]['delay_l']

            tpr_l = []
            fpr_l = []
            fnr_l = []
            delay_mean_l = []
            delay_std_l  = []

            for i in xrange(nPoints):
                tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
                fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )
                fnr_l.append( 100.0 - tpr_l[-1] )
                delay_mean_l.append( np.mean(delay_ll[i]) )
                delay_std_l.append( np.std(delay_ll[i]) )

            label = method+'_dim_'+str(dim)
            
            # visualization
            color = colors.next()
            shape = shapes.next()
            ax1 = fig.add_subplot(111)            
            plt.plot(fpr_l, tpr_l, '-'+shape+color, label=label, mec=color, ms=6, mew=2)
            plt.xlim([-1, 101])
            plt.ylim([-1, 101])
            plt.ylabel('True positive rate (percentage)', fontsize=22)
            plt.xlabel('False positive rate (percentage)', fontsize=22)
            plt.xticks([0, 50, 100], fontsize=22)
            plt.yticks([0, 50, 100], fontsize=22)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.legend(loc='lower right', prop={'size':20})

        if opt.bSavePdf is False:
            plt.show()
        else:
            print "Save pdf to Dropbox folder"
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('mv test.p* ~/Dropbox/HRL/')




    ## subject = 'gatsbii'
    ## task    = 'scooping'
    ## verbose = True
    

    ## l = data_viz(subject, task, verbose=verbose)
    ## ## l.audio_test()
    ## ## l.kinematics_test()
    ## ## l.reduce_cart()

    ## # set RF over EE
    ## l.extractLocalFeature()
    
