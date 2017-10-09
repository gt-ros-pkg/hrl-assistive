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

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system & utils
import os, sys, copy, random
import scipy, numpy as np
import hrl_lib.util as ut

# Private utils
## from hrl_anomaly_detection import util as util
## from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import util as util
from hrl_anomaly_detection import data_manager as dm
import hrl_anomaly_detection.IROS17_isolation.isolation_util as iutil
from hrl_execution_monitor import util as autil

from hrl_anomaly_detection.RAL18_detection import util as vutil


# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm


from sklearn import preprocessing
from joblib import Parallel, delayed
import gc

random.seed(3334)
np.random.seed(3334)


def get_data(subject_names, task_name, raw_data_path, save_data_path, param_dict):

    # load params (param_dict)
    data_dict  = param_dict['data_param']
    AE_dict    = param_dict['AE']
    data_renew = data_dict['renew']
    ae_renew   = param_dict['HMM']['renew']
    method     = param_dict['ROC']['methods'][0]
    nPoints    = param_dict['ROC']['nPoints']
    
    #------------------------------------------
    if os.path.isdir(save_data_path) is False:
        os.system('mkdir -p '+save_data_path)

    crossVal_pkl = os.path.join(save_data_path, 'cv_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)        
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getRawDataLOPO(subject_names, task_name, raw_data_path, \
                              save_data_path,\
                              downSampleSize=data_dict['downSampleSize'],\
                              handFeatures=data_dict['isolationFeatures'], \
                              rawFeatures=AE_dict['rawFeatures'], \
                              cut_data=data_dict['cut_data'], \
                              data_renew=data_renew, max_time=data_dict['max_time'],
                              ros_bag_image=True)

        (d['successData'], d['success_image_list']), (d['failureData'], d['failure_image_list']), \
          d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successRawDataList'], d['failureRawDataList'],\
                               d['successFileList'], d['failureFileList'],\
                               success_image_list = d['success_image_list'], \
                               failure_image_list = d['failure_image_list'])

        d['failure_labels']  = get_label_from_filename(d['failure_files'])

        ut.save_pickle(d, crossVal_pkl)

    print "Main data"
    print np.shape(d['successData']), np.shape(d['success_image_list'])
    print np.shape(d['failureData']), np.shape(d['failure_image_list'])

    ## if fine_tuning is False:
    td1, td2, td3 = vutil.get_ext_feeding_data(task_name, save_data_path, param_dict, d,
                                               raw_feature=True, ros_bag_image=True)

    # Get main and sub data dictionary
    td = {}
    for key in td1.keys():
        if key in ['success_image_list', 'failure_image_list',
                   'successRawDataList', 'failureRawDataList',
                   'successFileList', 'failureFileList',
                   'success_files', 'failure_files',
                   'failure_labels']:
            td[key] = td1[key]+td2[key]+td3[key]
        elif key in ['successData', 'failureData']:
            td[key] = np.vstack([np.swapaxes(td1[key],0,1),
                                 np.swapaxes(td2[key],0,1),
                                 np.swapaxes(td3[key],0,1)])
            td[key] = np.swapaxes(td[key],0,1)

    return d, td


def get_label_from_filename(file_names):

    labels = []
    for f in file_names:
        labels.append( int(f.split('/')[-1].split('_')[0]) )

    return labels

    

def get_detection_idx(save_data_path, main_data, sub_data, param_dict, verbose=False, renew=False):
    
    # load params (param_dict)
    data_dict  = param_dict['data_param']
    AE_dict    = param_dict['AE']
    data_renew = data_dict['renew']
    ae_renew   = param_dict['HMM']['renew']
    method     = param_dict['ROC']['methods'][0]
    nPoints    = param_dict['ROC']['nPoints']
    
    if ae_renew: clf_renew = True
    else:        clf_renew = param_dict['SVM']['renew']
    fine_tuning = False

    # Check the list of temporal data and images
    nDim = len(main_data['successData'])
    tp_ll = [[] for i in xrange(nPoints)]
    fp_ll = [[] for i in xrange(nPoints)]
    tn_ll = [[] for i in xrange(nPoints)]
    fn_ll = [[] for i in xrange(nPoints)]
    roc_l = []
    train_idx_ll = []
    test_idx_ll  = []

    detection_pkl = os.path.join(save_data_path, 'anomaly_idx.pkl')

    #-----------------------------------------------------------------------------------------
    # Anomaly Detection using lstm-dvae-phase
    #-----------------------------------------------------------------------------------------        
    # Leave-one-person-out cross validation
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(main_data['kFoldList']):

        if renew is False and os.path.isfile(detection_pkl) : break
        print "==================== ", idx, " ========================"
        
        # ------------------------------------------------------------------------------------------         
        # dim x sample x length
        normalTrainData   = main_data['successData'][:, normalTrainIdx, :]
        abnormalTrainData = main_data['failureData'][:, abnormalTrainIdx, :]
        normalTestData    = main_data['successData'][:, normalTestIdx, :]
        abnormalTestData  = main_data['failureData'][:, abnormalTestIdx, :]
        normalTrainData   = np.hstack([normalTrainData,
                                       copy.deepcopy(sub_data['successData'])])
        abnormalTrainData = np.hstack([abnormalTrainData,
                                       copy.deepcopy(sub_data['failureData'])])

        # shuffle
        np.random.seed(3334+idx)
        idx_list = range(len(normalTrainData[0]))
        np.random.shuffle(idx_list)
        normalTrainData = normalTrainData[:,idx_list]

        normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, scaler =\
          vutil.get_scaled_data(normalTrainData, abnormalTrainData,
                                normalTestData, abnormalTestData, aligned=False)

        trainData = [normalTrainData[:int(len(normalTrainData)*0.7)],
                     [0]*len(normalTrainData[:int(len(normalTrainData)*0.7)])]
        valData   = [normalTrainData[int(len(normalTrainData)*0.7):],
                     [0]*len(normalTrainData[int(len(normalTrainData)*0.7):])]
        testData  = [normalTestData, [0]*len(normalTestData)]

        # ------------------------------------------------------------------------------------------         
        # scaling info to reconstruct the original scale of data
        scaler_dict  = {'scaler': scaler, 'scale': 1, 'param_dict': main_data['raw_param_dict']}
        method       = 'lstm_dvae_phase'
        vae_logvar   = None

        weights_path = os.path.join(save_data_path,'model_weights_'+method+'_'+str(idx)+'.h5')
        
        if (method.find('lstm_vae')>=0 or method.find('lstm_dvae')>=0):
            dyn_ths     = True
            ad_method   = 'lower_bound'
            
            from hrl_execution_monitor.keras_util import lstm_dvae_phase as km

            autoencoder, vae_mean, _, enc_z_mean, enc_z_std, generator = \
              km.lstm_vae(trainData, valData, weights_path, patience=4, batch_size=256,
                          noise_mag=0.05, timesteps=1, sam_epoch=40,
                          x_std_div=4., x_std_offset=0.1, z_std=1.0,\
                          phase=1., z_dim=3, h1_dim=4, \
                          renew=ae_renew, fine_tuning=fine_tuning, plot=False,\
                          scaler_dict=scaler_dict)
        else:
            sys.exit()

        alpha = np.array([1.0]*nDim) #/float(nDim)
        ths_l = np.logspace(-1.0,2.4,nPoints) - 0.2

        from hrl_anomaly_detection.journal_isolation import detector as dt
        save_pkl = os.path.join(save_data_path, 'model_ad_scores_'+str(idx)+'.pkl')
        tp_l, tn_l, fp_l, fn_l, roc, train_anomaly_idx_l, test_anomaly_idx_l = \
          dt.anomaly_detection(autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator,
                               normalTrainData, valData[0], abnormalTrainData,\
                               normalTestData, abnormalTestData, \
                               ad_method, method,
                               1, alpha, ths_l=ths_l, save_pkl=save_pkl, stateful=True,
                               x_std_div = 4, x_std_offset=0.1, z_std=1.0, \
                               phase=1.0, plot=False, \
                               renew=clf_renew, dyn_ths=dyn_ths, batch_info=(True,256),\
                               param_dict=main_data['param_dict'], scaler_dict=scaler_dict,\
                               filenames=(np.array(main_data['success_files'])[normalTestIdx],
                                          np.array(main_data['failure_files'])[abnormalTestIdx]),\
                               return_idx=True)

        roc_l.append(roc)
        train_idx_ll.append(train_anomaly_idx_l)
        test_idx_ll.append(test_anomaly_idx_l)

        for i in xrange(len(ths_l)):
            tp_ll[i] += tp_l[i]
            fp_ll[i] += fp_l[i]
            tn_ll[i] += tn_l[i]
            fn_ll[i] += fn_l[i]
   
    #--------------------------------------------------------------------
    if renew or os.path.isfile(detection_pkl) is False:
        print "roc list ", roc_l
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

        fs_l = []
        for i in xrange(len(ths_l)):
            fs_l.append( (2.0*float(np.sum(tp_ll[i])))/ (2.0*float(np.sum(tp_ll[i])) + float(np.sum(fn_ll[i])) + float(np.sum(fp_ll[i]))) )

        dd = {}
        dd['tp_ll'] = tp_ll
        dd['fp_ll'] = fp_ll
        dd['tn_ll'] = tn_ll
        dd['fn_ll'] = fn_ll
        ## d['roc_l'] = roc_l
        dd['train_idx_ll'] = train_idx_ll
        dd['test_idx_ll']  = test_idx_ll
        dd['fs_l']  = fs_l
        dd['kFoldList'] = main_data['kFoldList']
        ut.save_pickle(dd, detection_pkl)
    else:
        dd = ut.load_pickle(detection_pkl)

    return dd['train_idx_ll'], dd['test_idx_ll'], dd['fs_l']

    
def get_isolation_data(save_data_path, main_data, sub_data,
                       train_idx_list, test_idx_list, ths_idx, param_dict):


    
    #-----------------------------------------------------------------------------------------
    # Data extraction
    #-----------------------------------------------------------------------------------------        
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(main_data['kFoldList']):
        print "==================== ", idx, " ========================"

        # detection indices
        train_idx = train_idx_list[idx][ths_idx]
        test_idx  = test_idx_list[idx][ths_idx]


        #Signal data
        abnormalTrainData = main_data['failureData'][:, abnormalTrainIdx, :]
        abnormalTrainData = np.hstack([abnormalTrainData,
                                       copy.deepcopy(sub_data['failureData'])])
        abnormalTrainLabels = np.array(main_data['failure_labels'])[abnormalTrainIdx].tolist()+\
          sub_data['failure_labels']
          
        abnormalTestData   = main_data['failureData'][:, abnormalTestIdx, :]
        abnormalTestLabels = np.array(main_data['failure_labels'])[abnormalTestIdx].tolist()

        ## print np.shape(train_idx_list)
        ## print np.shape(test_idx_list)
        ## print np.shape(abnormalTrainData), np.shape(abnormalTrainLabels)
        ## print np.shape(abnormalTestData), np.shape(abnormalTestLabels)

        #Image data

        #Extra images

        
        sys.exit()

    
    return train_data, test_data
                      
    # split data with 80:20 ratio, 3set
    kFold_list = d['kFold_list'][:1]

    # flattening image list
    success_image_list = autil.image_list_flatten( d.get('success_image_list',[]) )
    failure_image_list = autil.image_list_flatten( d.get('failure_image_list',[]) )

    failure_labels = []
    for f in d['failureFiles']:
        failure_labels.append( int( f.split('/')[-1].split('_')[0] ) )
    failure_labels = np.array( failure_labels )

    # Static feature selection for isolation
    feature_list = []
    for feature in param_dict['data_param']['staticFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    successData_static = np.array(d['successData'])[feature_list]
    failureData_static = np.array(d['failureData'])[feature_list]
    
    #-----------------------------------------------------------------------------------------
    # Dynamic feature selection for detection and isolation
    feature_idx_list = []
    success_data_ad = []
    failure_data_ad = []
    nDetector = len(param_dict['data_param']['handFeatures'])
    for i in xrange(nDetector):
        
        feature_idx_list.append([])
        for feature in param_dict['data_param']['handFeatures'][i]:
            feature_idx_list[i].append(data_dict['isolationFeatures'].index(feature))

        success_data_ad.append( copy.copy(d['successData'][feature_idx_list[i]]) )
        failure_data_ad.append( copy.copy(d['failureData'][feature_idx_list[i]]) )
        HMM_dict_local = copy.deepcopy(HMM_dict)
        HMM_dict_local['scale'] = param_dict['HMM']['scale'][i]
        
        # Training HMM, and getting classifier training and testing data
        dm.saveHMMinducedFeatures(kFold_list, success_data_ad[i], failure_data_ad[i],\
                                  task_name, save_data_path,\
                                  HMM_dict_local, data_renew, startIdx, nState, cov, \
                                  success_files=d['successFiles'], failure_files=d['failureFiles'],\
                                  noise_mag=noise_mag[i], suffix=str(i),\
                                  verbose=verbose, one_class=False)

    del d

    # ---------------------------------------------------------------
    # get data
    data_dict = {}
    data_pkl = os.path.join(save_data_path, 'isol_data.pkl')
    if os.path.isfile(data_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']:

        l_data = Parallel(n_jobs=1, verbose=10)\
          (delayed(iutil.get_hmm_isolation_data)(idx, kFold_list[idx], failure_data_ad, \
                                                 failureData_static, \
                                                 failure_labels,\
                                                 failure_image_list,\
                                                 task_name, save_data_path, param_dict, weight,\
                                                 single_detector=single_detector,\
                                                 n_jobs=-1, window_steps=window_steps, verbose=verbose\
                                                 ) for idx in xrange(len(kFold_list)) )
        
        data_dict = {}
        for i in xrange(len(l_data)):
            idx = l_data[i][0]
            data_dict[idx] = (l_data[i][1],l_data[i][2],l_data[i][3],l_data[i][4] )
            
        print "save pkl: ", data_pkl
        ut.save_pickle(data_dict, data_pkl)            
    else:
        data_dict = ut.load_pickle(data_pkl)
    

    # ---------------------------------------------------------------
    scores = []
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):
        print "kFold_list: ", idx

        (x_trains, y_train, x_tests, y_test) = data_dict[idx]         
        x_train = x_trains[0] 
        x_test  = x_tests[0] 
        print np.shape(x_train), np.shape(x_test)

        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)

        if type(x_train) is np.ndarray:
            x_train = x_train.tolist()
            x_test  = x_test.tolist()
        if type(y_train) is np.ndarray:
            y_train  = y_train.tolist()
            y_test   = y_test.tolist()
        
        ## from sklearn.svm import SVC
        ## clf = SVC(C=1.0, kernel='rbf') #, decision_function_shape='ovo')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)

        clf.fit(x_train, y_train)
        ## y_pred = clf.predict(x_test.tolist())
        score = clf.score(x_test, y_test)
        scores.append( score )
        print idx, " : score = ", score


    print scores
    print "Score mean = ", np.mean(scores), np.std(scores)

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    opt, args = p.parse_args()

    from hrl_anomaly_detection.journal_isolation.isolation_param import *
    # IROS2017
    subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/JOURNAL_ISOL/'+opt.task+'_1'


    window_steps= 5
    task_name = 'feeding'
    single_detector=True
    nb_classes = 12


    get_isolation_data(subject_names, task_name, raw_data_path, save_data_path,
                       param_dict, weight=1.0, single_detector=single_detector,
                       window_steps=window_steps, verbose=False)




## def get_isolation_data(idx, failureData, failureImages, failureLabels, failureIdx):



##     return idx, [x_train, x_train_img], y_train, [x_test, x_test_img], y_test
