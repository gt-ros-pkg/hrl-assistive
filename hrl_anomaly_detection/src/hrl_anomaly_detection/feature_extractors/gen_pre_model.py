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

import os
from hrl_anomaly_detection import util
from hrl_anomaly_detection.params import *
from hrl_anomaly_detection import data_manager as dm
import hrl_lib.util as ut


def getTaskFileList(root_path, task_name):
    # List up recorded files
    folder_list = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path,d))]        

    file_list = []
    for d in folder_list:

        if len(d.split('_')[1:]) > 1:            
            folder_name = d.split('_')[1] + '_' + d.split('_')[2]
        elif len(d.split('_')[1:]) == 1:            
            folder_name = d.split('_')[1]
        else:
            continue

        if task_name == folder_name: 
            files = os.listdir(os.path.join(root_path,d))
            for f in files:
                # pickle file name with full path
                pkl_file = os.path.join(root_path,d,f)
                file_list.append(pkl_file)
                
    print "--------------------------------------------"
    print "# of files: ", len(file_list)
    print "--------------------------------------------"    
    return file_list
    
            
def genPreTrainData(task_name, target_param_dict, raw_data_path, save_data_path, task_list, data_renew, \
                    rf_center, local_range,\
                    time_window, save_pkl, dim=3):

    downSampleSize = 200

    if os.path.isdir(save_data_path) is False:
        os.system('mkdir -p '+save_data_path)    

    if os.path.isfile(save_pkl):
        d = ut.load_pickle(save_pkl)
        return d

    data = None
    for task in task_list:

        if task == 'pushing_microblack':
            _, _, param_dict = getPushingMicroBlack(task, opt.bDataRenew, False, False, \
                                                    rf_center, local_range, dim=dim )
        elif task == 'feeding':
            _, _, param_dict = getFeeding(task, opt.bDataRenew, False, False, \
                                            rf_center, local_range, dim=dim )
        elif task == 'scooping':
            _, _, param_dict = getScooping(task, opt.bDataRenew, False, False, \
                                             rf_center, local_range, dim=dim )
        
        file_list = getTaskFileList(raw_data_path, task)
        
        # loading and time-sync    
        all_data_pkl     = os.path.join(save_data_path, task+'_all')
        _, all_data_dict = util.loadData(file_list, isTrainingData=False,
                                         downSampleSize=target_param_dict['data_param']['downSampleSize'],\
                                         local_range=local_range, rf_center=rf_center,\
                                         renew=param_dict['data_param']['renew'], save_pkl=all_data_pkl)

        success_features, failure_features, feature_dict = \
        dm.extractRawFeature(all_data_dict, param_dict['AE']['rawFeatures'], \
                             nSuccess=len(all_data_dict['timesList'])-1, \
                             nFailure=1, cut_data=None, \
                             scaling=False)
        success_features = np.swapaxes(success_features, 0, 1)
        failure_features = np.swapaxes(failure_features, 0, 1)
        
        if data is None:
            data = np.vstack([success_features, failure_features])
        else:
            data = np.vstack([ data, np.vstack([success_features, failure_features]) ])
        print "data: ", np.shape(data)
        ## allDataConv = dm.getTimeDelayData(allData, time_window)
        ## print "allDataConv: ", np.shape(allDataConv)

    dataDim = feature_dict['dataDim']
    data = np.swapaxes(data, 0,1) # dim x sample x length

    # scaling parameter
    feature_max = []
    feature_min = []
    count = 0
    for i, dimInfo in enumerate(dataDim):
        dim = dimInfo[1]
        for j in xrange(dim):
            ## print np.shape(data[count:count+dim]), count, dim
            feature_max.append( np.max(data[count:count+dim]) )
            feature_min.append( np.min(data[count:count+dim]) )
        count += dim
    
    # scaling
    scaled_data = []
    for i, feature in enumerate(data):
        if abs( feature_max[i] - feature_min[i]) < 1e-3:
            scaled_data.append( np.array(feature) )
        else:
            scaled_data.append( ( np.array(feature) - feature_min[i] )\
                                /( feature_max[i] - feature_min[i]) )

    # weighting
    ## count = 0
    ## for i, dimInfo in enumerate(dataDim):
    ##     dim = dimInfo[1]
    ##     temp = np.array(scaled_data[count:count+dim])
    ##     temp /= float(dim)
    ##     scaled_data[count:count+dim] = temp.tolist()
    ##     count += dim
    
    # sample x dim x length => n x window
    scaled_data = np.swapaxes(scaled_data, 0,1) 
    allDataConv = dm.getTimeDelayData(scaled_data, time_window)

    d = {}
    d['feature_info'] = dataDim
    d['feature_max'] = feature_max
    d['feature_min'] = feature_min    
    d['allDataConv'] = allDataConv
    d['task_list']   = task_list
    ut.save_pickle(d, save_pkl)

    return d

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()

    p.add_option('--task', action='store', dest='task', type='string', default='pushing_microwhite',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=3,
                 help='type the desired dimension')    
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    opt, args = p.parse_args()

    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'
    save_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+opt.task+'_data'
    task_list = ['pushing_microblack', 'feeding', 'scooping', 'pushing_toolcase']
    time_window = 4
    rf_center     = 'kinEEPos'        
    local_range    = 10.0

    _, _, param_dict = getPushingMicroWhite(opt.task, opt.bDataRenew, False, False, \
                                            rf_center, local_range, pre_train=True, dim=opt.dim )


    save_pkl     = os.path.join(save_data_path, opt.task+'_pretrain_data_'+str(opt.dim) )   
    d = genPreTrainData(opt.task, param_dict, raw_data_path, save_data_path, task_list, opt.bDataRenew, \
                    rf_center, local_range,\
                    time_window, save_pkl, dim=opt.dim)

    X_train = d['allDataConv']
    nDim    = len(X_train[1])
    print np.shape(X_train)

    from hrl_anomaly_detection.feature_extractors import auto_encoder as ae
    ml = ae.auto_encoder([nDim]+param_dict['AE']['layer_sizes'], \
                         param_dict['AE']['learning_rate'], param_dict['AE']['learning_rate_decay'],\
                         param_dict['AE']['momentum'], param_dict['AE']['dampening'], \
                         param_dict['AE']['lambda_reg'], param_dict['AE']['time_window'], \
                         max_iteration=param_dict['AE']['max_iteration'], \
                         min_loss=param_dict['AE']['min_loss'], cuda=True, verbose=True)

    AE_model = os.path.join(save_data_path, 'ae_pretrain_model.pkl')    
    if os.path.isfile(AE_model) or opt.bDataRenew:
        print "AE model exists: ", AE_model
        ml.fit(X_train, save_obs={'save': True, 'load': True, 'filename': AE_model})
    else:
        ml.fit(X_train, save_obs={'save': True, 'load': False, 'filename': AE_model})
        ## ml.save_params(AE_model)
