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
import numpy as np

from sklearn.grid_search import ParameterGrid
import time
from hrl_anomaly_detection.aws.cloud_search import CloudSearch
from hrl_anomaly_detection.hmm import learning_hmm as hmm

class CloudSearchForHMM(CloudSearch):
    def __init__(self, path_json, path_key, clust_name, user_name):
        CloudSearch.__init__(self, path_json, path_key, clust_name, user_name)

    #run data in cloud.
	#each node grabs file from their local path and runs the model
	#requires grab_data to be implemented correctly
	#n_inst is to create a fold. the way it generates fold can be changed
    def run_with_local_data(self, model, params, processed_data_path, nFiles):

        ## path_shell = 'export PATH='+os.path.expanduser('~')+'/catkin_ws/src/hrl-assistive/hrl_anomaly_detection/src/hrl_anomaly_detection/hmm'+':$PATH'
        path_shell = 'source ~/.bashrc'
        self.sync_run_shell(path_shell)
                
        ## from cross import cross_validate_local
        
        all_param = list(ParameterGrid(params))
        for param in all_param:
            for idx in xrange(nFiles):
                task = self.lb_view.apply(cross_validate_local, idx, processed_data_path, model, param)
                print task.get()
                self.all_tasks.append(task)
        return self.all_tasks

    ## def wait(self):

def cross_validate(train_data, test_data,  model, params):
    '''
    train_data : [x,y]
    '''

    train_data_x = train_data[0]
    train_data_y = train_data[1]
    test_data_x  = test_data[0]
    test_data_y  = test_data[1]
    
    model.set_params(**params)
    nEmissionDim = len(train_data_x)

    scale = 1.0
    cov_mult = [1.0]*(nEmissionDim**2)
    for key, value in six.iteritems(params): 
        if key is 'cov':
            cov_mult = [value]*(nEmissionDim**2)
        if key is 'scale':
            scale = value
            
    ret = model.fit(train_data_x*scale, cov_mult=cov_mult)
    if ret == 'Failure':
        return 0.0, params
    else:
        score = model.score(test_data_x*scale, test_data_y)    
        return score, params

def cross_validate_local(idx, processed_data_path, model, params):
    '''
    
    '''    
    dim   = 4
    for key, value in six.iteritems(params): 
        if key is 'dim':
            dim = value

    # Load data
    AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')
    d = ut.load_pickle(AE_proc_data)

    pooling_param_dict  = {'dim': dim} # only for AE

    # dim x sample x length
    normalTrainData, pooling_param_dict = dm.variancePooling(d['normTrainData'], \
                                                             pooling_param_dict)
    abnormalTrainData,_                 = dm.variancePooling(d['abnormTrainData'], pooling_param_dict)
    normalTestData,_                    = dm.variancePooling(d['normTestData'], pooling_param_dict)
    abnormalTestData,_                  = dm.variancePooling(d['abnormTestData'], pooling_param_dict)

    trainSet = [normalTrainData, [1.0]*len(normalTrainData) ]

    testData_x = np.vstack([ np.swapaxes(normalTestData, 0, 1), np.swapaxes(abnormalTestData, 0, 1) ])
    testData_x = np.swapaxes(testData_x, 0, 1)
    testData_y = [1.0]*len(normalTestData[0]) + [-1.0]*len(abnormalTestData[0])    
    testSet    = [testData_x, testData_y ]

    return cross_validate(trainSet, testSet, model, params)





if __name__ == '__main__':

    task = 'pushing'
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE'        
    
    model = hmm.learning_hmm(10, 10)
    parameters = {'nState': [10, 15, 20, 25], 'scale':np.arange(1.0, 10.0, 1.0), 'cov': [1.0, 2.0] }

    cloud = CloudSearchForHMM('/home/dpark/.starcluster/ipcluster/SecurityGroup:@sc-testdpark-us-east-1.json', '/home/dpark/HRL_ANOMALY.pem', 'testdpark', 'ubuntu') # cluster name, user name in aws node
    cloud.run_with_local_data(model, parameters, save_data_path, 9 )

    print cloud.get_completed_results()

    # wait until finishing parameter search
    while cloud.get_num_all_tasks() != cloud.get_num_tasks_completed():
        print cloud.get_completed_results()
        time.sleep(20)
    
    print cloud.get_completed_results()
    ## os.system( get )
    ## cloud.stop()

    print "Finished"
