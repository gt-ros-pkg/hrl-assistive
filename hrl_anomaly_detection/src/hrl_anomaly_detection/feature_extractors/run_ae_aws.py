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

from cloud_seaorch import CloudSearch
from hrl_anomaly_detection.aws.CloudSearch import CloudSearch
from hrl_anomaly_detection.feature_extractors.auto_encoder import auto_encoder

class CloudSearchForAE(CloudSearch):

    def __init__(self):
        CloudSearch.__init__(self)

	#run model from data in cloud.
	#each node grabs file from their local path and runs the model
	#requires grab_data to be implemented correctly
	#n_inst is to create a fold. the way it generates fold can be changed
	def run_with_local_data(self, model, params, n_inst, cv, path_file):
		from cross import cross_validate_local
		#from cross import grab_data
		splited = self.split(n_inst, cv)
		all_param = list(ParameterGrid(params))
		for param in all_param:
			for train, test in splited:
				task = self.lb_view.apply(cross_validate_local, train, test, path_file, model, param)
				self.all_tasks.append(task)
		return self.all_tasks
    

def getDataSet(subject_names, task_name, raw_data_path, \
                    processed_data_path, param_dict, cuda=True, verbose=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    feature_list = data_dict['feature_list']
    # AE
    AE_dict     = param_dict['AE']
    #------------------------------------------

                    
    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')

    d = ut.load_pickle(crossVal_pkl)
    successData = d['successData']
    failureData = d['failureData']
    aug_successData = d['aug_successData']
    aug_failureData = d['aug_failureData']
    kFold_list      = d['kFoldList']                                        

    
    


if __name__ == '__main__':




    parameters = {'learning_rate': [1e-6, 1e-6], 'momentum':[1e-6], 'dampening':[1e-6], \
                  'layer_sizes': [ [X.shape[0], 128,64,16], [X.shape[0], 64,32,16], [X.shape[0], 64,32,8] ] }
    model = SVC()


    cloud = CloudSearch('/root/.starcluster/ipcluster/SecurityGroup:@sc-freecluster-us-east-1.json', '~/HRK_ANOMALY.pem', 'dparkcluster', 'sgeadmin')
    cloud.run_with_local_data(model, parameters, 4898, 10, '/scratch/ubuntu/')


    while cloud.get_num_all_tasks() != cloud.get_num_tasks_completed():

        print cloud.get_completed_results()
        time.sleep(20)
    
    print cloud.get_completed_results()
    ## os.system( get )
    ## cloud.stop()

    print "Finished"
