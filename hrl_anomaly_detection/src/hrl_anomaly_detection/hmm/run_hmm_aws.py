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
    def run_with_local_data(self, params, processed_data_path, nFiles):

        from cross import cross_validate_local
        ## from hrl_anomaly_detection.hmm.run_hmm_aws import cross_validate_local
        ## from cross 
        
        model = None #hmm.learning_hmm(10, 10)

        all_param = list(ParameterGrid(params))
        for param in all_param:
            for idx in xrange(nFiles):
                print "--------------------- ", idx , " ---------------------------"
                task = self.lb_view.apply(cross_validate_local, idx, processed_data_path, model, param)
                print task.get()
                self.all_tasks.append(task)
        return self.all_tasks

    ## def wait(self):





if __name__ == '__main__':

    task = 'pushing'
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE'        
    
    parameters = {'nState': [10, 15, 20, 25], 'scale':np.arange(1.0, 10.0, 1.0), 'cov': [1.0, 2.0] }

    cloud = CloudSearchForHMM('/home/dpark/.starcluster/ipcluster/SecurityGroup:@sc-testdpark-us-east-1.json', '/home/dpark/HRL_ANOMALY.pem', 'testdpark', 'ubuntu') # cluster name, user name in aws node
    cloud.run_with_local_data(parameters, save_data_path, 9 )

    print cloud.get_completed_results()

    # wait until finishing parameter search
    while cloud.get_num_all_tasks() != cloud.get_num_tasks_completed():
        print cloud.get_completed_results()
        time.sleep(20)
    
    print cloud.get_completed_results()
    ## os.system( get )
    ## cloud.stop()

    print "Finished"
