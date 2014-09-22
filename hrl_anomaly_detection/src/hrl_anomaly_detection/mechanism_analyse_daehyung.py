#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import numpy as np
import glob

# ROS library
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut
import matplotlib.pyplot as plt
import common as co

import mechanism_analyse_RAM as mar


def get_all_data_by_chunk(mech_vec_list, mech_nm_list):

    data, _ = mar.create_blocked_dataset_semantic_classes(mech_vec_list,
                                    mech_nm_list, append_robot = True)    

    # there can be multiple chunks with a target, chunk is unique...
    mean_thresh_charlie_dict = {}    
    for chunk in data.uniquechunks:
        non_robot_idxs = np.where(['robot' not in i for i in data.chunks])[0] # if there is no robot, true 
        idxs = np.where(data.chunks[non_robot_idxs] == chunk)[0] # find same target samples in non_robot target samples
        train_trials = (data.samples[non_robot_idxs])[idxs]

        # skip empty set
        if (train_trials.shape)[0] == 0: continue

        mean_force_profile = np.mean(train_trials, 0)
        std_force_profile = np.std(train_trials, 0)
            
        if 'robot' in chunk:
            # remove the trailing _robot
            key = chunk[0:-6]
        else:
            key = chunk
        mean_thresh_charlie_dict[key] = (mean_force_profile * 0.,
                                         mean_force_profile, std_force_profile)


        
        

if __name__ == '__main__':

    root_path = os.environ['HRLBASEPATH']+'/'
    
    pkl_list = glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/*.pkl') + glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/perfect_perception/*.pkl') + glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/simulate_perception/*.pkl')

    r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
    mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36) #get vec_list, name_list

    get_all_data_by_chunk(mech_vec_list, mech_nm_list)
