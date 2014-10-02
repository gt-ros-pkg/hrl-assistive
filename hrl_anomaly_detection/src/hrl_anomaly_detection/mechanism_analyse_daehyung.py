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

# Machine learning
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters


import mechanism_analyse_RAM as mar


def get_all_blocked_detection():
    root_path = os.environ['HRLBASEPATH']+'/'
    
    pkl_list = glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/*.pkl') #+ glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/perfect_perception/*.pkl') + glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/simulate_perception/*.pkl')

    r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')


# Get mean force profile by chunks
def blocked_detection(mech_vec_list, mech_nm_list):
    
    data, _ = mar.create_blocked_dataset_semantic_classes(mech_vec_list,
                                    mech_nm_list, append_robot = True)    

    fig = plt.figure()
    ax  = fig.add_subplot(111, aspect='equal')
    
    # create the generator
    ## nfs = NFoldPartitioner(cvtype=1) # Split into 248 and 12 set
    ## spl = splitters.Splitter(attr='partitions')
    ## splits = [list(spl.generate(x)) for x in nfs.generate(data)]

    ## for l_wdata, l_vdata in splits:        
    ##     print len(l_wdata), len(l_vdata), len(data)
    
    non_robot_idxs = np.where(['robot' not in i for i in data.chunks])[0] # if there is no robot, true 
    ## non_robot_idxs = np.where(['robot' in i for i in data.chunks])[0] # if there is no robot, true 
    
    # chunks : experiment data name    
    # there can be multiple chunks with a target, chunk is unique...
    mean_thresh_charlie_dict = {}    
    for chunk in data.uniquechunks:

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

        ax.plot(mean_force_profile)
        
    ## d = {'mean_charlie': mean_thresh_charlie_dict}
    ## ut.save_pickle(d, 'non_robot_mean_dict.pkl')

    fig.savefig('/home/dpark/Dropbox/HRL/mech.pdf', format='pdf')    
    ## plt.show()

    
def blocked_detection_n_equals_1(mech_vec_list, mech_nm_list):
    data, _ = mar.create_blocked_dataset_semantic_classes(mech_vec_list, mech_nm_list, append_robot = False)
    nfs = NFoldPartitioner(cvtype=1, attr='targets') # 1-fold ?
    spl = splitters.Splitter(attr='partitions')
    splits = [list(spl.generate(x)) for x in nfs.generate(data)]
    
    ## splitter = NFoldSplitter(cvtype=1)
    ## label_splitter = NFoldSplitter(cvtype=1, attr='labels')
    mean_thresh_known_mech_dict = {}
    for l_wdata, l_vdata in splits:
        mean_thresh_known_mech_list = []
        Ms = mar.compute_Ms(data, l_vdata.targets[0], plot=True)
        break

        mechs = l_vdata.uniquechunks
        for m in mechs:
            n_std = 0.
            all_trials = l_vdata.samples[np.where(l_vdata.chunks == m)]
            le = all_trials.shape[1]
            for i in range(all_trials.shape[0]):
                one_trial = all_trials[i,:].reshape(1,le)
                mn_list, std_list = mar.estimate_theta(one_trial, Ms, plot=False)
                mn_arr, std_arr = np.array(mn_list), np.array(std_list)
                n_std = max(n_std, np.max(np.abs(all_trials - mn_arr) / std_arr))

            mean_thresh_known_mech_dict[m] = (Ms, n_std) # store on a per mechanism granularity
            print 'n_std for', m, ':', n_std
            print 'max error force for', m, ':', np.max(n_std*std_arr[2:])

    ## d = ut.load_pickle('blocked_thresh_dict.pkl')
    ## d['mean_known_mech'] = mean_thresh_known_mech_dict
    ## ut.save_pickle(d, 'blocked_thresh_dict.pkl')
    

def get_discrete_test_from_mean_dict(pkl_file):

    data = ut.load_pickle(pkl_file)

    mean_charlie    = data['mean_charlie']
    mean_known_mech = data['mean_known_mech']

    
    print mean_charlie['ikea_cabinet_noisy_cody'] # Force profile (mean * 0.0, mean, std)
    print mean_known_mech['ikea_cabinet_noisy_cody']
    

    ## for key in mean_dict.keys():

    ##     mean_force_profile = mean_dict[key][1]
    ##     std_force_profile = mean_dict[key][2]

        

    

if __name__ == '__main__':

    root_path = os.environ['HRLBASEPATH']+'/'
    
    pkl_list = glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/*.pkl') #+ glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/perfect_perception/*.pkl') + glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/simulate_perception/*.pkl')
    ## pkl_list = glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/perfect_perception/*.pkl') #+ glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/simulate_perception/*.pkl')
    ## pkl_list = glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/simulate_perception/*.pkl')

    ## root_path = os.environ['HRLBASEPATH']    
    ## ## pkl_list = glob.glob(root_path+'_data/usr/advait/ram_www/RAM_db_of_different_kinds/RAM_db/*_new.pkl')
    ## pkl_list = glob.glob(root_path+'_data/usr/advait/ram_www/*_new.pkl')

    
    r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
    mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36) #get vec_list, name_list

    blocked_detection(mech_vec_list, mech_nm_list)
    #blocked_detection_n_equals_1(mech_vec_list, mech_nm_list)
    
    ## get_discrete_test_from_mean_dict('non_robot_mean_dict.pkl')
    ## get_discrete_test_from_mean_dict('blocked_thresh_dict.pkl')

    # Get normal data from 3 door, two robot, and one human 
    # Get collision data from any case
