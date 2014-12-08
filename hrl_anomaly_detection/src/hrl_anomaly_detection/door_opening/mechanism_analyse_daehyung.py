#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import math
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

# Matplot
import matplotlib.pyplot as pp
import hrl_lib.matplotlib_util as mpu


import advait.mechanism_analyse_RAM as mar
import advait.ram_db as rd
import advait.mechanism_analyse_advait as maa
import advait.arm_trajectories_ram as atr

data_path = os.environ['HRLBASEPATH']+'_data/usr/advait/ram_www/data_from_robot_trials/'
root_path = os.environ['HRLBASEPATH']+'/'


def get_a_blocked_detection(mech, ang_interval=1.0):

    cls = mech

    # collision w/ box
    if mech == 'lab_fridge_cody':
        pkl_nm = data_path+'robot_trials/lab_fridge_collision_box/pull_trajectories_lab_refrigerator_2010Dec10_044022_new.pkl'
        one_pkl_nm = data_path + 'robot_trials/perfect_perception/lab_fridge_cody_new.pkl'

    # collision w/ chair
    elif mech == 'lab_fridge_cody':
        pkl_nm = data_path+'robot_trials/lab_fridge_collision_chair/pull_trajectories_lab_refrigerator_2010Dec10_042926_new.pkl'
        one_pkl_nm = data_path + 'robot_trials/perfect_perception/lab_fridge_cody_new.pkl'

    # No collision
    elif mech == 'ikea_cabinet_pr2':
        pkl_nm = data_path+'robot_trials/ikea_cabinet/pr2_pull_2010Dec08_204324_new.pkl'
        one_pkl_nm = data_path + 'robot_trials/perfect_perception/ikea_cabinet_pr2_new.pkl'

    ## # locked
    ## elif mech == 'kitchen_cabinet_cody':
    ##     pkl_nm = data_path+'robot_trials/kitchen_cabinet_locked/pull_trajectories_kitchen_cabinet_2010Dec11_233625_new.pkl'
    ##     one_pkl_nm = data_path + 'robot_trials/perfect_perception/kitchen_cabinet_cody_new.pkl'
    ## elif mech == 'kitchen_cabinet_pr2':
    ##     pkl_nm = data_path + 'robot_trials/kitchen_cabinet_locked/pr2_pull_2010Dec12_005340_new.pkl'
    ##     one_pkl_nm = data_path + 'robot_trials/perfect_perception/kitchen_cabinet_pr2_new.pkl'

        
    ## # collision w/ chair
    ## elif mech == 'kitchen_cabinet_cody':
    ##     pkl_nm = data_path+'robot_trials/hsi_kitchen_collision_chair/pull_trajectories_kitchen_cabinet_2010Dec10_060852_new.pkl'
    ##     one_pkl_nm = pth + 'RAM_db/robot_trials/perfect_perception/kitchen_cabinet_cody_new.pkl'

    ## # collision w/ box
    ## elif mech == 'kitchen_cabinet_cody':
    ##     pkl_nm = data_path+'robot_trials/hsi_kitchen_collision_box/pull_trajectories_kitchen_cabinet_2010Dec10_060454_new.pkl'
    ##     one_pkl_nm = data_path + 'robot_trials/perfect_perception/kitchen_cabinet_cody_new.pkl'

    # collision w/ box
    elif mech == 'kitchen_cabinet_pr2':
        pkl_nm = data_path + 'robot_trials/hsi_kitchen_collision_pr2/pr2_pull_2010Dec10_071602_new.pkl'
        one_pkl_nm = data_path + 'robot_trials/perfect_perception/kitchen_cabinet_pr2.pkl'
        ## pkl_nm = data_path + 'robot_trials/hsi_kitchen_collision_pr2/pr2_pull_2010Dec10_071602_new.pkl'
        ## one_pkl_nm = data_path + 'robot_trials/perfect_perception/kitchen_cabinet_pr2.pkl'
        ## pkl_nm = '/home/dpark/Dropbox/HRL/pr2_pull_2010Dec10_071602_new.pkl'

    else:
        print "No available data"
        sys.exit()

        
    max_ang = math.radians(30)
    
    pull_dict = ut.load_pickle(pkl_nm)
    typ = 'rotary'
    pr2_log =  'pr2' in pkl_nm
    h_config, h_ftan = atr.force_trajectory_in_hindsight(pull_dict,
                                                   typ, pr2_log)

    h_config = np.array(h_config)
    h_ftan = np.array(h_ftan)
    h_ftan = h_ftan[h_config < max_ang]
    h_config = h_config[h_config < max_ang] # cut
    bin_size = math.radians(ang_interval)
    h_config_degrees = np.degrees(h_config)
    ftan_raw = h_ftan
    
    # resampling with specific interval
    h_config, h_ftan = maa.bin(h_config, h_ftan, bin_size, np.mean, True) 
       
    return h_config, h_ftan
    

def get_all_blocked_detection(): # human

    pkl_list = glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/*_new.pkl')
    r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
    mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36) #get vec_list, name_list

    data, labels, chunks = create_blocked_dataset_semantic_classes(mech_vec_list,
                                    mech_nm_list, append_robot = True)    

    return np.array(data), labels, chunks

#---------------- blocked analysis --------------------
#
# lets perform this analysis for freezer, fridge, and office cabinet
# class. I have maximum data for these classes.
#
#-----------------------------------------------------

def create_blocked_dataset_semantic_classes(mech_vec_list,
                                            mech_nm_list, append_robot):
    all_vecs = np.column_stack(mech_vec_list)
    lab_num = 0
    chunk_num = 0
    labels = []
    feat_list = []
    chunks = []
    labels_test = []
    feat_list_test = []
    chunks_test = []
    for i, v_mat in enumerate(mech_vec_list):
        nm = mech_nm_list[i]
        if nm not in rd.tags_dict: #name filtering
            print nm + ' is not in tags_dict'
            #raw_input('Hit ENTER to continue')
            continue
        tags = rd.tags_dict[nm]
        if 'recessed' in nm:
            continue
        if 'HSI_Executive_Board_Room_Cabinet_Left' in nm:
            continue

        if rd.ig in tags or rd.te in tags:
            continue

        if rd.k in tags and rd.r in tags:
            #lab_str = 'Refrigerator'
            lab_str = 'Fridge'
            lab_num = 0
        elif rd.k in tags and rd.f in tags:
            lab_str = 'Freezer'
            lab_num = 1
        elif rd.k in tags and rd.c in tags:
            lab_str = 'Kitchen Cabinet'
            lab_num = 2
        elif rd.o in tags and rd.c in tags:
            lab_str = 'Office Cabinet'
            lab_num = 3
            if 'HSI_kitchen_cabinet_left' in nm:
                v_mat = 1.0*v_mat + 0.
        elif rd.do in tags and rd.s in tags:
            lab_str = 'Springloaded Door'
            lab_num = 4
        else:
            continue
        for v in v_mat.T:
            if rd.te in tags:
                labels_test.append(lab_str)
                chunks_test.append(mech_nm_list[i])
            else:
                labels.append(lab_str)
                if rd.ro in tags and append_robot:
                    chunks.append(mech_nm_list[i]+'_robot')
                else:
                    chunks.append(mech_nm_list[i])
        if rd.te in tags:
            feat_list_test.append(v_mat)
        else:
            feat_list.append(v_mat)
        print '-------------------------'
        print 'nm:', nm
        if nm == 'HSI_kitchen_cabinet_right':
            print '####################33'
            print '####################33'
            print '####################33'
            
    #chunks=None
    feats = np.column_stack(feat_list)

    # (length x samples), mechanism tags, mechanism+actor tags
    return feats, labels, chunks


def get_trans_mat(vecs, nState):
    # Still it's not correct since I am using observations instead of hidden states.
    # vecs = number_of_data x profile_length
    
    #init
    discrete_max  = 0.0
    discrete_vecs = np.zeros(vecs.shape)

    # discretization
    discrete_max = np.nanmax(vecs)

    # Non-negative states
    ## state_table = np.arange(0.0, discrete_max+0.000001, resol)
    state_table = np.linspace(0.0, discrete_max, nState)

    # Reset transition probability matrix
    trans_size = len(state_table) #int(np.ceil(discrete_max / resol)) + 1
    trans_mat  = np.zeros((trans_size, trans_size))
    trans_prob_mat = np.zeros((trans_size, trans_size))

    # Discretization and Update transition probability matrix
    n,m = vecs.shape
    for i in xrange(n):
        for j in xrange(m):
            if math.isnan(vecs[i][j]): 
                ## discrete_vecs[i][j] = vecs[i][j]                
                ## continue
                discrete_vecs[i][j] = (vecs[i][j-1] + vecs[i][j+1]) / 2.0
                if math.isnan(discrete_vecs[i][j]): 
                    print "we found nan"
                    sys.exit()
            else:                        
                discrete_vecs[i][j], _ = find_nearest(state_table, vecs[i][j])

            if j != 0:
                _, x_idx = find_nearest(state_table, discrete_vecs[i][j-1])
                _, y_idx = find_nearest(state_table, discrete_vecs[i][j])                

                # NOTE: really need?
                if x_idx <= y_idx:
                    trans_mat[x_idx,y_idx] += 1.0                

    # Set transition probability matrix
    for i in xrange(trans_size):
        total = np.sum(trans_mat[i,:])
        if total == 0: 
            ## trans_prob_mat[i,:] = 1.0 / float(trans_size)                    
            trans_prob_mat[i,i:] = 1.0 / float(trans_size-(i))                    
            ## trans_prob_mat[i,i] += 0.5
            ## if i!=0: trans_prob_mat[i,i-1] += 0.25
            ## if i!=trans_size: trans_prob_mat[i,i+1] += 0.25
            ## trans_prob_mat[i,:] = trans_prob_mat[i,:] / np.sum(trans_prob_mat[i,:])
        else:
            trans_prob_mat[i,:] = trans_mat[i,:] / total

    return trans_prob_mat, discrete_vecs


def approx_missing_value(vecs):

    new_vecs = np.zeros(vecs.shape)
    
    n,m = vecs.shape
    for i in xrange(n):
        for j in xrange(m):
            if math.isnan(vecs[i][j]): 
                new_vecs[i][j] = (vecs[i][j-1] + vecs[i][j+1]) / 2.0
                if math.isnan(new_vecs[i][j]): 
                    print "we found nan"
                    sys.exit()
            else:                        
                new_vecs[i][j] = vecs[i][j]
    
    return new_vecs

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx








    

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

        

def generate_roc_curve(mech_vec_list, mech_nm_list,
                       semantic_range = np.arange(0.2, 2.7, 0.3),
                       mech_range = np.arange(0.2, 6.5, 0.7),
                       n_prev_trials = 1, prev_c = 'r',
                       plot_prev=True, sem_c = 'b', sem_m = '+',
                       plot_semantic=True, semantic_label='operating 1st time and \n known mechanism class'):

    t_nm_list, t_mech_vec_list = [], []
    for i, nm in enumerate(mech_nm_list):
        ## print 'nm:', nm
        if 'known' in nm:
            continue
        t_nm_list.append(nm)
        t_mech_vec_list.append(mech_vec_list[i])

    data, _ = mar.create_blocked_dataset_semantic_classes(t_mech_vec_list, t_nm_list, append_robot = False)
    
    ## label_splitter = NFoldSplitter(cvtype=1, attr='labels')
    thresh_dict = ut.load_pickle('blocked_thresh_dict.pkl') # human + robot data
    mean_charlie_dict = thresh_dict['mean_charlie']
    mean_known_mech_dict = thresh_dict['mean_known_mech']

    #---------------- semantic class prior -------------
    if plot_semantic:
        fp_l_l = []
        mn_l_l = []
        err_l_l = []
        mech_fp_l_l = []
        mech_mn_l_l = []
        mech_err_l_l = []

        nfs = NFoldPartitioner(cvtype=1, attr='targets') # 1-fold ?
        label_splitter = splitters.Splitter(attr='partitions')            
        splits = [list(label_splitter.generate(x)) for x in nfs.generate(data)]            

        # Grouping by labels
        for l_wdata, l_vdata in splits: #label_splitter(data):

            print "Number of data: ", len(l_vdata.chunks)
        
            # Why zero??? Do we want specific chunk?  -> changed into 10
            lab = l_vdata.targets[0] # all same label
            chunk = l_vdata.chunks[0] # chunk should be independant!!
            trials = l_vdata.samples 

            if lab == 'Refrigerator':
                lab = 'Fridge'

            ## tot_mean = None
            ## tot_std  = None
            ## for chunk in l_vdata.chunks:
            ##     _, mean, std =  mean_charlie_dict[chunk] # mean except the specified chunk in same class
            ##     if tot_mean is None:
            ##         tot_mean = mean
            ##         tot_std  = std
            ##     else:
            ##         tot_mean += mean
            ##         tot_std += std

            ##     print chunk, mean[0], tot_mean[0]

            ## mean = tot_mean/float(len(l_vdata.chunks))
            ## std = tot_std/float(len(l_vdata.chunks))
            ## print mean[0], tot_mean[0], float(len(l_vdata.chunks))
            ## sys.exit()
            
            # Select evaluation chunk for the ROC ? 
            ## _, mean, std =  mean_charlie_dict[lab]
            _, mean, std =  mean_charlie_dict[chunk]

            # cutting into the same length
            min_len = min(len(mean), trials.shape[1])
            trials = trials[:,:min_len]
            mean = mean[:min_len]
            std = std[:min_len] #???

            mn_list = []
            fp_list, err_list = [], []
            for n in semantic_range:
                err = (mean + n*std) - trials                    
                #false_pos = np.sum(np.any(err<0, 1))
                #tot = trials.shape[0]
                false_pos = np.sum(err<0) # Count false cases
                tot = trials.shape[0] * trials.shape[1]
                fp_list.append(false_pos/(tot*0.01))
                err = err[np.where(err>0)] 
                err_list.append(err.flatten())
                mn_list.append(np.mean(err))
            err_l_l.append(err_list)
            fp_l_l.append(fp_list)
            mn_l_l.append(mn_list)

        
            
        ll = [[] for i in err_l_l[0]]  # why 0?
        for i,e in enumerate(err_l_l): # labels
            for j,l in enumerate(ll):  # multiplier range
                l.append(e[j])

        std_list = []
        for l in ll:
            std_list.append(np.std(np.concatenate(l).flatten()))

        mn_list = np.mean(np.row_stack(mn_l_l), 0).tolist() # means into a row
        fp_list = np.mean(np.row_stack(fp_l_l), 0).tolist()
        #pp.errorbar(fp_list, mn_list, std_list)

        ## mn_list = np.array(mn_l_l).flatten()
        ## fp_list = np.array(fp_l_l).flatten()
        
        pp.plot(fp_list, mn_list, '--'+sem_m+sem_c, label= semantic_label,
                mec=sem_c, ms=8, mew=2)
        #pp.plot(fp_list, mn_list, '-ob', label='with prior')

    #---------------- mechanism knowledge prior -------------
    if plot_prev:
        
        t_nm_list, t_mech_vec_list = [], []
        for i, nm in enumerate(mech_nm_list):
            ## print 'nm:', nm
            if 'known' in nm:
                t_nm_list.append(nm)
                t_mech_vec_list.append(mech_vec_list[i])
        if t_nm_list == []:
            t_mech_vec_list = mech_vec_list
            t_nm_list = mech_nm_list

        data, _ = mar.create_blocked_dataset_semantic_classes(t_mech_vec_list, t_nm_list, append_robot = False)
        
        ## chunk_splitter = NFoldSplitter(cvtype=1, attr='chunks')        
        nfs = NFoldPartitioner(cvtype=1, attr='chunks') # 1-fold ?
        chunk_splitter = splitters.Splitter(attr='partitions')            
        splits = [list(label_splitter.generate(x)) for x in nfs.generate(data)]            
        
        err_mean_list = []
        err_std_list = []
        fp_list = []
        for n in mech_range:
            false_pos = 0
            n_trials = 0
            err_list = []
            for _, l_vdata in splits: #chunk_splitter(data):
                lab = l_vdata.targets[0]
                trials = l_vdata.samples
                m = l_vdata.chunks[0]
                #one_trial = trials[0].reshape(1, len(trials[0]))
                one_trial = trials[0:n_prev_trials]

                ## print n, ": ", lab, chunk
                
                Ms, n_std = mean_known_mech_dict[m]
                mn_list, std_list = mar.estimate_theta(one_trial, Ms, plot=False, add_var = 0.0)
                mn_mech_arr = np.array(mn_list)
                std_mech_arr = np.array(std_list)

    #            trials = trials[:,:len(mn_mech_arr)]
                min_len = min(len(mn_mech_arr), trials.shape[1])
                trials = trials[:,:min_len]
                mn_mech_arr = mn_mech_arr[:min_len]
                std_mech_arr = std_mech_arr[:min_len]

                for t in trials:
                    err = (mn_mech_arr + n*std_mech_arr) - t
                    #false_pos += np.any(err<0)
                    #n_trials += 1
                    false_pos += np.sum(err<0)
                    n_trials += len(err)
                    err = err[np.where(err>0)]
                    err_list.append(err)

            e_all = np.concatenate(err_list)
            err_mean_list.append(np.mean(e_all))
            err_std_list.append(np.std(e_all))
            fp_list.append(false_pos/(n_trials*0.01))

        #pp.plot(fp_list, err_mean_list, '-o'+prev_c, label='knowledge of mechanism and \n opened earlier %d times'%n_prev_trials)
        pp.plot(fp_list, err_mean_list, '-o'+prev_c, mec=prev_c,
                ms=5, label='operating 2nd time and \n known mechanism identity')
        #pp.plot(fp_list, err_mean_list, '-or', label='with prior')


    pp.xlabel('False positive rate (percentage)', fontsize=22)
    pp.ylabel('Mean excess force (Newtons)', fontsize=22)
    pp.xlim(-0.5,45)
    mpu.legend()

    
def generate_roc_curve_no_prior(mech_vec_list, mech_nm_list):
    #pp.figure()
    data, _ = mar.create_blocked_dataset_semantic_classes(mech_vec_list, mech_nm_list, append_robot = False)
    ## chunk_splitter = NFoldSplitter(cvtype=1, attr='chunks')

    err_mean_list = []
    err_std_list = []
    fp_list = []
    all_trials = data.samples
    n_trials = all_trials.shape[0] * all_trials.shape[1]
    le = all_trials.shape[1]
    for n in np.arange(0.1, 1.7, 0.15):
        err = (all_trials[:,0]*n).T - all_trials.T + 2.
        false_pos = np.sum(err<0)
        err = err[np.where(err>0)]
        err_mean_list.append(np.mean(err))
        err_std_list.append(np.std(err))
        fp_list.append(false_pos/(n_trials*0.01))

    pp.plot(fp_list, err_mean_list, ':xy', mew=2, ms=8, label='No prior (ratio of \n initial force)', mec='y')

    err_mean_list = []
    err_std_list = []
    fp_list = []
    for f in np.arange(2.5, 45, 5.):
        err = f - all_trials
        false_pos = np.sum(err<0)
        err = err[np.where(err>0)]
        err_mean_list.append(np.mean(err))
        err_std_list.append(np.std(err))
        fp_list.append(false_pos/(n_trials*0.01))

    pp.plot(fp_list, err_mean_list, '-.^g', ms=8, label='No prior (constant)', mec='g')

    pp.xlabel('False positive rate (percentage)')
    pp.ylabel('Mean excess force (Newtons)')
    #mpu.legend(display_mode='less_space', draw_frame=False)
    mpu.legend()
    pp.xlim(-0.5,45)
    

def get_data(pkl_file, mech_class='Office Cabinet', verbose=False, renew=False):

    ######################################################    
    # Get Training Data
    if os.path.isfile(pkl_file) and renew==False:
        print "Saved pickle found"
        data = ut.load_pickle(pkl_file)
        data_vecs = data['data_vecs']
        data_mech = data['data_mech']
        data_chunks = data['data_chunks']
    else:        
        print "No saved pickle found"        
        data_vecs, data_mech, data_chunks = get_all_blocked_detection()
        data = {}
        data['data_vecs'] = data_vecs
        data['data_mech'] = data_mech
        data['data_chunks'] = data_chunks
        ut.save_pickle(data,pkl_file)

    ## from collections import OrderedDict
    ## print list(OrderedDict.fromkeys(data_mech)), len(list(OrderedDict.fromkeys(data_mech)))
    ## print list(OrderedDict.fromkeys(data_chunks)), len(list(OrderedDict.fromkeys(data_chunks)))
    ## print len(data_mech), data_vecs.shape
    ## sys.exit()
    
    # Filtering
    idxs = np.where([mech_class in i for i in data_mech])[0].tolist()
    print "Load ", mech_class

    ## print data_mech
    ## print data_vecs.shape, np.array(data_mech).shape, np.array(data_chunks).shape
    data_vecs = data_vecs[:,idxs]
    data_mech = [data_mech[i] for i in idxs]
    data_chunks = [data_chunks[i] for i in idxs]

    ## X data
    data_vecs = np.array([data_vecs.T]) # category x number_of_data x profile_length
    data_vecs[0] = approx_missing_value(data_vecs[0])    

    ## ## time step data
    ## m, n = data_vecs[0].shape
    ## aXData = np.array([np.arange(0.0,float(n)-0.0001,1.0).tolist()] * m)

    if verbose==True:
        print data_vecs.shape, np.array(data_mech).shape, np.array(data_chunks).shape
    
    return data_vecs, data_mech, data_chunks

    
if __name__ == '__main__':

    import optparse
    import glob
    p = optparse.OptionParser()
    p.add_option('--fig_roc_human', action='store_true',
                 dest='fig_roc_human',
                 help='generate ROC like curve from the BIOROB dataset.')
    opt, args = p.parse_args()
       

    if opt.fig_roc_human:
        pkl_list = glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/*_new.pkl')
        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)
        mpu.set_figure_size(10, 7.)

        pp.figure()
        generate_roc_curve_no_prior(mech_vec_list, mech_nm_list)
        generate_roc_curve(mech_vec_list, mech_nm_list)
        f = pp.gcf()
        f.subplots_adjust(bottom=.15, top=.96, right=.98, left=0.15)
        pp.savefig('roc_compare.pdf')
        pp.show()

    else:
            
        pkl_list = glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/*_new.pkl') #+ glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/perfect_perception/*_new.pkl') + glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/simulate_perception/*_new.pkl')

        ## root_path = os.environ['HRLBASEPATH']    
        ## ## pkl_list = glob.glob(root_path+'_data/usr/advait/ram_www/RAM_db_of_different_kinds/RAM_db/*_new.pkl')
        ## pkl_list = glob.glob(root_path+'_data/usr/advait/ram_www/*_new.pkl')


        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36) #get vec_list, name_list

        ## get_all_blocked_detection()
        ## blocked_detection(mech_vec_list, mech_nm_list)
        #blocked_detection_n_equals_1(mech_vec_list, mech_nm_list)

        ## get_discrete_test_from_mean_dict('non_robot_mean_dict.pkl')
        ## get_discrete_test_from_mean_dict('blocked_thresh_dict.pkl')

        # Get normal data from 3 door, two robot, and one human 
        # Get collision data from any case


