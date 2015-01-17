#!/usr/bin/python

import sys, os, copy
import numpy as np, math
import glob
import socket
import time
import random 

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters

# Util
import hrl_lib.util as ut
import matplotlib.pyplot as pp
import matplotlib as mpl

import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad
import hrl_anomaly_detection.advait.mechanism_analyse_RAM as mar
from hrl_anomaly_detection.HMM.learning_hmm import learning_hmm
from hrl_anomaly_detection.HMM.anomaly_checker import anomaly_checker
import hrl_anomaly_detection.door_opening.door_open_common as doc
import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
import hrl_lib.matplotlib_util as mpu
import hrl_anomaly_detection.advait.mechanism_analyse_advait as maa

roc_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/roc_sig_0_3/door_roc_data'
    
def get_interp_data(x,y, ang_interval=0.25):

    # Cubic-spline interpolation
    from scipy import interpolate
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(x[0], x[-1], ang_interval)
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew   


def genCrossValData(data_path, cross_data_path, human_only=True, bSimBlock=False, ang_interval=1.0):

    if os.path.isdir(cross_data_path) == False:
        os.system('mkdir -p '+cross_data_path)

    save_file = os.path.join(cross_data_path, 'data_1.pkl')        
    if os.path.isfile(save_file) is True: return
    
    # human and robot data
    if human_only:
        pkl_list = glob.glob(data_path+'RAM_db/*_new.pkl') + \
          glob.glob(data_path+'RAM_db/robot_trials/perfect_perception/*_new.pkl') + \
          glob.glob(data_path+'RAM_db/robot_trials/simulate_perception/*_new.pkl')
    else:
        pkl_list = glob.glob(data_path+'RAM_db/*_new.pkl') + \
          glob.glob(data_path+'RAM_db/robot_trials/perfect_perception/*_new.pkl')

    r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
    mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36) #get vec_list, name_list

    # data consists of (mech_vec_matrix?, label_string(Freezer...), mech_name)
    data, _ = mar.create_blocked_dataset_semantic_classes(mech_vec_list,
                                                      mech_nm_list, append_robot = True)    

    # create the generator
    #label_splitter = NFoldSplitter(cvtype=1, attr='labels')
    nfs = NFoldPartitioner(cvtype=1) # 1-fold ?
    spl = splitters.Splitter(attr='partitions')
    splits = [list(spl.generate(x)) for x in nfs.generate(data)] # split by chunk
        
    d = {}
    count = 0
    for l_wdata, l_vdata in splits:

        if human_only:       
            if 'robot' in l_vdata.chunks[0]: 
                print "Pass a robot chunk"
                continue
        else:
            if 'robot' not in l_vdata.chunks[0]: 
                print "Pass a non-robot chunk"
                continue
            
        count += 1
        non_robot_idxs = np.where(['robot' not in i for i in l_wdata.chunks])[0] # if there is no robot, true 
        # find same target samples in non_robot target samples        
        idxs = np.where(l_wdata.targets[non_robot_idxs] == l_vdata.targets[0])[0] 

        train_trials = (l_wdata.samples[non_robot_idxs])[idxs]
        test_trials  = l_vdata.samples # chunk                                       
        chunk = l_vdata.chunks[0]
        target = l_vdata.targets[0]


        if ang_interval < 1.0:
            # resampling with specific interval
            bin_size = ang_interval
            h_config = np.arange(0.0, nMaxStep, 1.0)
            new_test_trials = []

            for i in xrange(len(test_trials)):
                new_h_config, new_test_trial = get_interp_data(h_config, test_trials[i]) 
                new_test_trials.append(new_test_trial)
        else:
            new_test_trials = test_trials

        if bSimBlock:                
            new_test_trials, test_anomaly_idx, org_test_trials = \
              simulated_block_conv(new_test_trials, \
                                   int(len(new_test_trials[0])*0.2), \
                                   int(len(new_test_trials[0])*0.8), \
                                   ang_interval, \
                                   nRandom=5) 
        else:
            test_anomaly_idx = []
            org_test_trials = test_trials

        #SAVE!!
        d['train_trials']     = train_trials
        d['test_trials']      = new_test_trials
        d['test_anomaly_idx'] = test_anomaly_idx
        d['test_trials_no_simblock']  = org_test_trials            
        d['chunk'] = chunk
        d['target'] = target
        save_file = os.path.join(cross_data_path, 'data_'+str(count)+'.pkl')
        ut.save_pickle(d, save_file)


            
def simulated_block_conv(trials, nMinStep, nMaxStep, ang_interval, nRandom=5):
    print "Convert into simulated block data"

    rnd_block_l = []
    rnd_slope_l = []
    for n in xrange(nRandom):
        while True:
            blocked_step = random.randint(nMinStep, nMaxStep)
            if blocked_step in rnd_block_l: continue
            else: 
                rnd_block_l.append(blocked_step)
                break

        while True:
            blocked_slope = random.uniform(0.0, 1.5)
            if blocked_slope in rnd_slope_l: continue
            else:
                rnd_slope_l.append(blocked_slope)
                break

    new_trials = None
    new_anomaly_pts = []
    org_trials = None
    for trial in trials:
        for n,s in zip(rnd_block_l,rnd_slope_l):

            f_trial = trial[:n]

            nRemLength = len(trial) - n
            x = (np.arange(0.0, nRemLength, 1.0)+1.0) * ang_interval
            ## b_trial = x*s + f_trial[-1]
            ## b_trial = 0.1*(np.exp(x)-1.0) + trial[n:]
            b_trial = x*0.1 + trial[n:]

            ## # Restrict max
            ## for i, sample in enumerate(b_trial):
            ##     if sample > 15.0: b_trial[i] = 15.0

            new_trial = np.hstack([f_trial, b_trial])
            if new_trials is None:
                new_trials = new_trial
                org_trials = trial
            else:
                new_trials = np.vstack([new_trials, new_trial])
                org_trials = np.vstack([org_trials, trial])

            new_anomaly_pts.append(n)

            ## pp.figure()                    
            ## x = np.arange(0.0, len(trial), 1.0)*ang_interval
            ## print x.shape, new_trial.shape, n
            ## pp.plot(x, new_trial,'r')
            ## pp.plot(x, trial,'b')
            ## pp.plot([n-1,n-1],[0,10.0])
            ## pp.show()

            

    return new_trials, new_anomaly_pts, org_trials
    #return trials, new_anomaly_pts

    
def tuneCrossValHMM(cross_data_path, cross_test_path, nState, nMaxStep, trans_type="left_right"):

    if not(os.path.isdir(cross_test_path+'/'+str(nState))):
        os.system('mkdir -p '+cross_test_path+'/'+str(nState)) 
        time.sleep(0.5)
    elif hcu.is_file(cross_test_path+'/'+str(nState), 'complete'):
        print "#############################################################################"
        print "All file exist "
        print "#############################################################################"        
        return

    ## Load data pickle
    train_data = []
    test_data = []
    test_idx_list = []        
    for f in os.listdir(cross_data_path):
        if f.endswith(".pkl"):
            test_num = f.split('_')[-1].split('.')[0]
            
            # Load data
            d = ut.load_pickle( os.path.join(cross_data_path,f) )
            train_trials = d['train_trials']
            test_trials  = d['test_trials']
            chunk        = d['chunk'] 
            target       = d['target']

            test_idx_list.append(test_num)
            train_data.append(train_trials)
            test_data.append(test_trials)

            
    ####################################################################
    strMachine = socket.gethostname()+"_"+str(os.getpid())    

    count = 0
    for i in xrange(len(train_data)):

        B_tune_pkl = cross_test_path+'/'+str(nState)+'/B_tune_data_'+str(test_idx_list[i])+'.pkl'
        mutex_file = cross_test_path+'/'+str(nState)+'/running_'+str(test_idx_list[i])+'_'+strMachine+'.txt'                 
        if os.path.isfile(B_tune_pkl): 
            count += 1
            continue
        elif hcu.is_file(cross_test_path+'/'+str(nState), 'running_'+str(test_idx_list[i])): 
            print "#############################################################################"
            print "Another machine Is Running already, ignore this : " , nState
            print "#############################################################################"
            continue
        else:
            os.system('touch '+mutex_file)
        
        lh = learning_hmm(aXData=train_data[i], nState=nState, 
                          nMaxStep=nMaxStep, trans_type=trans_type)            

        ## lh.fit(lh.aXData, verbose=False)    
        
        lh.param_optimization(save_file=B_tune_pkl)

        os.system('rm '+mutex_file)

    if count == len(train_data):
        print "#############################################################################"
        print "All file exist "
        print "#############################################################################"        
        os.system('touch '+os.path.join(cross_test_path,str(nState),'complete.txt'))
        

def load_cross_param(cross_data_path, cross_test_path, nMaxStep, trans_type, test=False):

    # Get the best param for training set
    test_idx_list         = []
    train_data            = []
    test_data             = []
    test_anomaly_idx_data = []
    org_test_data         = []
    B_list        = []
    nState_list   = []
    score_list    = []

    #-----------------------------------------------------------------        
    for f in os.listdir(cross_data_path):
        if f.endswith(".pkl"):
            test_num = f.split('_')[-1].split('.')[0]

            # Load data
            try:
                d = ut.load_pickle( os.path.join(cross_data_path,f) )
            except:
                d = ut.load_pickle( os.path.join(cross_data_path,f) )
                
            train_trials     = d['train_trials']
            test_trials      = d['test_trials']
            test_anomaly_idx = d.get('test_anomaly_idx', [nMaxStep]*len(test_trials))            
            org_test_trials  = d['test_trials_no_simblock']            
            chunk            = d['chunk'] 
            target           = d['target']

            if test_anomaly_idx == []: test_anomaly_idx = [nMaxStep]*len(test_trials)            
            
            test_idx_list.append(test_num)
            train_data.append(train_trials)
            test_data.append(test_trials)            
            test_anomaly_idx_data.append(test_anomaly_idx)
            org_test_data.append(org_test_trials)            

            # find parameters with a minimum score
            min_score  = 10000.0
            min_nState = None
            dir_list = os.listdir(cross_test_path)
            for d in dir_list:
                if os.path.isdir(cross_test_path+'/'+d) is not True: continue
                if not(str(10) in d) and test==True: continue
                if "roc" in d: continue
                if "ab" in d: continue
                if "nFuture" in d: continue

                f_pkl = os.path.join(cross_test_path, d, 'B_tune_data_'+str(test_num)+'.pkl')
                hcu.wait_file(f_pkl)                                                            
                param_dict = ut.load_pickle(f_pkl)

                if min_nState == None:
                    min_nState = param_dict['nState']
                    min_score  = param_dict['score']
                    min_B      = param_dict['B']
                elif min_score > param_dict['score']:
                    min_nState = param_dict['nState']
                    min_score  = param_dict['score']
                    min_B      = param_dict['B']
                                                            
            B_list.append(min_B)
            nState_list.append(min_nState)
            score_list.append(min_score)


    print "Load cross validation params complete"
    return test_idx_list, train_data, test_data, test_anomaly_idx_data, org_test_data, B_list, nState_list
    

    
    
        
def get_threshold_by_cost(cross_data_path, cross_test_path, cost_ratios, nMaxStep, \
                          trans_type, nFutureStep, aws=False, test=False):

    # Get the best param for training set
    test_idx_list, train_data, _, _, _, B_list, nState_list = load_cross_param(cross_data_path, \
                                                                                 cross_test_path, \
                                                                                 nMaxStep, \
                                                                                 trans_type)
        
    #-----------------------------------------------------------------            
    print "------------------------------------------------------"
    print "Loaded all best params B and nState"
    print "------------------------------------------------------"

    strMachine = socket.gethostname()+"_"+str(os.getpid())    
    X_test = np.arange(0.0, 36.0, 1.0)
    start_step = 2
    
    sig_mult        = np.arange(5.0, 50.0+0.00001, 2.0)
    sig_offset      = np.arange(0.0, 1.5+0.00001, 0.1)

    ## sig_mult   = np.arange(0.5, 10.0+0.00001, 0.1)
    ## sig_offset = [0.0]
    
    param_list = []
    for a in sig_mult:
        for b in sig_offset:
            param_list.append([a,b])

    #-----------------------------------------------------------------        
    for i, test_idx in enumerate(test_idx_list):

        tune_res_path = os.path.join(cross_test_path, 'nFuture_'+str(nFutureStep), "ab_for_d_"+str(test_idx))
        if not(os.path.isdir(tune_res_path)):
            os.system('mkdir -p '+tune_res_path) 
            time.sleep(0.5)

        # init anomaly checked list
        ac_res = False
        ## false_pos = np.zeros((len(param_list), len(train_data[i]), len(train_data[i][0])-start_step))
        fp_l_l  = None
        err_l_l = None
        bAnomaly_l_l = []                       
        fp_param = None
        err_param = None
        
        for c_idx, cost_ratio in enumerate(cost_ratios):
            
            tune_res_file = "ab_for_d_"+str(test_idx)+"_cratio_"+str(cost_ratio)+'.pkl'
            tune_res_file = os.path.join(tune_res_path, tune_res_file)

            mutex_file_part = 'running_'+str(test_idx)+"_cratio_"+str(cost_ratio)
            mutex_file_full = mutex_file_part+"_"+strMachine+'.txt'        
            mutex_file = os.path.join(tune_res_path,mutex_file_full)

            if os.path.isfile(tune_res_file): continue
            elif hcu.is_file(tune_res_path, mutex_file_part): continue
            elif os.path.isfile(mutex_file): continue
            os.system('touch '+mutex_file)

            # For AWS
            if aws:
                if hcu.is_file_w_time(tune_res_path, mutex_file_part, exStrName=mutex_file_full, \
                                      loop_time=1.0, wait_time=15.0, priority_check=True):
                    os.system('rm '+mutex_file)
                    continue

            # --------------------------------------------------------                
            if ac_res == False:

                ac_res = True
                print "Get train data ", test_idx
                nState = nState_list[i]
                B      = B_list[i]

                ## min_cost = 10000
                ## min_fp   = None
                ## min_err  = None

                # Set a learning object
                lh = None
                lh = learning_hmm(aXData=train_data[i], nState=nState, \
                                  nMaxStep=nMaxStep,\
                                  nFutureStep=nFutureStep,\
                                  nCurrentStep=nCurrentStep, trans_type=trans_type)
                lh.fit(lh.aXData, B=B, verbose=False)    

                for j, trial in enumerate(train_data[i]):

                    # Init checker
                    ac = anomaly_checker(lh)

                    # Simulate each profile
                    for k in xrange(len(trial)):
                        # Update buffer
                        ac.update_buffer(trial[:k+1])

                        if k>= start_step:                    
                            # check anomaly score
                            bAnomaly_l, err_l = ac.check_anomaly_batch(trial[k], param_list)

                            if fp_l_l is None:
                                fp_l_l = bAnomaly_l
                                err_l_l = err_l
                            else:
                                fp_l_l = np.vstack([fp_l_l, bAnomaly_l])
                                err_l_l = np.vstack([err_l_l, err_l])
                            ## if bAnomaly: 
                            ##     false_pos[j, k-start_step] = 1.0 
                            ## else:
                            ##     err_l.append(mean_err)

                fp_param = np.mean(fp_l_l, axis=0)
                err_param = np.sum(np.array(err_l_l), axis=0) / (len(fp_l_l)-np.sum(fp_l_l, axis=0))
                                                        
            # --------------------------------------------------------

            cost_param = cost_ratio*fp_param + (1.0-cost_ratio)*err_param
            min_idx  = np.where(cost_param == cost_param.min())[0][0]

            if cost_param.min() != cost_param[min_idx]:
                print "ERROR wrong cost param!!!!!!!!!!!!!!!!!!!!!"
                sys.exit()

            tune_res_dict = {}
            tune_res_dict['test_idx'] = test_idx
            tune_res_dict['min_cost'] = cost_param[min_idx]
            tune_res_dict['min_sig_mult'] = param_list[min_idx][0]
            tune_res_dict['min_sig_offset'] = param_list[min_idx][1]
            tune_res_dict['min_fp'] = fp_param[min_idx]
            tune_res_dict['min_err'] = err_param[min_idx]

            ut.save_pickle(tune_res_dict, tune_res_file)
            os.system('rm '+mutex_file)


def get_roc_by_cost(cross_data_path, cross_test_path, cost_ratio, nMaxStep, \
                    trans_type, nFutureStep=5, aws=False, bSimBlock=False, ang_interval=1.0, \
                    sig_mult=None):

    # Get the best param for training set
    test_idx_list, train_data, test_data, test_anomaly_idx_data, org_test_data, B_list, nState_list = \
      load_cross_param(cross_data_path, cross_test_path, nMaxStep, trans_type)   

    #-----------------------------------------------------------------
    strMachine = socket.gethostname()+"_"+str(os.getpid())
    bComplete  = True
    start_step = 2 * int(1.0/ang_interval)
    
    for i, test_idx in enumerate(test_idx_list):
        
        roc_res_path = os.path.join(cross_test_path, 'nFuture_'+str(nFutureStep), "roc_for_d_"+str(test_idx))
        if not(os.path.isdir(roc_res_path)):
            os.system('mkdir -p '+roc_res_path) 
            time.sleep(0.5)
        
        # Check saved or mutex files
        if sig_mult != None:
            roc_res_file = os.path.join(roc_res_path, "roc_"+str(test_idx)+ \
                                        "_sig_mult_"+str(sig_mult)+'.pkl')
            mutex_file_part = "running_"+str(test_idx)+"_sig_mult_"+str(sig_mult)
        else:
            roc_res_file = os.path.join(roc_res_path, "roc_"+str(test_idx)+ \
                                        "_cratio_"+str(cost_ratio)+'.pkl')
            mutex_file_part = "running_"+str(test_idx)+"_cratio_"+str(cost_ratio)  
            
        mutex_file_full = mutex_file_part+"_"+strMachine+'.txt'                     
        mutex_file = os.path.join(roc_res_path, mutex_file_full)

        if os.path.isfile(roc_res_file): continue
        elif hcu.is_file(roc_res_path, mutex_file_part):
            bComplete = False
            continue
        elif os.path.isfile(mutex_file): 
            bComplete = False
            continue
        os.system('touch '+mutex_file)

        # For AWS
        if aws:
            if hcu.is_file_w_time(roc_res_path, mutex_file_part, exStrName=mutex_file_full, loop_time=1.0, wait_time=15.0, priority_check=True):
                os.system('rm '+mutex_file)
                continue       
    
        tune_res_file = "ab_for_d_"+str(test_idx)+"_cratio_"+str(cost_ratio)+'.pkl'
        tune_res_file = os.path.join(cross_test_path, 'nFuture_'+str(nFutureStep), "ab_for_d_"+str(test_idx), tune_res_file)

        if os.path.isfile(tune_res_file) is False: 
            bComplete = False
            os.system('rm '+mutex_file)            
            continue
        
        ## hcu.wait_file(tune_res_file)
        param_dict = ut.load_pickle(tune_res_file)
        min_sig_mult    = param_dict['min_sig_mult']
        min_sig_offset  = param_dict['min_sig_offset']

        if test_idx != param_dict['test_idx']:
            print "------------------------------------------------------"
            print "Test index is not same: ", test_idx, param_dict['test_idx']
            print "------------------------------------------------------"

        nState = nState_list[i]
        B      = B_list[i]

        
        lh = learning_hmm(aXData=train_data[i], nState=nState, \
                          nMaxStep=nMaxStep, \
                          nFutureStep=nFutureStep, \
                          trans_type=trans_type)
        lh.fit(lh.aXData, B=B, verbose=False)    

        # Init variables
        ## false_pos  = np.zeros((len(test_data[i]), len(test_data[i][0])-start_step))        
        ## true_neg   = []
        false_pos  = None
        sef_l      = [] # simulated excess force
        sat_l      = [] # simulated anomaly time
        err_l      = []        
        X_test = np.arange(0.0, len(test_data[i][0]), 1.0) * ang_interval
        test_anomaly_idx = test_anomaly_idx_data[i]        
        
        for j, trial in enumerate(test_data[i]):

            #temp
            if sig_mult != None:
                min_sig_mult = sig_mult
                min_sig_offset = 0.0

            org_trial = org_test_data[i][j]

            
            # Init checker
            ac = anomaly_checker(lh, sig_mult=min_sig_mult, sig_offset=min_sig_offset)
            fp_l = np.zeros((test_anomaly_idx[j]-start_step))
            ## tn_l = np.zeros((nMaxStep-test_anomaly_idx[j]))

            # Simulate each profile
            for k in xrange(len(trial)):
                # Update buffer
                ac.update_buffer(trial[:k+1])

                if k>= start_step:                    
                    # check anomaly score
                    bAnomaly, mean_err, _ = ac.check_anomaly(trial[k])

                    if bAnomaly == 0.0: err_l.append(mean_err)
                    
                    if bAnomaly and k < test_anomaly_idx[j]: 
                        fp_l[k-start_step] = 1.0 
                    elif bAnomaly and k >= test_anomaly_idx[j]:
                        
                        sef_l.append(trial[k]-org_trial[k])
                        sat_l.append((k-(test_anomaly_idx[j]-1))*ang_interval)
                        break                                                                        
                    ## elif bAnomaly is False and k >= test_anomaly_idx[j]:
                        ## tn_l[k-test_anomaly_idx[j]] = 1.0 

            if false_pos is None:
                false_pos = fp_l
                ## true_neg  = tn_l
            else:
                false_pos = np.hstack([false_pos, fp_l])
                ## true_neg  = np.hstack([true_neg, tn_l])

        print "--------------------"
        print "Test done: ", test_idx, " mean_fp: ", np.mean(false_pos), \
          "err: ", np.mean(np.array(err_l)) #, " mean_sim_force: ", np.mean(np.array(sef_l))
        print "--------------------"
        
                    
        roc_res_dict = {}
        roc_res_dict['test_idx'] = test_idx
        roc_res_dict['cost_ratio'] = cost_ratio
        roc_res_dict['min_sig_mult'] = min_sig_mult
        roc_res_dict['min_sig_offset'] = min_sig_offset
        
        roc_res_dict['false_positive'] = false_pos
        roc_res_dict['sim_mean_force'] = sef_l
        roc_res_dict['sim_anomaly_time'] = sat_l
        roc_res_dict['force_error'] = err_l
        ut.save_pickle(roc_res_dict, roc_res_file)
        os.system('rm '+mutex_file)


    if bComplete:
        t_false_pos  = None
        ## t_true_neg   = None
        t_sef_l      = []        
        t_sat_l      = []        
        t_err_l      = []        
        
        for i, test_idx in enumerate(test_idx_list):
            # Check saved or mutex files
            roc_res_path = os.path.join(cross_test_path, 'nFuture_'+str(nFutureStep), "roc_for_d_"+str(test_idx))

            if sig_mult != None:
                roc_res_file = os.path.join(roc_res_path, "roc_"+str(test_idx)+ \
                                            "_sig_mult_"+str(sig_mult)+'.pkl')
            else:
                roc_res_file = os.path.join(roc_res_path, "roc_"+str(test_idx)+ \
                                            "_cratio_"+str(cost_ratio)+'.pkl')

            roc_dict = ut.load_pickle(roc_res_file)

            if t_false_pos is None:                
                t_false_pos = roc_dict['false_positive']
                ## t_true_neg = np.array(roc_dict.get('true_negative',[0]))
            else:
                ## print roc_dict['min_sig_mult'], roc_dict['min_sig_offset'], np.mean(np.array(roc_dict['false_positive']))*100.0, " : ", roc_dict['cost_ratio'], test_idx
                t_false_pos = np.hstack([t_false_pos, roc_dict['false_positive']])
                ## t_true_neg  = np.vstack([t_true_neg, np.array(roc_dict.get('true_negative',[0]))])

            t_sef_l += roc_dict.get('sim_mean_force', [0.0])                
            t_sat_l += roc_dict.get('sim_anomaly_time', [0.0])                
            t_err_l += roc_dict['force_error']

        fp  = np.mean(t_false_pos.flatten()) * 100.0
        ## tn  = np.mean(t_true_neg.flatten()) * 100.0
        sef = np.mean(np.array(t_sef_l).flatten())
        sat = np.mean(np.array(t_sat_l).flatten())
        err = np.mean(np.array(t_err_l).flatten())

        return fp, sef, sat, err

    ## return fp, err
    return 0., 0., 0., 0.

    
def generate_roc_curve(cross_data_path, cross_test_path, future_steps, cost_ratios, ROC_target, nMaxStep=36, \
                       trans_type='full', bSimBlock=False, bPlot=False, bAWS=False, \
                       ang_interval=1.0):

    if "human" in ROC_target:
        genCrossValData(data_path, cross_data_path, bSimBlock=bSimBlock, ang_interval=ang_interval)
    elif "robot" in ROC_target:
        genCrossValData(data_path, cross_data_path, human_only=False, bSimBlock=bSimBlock, \
                        ang_interval=ang_interval)
    else:
        print "No task defined: ", ROC_target
        sys.exit()
            
    
    # 1) HMM param optimization                
    for nState in xrange(10,35,1):        
        tuneCrossValHMM(cross_data_path, cross_test_path, nState, nMaxStep, trans_type)
    
    # --------------------------------------------------------            
    import itertools
    colors = itertools.cycle(['g', 'm', 'c', 'k'])
    shapes = itertools.cycle(['x','v', 'o', '+'])


    for nFutureStep in future_steps:

        # Evaluate threshold in terms of training set
        get_threshold_by_cost(cross_data_path, cross_test_path, cost_ratios, \
                              nMaxStep, trans_type, \
                              nFutureStep=nFutureStep, aws=bAWS)

        # --------------------------------------------------------
        fp_list = []
        ## tn_list = []
        sef_list = []
        sat_list = []
        err_list = []

        for cost_ratio in cost_ratios:
            fp, sef, sat, err = get_roc_by_cost(cross_data_path, cross_test_path, \
                                                cost_ratio, nMaxStep, \
                                                trans_type, nFutureStep=nFutureStep, \
                                                aws=bAWS, bSimBlock=bSimBlock, \
                                                ang_interval=ang_interval)
            fp_list.append(fp)
            ## tn_list.append(tn)
            sef_list.append(sef)
            sat_list.append(sat)
            err_list.append(err)

        ## sig_mults   = np.arange(5.0, 50.0+0.00001, 2.0)    
        ## cost_ratio = 1.0
        ## for sig_mult in sig_mults:
        ##     fp, sef, sat, err = get_roc_by_cost(cross_data_path, cross_test_path, \
        ##                                   cost_ratio, nMaxStep, \
        ##                                   trans_type, nFutureStep=nFutureStep, \
        ##                                   aws=bAWS, bSimBlock=bSimBlock, \
        ##                                   ang_interval=ang_interval, sig_mult=sig_mult)
        ##     fp_list.append(fp)
        ##     ## tn_list.append(tn)
        ##     sef_list.append(sef)
        ##     sat_list.append(sat)
        ##     err_list.append(err)        

        #---------------------------------------
        if bPlot:

            color = colors.next()
            shape = shapes.next()

            idx_list = sorted(range(len(err_list)), key=lambda k: err_list[k])
            ## idx_list = sorted(range(len(fp_list)), key=lambda k: fp_list[k])
            sorted_fp_list  = [fp_list[i] for i in idx_list]
            sorted_sef_list = [sef_list[i] for i in idx_list]
            sorted_sat_list = [sat_list[i] for i in idx_list]
            sorted_err_list = [err_list[i] for i in idx_list]            

            semantic_label=str(nFutureStep)+' step PHMM anomaly detection', 
            sem_l='-'; sem_c=color; sem_m=shape                        

            if bSimBlock:
                
                ## pp.plot(cost_ratios, sef_list, sem_l+sem_m+sem_c, label= semantic_label,
                ##         mec=sem_c, ms=6, mew=2)                
                pp.plot(sorted_fp_list, sorted_sat_list, sem_l+sem_m+sem_c, label= semantic_label,
                        mec=sem_c, ms=6, mew=2)
            else:
                pp.plot(sorted_fp_list, sorted_err_list, sem_l+sem_m+sem_c, label= semantic_label,
                        mec=sem_c, ms=6, mew=2)
                
    #---------------------------------------            
    if bPlot:
        ## pp.plot(fp_list, mn_list, '--'+sem_m, label= semantic_label,
        ##         ms=6, mew=2)
        ## pp.legend(loc='best',prop={'size':16})
        pp.legend(loc=1,prop={'size':14})
        pp.xlim(-0.1,5)
        ## pp.ylim(0.,4)            
        pp.show()
        ## pp.savefig('robot_roc_sig_0_3.pdf')

            

    
    
    
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='renew',
                 default=False, help='Renew pickle files.')
    p.add_option('--cross_val', '--cv', action='store_true', dest='bCrossVal',
                 default=False, help='N-fold cross validation for parameter')
    p.add_option('--fig_roc_human', action='store_true', dest='bROCHuman', 
                 default=False, help='generate ROC like curve from the BIOROB dataset.')
    p.add_option('--fig_roc_robot', action='store_true', dest='bROCRobot',
                 default=False, help='Plot roc curve wrt robot data')
    p.add_option('--simulated_block', '--sb', action='store_true', dest='bSimBlock',
                 default=False, help='Add simulated & blocked data')
    p.add_option('--fig_roc_plot', '--plot', action='store_true', dest='bROCPlot',
                 default=False, help='Plot roc curve wrt robot data')
    p.add_option('--aws', action='store_true', dest='bAWS',
                 default=False, help='Use amazon cloud computing service')
    p.add_option('--block', '--b', action='store_true', dest='bUseBlockData',
                 default=False, help='Use blocked data')
    p.add_option('--animation', '--ani', action='store_true', dest='bAnimation',
                 default=False, help='Plot by time using animation')

    p.add_option('--optimize_mv', '--mv', action='store_true', dest='bOptMeanVar',
                 default=False, help='Optimize mean and vars for B matrix')
    p.add_option('--approx_pred', '--ap', action='store_true', dest='bApproxObsrv',
                 default=False, help='Approximately compute the distribution of multi-step observations')
    p.add_option('--fig_roc_phmm_comp_plot', '--pc_plot', action='store_true', dest='bROCPHMMPlot',
                 default=False, help='Plot phmm comparison roc curve wrt robot data')
    p.add_option('--all_path_plot', '--all', action='store_true', dest='bAllPlot',
                 default=False, help='Plot all paths')
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out everything')
    opt, args = p.parse_args()

    ## Init variables
    ## data_path = os.environ['HRLBASEPATH']+'_data/usr/advait/ram_www/data_from_robot_trials/'
    data_path = os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook/'
    root_path = os.environ['HRLBASEPATH']+'/'
    nMaxStep  = 36 # total step of data. It should be automatically assigned...
    nFutureStep = 8
    nCurrentStep = 14  #14
    ## trans_type = "left_right"
    trans_type = "full"


    # for block test
    if opt.bUseBlockData or opt.bAllPlot:    
        nClass = 2
        cls = doc.class_list[nClass]
        mech = 'kitchen_cabinet_pr2'
        ## mech = 'kitchen_cabinet_cody'
        ## mech = 'ikea_cabinet_pr2'
    else:
        nClass = 2
        cls = doc.class_list[nClass]


    if opt.bCrossVal is False and opt.bROCHuman is False and opt.bROCRobot is False: 
        
        pkl_file  = "mech_class_"+doc.class_dir_list[nClass]+".pkl"      
        data_vecs, _, _ = mad.get_data(pkl_file, mech_class=cls, renew=opt.renew) # human data       
        B_tune_pkl = "B_tune_"+doc.class_dir_list[nClass]+".pkl"        
        
        if os.path.isfile(B_tune_pkl) is False:
            lh = learning_hmm(aXData=data_vecs[0], nState=30, 
                              nMaxStep=nMaxStep, nFutureStep=nFutureStep, 
                              nCurrentStep=nCurrentStep, trans_type=trans_type)    
            lh.param_optimization(save_file=B_tune_pkl)
        else:                       
            A, B, pi, nState = doc.get_hmm_init_param(mech_class=cls, pkl_file=B_tune_pkl)        
            ## B = mad.get_trans_mat(data_vecs[0], nState)
            ## print np.array(B).shape, nState
            ## sys.exit()
            
        # Training 
        lh = learning_hmm(aXData=data_vecs[0], nState=nState, 
                          nMaxStep=nMaxStep, nFutureStep=nFutureStep, 
                          nCurrentStep=nCurrentStep, trans_type=trans_type)    

    ###################################################################################            
    if (opt.bROCHuman or opt.bCrossVal) and opt.bSimBlock is False: 
        print "------------- ROC HUMAN -------------"
        ROC_target = "human"
        cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_'+ROC_target+'_cross_data'
        cross_test_path = os.path.join(cross_data_path,ROC_target+'_'+trans_type)        

        future_steps = [1, 2, 4]             
        ## cost_ratios = [1.0]
        cost_ratios = [1.0, 0.999, 0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]
        ang_interval = 1.0
        
        
        if opt.bROCPlot:
            pkl_list = glob.glob(data_path+'RAM_db/*_new.pkl')
            s_range = np.arange(0.05, 5.0, 0.3) 
            m_range = np.arange(0.1, 3.8, 0.6)

            r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
            mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)

            ## mpu.set_figure_size(10, 7.)
            
            pp.figure()                    
            ## mar.generate_roc_curve_no_prior(mech_vec_list, mech_nm_list)
            mar.generate_roc_curve(mech_vec_list, mech_nm_list, plot_prev=False)
            f = pp.gcf()
            f.subplots_adjust(bottom=.15, top=.96, right=.98, left=0.15)

        generate_roc_curve(cross_data_path, cross_test_path, future_steps, cost_ratios, ROC_target, \
                           nMaxStep=nMaxStep, trans_type=trans_type, \
                           bSimBlock=opt.bSimBlock, bPlot=opt.bROCPlot, ang_interval=ang_interval)
            

    ###################################################################################            
    elif opt.bROCRobot and opt.bSimBlock is False:
        print "------------- ROC HUMAN -------------"
        ROC_target = "robot"        
        cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_'+ROC_target+'_cross_data'
        cross_test_path = os.path.join(cross_data_path,ROC_target+'_'+trans_type)        
        future_steps = [1, 2, 4, 8] 
        cost_ratios = [1.0, 0.9999, 0.999, 0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]
        ang_interval = 1.0
        
        #--------------------------------------------------------------------------------
        if opt.bROCPlot:
            s_range = np.arange(0.05, 5.0, 0.3) 
            m_range = np.arange(0.1, 3.8, 0.6)
            
            pp.figure()

            if False:
                pkl_list = glob.glob(data_path+'RAM_db/robot_trials/simulate_perception/*_new.pkl')
                r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
                mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)        
                
                mar.generate_roc_curve(mech_vec_list, mech_nm_list,
                                   s_range, m_range, sem_c='c', sem_m='^',
                                   ## semantic_label = 'operating 1st time with \n 
                                   ## uncertainty in state estimation', \
                                   ## plot_prev=False)
                                   semantic_label = 'probabilistic model with \n uncertainty in state estimation', \
                                   plot_prev=False)

            # advait
            pkl_list = glob.glob(data_path+'RAM_db/robot_trials/perfect_perception/*_new.pkl')
            r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
            mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)
            mad.generate_roc_curve(mech_vec_list, mech_nm_list,
                                    s_range, m_range, sem_c='b',
                                    ## semantic_label = 'operating 1st time with \n accurate state estimation',
                                    semantic_label = 'probabilistic model with \n accurate state estimation',
                                    plot_prev=False)                                                   
        
        generate_roc_curve(cross_data_path, cross_test_path, future_steps, cost_ratios, ROC_target, \
                           nMaxStep=nMaxStep, trans_type=trans_type, \
                           bSimBlock=opt.bSimBlock, bPlot=opt.bROCPlot, ang_interval=ang_interval)

            
    ###################################################################################                    
    elif (opt.bROCHuman or opt.bCrossVal) and opt.bSimBlock:
        print "------------- ROC HUMAN with simulated block data-------------"
        ROC_target = "human_block"        
        cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_'+ROC_target+'_cross_data'
        cross_test_path = os.path.join(cross_data_path,ROC_target+'_'+trans_type)        

        future_steps = [1, 2, 4, 8]             
        cost_ratios = [1.0, 0.999, 0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]
        ang_interval = 1.0

        if opt.bROCPlot:
            pp.figure()
            f = pp.gcf()
            f.subplots_adjust(bottom=.15, top=.96, right=.98, left=0.15)
            
        generate_roc_curve(cross_data_path, cross_test_path, future_steps, cost_ratios, ROC_target, \
                           nMaxStep=nMaxStep, trans_type=trans_type, \
                           bSimBlock=opt.bSimBlock, bPlot=opt.bROCPlot, ang_interval=ang_interval)

                           
    ###################################################################################                    
    elif opt.bROCRobot and opt.bSimBlock:
        print "------------- ROC HUMAN with simulated block data-------------"
        ROC_target = "robot_block"        
        cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_'+ROC_target+'_cross_data'
        cross_test_path = os.path.join(cross_data_path,ROC_target+'_'+trans_type)        

        future_steps = [1, 2, 4, 8]             
        cost_ratios = [1.0, 0.999, 0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]
        ang_interval = 1.0

        if opt.bROCPlot:
            pp.figure()
            f = pp.gcf()
            f.subplots_adjust(bottom=.15, top=.96, right=.98, left=0.15)
            
        generate_roc_curve(cross_data_path, cross_test_path, future_steps, cost_ratios, ROC_target, \
                           nMaxStep=nMaxStep, trans_type=trans_type, \
                           bSimBlock=opt.bSimBlock, bPlot=opt.bROCPlot, ang_interval=ang_interval)
                
                           
        
    ###################################################################################                    
    elif opt.bUseBlockData:

        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    

        ######################################################    
        # Test data
        h_config, h_ftan = mad.get_a_blocked_detection(mech, ang_interval=0.25) # robot blocked test data
        h_config =  np.array(h_config)*180.0/3.14

        # Training data            
        h_ftan   = data_vecs[0][2,:].tolist() # ikea cabinet door openning data
        h_config = np.arange(0,float(len(h_ftan)), 1.0)

        x_test = h_ftan[:nCurrentStep]
        x_test_next = h_ftan[nCurrentStep:nCurrentStep+lh.nFutureStep]
        x_test_all  = h_ftan
                
        if opt.bAnimation:

            ## x,y = get_interp_data(h_config, h_ftan)
            x,y = h_config, h_ftan
            ac = anomaly_checker(lh, sig_mult=6.0, sig_offset=1.0)
            ac.simulation(x,y)
            
            ## lh.animated_path_plot(x_test_all, opt.bAniReload)
        
        elif opt.bApproxObsrv:
            start_time = time.clock()
            lh.init_plot(bAni=opt.bAnimation)            

            x_pred, x_pred_prob = lh.multi_step_approximated_predict(x_test,
                                                                     full_step=True, 
                                                                     verbose=opt.bVerbose)

            elapsed = []
            elapsed.append(time.clock() - start_time)

            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), 
                                    x_pred_prob, np.array(x_test_next), 
                                    X_test_all=x_test_all)

            elapsed.append(time.clock() - elapsed[-1])        
            print elapsed
            
            lh.final_plot()
        else:               
            lh.init_plot(bAni=opt.bAnimation)            
            x_pred, x_pred_prob = lh.multi_step_predict(x_test, verbose=opt.bVerbose)
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), 
                                    x_pred_prob, np.array(x_test_next), 
                                    X_test_all=x_test_all)
            lh.final_plot()
        
            
        
    ###################################################################################            
    elif opt.bApproxObsrv:
        if lh.nFutureStep <= 1: print "Error: nFutureStep should be over than 2."
        
        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    
        
        for i in xrange(1,31,2):
            
            x_test      = data_vecs[0][i,:nCurrentStep].tolist()
            x_test_next = data_vecs[0][i,nCurrentStep:nCurrentStep+lh.nFutureStep].tolist()
            x_test_all  = data_vecs[0][i,:].tolist()

            x_pred, x_pred_prob = lh.multi_step_approximated_predict(x_test, 
                                                                     full_step=True, 
                                                                     verbose=opt.bVerbose)
            
            lh.init_plot()            
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), x_pred_prob, np.array(x_test_next))
            lh.final_plot()
                            
            
    ###################################################################################            
    ###################################################################################                    
    elif opt.bOptMeanVar:
        print "------------- Optimize B matrix -------------"

        # Save file name
        import socket, time
        host_name = socket.gethostname()
        t=time.gmtime()                
        os.system('mkdir -p /home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune_'+ \
                  doc.class_dir_list[nClass])
        save_file = os.path.join('/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune_'+ \
                                 doc.class_dir_list[nClass],
                                 host_name+'_'+str(t[0])+str(t[1])+str(t[2])+'_'
                                 +str(t[3])+str(t[4])+str(t[5])+'.pkl')

        lh.param_optimization(save_file=save_file)
                

    ###################################################################################            
    elif opt.bROCPHMMPlot:
        pkl_list = glob.glob(data_path+'RAM_db/robot_trials/perfect_perception/*_new.pkl')
        s_range = np.arange(0.05, 3.8, 0.2) 
        m_range = np.arange(0.1, 3.8, 0.6)        
        
        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)

        ## mpu.set_figure_size(26, 14.)        
        pp.figure()        

        # Set the default color cycle
        import itertools
        colors = itertools.cycle(['r', 'g', 'b'])
        shapes = itertools.cycle(['x','v', 'o'])
        ## colors = itertools.cycle(['r', 'g', 'b', 'y'])
        ## shapes = itertools.cycle(['x','v', 'o', '+'])
        lines  = itertools.cycle(['--','-'])
        ## mpl.rcParams['axes.color_cycle'] = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
        ## pp.gca().set_color_cycle(['r', 'g', 'b', 'y', 'm', 'c', 'k'])

        sig_offs = [0, 3]
        for sig_off in sig_offs:
            roc_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/roc_sig_0_'+str(sig_off)+'/door_roc_data'
            line = lines.next()

            ## for i in xrange(1,9,3):
            for i in [1,4,8]:
                color = colors.next()
                shape = shapes.next()
                roc_root_path = roc_data_path+'_'+str(i)
                generate_roc_curve(mech_vec_list, mech_nm_list, \
                                   nFutureStep=i,
                                   semantic_range = np.arange(0.2, 2.7, 0.3), bPlot=True,
                                   roc_root_path=roc_root_path, semantic_label=str(i)+ \
                                   ' step PHMM with 0.'+str(sig_off)+' offset', 
                                   sem_l=line,sem_c=color,sem_m=shape)
            
        ## pp.xlim(-0.5,27)
        pp.xlim(-0.5,5)
        pp.ylim(0.,5)
        pp.legend(loc='best',prop={'size':14})
        pp.xlabel('False positive rate (percentage)', fontsize=22)
        pp.ylabel('Mean excess force (Newtons)', fontsize=22)
        pp.savefig('robot_roc_phmm_comp.pdf')
        pp.show()
            

    ###################################################################################            
    elif opt.bAllPlot:

        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    
        lh.all_path_plot(lh.aXData)
        lh.final_plot()
                
    else:
        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    
        ## lh.path_plot(data_vecs[0], data_vecs[0,:,3])

        for i in xrange(18,31,2):
            
            x_test      = data_vecs[0][i,:nCurrentStep].tolist()
            x_test_next = data_vecs[0][i,nCurrentStep:nCurrentStep+lh.nFutureStep].tolist()
            x_test_all  = data_vecs[0][i,:].tolist()
                
            x_pred, x_pred_prob = lh.multi_step_predict(x_test, verbose=opt.bVerbose)
            lh.init_plot()            
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), x_pred_prob, np.array(x_test_next))
            lh.final_plot()



    ## # Compute mean and std
    ## mu    = np.zeros((nMaxStep,1))
    ## sigma = np.zeros((nMaxStep,1))
    ## index = 0
    ## m_init = 0
    ## while (index < nMaxStep):
    ##     temp_vec = lh.aXData[:,(m_init):(m_init + 1)] 
    ##     m_init = m_init + 1

    ##     mu[index] = np.mean(temp_vec)
    ##     sigma[index] = np.std(temp_vec)
    ##     index = index+1

    ## for i in xrange(len(mu)):
    ##     print mu[i],sigma[i]

            
    ## print lh.A
    ## print lh.B
    ## print lh.pi
            
    ## print lh.mean_path_plot(lh.mu, lh.sigma)
        
    ## print x_test
    ## print x_test[-4:]

    ## fig = plt.figure(1)
    ## ax = fig.add_subplot(111)

    ## ax.plot(obsrv_range, future_prob)
    ## plt.show()






## and False:
##         print "------------- Cross Validation -------------"

##         # Save file name
##         import socket, time
##         host_name = socket.gethostname()
##         t=time.gmtime()                
##         save_file = os.path.join('/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune',
##                                  host_name+'_'+str(t[0])+str(t[1])+str(t[2])+'_'
##                                  +str(t[3])+str(t[4])+'.pkl')

##         # Random step size
##         step_size_list_set = []
##         for i in xrange(300):
##             step_size_list = [1] * lh.nState
##             while sum(step_size_list)!=lh.nMaxStep:
##                 ## idx = int(random.gauss(float(lh.nState)/2.0,float(lh.nState)/2.0/2.0))
##                 idx = int(random.randrange(0, lh.nState, 1))
                
##                 if idx < 0 or idx >= lh.nState: 
##                     continue
##                 else:
##                     step_size_list[idx] += 1                
##             step_size_list_set.append(step_size_list)                    

        
##         ## tuned_parameters = [{'nState': [20,25,30,35], 'nFutureStep': [1], 
##         ##                      'fObsrvResol': [0.05,0.1,0.15,0.2,0.25], 'nCurrentStep': [5,10,15,20,25]}]
##         tuned_parameters = [{'nState': [lh.nState], 'nFutureStep': [1], 
##                              'fObsrvResol': [0.05,0.1,0.15,0.2], 'step_size_list': step_size_list_set}]        

##         ## tuned_parameters = [{'nState': [20,30], 'nFutureStep': [1], 'fObsrvResol': [0.1]}]
##         lh.param_estimation(tuned_parameters, 10, save_file=save_file)

##     elif opt.bCrossVal:



    ## elif opt.bROCHuman:
    ##     # Need to copy block data from ../advait
        
    ##     pkl_list = glob.glob(data_path+'RAM_db/*_new.pkl')
    ##     s_range = np.arange(0.05, 5.0, 0.3) 
    ##     m_range = np.arange(0.1, 3.8, 0.6)
        
    ##     r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
    ##     mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)

    ##     mpu.set_figure_size(10, 7.)
    ##     nFutureStep = 8

        
    ##     ## generate_roc_curve(mech_vec_list, mech_nm_list, \
    ##     ##                    nFutureStep=nFutureStep,fObsrvResol=fObsrvResol,
    ##     ##                    semantic_range = np.arange(0.2, 2.7, 0.3), bPlot=opt.bROCPlot,
    ##     ##                    roc_root_path=roc_root_path, semantic_label=str(nFutureStep)+ \
    ##     ##                    ' step PHMM with \n accurate state estimation', 
    ##     ##                    sem_c=color,sem_m=shape)

        
    ##     pp.figure()
    ##     mar.generate_roc_curve_no_prior(mech_vec_list, mech_nm_list)
    ##     mar.generate_roc_curve(mech_vec_list, mech_nm_list)
    ##     f = pp.gcf()
    ##     f.subplots_adjust(bottom=.15, top=.96, right=.98, left=0.15)
    ##     ## pp.savefig('roc_compare.pdf')
    ##     pp.show()

        

## def generate_roc_curve(mech_vec_list, mech_nm_list,                        
##                        nFutureStep, fObsrvResol,
##                        semantic_range = np.arange(0.2, 2.7, 0.3),
##                        target_class=['Freezer','Fridge','Office Cabinet'],
##                        bPlot=False, roc_root_path=roc_data_path,
##                        semantic_label='PHMM anomaly detection w/ known mechanisum class', 
##                        sem_l='-',sem_c='r',sem_m='*', trans_type="left_right"):

##     start_step = 2
    
##     t_nm_list, t_mech_vec_list = [], []
##     for i, nm in enumerate(mech_nm_list):
##         ## print 'nm:', nm
##         if 'known' in nm:
##             continue
##         t_nm_list.append(nm)
##         t_mech_vec_list.append(mech_vec_list[i])

##     data, _ = mar.create_blocked_dataset_semantic_classes(t_mech_vec_list, t_nm_list, append_robot = False)

 
##     #---------------- semantic class prior -------------
##     # init containers
##     fp_l_l = []
##     mn_l_l = []
##     err_l_l = []
##     mech_fp_l_l = []
##     mech_mn_l_l = []
##     mech_err_l_l = []

##     # splitter
##     nfs = NFoldPartitioner(cvtype=1, attr='targets') # 1-fold ?
##     label_splitter = splitters.Splitter(attr='partitions')            
##     splits = [list(label_splitter.generate(x)) for x in nfs.generate(data)]            

##     X_test = np.arange(0.0, 36.0, 1.0)

##     # Run by class
##     for l_wdata, l_vdata in splits: #label_splitter(data):

##         mech_class = l_vdata.targets[0]
##         trials = l_vdata.samples # all data
    
##         # check existence of computed result
##         idx = doc.class_list.index(mech_class)        
##         if mech_class not in target_class: continue
##         ## elif os.path.isfile('roc_'+doc.class_dir_list[idx]+'.pkl'): continue
##         ## elif os.path.isfile('roc_'+doc.class_dir_list[idx]+'_complete'): continue
        
##         # cutting into the same length
##         trials = trials[:,:36]

##         pkl_file  = "mech_class_"+doc.class_dir_list[idx]+".pkl"        
##         data_vecs, data_mech, data_chunks = mad.get_data(pkl_file, mech_class=mech_class, renew=opt.renew) # human data
##         A, B, pi, nState = doc.get_hmm_init_param(mech_class=mech_class)        

##         print "-------------------------------"
##         print "Mech class: ", mech_class
##         print "Data size: ", np.array(data_vecs).shape
##         print "-------------------------------"
        
##         # Training 
##         lh = learning_hmm(aXData=data_vecs[0], nState=nState, 
##                           nMaxStep=nMaxStep, nFutureStep=nFutureStep, 
##                           fObsrvResol=fObsrvResol, nCurrentStep=nCurrentStep, trans_type=trans_type)    

##         lh.fit(lh.aXData, A=A, B=B, pi=pi, verbose=opt.bVerbose)                
        
##         mn_list = []
##         fp_list, err_list = [], []        
        
##         for n in semantic_range:
##             print "n: ", n

##             if os.path.isdir(roc_root_path) == False:
##                 os.system('mkdir -p '+roc_root_path)

##             # check saved file
##             target_pkl = roc_root_path+'/'+'fp_'+doc.class_dir_list[idx]+'_n_'+str(n)+'.pkl'

##             # mutex file
##             host_name = socket.gethostname()
##             mutex_file = roc_root_path+'/'+host_name+'_mutex_'+doc.class_dir_list[idx]+'_'+str(n)
                        
##             if os.path.isfile(target_pkl) == False \
##                 and hcu.is_file(roc_root_path, 'mutex_'+doc.class_dir_list[idx]+'_'+str(n)) == False: 
            
##                 os.system('touch '+mutex_file)

##                 # Init variables
##                 false_pos = np.zeros((len(trials), len(trials[0])-start_step))
##                 tot = trials.shape[0] * trials.shape[1]
##                 err_l = []
                    
##                 # Gives all profiles
##                 for i, trial in enumerate(trials):

##                     # Init checker
##                     ac = anomaly_checker(lh, sig_mult=n)

##                     # Simulate each profile
##                     for j in xrange(len(trial)):
##                         # Update buffer
##                         ac.update_buffer(X_test[:j+1], trial[:j+1])

##                         if j>= start_step:                    
##                             # check anomaly score
##                             bFlag, max_err, fScore = ac.check_anomaly(trial[j])
##                             if bFlag: 
##                                 false_pos[i, j-start_step] = 1.0 
##                             else:
##                                 err_l.append(max_err)

##                             print "(",i,"/",len(trials)," ",j, ") : ", false_pos[i, j-start_step], max_err

##                 # save data & remove mutex file
##                 d = {}
##                 d['false_pos'] = false_pos
##                 d['tot'] = tot
##                 d['err_l'] = err_l
##                 d['n'] = n
##                 ut.save_pickle(d, target_pkl)
##                 os.system('rm '+mutex_file)

##             elif os.path.isfile(target_pkl) == True:
                
##                 d = ut.load_pickle(target_pkl)
##                 false_pos = d['false_pos']
##                 tot   = d['tot']  
##                 err_l = d['err_l']  
##                 n     = d['n']  
                
##                 fp_list.append(np.sum(false_pos)/(tot*0.01))
##                 err_list.append(err_l)
##                 ## mn_list.append(np.mean(np.array(err_l)))

##                 tot_e = 0.0
##                 tot_e_cnt = 0.0
##                 for e in err_l:
##                     if np.isnan(e) == False:
##                         tot_e += e 
##                         tot_e_cnt += 1.0
##                 mn_list.append(tot_e/tot_e_cnt)
                
##             else:
##                 print "Mutex exists"
##                 continue
                
##         fp_l_l.append(fp_list)
##         err_l_l.append(err_list)
##         mn_l_l.append(mn_list)
        
##     ## ll = [[] for i in err_l_l[0]]  # why 0?
##     ## for i,e in enumerate(err_l_l): # labels
##     ##     for j,l in enumerate(ll):  # multiplier range
##     ##         l.append(e[j])

##     if bPlot:
##         mn_list = np.mean(np.row_stack(mn_l_l), 0).tolist() # means into a row
##         fp_list = np.mean(np.row_stack(fp_l_l), 0).tolist()                
##         pp.plot(fp_list, mn_list, sem_l+sem_m+sem_c, label= semantic_label,
##                 mec=sem_c, ms=6, mew=2)
##         ## pp.plot(fp_list, mn_list, '--'+sem_m, label= semantic_label,
##         ##         ms=6, mew=2)
##         ## pp.legend(loc='best',prop={'size':16})
##         pp.legend(loc=1,prop={'size':14})
