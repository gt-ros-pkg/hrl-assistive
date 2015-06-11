#!/usr/bin/python

import sys, os, copy
import numpy as np, math
import glob
import socket
import time
import random 

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

# Machine learning library
import mlpy
import scipy as scp
from scipy import interpolate       
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters
from joblib import Parallel, delayed

# Util
import hrl_lib.util as ut
import matplotlib
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon

import data_manager as dm
import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
from hrl_anomaly_detection.HMM.learning_hmm_multi import learning_hmm_multi



def fig_roc(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
            prefix, nState=20, \
            threshold_mult = np.arange(0.05, 1.2, 0.05), opr='robot', attr='id', bPlot=False, \
            cov_mult=[1.0, 1.0, 1.0, 1.0], renew=False, test=False, disp=None, rm_run=False, sim=False):
    
    # For parallel computing
    strMachine = socket.gethostname()+"_"+str(os.getpid())    
    trans_type = "left_right"
    
    # Check the existance of workspace
    cross_test_path = os.path.join(cross_data_path, str(nState)+'_'+test_title)
    if os.path.isdir(cross_test_path) == False:
        os.system('mkdir -p '+cross_test_path)

    if rm_run == True:                            
        os.system('rm '+os.path.join(cross_test_path, 'running')+'*')
        

    # anomaly check method list
    use_ml_pkl = False
    false_dataSet = None
    count = 0        
    
    for method in check_methods:        
        for i in xrange(nDataSet):

            pkl_file = os.path.join(cross_data_path, "dataSet_"+str(i))
            dd = ut.load_pickle(pkl_file)

            train_aXData1 = dd['ft_force_mag_train_l']
            train_aXData2 = dd['audio_rms_train_l'] 
            train_chunks  = dd['train_chunks']
            test_aXData1 = dd['ft_force_mag_test_l']
            test_aXData2 = dd['audio_rms_test_l'] 
            test_chunks  = dd['test_chunks']

            # min max scaling for training data
            aXData1_scaled, min_c1, max_c1 = dm.scaling(train_aXData1, scale=10.0)
            aXData2_scaled, min_c2, max_c2 = dm.scaling(train_aXData2, scale=10.0)    
            labels = [True]*len(train_aXData1)
            train_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, train_chunks, labels)

            # test data!!
            aXData1_scaled, _, _ = dm.scaling(test_aXData1, min_c1, max_c1, scale=10.0)
            aXData2_scaled, _, _ = dm.scaling(test_aXData2, min_c2, max_c2, scale=10.0)    
            labels = [False]*len(test_aXData1)
            test_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, test_chunks, labels)

            if sim == True:
                false_aXData1 = dd['ft_force_mag_sim_false_l']
                false_aXData2 = dd['audio_rms_sim_false_l'] 
                false_chunks  = dd['sim_false_chunks']
                false_anomaly_start = dd['anomaly_start_idx']
            
                # generate simulated data!!
                aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
                aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)    
                labels = [False]*len(false_aXData1)
                false_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, false_chunks, labels)
                false_dataSet.sa['anomaly_idx'] = false_anomaly_start
            else:
                false_aXData1 = dd['ft_force_mag_false_l']
                false_aXData2 = dd['audio_rms_false_l'] 
                false_chunks  = dd['false_chunks']
                false_anomaly_start = dd['anomaly_start_idx']                

                # generate simulated data!!
                aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
                aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)    
                labels = [False]*len(false_aXData1)
                false_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, false_chunks, labels)
                false_dataSet.sa['anomaly_idx'] = false_anomaly_start

            for check_dim in check_dims:
            
                lhm = None

                for ths in threshold_mult:

                    # save file name
                    res_file = prefix+'_dataset_'+str(i)+'_'+method+'_roc_'+opr+'_dim_'+str(check_dim)+'_ths_'+ \
                      str(ths)+'.pkl'
                    res_file = os.path.join(cross_test_path, res_file)

                    mutex_file_part = 'running_dataset_'+str(i)+'_dim_'+str(check_dim)+'_ths_'+str(ths)+'_'+method
                    mutex_file_full = mutex_file_part+'_'+strMachine+'.txt'
                    mutex_file      = os.path.join(cross_test_path, mutex_file_full)
                                        
                    if os.path.isfile(res_file): 
                        count += 1            
                        continue
                    elif hcu.is_file(cross_test_path, mutex_file_part): 
                        continue
                    ## elif os.path.isfile(mutex_file): continue
                    os.system('touch '+mutex_file)

                    if lhm is None:
                        if check_dim is not 2:
                            x_train1 = train_dataSet.samples[:,check_dim,:]
                            lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, nEmissionDim=1, \
                                                     check_method=method)
                            if check_dim==0: lhm.fit(x_train1, cov_mult=[cov_mult[0]]*4, use_pkl=use_ml_pkl)
                            elif check_dim==1: lhm.fit(x_train1, cov_mult=[cov_mult[3]]*4, use_pkl=use_ml_pkl)
                        else:
                            x_train1 = train_dataSet.samples[:,0,:]
                            x_train2 = train_dataSet.samples[:,1,:]
                            lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, check_method=method)
                            lhm.fit(x_train1, x_train2, cov_mult=cov_mult, use_pkl=use_ml_pkl)            

                    if (disp == 'test' or disp == 'false') and method == 'progress':
                        if disp == 'test':
                            if check_dim == 2:
                                x_test1 = test_dataSet.samples[:,0]
                                x_test2 = test_dataSet.samples[:,1]
                            else:
                                x_test1 = test_dataSet.samples[:,check_dim]
                        elif disp == 'false':
                            if check_dim == 2:
                                x_test1 = false_dataSet.samples[:,0]
                                x_test2 = false_dataSet.samples[:,1]
                            else:
                                x_test1 = false_dataSet.samples[:,check_dim]

                                                
                        lhm.likelihood_disp(x_test1, x_test2, ths, scale1=[min_c1, max_c1, scale], \
                                            scale2=[min_c2, max_c2, scale])
                        

                    if test:
                        tp, fn, fp, tn, delay_l = anomaly_check_online_test(lhm, test_dataSet, \
                                                                               false_dataSet, \
                                                                               ths, \
                                                                               check_dim=check_dim)
                    elif onoff_type == 'online':
                        tp, fn, fp, tn, delay_l, _ = anomaly_check_online(lhm, test_dataSet, \
                                                                       false_dataSet, \
                                                                       ths, \
                                                                       check_dim=check_dim)
                    else:
                        tp, fn, fp, tn, delay_l = anomaly_check_offline(lhm, test_dataSet, \
                                                                        false_dataSet, \
                                                                        ths, \
                                                                        check_dim=check_dim)

                    d = {}
                    d['fn']    = fn
                    d['tn']    = tn
                    d['tp']    = tp
                    d['fp']    = fp
                    d['ths']   = ths
                    d['delay_l'] = delay_l

                    try:
                        ut.save_pickle(d,res_file)        
                    except:
                        print "There is the targeted pkl file"
                        
                    os.system('rm '+mutex_file)
                    print "-----------------------------------------------"

    if count == len(threshold_mult)*len(check_methods)*nDataSet*len(check_dims):
        print "#############################################################################"
        print "All file exist ", count
        print "#############################################################################"        
    else:
        return
        
    if bPlot:

        import itertools
        colors = itertools.cycle(['g', 'm', 'c', 'k'])
        shapes = itertools.cycle(['x','v', 'o', '+'])
        
        fig = pp.figure()

        if len(check_methods) >= len(check_dims): nClass = len(check_methods)
        else: nClass = len(check_dims)

        for n in range(nClass):

            if len(check_methods) >= len(check_dims): 
                method = check_methods[n]
                check_dim = check_dims[0]
            else: 
                method = check_methods[0]
                check_dim = check_dims[n]
                
                
            fn_l = np.zeros(len(threshold_mult))
            tp_l = np.zeros(len(threshold_mult))
            tn_l = np.zeros(len(threshold_mult))
            fp_l = np.zeros(len(threshold_mult))

            delay_l = np.zeros(len(threshold_mult)); delay_cnt = np.zeros(len(threshold_mult))
            ## err_l = np.zeros(len(threshold_mult));   err_cnt = np.zeros(len(threshold_mult))
                
            for i in xrange(nDataSet):

                for j, ths in enumerate(threshold_mult):
                    # save file name
                    res_file = prefix+'_dataset_'+str(i)+'_'+method+'_roc_'+opr+'_dim_'+str(check_dim)+'_ths_'+ \
                      str(ths)+'.pkl'
                    res_file = os.path.join(cross_test_path, res_file)

                    d = ut.load_pickle(res_file)
                    fn_l[j] += d['fn']; tp_l[j] += d['tp'] 
                    tn_l[j] += d['tn']; fp_l[j] += d['fp'] 
                    delay_l[j] += np.sum(d['delay_l']); delay_cnt[j] += float(len(d['delay_l']))  

                    ## print ths, " : ", d['tn'], d['fn'], " : ", d['delay_l']
                    ## # Exclude wrong detection cases
                    ## if delay == []: continue

            tpr_l = np.zeros(len(threshold_mult))
            fpr_l = np.zeros(len(threshold_mult))
            npv_l = np.zeros(len(threshold_mult))
            detect_l = np.zeros(len(threshold_mult))
                    
            for i in xrange(len(threshold_mult)):
                if tp_l[i]+fn_l[i] != 0:
                    tpr_l[i] = tp_l[i]/(tp_l[i]+fn_l[i])*100.0

                if fp_l[i]+tn_l[i] != 0:
                    fpr_l[i] = fp_l[i]/(fp_l[i]+tn_l[i])*100.0

                if tn_l[i]+fn_l[i] != 0:
                    npv_l[i] = tn_l[i]/(tn_l[i]+fn_l[i])*100.0

                if delay_cnt[i] == 0:
                    delay_l[i] = 0
                else:                    
                    delay_l[i] = delay_l[i]/delay_cnt[i]

                if tn_l[i] + fn_l[i] + fp_l[i] != 0:
                    detect_l[i] = (tn_l[i]+fn_l[i])/(tn_l[i] + fn_l[i] + fp_l[i])*100.0

            idx_list = sorted(range(len(fpr_l)), key=lambda k: fpr_l[k])
            sorted_tpr_l   = np.array([tpr_l[k] for k in idx_list])
            sorted_fpr_l   = np.array([fpr_l[k] for k in idx_list])
            sorted_npv_l   = np.array([npv_l[k] for k in idx_list])
            sorted_delay_l = [delay_l[k] for k in idx_list]
            sorted_detect_l = [detect_l[k] for k in idx_list]

            color = colors.next()
            shape = shapes.next()

            ## if i==0: semantic_label='Force only'
            ## elif i==1: semantic_label='Sound only'
            ## else: semantic_label='Force and sound'
            ## pp.plot(sorted_fn_l, sorted_delay_l, '-'+shape+color, label=method, mec=color, ms=8, mew=2)
            ## if method == 'global': label = 'fixed threshold'
            ## if method == 'progress': label = 'progress based threshold'
            label = method+"_"+str(check_dim)

            if test:
                pp.plot(sorted_npv_l, sorted_delay_l, '-'+shape+color, label=label, mec=color, ms=8, mew=2)
                ##pp.plot(sorted_detect_l, sorted_delay_l, '-'+shape+color, label=label, mec=color, ms=8, mew=2)
                ##pp.plot(sorted_npv_l, sorted_detect_l, '-'+shape+color, label=label, mec=color, ms=8, mew=2)
            else:
                pp.plot(sorted_fpr_l, sorted_tpr_l, '-'+shape+color, label=label, mec=color, ms=8, mew=2)
                #pp.plot(sorted_ths_l, sorted_tn_l, '-'+shape+color, label=method, mec=color, ms=8, mew=2)



        ## fp_l = fp_l[:,0]
        ## tp_l = tp_l[:,0]
        
        ## from scipy.optimize import curve_fit
        ## def sigma(e, k ,n, offset): return k*((e+offset)**n)
        ## param, var = curve_fit(sigma, fp_l, tp_l)
        ## new_fp_l = np.linspace(fp_l.min(), fp_l.max(), 50)        
        ## pp.plot(new_fp_l, sigma(new_fp_l, *param))

        if test == False:
            ## pp.xlim([0.0, 40])
            pp.ylim([0.0, 100])        
            pp.xlabel('False Positive Rate (Percentage)', fontsize=16)
            pp.ylabel('True Positive Rate (Percentage)', fontsize=16)    

        pp.legend(loc=4,prop={'size':16})
        
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
        ## pp.show()
        
    return


def fig_roc_all(cross_root_path, all_task_names, test_title, nState, threshold_mult, check_methods, \
                check_dims, an_type=None, force_an=None, sound_an=None, renew=False, sim=False):
                    
    import itertools
    colors = itertools.cycle(['g', 'm', 'c', 'k'])
    shapes = itertools.cycle(['x','v', 'o', '+'])
    threshold_mult = threshold_mult.tolist()
    
    fig = pp.figure()

    if len(check_methods) > len(check_dims): nClass = len(check_methods)
    else: nClass = len(check_dims)

    for n in range(nClass):

        if len(check_methods) > len(check_dims): 
            method = check_methods[n]
            check_dim = check_dims[0]
        else: 
            method = check_methods[0]
            check_dim = check_dims[n]

        fn_l = np.zeros(len(threshold_mult))
        tp_l = np.zeros(len(threshold_mult))
        tn_l = np.zeros(len(threshold_mult))
        fp_l = np.zeros(len(threshold_mult))

        delay_l = np.zeros(len(threshold_mult)); delay_cnt = np.zeros(len(threshold_mult))
        ## err_l = np.zeros(len(threshold_mult));   err_cnt = np.zeros(len(threshold_mult))

        if sim:
            save_pkl_file = os.path.join(cross_root_path,test_title+'_'+method+'_'+str(check_dim)+'.pkl')
        else:
            save_pkl_file = os.path.join(cross_root_path,test_title+'_real_'+method+'_'+str(check_dim)+'.pkl')
            
        if os.path.isfile(save_pkl_file) == False:        
            # Collect data
            for task_name in all_task_names:

                if sim:
                    cross_data_path = os.path.join(cross_root_path, 'multi_sim_'+task_name, test_title)
                else:
                    cross_data_path = os.path.join(cross_root_path, 'multi_'+task_name, test_title)

                t_dirs = os.listdir(cross_data_path)
                for t_dir in t_dirs:
                    if t_dir.find(test_title)>=0:
                        break

                cross_test_path = os.path.join(cross_data_path, t_dir)



                pkl_files = sorted([d for d in os.listdir(cross_test_path) if os.path.isfile(os.path.join( \
                    cross_test_path,d))])

                for pkl_file in pkl_files:

                    if pkl_file.find('txt') >= 0:
                        print "There is running file!!!"
                        print cross_test_path
                        res_file = os.path.join(cross_test_path, pkl_file)
                        os.system('rm '+res_file)
                        sys.exit()

                    # method
                    c_method = pkl_file.split('_roc')[0].split('_')[-1]
                    if c_method != method: continue
                    # dim
                    c_dim = int(pkl_file.split('dim_')[-1].split('_ths')[0])
                    if c_dim != check_dim: continue

                    # ths
                    ths = float(pkl_file.split('ths_')[-1].split('.pkl')[0])

                    # find close index
                    for i, t_thres in enumerate(threshold_mult):
                        if abs(t_thres - ths) < 0.00001:
                            idx = i
                            break

                    res_file = os.path.join(cross_test_path, pkl_file)

                    d = ut.load_pickle(res_file)
                    fn_l[idx] += d['fn']; tp_l[idx] += d['tp'] 
                    tn_l[idx] += d['tn']; fp_l[idx] += d['fp'] 
                    delay_l[idx] += np.sum(d['delay_l']); delay_cnt[idx] += float(len(d['delay_l']))  
            data = {}
            data['fn_l'] = fn_l
            data['tp_l'] = tp_l
            data['tn_l'] = tn_l
            data['fp_l'] = fp_l
            data['delay_l'] = delay_l
            data['delay_cnt'] = delay_cnt
            ut.save_pickle(data, save_pkl_file)
        else:
            data = ut.load_pickle(save_pkl_file)
            fn_l = data['fn_l']
            tp_l = data['tp_l'] 
            tn_l = data['tn_l'] 
            fp_l = data['fp_l'] 
            delay_l = data['delay_l'] 
            delay_cnt = data['delay_cnt'] 
                

        tpr_l = np.zeros(len(threshold_mult))
        fpr_l = np.zeros(len(threshold_mult))
        npv_l = np.zeros(len(threshold_mult))
        detect_l = np.zeros(len(threshold_mult))

        for i in xrange(len(threshold_mult)):
            if tp_l[i]+fn_l[i] != 0:
                tpr_l[i] = tp_l[i]/(tp_l[i]+fn_l[i])*100.0

            if fp_l[i]+tn_l[i] != 0:
                fpr_l[i] = fp_l[i]/(fp_l[i]+tn_l[i])*100.0

            if tn_l[i]+fn_l[i] != 0:
                npv_l[i] = tn_l[i]/(tn_l[i]+fn_l[i])*100.0

            if delay_cnt[i] == 0:
                delay_l[i] = 0
            else:                    
                delay_l[i] = delay_l[i]/delay_cnt[i]

            if tn_l[i] + fn_l[i] + fp_l[i] != 0:
                detect_l[i] = (tn_l[i]+fn_l[i])/(tn_l[i] + fn_l[i] + fp_l[i])*100.0
                
        idx_list = sorted(range(len(fpr_l)), key=lambda k: fpr_l[k])
        sorted_tpr_l   = np.array([tpr_l[k] for k in idx_list])
        sorted_fpr_l   = np.array([fpr_l[k] for k in idx_list])
        sorted_npv_l   = np.array([npv_l[k] for k in idx_list])
        sorted_delay_l = [delay_l[k] for k in idx_list]
        sorted_detect_l = [detect_l[k] for k in idx_list]

        color = colors.next()
        shape = shapes.next()

        if test_title.find('dim')>=0:
            if check_dim == 0:
                label = 'Force only'
            elif check_dim == 1:
                label = 'Sound only'
            elif check_dim == 2:
                label = 'Force & sound'
            else:
                label = method +"_"+str(check_dim)                
        else:
            if method == 'globalChange':
                label = 'Fixed threshold & \n change detection'
            elif method == 'change':
                label = 'Change detection'
            elif method == 'global':
                label = 'Fixed threshold \n detection'
            elif method == 'progress':
                label = 'Progress-based \n detection'
            else:
                label = method +"_"+str(check_dim)
        pp.plot(sorted_fpr_l, sorted_tpr_l, '-'+shape+color, label=label, mec=color, ms=8, mew=2)

    pp.legend(loc=4,prop={'size':16})
    pp.xlabel('False Positive Rate (Percentage)', fontsize=16)
    pp.ylabel('True Positive Rate (Percentage)', fontsize=16)    

    fig.savefig('test.pdf')
    fig.savefig('test.png')
    os.system('cp test.p* ~/Dropbox/HRL/')
    #pp.show()
 
       
#---------------------------------------------------------------------------------------#        
def fig_eval(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
             prefix, nState=20, \
             opr='robot', attr='id', bPlot=False, \
             cov_mult=[1.0, 1.0, 1.0, 1.0], renew=False, test=False, disp=None, rm_run=False, sim=False):
    
    # For parallel computing
    strMachine = socket.gethostname()+"_"+str(os.getpid())    
    trans_type = "left_right"
    
    # Check the existance of workspace
    cross_test_path = os.path.join(cross_data_path, str(nState)+'_'+test_title)
    if os.path.isdir(cross_test_path) == False:
        os.system('mkdir -p '+cross_test_path)

    if rm_run == True:                            
        os.system('rm '+os.path.join(cross_test_path, 'running')+'*')
        
    # anomaly check method list
    use_ml_pkl = False
    false_dataSet = None
    count = 0        
    
    for method in check_methods:        
        for i in xrange(nDataSet):

            pkl_file = os.path.join(cross_data_path, "dataSet_"+str(i))
            dd = ut.load_pickle(pkl_file)

            train_aXData1 = dd['ft_force_mag_train_l']
            train_aXData2 = dd['audio_rms_train_l'] 
            train_chunks  = dd['train_chunks']
            test_aXData1 = dd['ft_force_mag_test_l']
            test_aXData2 = dd['audio_rms_test_l'] 
            test_chunks  = dd['test_chunks']

            # min max scaling for training data
            aXData1_scaled, min_c1, max_c1 = dm.scaling(train_aXData1, scale=10.0)
            aXData2_scaled, min_c2, max_c2 = dm.scaling(train_aXData2, scale=10.0)    
            labels = [True]*len(train_aXData1)
            train_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, train_chunks, labels)

            # test data!!
            aXData1_scaled, _, _ = dm.scaling(test_aXData1, min_c1, max_c1, scale=10.0)
            aXData2_scaled, _, _ = dm.scaling(test_aXData2, min_c2, max_c2, scale=10.0)    
            labels = [False]*len(test_aXData1)
            test_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, test_chunks, labels)

            if sim == True:
                false_aXData1 = dd['ft_force_mag_sim_false_l']
                false_aXData2 = dd['audio_rms_sim_false_l'] 
                false_chunks  = dd['sim_false_chunks']
                false_anomaly_start = dd['anomaly_start_idx']
            
                # generate simulated data!!
                aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
                aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)    
                labels = [False]*len(false_aXData1)
                false_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, false_chunks, labels)
                false_dataSet.sa['anomaly_idx'] = false_anomaly_start
            else:
                false_aXData1 = dd['ft_force_mag_false_l']
                false_aXData2 = dd['audio_rms_false_l'] 
                false_chunks  = dd['false_chunks']
                false_anomaly_start = dd['anomaly_start_idx']                

                aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
                aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)    
                labels = [False]*len(false_aXData1)
                false_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, false_chunks, labels)
                false_dataSet.sa['anomaly_idx'] = false_anomaly_start

            for check_dim in check_dims:
            
                lhm = None

                # save file name
                res_file = prefix+'_dataset_'+str(i)+'_'+method+'_roc_'+opr+'_dim_'+str(check_dim)+'.pkl'
                res_file = os.path.join(cross_test_path, res_file)

                mutex_file_part = 'running_dataset_'+str(i)+'_dim_'+str(check_dim)+'_'+method
                mutex_file_full = mutex_file_part+'_'+strMachine+'.txt'
                mutex_file      = os.path.join(cross_test_path, mutex_file_full)

                if os.path.isfile(res_file): 
                    count += 1            
                    continue
                elif hcu.is_file(cross_test_path, mutex_file_part): 
                    continue
                ## elif os.path.isfile(mutex_file): continue
                os.system('touch '+mutex_file)

                if lhm is None:
                    if check_dim is not 2:
                        x_train1 = train_dataSet.samples[:,check_dim,:]
                        lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, nEmissionDim=1, \
                                                 check_method=method)
                        if check_dim==0: lhm.fit(x_train1, cov_mult=[cov_mult[0]]*4, use_pkl=use_ml_pkl)
                        elif check_dim==1: lhm.fit(x_train1, cov_mult=[cov_mult[3]]*4, use_pkl=use_ml_pkl)
                    else:
                        x_train1 = train_dataSet.samples[:,0,:]
                        x_train2 = train_dataSet.samples[:,1,:]
                        lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, check_method=method)
                        lhm.fit(x_train1, x_train2, cov_mult=cov_mult, use_pkl=use_ml_pkl)            

                # find a minimum sensitivity gain
                if check_dim == 2:
                    x_test1 = test_dataSet.samples[:,0]
                    x_test2 = test_dataSet.samples[:,1]
                else:
                    x_test1 = test_dataSet.samples[:,check_dim]

                max_ths = 0
                n = len(x_test1)
                for i in range(n):
                    m = len(x_test1[i])

                    # anomaly_check only returns anomaly cases only
                    for j in range(2,m):                    

                        if check_dim == 2:            
                            ths = lhm.max_sensitivity_gain(x_test1[i][:j], x_test2[i][:j])   
                        else:
                            ths = lhm.max_sensitivity_gain(x_test1[i][:j])

                        if max_ths < ths:
                            max_ths = ths
                            print "Maximum threshold: ", max_ths

                            

                if test:
                    tp, fn, fp, tn, delay_l = anomaly_check_online_test(lhm, [], \
                                                                           false_dataSet, \
                                                                           max_ths, \
                                                                           check_dim=check_dim)
                elif onoff_type == 'online':
                    tp, fn, fp, tn, delay_l, false_detection_l = anomaly_check_online(lhm, [], \
                                                                                      false_dataSet, \
                                                                                      max_ths, \
                                                                                      check_dim=check_dim)
                else:
                    tp, fn, fp, tn, delay_l = anomaly_check_offline(lhm, [], \
                                                                    false_dataSet, \
                                                                    max_ths, \
                                                                    check_dim=check_dim)

                d = {}
                d['fn']    = fn
                d['tn']    = tn
                d['tp']    = tp
                d['fp']    = fp
                d['ths']   = ths
                d['delay_l'] = delay_l
                d['false_detection_l'] = false_detection_l

                try:
                    ut.save_pickle(d,res_file)        
                except:
                    print "There is the targeted pkl file"

                os.system('rm '+mutex_file)
                print "-----------------------------------------------"

    if count == len(check_methods)*nDataSet*len(check_dims):
        print "#############################################################################"
        print "All file exist ", count
        print "#############################################################################"        
    else:
        return
        

    if bPlot:

        fig = pp.figure()
        
        fn_l = np.zeros(nDataSet)
        tp_l = np.zeros(nDataSet)
        tn_l = np.zeros(nDataSet)
        fp_l = np.zeros(nDataSet)
        delay_l = []
        fd_l = []
        fpr_l = np.zeros(nDataSet)
        
        for i in xrange(nDataSet):
            # save file name
            res_file = prefix+'_dataset_'+str(i)+'_'+method+'_roc_'+opr+'_dim_'+str(check_dim)+'.pkl'
            res_file = os.path.join(cross_test_path, res_file)

            d = ut.load_pickle(res_file)
            fn_l[i] = d['fn']; tp_l[i] = d['tp'] 
            tn_l[i] = d['tn']; fp_l[i] = d['fp'] 
            delay_l.append([d['delay_l']])
            fd_l.append([d['false_detection_l']])
            print d['false_detection_l']

        for i in xrange(nDataSet):
            if fp_l[i]+tn_l[i] != 0:
                fpr_l[i] = fp_l[i]/(fp_l[i]+tn_l[i])*100.0


        tot_fpr = np.sum(fp_l)/(np.sum(fp_l)+np.sum(tn_l))*100.0
        
        pp.ylim([0.0, 100])                   
        pp.bar(range(nDataSet+1), np.hstack([fpr_l,np.array([tot_fpr])]))

        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
        #pp.show()
        



#---------------------------------------------------------------------------------------#    
def fig_eval_all(cross_root_path, all_task_names, test_title, nState, check_methods, \
                 check_dims, an_type=None, force_an=None, sound_an=None, renew=False, sim=False):
                    
    fig = pp.figure()

    if len(check_methods) >= len(check_dims): nClass = len(check_methods)
    else: nClass = len(check_dims)

    for n in range(nClass):

        if len(check_methods) >= len(check_dims): 
            method = check_methods[n]
            check_dim = check_dims[0]
        else: 
            method = check_methods[0]
            check_dim = check_dims[n]

        fn_l = np.zeros(len(all_task_names))
        tp_l = np.zeros(len(all_task_names))
        tn_l = np.zeros(len(all_task_names))
        fp_l = np.zeros(len(all_task_names))
        fdr_l = np.zeros(len(all_task_names)) # false detection rate

        if sim:
            save_pkl_file = os.path.join(cross_root_path,test_title+'_'+method+'_'+str(check_dim)+'.pkl')
        else:
            save_pkl_file = os.path.join(cross_root_path,test_title+'_real_'+method+'_'+str(check_dim)+'.pkl')

        if os.path.isfile(save_pkl_file) == False:        
            # Collect data
            for task_num, task_name in enumerate(all_task_names):

                if sim:
                    cross_data_path = os.path.join(cross_root_path, 'multi_sim_'+task_name, test_title)
                else:
                    cross_data_path = os.path.join(cross_root_path, 'multi_'+task_name, test_title)

                t_dirs = os.listdir(cross_data_path)
                for t_dir in t_dirs:
                    if t_dir.find(test_title)>=0:
                        break

                cross_test_path = os.path.join(cross_data_path, t_dir)

                pkl_files = sorted([d for d in os.listdir(cross_test_path) if os.path.isfile(os.path.join( \
                    cross_test_path,d))])

                for pkl_file in pkl_files:

                    if pkl_file.find('txt') >= 0:
                        print "There is running file!!!"
                        print cross_test_path
                        res_file = os.path.join(cross_test_path, pkl_file)
                        os.system('rm '+res_file)
                        sys.exit()

                    # method
                    c_method = pkl_file.split('_roc')[0].split('_')[-1]
                    if c_method != method: continue
                    # dim
                    c_dim = int(pkl_file.split('dim_')[-1].split('.pkl')[0])
                    if c_dim != check_dim: continue

                    res_file = os.path.join(cross_test_path, pkl_file)

                    d = ut.load_pickle(res_file)
                    fn_l[task_num] += d['fn']; tp_l[task_num] += d['tp'] 
                    tn_l[task_num] += d['tn']; fp_l[task_num] += d['fp'] 

                    print task_num, d['false_detection_l']
                    ## fdr_l[task_num] = d['false_detection_l']

            data = {}
            data['fn_l'] = fn_l
            data['tp_l'] = tp_l
            data['tn_l'] = tn_l
            data['fp_l'] = fp_l
            ut.save_pickle(data, save_pkl_file)
        else:
            data = ut.load_pickle(save_pkl_file)
            fn_l = data['fn_l']
            tp_l = data['tp_l'] 
            tn_l = data['tn_l'] 
            fp_l = data['fp_l'] 

        tpr_l = np.zeros(len(all_task_names))
        fpr_l = np.zeros(len(all_task_names))
        npv_l = np.zeros(len(all_task_names))
        detect_l = np.zeros(len(all_task_names))

        for i in xrange(len(all_task_names)):
            if tp_l[i]+fn_l[i] != 0:
                tpr_l[i] = tp_l[i]/(tp_l[i]+fn_l[i])*100.0

            if fp_l[i]+tn_l[i] != 0:
                fpr_l[i] = fp_l[i]/(fp_l[i]+tn_l[i])*100.0

            if tn_l[i]+fn_l[i] != 0:
                npv_l[i] = tn_l[i]/(tn_l[i]+fn_l[i])*100.0

            if tn_l[i] + fn_l[i] + fp_l[i] != 0:
                detect_l[i] = (tn_l[i]+fn_l[i])/(tn_l[i] + fn_l[i] + fp_l[i])*100.0
                
        pp.bar(range(len(all_task_names)), fpr_l)

    ## pp.xlabel('False Positive Rate (Percentage)', fontsize=16)
    ## pp.ylabel('True Positive Rate (Percentage)', fontsize=16)    

    fig.savefig('test.pdf')
    fig.savefig('test.png')
    os.system('cp test.p* ~/Dropbox/HRL/')
    #pp.show()
 
#-------------------------------------------------------------------------------------------------------
def fig_roc_offline(cross_data_path, \
                    true_aXData1, true_aXData2, true_chunks, \
                    false_aXData1, false_aXData2, false_chunks, \
                    prefix, nState=20, \
                    threshold_mult = np.arange(0.05, 1.2, 0.05), opr='robot', attr='id', bPlot=False, \
                    cov_mult=[1.0, 1.0, 1.0, 1.0]):

    # For parallel computing
    strMachine = socket.gethostname()+"_"+str(os.getpid())    
    trans_type = "left_right"
    
    # Check the existance of workspace
    cross_test_path = os.path.join(cross_data_path, str(nState))
    if os.path.isdir(cross_test_path) == False:
        os.system('mkdir -p '+cross_test_path)

    # min max scaling for true data
    aXData1_scaled, min_c1, max_c1 = dm.scaling(true_aXData1, scale=10.0)
    aXData2_scaled, min_c2, max_c2 = dm.scaling(true_aXData2, scale=10.0)    
    labels = [True]*len(true_aXData1)
    true_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, true_chunks, labels)
    print "Scaling data: ", np.shape(true_aXData1), " => ", np.shape(aXData1_scaled)
    
    # min max scaling for false data
    aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0, verbose=True)
    aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)    
    labels = [False]*len(false_aXData1)
    false_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, false_chunks, labels)

    # K random training-test set
    splits = []
    for i in xrange(40):
        test_dataSet  = Dataset.random_samples(true_dataSet, len(false_aXData1))
        train_ids = [val for val in true_dataSet.sa.id if val not in test_dataSet.sa.id] 
        train_ids = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
        train_dataSet = true_dataSet[train_ids]
        splits.append([train_dataSet, test_dataSet])
        
    ## Multi dimension
    for i in xrange(3):
        count = 0
        for ths in threshold_mult:

            # save file name
            res_file = prefix+'_roc_'+opr+'_dim_'+str(i)+'_ths_'+str(ths)+'.pkl'
            res_file = os.path.join(cross_test_path, res_file)

            mutex_file_part = 'running_dim_'+str(i)+'_ths_'+str(ths)
            mutex_file_full = mutex_file_part+'_'+strMachine+'.txt'
            mutex_file      = os.path.join(cross_test_path, mutex_file_full)

            if os.path.isfile(res_file): 
                count += 1            
                continue
            elif hcu.is_file(cross_test_path, mutex_file_part): continue
            elif os.path.isfile(mutex_file): continue
            os.system('touch '+mutex_file)

            print "---------------------------------"
            print "Total splits: ", len(splits)


            ## fn_ll = []
            ## tn_ll = []
            ## fn_err_ll = []
            ## tn_err_ll = []
            ## for j, (l_wdata, l_vdata) in enumerate(splits):
            ##     fn_ll, tn_ll, fn_err_ll, tn_err_ll = anomaly_check_offline(j, l_wdata, l_vdata, nState, \
            ##                                                            trans_type, ths, false_dataSet, \
            ##                                                            check_dim=i)
            ##     print np.mean(fn_ll), np.mean(tn_ll)
            ## sys.exit()
                                  
            n_jobs = -1
            r = Parallel(n_jobs=n_jobs)(delayed(anomaly_check_offline)(j, l_wdata, l_vdata, nState, \
                                                                       trans_type, ths, false_dataSet, \
                                                                       cov_mult=cov_mult, check_dim=i) \
                                        for j, (l_wdata, l_vdata) in enumerate(splits))
            fn_ll, tn_ll, fn_err_ll, tn_err_ll = zip(*r)

            import operator
            fn_l = reduce(operator.add, fn_ll)
            tn_l = reduce(operator.add, tn_ll)
            fn_err_l = reduce(operator.add, fn_err_ll)
            tn_err_l = reduce(operator.add, tn_err_ll)

            d = {}
            d['fn']  = np.mean(fn_l)
            d['tp']  = 1.0 - np.mean(fn_l)
            d['tn']  = np.mean(tn_l)
            d['fp']  = 1.0 - np.mean(tn_l)

            if fn_err_l == []:         
                d['fn_err'] = 0.0
            else:
                d['fn_err'] = np.mean(fn_err_l)

            if tn_err_l == []:         
                d['tn_err'] = 0.0
            else:
                d['tn_err'] = np.mean(tn_err_l)

            ut.save_pickle(d,res_file)        
            os.system('rm '+mutex_file)
            print "-----------------------------------------------"

        if count == len(threshold_mult):
            print "#############################################################################"
            print "All file exist ", count
            print "#############################################################################"        

        
    if count == len(threshold_mult) and bPlot:

        import itertools
        colors = itertools.cycle(['g', 'm', 'c', 'k'])
        shapes = itertools.cycle(['x','v', 'o', '+'])
        
        fig = pp.figure()
        
        for i in xrange(3):
            fp_l = []
            tp_l = []
            err_l = []
            for ths in threshold_mult:
                res_file   = prefix+'_roc_'+opr+'_dim_'+str(i)+'_'+'ths_'+str(ths)+'.pkl'
                res_file   = os.path.join(cross_test_path, res_file)

                d = ut.load_pickle(res_file)
                tp  = d['tp'] 
                fn  = d['fn'] 
                fp  = d['fp'] 
                tn  = d['tn'] 
                fn_err = d['fn_err']         
                tn_err = d['tn_err']         

                fp_l.append([fp])
                tp_l.append([tp])
                err_l.append([fn_err])

            fp_l  = np.array(fp_l)*100.0
            tp_l  = np.array(tp_l)*100.0

            color = colors.next()
            shape = shapes.next()
            semantic_label='likelihood detection \n with known mechanism class '+str(i)
            pp.plot(fp_l, tp_l, shape+color, label= semantic_label, mec=color, ms=8, mew=2)



        ## fp_l = fp_l[:,0]
        ## tp_l = tp_l[:,0]
        
        ## from scipy.optimize import curve_fit
        ## def sigma(e, k ,n, offset): return k*((e+offset)**n)
        ## param, var = curve_fit(sigma, fp_l, tp_l)
        ## new_fp_l = np.linspace(fp_l.min(), fp_l.max(), 50)        
        ## pp.plot(new_fp_l, sigma(new_fp_l, *param))

        
        pp.xlabel('False positive rate (percentage)')
        pp.ylabel('True positive rate (percentage)')    
        ## pp.xlim([0, 30])
        pp.legend(loc=1,prop={'size':14})
        
        pp.show()
                            
    return

    ########################################################################################    

## def fig_roc(cross_data_path, aXData1, aXData2, chunks, labels, prefix, nState=20, \
##             threshold_mult = np.arange(0.05, 1.2, 0.05), opr='robot', attr='id', bPlot=False):

##     # For parallel computing
##     strMachine = socket.gethostname()+"_"+str(os.getpid())    
##     trans_type = "left_right"
    
##     # Check the existance of workspace
##     cross_test_path = os.path.join(cross_data_path, str(nState))
##     if os.path.isdir(cross_test_path) == False:
##         os.system('mkdir -p '+cross_test_path)

##     # min max scaling
##     aXData1_scaled, min_c1, max_c1 = dm.scaling(aXData1)
##     aXData2_scaled, min_c2, max_c2 = dm.scaling(aXData2)    
##     dataSet    = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, chunks, labels)

##     # Cross validation   
##     nfs    = NFoldPartitioner(cvtype=1,attr=attr) # 1-fold ?
##     spl    = splitters.Splitter(attr='partitions')
##     splits = [list(spl.generate(x)) for x in nfs.generate(dataSet)] # split by chunk

##     count = 0
##     for ths in threshold_mult:
    
##         # save file name
##         res_file = prefix+'_roc_'+opr+'_'+'ths_'+str(ths)+'.pkl'
##         res_file = os.path.join(cross_test_path, res_file)
        
##         mutex_file_part = 'running_ths_'+str(ths)
##         mutex_file_full = mutex_file_part+'_'+strMachine+'.txt'
##         mutex_file      = os.path.join(cross_test_path, mutex_file_full)
        
##         if os.path.isfile(res_file): 
##             count += 1            
##             continue
##         elif hcu.is_file(cross_test_path, mutex_file_part): continue
##         elif os.path.isfile(mutex_file): continue
##         os.system('touch '+mutex_file)

##         print "---------------------------------"
##         print "Total splits: ", len(splits)

##         n_jobs = -1
##         r = Parallel(n_jobs=n_jobs)(delayed(anomaly_check)(i, l_wdata, l_vdata, nState, trans_type, ths) \
##                                     for i, (l_wdata, l_vdata) in enumerate(splits)) 
##         fp_ll, err_ll = zip(*r)

        
##         import operator
##         fp_l = reduce(operator.add, fp_ll)
##         err_l = reduce(operator.add, err_ll)
        
##         d = {}
##         d['fp']  = np.mean(fp_l)
##         if err_l == []:         
##             d['err'] = 0.0
##         else:
##             d['err'] = np.mean(err_l)

##         ut.save_pickle(d,res_file)        
##         os.system('rm '+mutex_file)
##         print "-----------------------------------------------"

##     if count == len(threshold_mult):
##         print "#############################################################################"
##         print "All file exist ", count
##         print "#############################################################################"        

        
##     if count == len(threshold_mult) and bPlot:

##         fp_l = []
##         err_l = []
##         for ths in threshold_mult:
##             res_file   = prefix+'_roc_'+opr+'_'+'ths_'+str(ths)+'.pkl'
##             res_file   = os.path.join(cross_test_path, res_file)

##             d = ut.load_pickle(res_file)
##             fp  = d['fp'] 
##             err = d['err']         

##             fp_l.append([fp])
##             err_l.append([err])

##         fp_l  = np.array(fp_l)*100.0
##         sem_c = 'b'
##         sem_m = '+'
##         semantic_label='likelihood detection \n with known mechanism class'
##         pp.figure()
##         pp.plot(fp_l, err_l, '--'+sem_m+sem_c, label= semantic_label, mec=sem_c, ms=8, mew=2)
##         pp.xlabel('False positive rate (percentage)')
##         pp.ylabel('Mean excess log likelihood')    
##         ## pp.xlim([0, 30])
##         pp.show()
                            
##     return

    
## def anomaly_check_offline(i, l_wdata, l_vdata, nState, trans_type, ths, false_dataSet=None, 
##                           cov_mult=[1.0, 1.0, 1.0, 1.0], check_dim=2):

##     # Cross validation
##     if check_dim is not 2:
##         x_train1 = l_wdata.samples[:,check_dim,:]

##         lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, nEmissionDim=1)
##         if check_dim==0: lhm.fit(x_train1, cov_mult=[cov_mult[0]]*4)
##         elif check_dim==1: lhm.fit(x_train1, cov_mult=[cov_mult[3]]*4)
##     else:
##         x_train1 = l_wdata.samples[:,0,:]
##         x_train2 = l_wdata.samples[:,1,:]

##         lhm = learning_hmm_multi(nState=nState, trans_type=trans_type)
##         lhm.fit(x_train1, x_train2, cov_mult=cov_mult)
       
##     fn_l  = []
##     tn_l  = []
##     fn_err_l = []
##     tn_err_l = []

##     # True data
##     if check_dim == 2:
##         x_test1 = l_vdata.samples[:,0]
##         x_test2 = l_vdata.samples[:,1]
##     else:
##         x_test1 = l_vdata.samples[:,check_dim]

##     n,_ = np.shape(x_test1)
##     for i in range(n):
##         if check_dim == 2:
##             fn, err = lhm.anomaly_check(x_test1[i:i+1], x_test2[i:i+1], ths_mult=ths)           
##         else:
##             fn, err = lhm.anomaly_check(x_test1[i:i+1], ths_mult=ths)           

##         fn_l.append(fn)
##         if err != 0.0: fn_err_l.append(err)

##     # False data
##     if check_dim == 2:
##         x_test1 = false_dataSet.samples[:,0]
##         x_test2 = false_dataSet.samples[:,1]
##     else:
##         x_test1 = false_dataSet.samples[:,check_dim]
        
##     n = len(x_test1)
##     for i in range(n):
##         if check_dim == 2:            
##             tn, err = lhm.anomaly_check(np.array([x_test1[i]]), np.array([x_test2[i]]), ths_mult=ths)           
##         else:
##             tn, err = lhm.anomaly_check(np.array([x_test1[i]]), ths_mult=ths)           
            
##         tn_l.append(tn)
##         if err != 0.0: tn_err_l.append(err)

##     return fn_l, tn_l, fn_err_l, tn_err_l


def anomaly_check_offline(lhm, test_dataSet, false_dataSet, ths, check_dim=2):

    tp = 0.0
    fn = 0.0
    fp = 0.0
    tn = 0.0
    
    err_l = []
    delay_l = []

    # 1) Use True data to get true negative rate
    if test_dataSet != []:    
        if check_dim == 2:
            x_test1 = test_dataSet.samples[:,0]
            x_test2 = test_dataSet.samples[:,1]
        else:
            x_test1 = test_dataSet.samples[:,check_dim]

        n = len(x_test1)
        for i in range(n):
            # anomaly_check only returns anomaly cases only
            if check_dim == 2:            
                an, err = lhm.anomaly_check(x_test1[i], x_test2[i], ths_mult=ths)   
            else:
                an, err = lhm.anomaly_check(x_test1[i], ths_mult=ths)           

            if an == 1.0:   fn += 1.0
            elif an == 0.0: tp += 1.0
    
    # 2) Use False data to get true negative rate
    if false_dataSet != []:
        if check_dim == 2:
            x_test1 = false_dataSet.samples[:,0]
            x_test2 = false_dataSet.samples[:,1]
        else:
            x_test1 = false_dataSet.samples[:,check_dim]
        anomaly_idx = false_dataSet.sa.anomaly_idx
        
        n = len(x_test1)
        for i in range(n):

            # anomaly_check only returns anomaly cases only
            delay = 0
            if check_dim == 2:            
                an, err = lhm.anomaly_check(x_test1[i], x_test2[i], ths_mult=ths)   
            else:
                an, err = lhm.anomaly_check(x_test1[i], ths_mult=ths)           

            if an == 1.0:   tn += 1.0
            elif an == 0.0: fp += 1.0
                        
    return tp, fn, fp, tn, delay_l


def anomaly_check_online(lhm, test_dataSet, false_dataSet, ths, check_dim=2):

    tp = 0.0
    fn = 0.0
    fp = 0.0
    tn = 0.0
    
    err_l = []
    delay_l = []
    false_detection_l = []

    # 1) Use True data to get true negative rate
    if test_dataSet != []:    
        if check_dim == 2:
            x_test1 = test_dataSet.samples[:,0]
            x_test2 = test_dataSet.samples[:,1]
        else:
            x_test1 = test_dataSet.samples[:,check_dim]

        n = len(x_test1)
        for i in range(n):
            m = len(x_test1[i])

            # anomaly_check only returns anomaly cases only
            for j in range(2,m):                    

                if check_dim == 2:            
                    an, err = lhm.anomaly_check(x_test1[i][:j], x_test2[i][:j], ths_mult=ths)   
                else:
                    an, err = lhm.anomaly_check(x_test1[i][:j], ths_mult=ths)           

                if an == 1.0:   fn += 1.0
                elif an == 0.0: tp += 1.0

                
    # 2) Use False data to get true negative rate
    if false_dataSet != []:
        if check_dim == 2:
            x_test1 = false_dataSet.samples[:,0]
            x_test2 = false_dataSet.samples[:,1]
        else:
            x_test1 = false_dataSet.samples[:,check_dim]
        anomaly_idx = false_dataSet.sa.anomaly_idx

        false_detection_l = np.zeros(len(x_test1))

        n = len(x_test1)
        for i in range(n):
            m = len(x_test1[i])

            # anomaly_check only returns anomaly cases only
            delay = 0
            for j in range(2,m):                    

                if check_dim == 2:            
                    an, err = lhm.anomaly_check(x_test1[i][:j], x_test2[i][:j], ths_mult=ths)   
                else:
                    an, err = lhm.anomaly_check(x_test1[i][:j], ths_mult=ths)           

                delay = j-anomaly_idx[i]

                if delay >= 0:
                    if an == 1.0:
                        tn += 1.0
                        delay_l.append(delay)
                    elif an == 0.0:
                        fp += 1.0
                else:
                    if an == 1.0:
                        fn += 1.0
                    elif an == 0.0:
                        tp += 1.0    
                    ## err_l.append(err)

                if an == 1.0: false_detection_l[i] = True

    return tp, fn, fp, tn, delay_l, false_detection_l


def anomaly_check_online_test(lhm, test_dataSet, false_dataSet, ths, check_dim=2):

    tp = 0.0
    fn = 0.0
    fp = 0.0
    tn = 0.0

    err_l = []
    delay_l = []

    ## # 1) Use True data to get true negative rate
    ## if test_dataSet != []:    
    ## if check_dim == 2:
    ##     x_test1 = test_dataSet.samples[:,0]
    ##     x_test2 = test_dataSet.samples[:,1]
    ## else:
    ##     x_test1 = test_dataSet.samples[:,check_dim]

    ## n = len(x_test1)
    ## for i in range(n):
    ##     m = len(x_test1[i])

    ##     # anomaly_check only returns anomaly cases only
    ##     for j in range(2,m):                    

    ##         if check_dim == 2:            
    ##             an, err = lhm.anomaly_check(x_test1[i][:j], x_test2[i][:j], ths_mult=ths)   
    ##         else:
    ##             an, err = lhm.anomaly_check(x_test1[i][:j], ths_mult=ths)           
            
    ##         if an == 1.0:   fn += 1.0
    ##         elif an == 0.0: tp += 1.0

                
    # 2) Use False data to get true negative rate
    if false_dataSet != []:
        if check_dim == 2:
            x_test1 = false_dataSet.samples[:,0]
            x_test2 = false_dataSet.samples[:,1]
        else:
            x_test1 = false_dataSet.samples[:,check_dim]
        anomaly_idx = false_dataSet.sa.anomaly_idx

        n = len(x_test1)
        for i in range(n):
            m = len(x_test1[i])

            # anomaly_check only returns anomaly cases only
            delay = 0
            for j in range(2,m):                    

                if check_dim == 2:            
                    an, err = lhm.anomaly_check(x_test1[i][:j], x_test2[i][:j], ths_mult=ths)   
                else:
                    an, err = lhm.anomaly_check(x_test1[i][:j], ths_mult=ths)           

                delay = j-anomaly_idx[i]

                if an == 1.0: break


            if an == 1.0:
                if delay >= 0:
                    tn += 1.0
                    delay_l.append(delay)                
                else:
                    fn += 1.0
            elif an == 0.0:
                print "Error with anomaly check"
                fp += 1.0

    return tp, fn, fp, tn, delay_l
    
    
def anomaly_check(i, l_wdata, l_vdata, nState, trans_type, ths):

    # Cross validation
    x_train1 = l_wdata.samples[:,0,:]
    x_train2 = l_wdata.samples[:,1,:]

    lhm = learning_hmm_multi(nState=nState, trans_type=trans_type)
    lhm.fit(x_train1, x_train2)

    x_test1 = l_vdata.samples[:,0,:]
    x_test2 = l_vdata.samples[:,1,:]
    n,m = np.shape(x_test1)
    
    fp_l  = []
    err_l = []
    for i in range(n):
        for j in range(2,m,1):
            fp, err = lhm.anomaly_check(x_test1[i:i+1,:j], x_test2[i:i+1,:j], ths_mult=ths)           
            fp_l.append(fp)
            if err != 0.0: err_l.append(err)
                
    return fp_l, err_l
    

    

def plot_audio(time_list, data_list, title=None, chunk=1024, rate=44100.0, max_int=32768.0 ):

    import librosa
    from librosa import feature


    data_seq = data_list.flatten()
    t = np.arange(0.0, len(data_seq), 1.0)/rate    
    
    # find init
    pp.figure()
    ax1 =pp.subplot(411)
    pp.plot(t, data_seq)
    ## pp.plot(time_list, data_list)
    ## pp.stem([idx_start, idx_end], [f[idx_start], f[idx_end]], 'k-*', bottom=0)
    ax1.set_xlim([0, t[-1]])
    if title is not None: pp.title(title)


    #========== Spectrogram =========================
    from matplotlib.mlab import complex_spectrum, specgram, magnitude_spectrum
    ax = pp.subplot(412)        
    pp.specgram(data_seq, NFFT=chunk, Fs=rate)
    ax.set_ylim([0,5000])
    ax = pp.subplot(413)       
    S = librosa.feature.melspectrogram(data_seq, sr=rate, n_fft=chunk, n_mels=30)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    librosa.display.specshow(log_S, sr=rate, hop_length=8, x_axis='time', y_axis='mel')
    ## ax.set_ylim([0,5000])

    ## ax = pp.subplot(414)            
    ## bands = np.arange(0, 10) * 100
    ## hists = []
    ## for i, data in enumerate(data_list):        


    ##     S = librosa.feature.melspectrogram(data, sr=rate, n_fft=chunk, n_mels=30, fmin=100, fmax=5000)
    ##     log_S = librosa.logamplitude(S, ref_power=np.max)
        
    ##     ## new_data = np.hstack([data/max_int, np.zeros(len(data))]) # zero padding
    ##     ## fft = np.fft.fft(new_data)  # FFT          
    ##     ## fftr=10*np.log10(abs(fft.real))[:len(new_data)/2]
    ##     ## freq=np.fft.fftfreq(np.arange(len(new_data)).shape[-1])[:len(new_data)/2]
        
    ##     ## ## print fftr.shape, freq.shape
    ##     ## hists.append(S[:,1])
    ##     if hists == []:
    ##         hists = log_S[:,1:2]
    ##     else:
    ##         hists = np.hstack([hists, log_S[:,1:2]])
    ##     print log_S.shape, S.shape, hists.shape

    ##     ## #count bin
    ##     ## hist, hin_edges = np.histogram(freq, weights=fftr, bins=bands, density=True)
    ##     ## hists.append(hist)
        
    ## pp.imshow(hists, origin='down')
        
        

        
    #========== RMS =========================
    ax2 = pp.subplot(414)    
    rms_list = []
    for i, data in enumerate(data_list):
        rms_list.append(get_rms(data))
    t = np.arange(0.0, len(data_list), 1.0)*chunk/rate    
    pp.plot(t, rms_list) 

    #========== MFCC =========================
    ax = pp.subplot(412)
    mfcc_feat = librosa.feature.mfcc(data_seq, n_mfcc=4, sr=rate, n_fft=1024)
    ## mfcc_feat = feature.mfcc(data_list, sr=rate, n_mfcc=13, n_fft=1024, )
    pp.imshow(mfcc_feat, origin='down')
    ## ax.set_xlim([0, t[-1]*100])

    ax = pp.subplot(413)
    S = feature.melspectrogram(data_list, sr=rate, n_fft=1024, hop_length=1, n_mels=128)
    log_S = librosa.logamplitude(S, ref_power=np.max)        
    ## mfcc_feat = librosa.feature.mfcc(S=log_S, n_mfcc=20, sr=rate, n_fft=1024)
    mfcc_feat = librosa.feature.delta(mfcc_feat)
    pp.imshow(mfcc_feat, origin='down')

    ax = pp.subplot(414)
    mfcc_feat = librosa.feature.delta(mfcc_feat, order=2)
    pp.imshow(mfcc_feat, origin='down')

    
    
    
    ## librosa.display.specshow(log_S, sr=rate, hop_length=256, x_axis='time', y_axis='mel')

    ## print log_S.shape
    
    ## pp.psd(data_list, NFFT=chunk, Fs=rate, noverlap=0)    
    ## complex_spectrum(data_list, Fs=rate)
    ## ax2.set_xlim([0, t[-1]])
    ## ax2.set_ylim([0, 30000])
    
    
    ## from features import mfcc, logfbank        
    ## mfcc_feat = mfcc(data_list, rate, numcep=13)
    ## ax4 = pp.subplot(414)
    ## pp.imshow(mfcc_feat.T, origin='down')
    ## ax4.set_xlim([0, t[-1]*100])

    ## ax4 = pp.subplot(414)
    ## fbank_feat = logfbank(data_list, rate, winlen=float(chunk)/rate, nfft=chunk, lowfreq=10, highfreq=3000)    
    ## pp.imshow(fbank_feat.T, origin='down')
    ## ax4.set_xlim([0, t[-1]*100])



    
    ## pp.subplot(412)
    ## for k in xrange(len(audio_data_cut)):            
    ##     cur_time = time_range + audio_time_cut[0] + float(k)*1024.0/44100.0
    ##     pp.plot(cur_time, audio_data_cut[k], 'b.')

    ## pp.subplot(413)
    ## pp.plot(audio_time_cut, np.mean((np.array(audio_data_cut)),axis=1))

    ## pp.subplot(414)
    ## pp.plot(audio_time_cut, np.std((np.array(audio_data_cut)),axis=1))
    ## pp.stem([idx_start, idx_end], [max(audio_data_l[i][idx_start]), max(audio_data_l[i][idx_end])], 'k-*', bottom=0)
    ## pp.title(names[i])
    ## pp.plot(audio_freq_l[i], audio_amp_l[i])
    pp.show()


def plot_one(data1, data2, false_data1=None, false_data2=None, data_idx=0, labels=None, freq=43.0):

    fig = pp.figure()
    plt.rc('text', usetex=True)
    
    ax1 = pp.subplot(211)
    if false_data1 is None:
        data = data1[data_idx]
    else:
        data = false_data1[data_idx]        

    x   = np.arange(0., float(len(data))) * (1./freq)
        
    pp.plot(x, data, 'b', linewidth=1.5, label='Force')
    #ax1.set_xlim([0, len(data)])
    ax1.set_ylim([0, np.amax(data)*1.1])
    pp.grid()
    ax1.set_ylabel("Magnitude [N]", fontsize=18)

        
    ax2 = pp.subplot(212)
    if false_data2 is None:
        data = data2[data_idx]
    else:
        data = false_data2[data_idx]
    
    pp.plot(x, data, 'b', linewidth=1.5, label='Sound')
    #ax2.set_xlim([0, len(data)])
    pp.grid()
    ax2.set_ylabel("RMS", fontsize=18)
    ax2.set_xlabel("Time step [sec]", fontsize=18)
    
    ax1.legend(prop={'size':18})
    ax2.legend(prop={'size':18})
    
    fig.savefig('test.pdf')
    fig.savefig('test.png')
    pp.show()

    

    
def plot_all(data1, data2, false_data1=None, false_data2=None, labels=None, distribution=False, freq=43.0):

        ## # find init
        ## pp.figure()
        ## pp.subplot(211)
        ## pp.plot(f)
        ## pp.stem([idx_start, idx_end], [f[idx_start], f[idx_end]], 'k-*', bottom=0)
        ## pp.title(names[i])
        ## pp.subplot(212)
        ## pp.plot(force[2,:])
        ## pp.show()
        
        ## plot_audio(audio_time_cut, audio_data_cut, chunk=CHUNK, rate=RATE, title=names[i])

    def vectors_to_mean_sigma(vecs):
        data = np.array(vecs)
        m,n = np.shape(data)
        mu  = np.zeros(n)
        sig = np.zeros(n)
        
        for i in xrange(n):
            mu[i]  = np.mean(data[:,i])
            sig[i] = np.std(data[:,i])
        
        return mu, sig


    if false_data1 is None:
        data = data1[0]
    else:
        data = false_data1[0]                
    x   = np.arange(0., float(len(data))) * (1./freq)
        
        
    fig = pp.figure()
    plt.rc('text', usetex=True)

    #-----------------------------------------------------------------
    ax1 = pp.subplot(211)
    if distribution:
        x       = range(len(data1[0]))
        mu, sig = vectors_to_mean_sigma(data1)        
        ax1.fill_between(x, mu-sig, mu+sig, facecolor='green', edgecolor='1.0', \
                         alpha=0.5, interpolate=True)            
    else:
        for i, d in enumerate(data1):
            if not(labels is not None and labels[i] == False):
                true_line, = pp.plot(x, d, label='Normal data')

    # False data
    if false_data1 is not None:
        for i, d in enumerate(false_data1):
            pp.plot(x. d, color='k', linewidth=1.0)
                
    ## # False data
    ## for i, d in enumerate(data1):
    ##     if labels is not None and labels[i] == False:
    ##         pp.plot(d, label=str(i), color='k', linewidth=1.0)
    ax1.set_ylabel("Force [N]", fontsize=18)

    ## true_line = ax1.plot([], [], color='green', alpha=0.5, linewidth=10, label='Normal data') #fake for legend
    ## false_line = ax1.plot([], [], color='k', linewidth=10, label='Abnormal data') #fake for legend    
    ## ax1.legend()

        
    #-----------------------------------------------------------------
    ax2 = pp.subplot(212)
    if distribution:
        x       = range(len(data2[0]))
        mu, sig = vectors_to_mean_sigma(data2)        
        ax2.fill_between(x, mu-sig, mu+sig, facecolor='green', edgecolor='1.0', \
                         alpha=0.5, interpolate=True)            
    else:        
        for i, d in enumerate(data2):
            if not(labels is not None and labels[i] == False):
                pp.plot(x, d)

    # for false data
    if false_data2 is not None:
        for i, d in enumerate(false_data2):
            pp.plot(x, d, color='k', linewidth=1.0)
            
    ax2.set_ylabel("Sound [RMS]", fontsize=18)
    ax2.set_xlabel("Time step [sec]", fontsize=18)

    ## true_line = ax2.plot([], [], color='green', alpha=0.5, linewidth=10, label='Normal data') #fake for legend
    ## false_line = ax2.plot([], [], color='k', linewidth=10, label='Abnormal data') #fake for legend
    ## ax2.legend()

    fig.savefig('tool_case_normal.pdf')
    fig.savefig('tool_case_normal.png')    
    pp.show()
    
    


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--roc_online_simulated_dim_check', '--ronsimdim', action='store_true', \
                 dest='bRocOnlineSimDimCheck', default=False, 
                 help='Plot online ROC by simulated anomaly form dim comparison')    
    p.add_option('--roc_online_simulated_method_check', '--ronsimmthd', action='store_true', \
                 dest='bRocOnlineSimMethodCheck',
                 default=False, help='Plot online ROC by simulated anomaly')    
    p.add_option('--roc_online_method_check', '--ronmthd', action='store_true', \
                 dest='bRocOnlineMethodCheck',
                 default=False, help='Plot online ROC by real anomaly')    
    p.add_option('--roc_offline_method_check', '--roffmthd', action='store_true', \
                 dest='bRocOfflineMethodCheck',
                 default=False, help='Plot offline ROC by real anomaly')    
    p.add_option('--online_method_check', '--omc', action='store_true', \
                 dest='bOnlineMethodCheck',
                 default=False, help='Plot offline ROC by real anomaly')    
    p.add_option('--test', action='store_true', \
                 dest='bTest',
                 default=False, help='Plot online ROC by simulated anomaly')    
    p.add_option('--all_plot', '--all', action='store_true', dest='bAllPlot',
                 default=False, help='Plot all data')
    p.add_option('--one_plot', '--one', action='store_true', dest='bOnePlot',
                 default=False, help='Plot one data')
    p.add_option('--plot', '--p', action='store_true', dest='bPlot',
                 default=False, help='Plot')
    p.add_option('--rm_running', '--rr', action='store_true', dest='bRemoveRunning',
                 default=False, help='Remove all the running files')
    

    p.add_option('--abnormal', '--an', action='store_true', dest='bAbnormal',
                 default=False, help='Renew pickle files.')
    p.add_option('--simulated_abnormal', '--sim_an', action='store_true', dest='bSimAbnormal',
                 default=False, help='.')
    p.add_option('--animation', '--ani', action='store_true', dest='bAnimation',
                 default=False, help='Plot by time using animation')
    p.add_option('--roc_human', '--rh', action='store_true', dest='bRocHuman',
                 default=False, help='Plot by a figure of ROC human')
    p.add_option('--roc_online_robot', '--ron', action='store_true', dest='bRocOnlineRobot',
                 default=False, help='Plot by a figure of ROC robot')
    p.add_option('--roc_offline_robot', '--roff', action='store_true', dest='bRocOfflineRobot',
                 default=False, help='Plot by a figure of ROC robot')
    p.add_option('--path_disp', '--pd', action='store_true', dest='bPathDisp',
                 default=False, help='Plot all path')
    p.add_option('--progress_diff', '--prd', action='store_true', dest='bProgressDiff',
                 default=False, help='Plot progress difference')
    p.add_option('--fftdisp', '--fd', action='store_true', dest='bFftDisp',
                 default=False, help='Plot')
    p.add_option('--use_ml_pkl', '--mp', action='store_true', dest='bUseMLObspickle',
                 default=False, help='Use pre-trained object file')
    opt, args = p.parse_args()


    ## data_path = os.environ['HRLBASEPATH']+'/src/projects/anomaly/test_data/'
    cross_root_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/robot'
    all_task_names  = ['microwave_black', 'microwave_white', 'lab_cabinet', 'wallsw', 'switch_device', \
                       'switch_outlet', 'case', 'lock_wipes', 'lock_huggies', 'toaster_white', 'glass_case']
    ## all_task_names  = ['microwave_white']
                
    class_num = 2
    task  = 1

    if class_num == 0:
        class_name = 'door'
        task_names = ['microwave_black', 'microwave_white', 'lab_cabinet']
        f_zero_size = [8, 5, 10]
        f_thres     = [1.0, 1.7, 3.0]
        audio_thres = [1.0, 1.0, 1.0]
        cov_mult = [[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0]]
        nState_l    = [20, 20, 10]
    elif class_num == 1: 
        class_name = 'switch'
        task_names = ['wallsw', 'switch_device', 'switch_outlet']
        f_zero_size = [5, 18, 7]
        f_thres     = [0.7, 0.5, 1.0]
        audio_thres = [1.0, 0.7, 0.0015]
        cov_mult = [[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0]]
        nState_l    = [20, 20, 20]
    elif class_num == 2:        
        class_name = 'lock'
        task_names = ['case', 'lock_wipes', 'lock_huggies']
        f_zero_size = [5, 5, 5]
        f_thres     = [1.0, 1.0, 1.35]
        audio_thres = [1.0, 1.0, 1.0]
        cov_mult = [[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0]]
        nState_l    = [20, 20, 20]
    elif class_num == 3:        
        class_name = 'complex'
        task_names = ['toaster_white', 'glass_case']
        f_zero_size = [5, 3, 8]
        f_thres     = [1.0, 1.5, 1.35]
        audio_thres = [1.0, 1.0, 1.0]
        cov_mult    = [[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0]]
        nState_l    = [20, 20, 20] #glass 10?
    elif class_num == 4:        
        class_name = 'button'
        task_names = ['joystick', 'keyboard']
        f_zero_size = [5, 5, 8]
        f_thres     = [1.35, 1.35, 1.35]
        audio_thres = [1.0, 1.0, 1.0]
        cov_mult    = [[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0]]
        nState_l    = [20, 20, 20]
    else:
        print "Please specify right task."
        sys.exit()

    scale = 10.0       
    freq  = 43.0 #Hz
    
    # Load data
    pkl_file  = os.path.join(cross_root_path,task_names[task]+"_data.pkl")    
    data_path = os.environ['HRLBASEPATH']+'/src/projects/anomaly/test_data/robot_20150213/'+class_name+'/'
      
    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    if opt.bTest: 
        
        print "ROC Offline Robot with simulated anomalies"
        cross_data_path = os.path.join(cross_root_path, 'multi_sim_'+task_names[task])
        nState          = nState_l[task]
        threshold_mult  = np.logspace(0.1, 1.5, 30, endpoint=True) 
        attr            = 'id'
        onoff_type      = 'online'
        check_methods   = ['global', 'progress']
        check_dims      = [2]
        test_title      = 'online_method_test'

        fig_roc(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
                task_names[task], nState, threshold_mult, \
                opr='robot', attr='id', bPlot=opt.bPlot, cov_mult=cov_mult[task], renew=False, test=True,
                sim=True)
                    
    elif opt.bRocOnlineSimDimCheck: 
        
        print "ROC Offline Robot with simulated anomalies"
        test_title      = 'online_dim_comp'
        cross_data_path = os.path.join(cross_root_path, 'multi_sim_'+task_names[task], test_title)
        nState          = nState_l[task]
        threshold_mult  = np.logspace(0.1, 2.0, 30, endpoint=True) - 5.0 
        attr            = 'id'
        onoff_type      = 'online'
        check_methods   = ['progress']
        check_dims      = [0,1,2]
        an_type         = 'both'
        force_an        = ['inelastic', 'inelastic_continue', 'elastic', 'elastic_continue']
        sound_an        = ['rndsharp', 'rnddull'] 
        disp            = 'None'

        true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet \
          = dm.loadData(pkl_file, data_path, task_names[task], f_zero_size[task], f_thres[task], \
                        audio_thres[task], cross_data_path, an_type, force_an, sound_an)

                        
        if opt.bAllPlot is not True:
            fig_roc(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
                    task_names[task], nState, threshold_mult, \
                    opr='robot', attr='id', bPlot=opt.bPlot, cov_mult=cov_mult[task], renew=False, \
                    disp=disp, rm_run=opt.bRemoveRunning, sim=True)
        else:
            fig_roc_all(cross_root_path, all_task_names, test_title, nState, threshold_mult, check_methods, \
                        check_dims, an_type, force_an, sound_an, sim=True)

                            
            
    #---------------------------------------------------------------------------           
    elif opt.bRocOnlineSimMethodCheck:
        
        print "ROC Online Robot with simulated anomalies"
        test_title      = 'online_method_comp'
        cross_data_path = os.path.join(cross_root_path, 'multi_sim_'+task_names[task], test_title)
        nState          = nState_l[task]
        threshold_mult  = np.logspace(-1.0, 2.5, 30, endpoint=True) -2.0
        attr            = 'id'
        onoff_type      = 'online'
        check_methods   = ['change', 'global', 'globalChange', 'progress']
        check_dims      = [2]
        an_type         = 'both'
        force_an        = ['normal', 'inelastic', 'inelastic_continue', 'elastic', 'elastic_continue']
        sound_an        = ['normal', 'rndsharp', 'rnddull'] 
        disp            = 'None'

        # temp
        ## threshold_mult  = [2.0]
        ## check_methods   = ['progress']
        ## disp            = 'test'
            
        true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet \
          = dm.loadData(pkl_file, data_path, task_names[task], f_zero_size[task], f_thres[task], \
                        audio_thres[task], cross_data_path, an_type, force_an, sound_an)

        if opt.bAllPlot is not True:
            fig_roc(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
                    task_names[task], nState, threshold_mult, \
                    opr='robot', attr='id', bPlot=opt.bPlot, cov_mult=cov_mult[task], renew=False, \
                    disp=disp, rm_run=opt.bRemoveRunning, sim=True)
        else:
            fig_roc_all(cross_root_path, all_task_names, test_title, nState, threshold_mult, check_methods, \
                        check_dims, an_type, force_an, sound_an, sim=True)


    #---------------------------------------------------------------------------           
    elif opt.bRocOnlineMethodCheck:
        
        print "ROC Online Robot with real anomalies"
        test_title      = 'online_method_comp'
        cross_data_path = os.path.join(cross_root_path, 'multi_'+task_names[task], test_title)
        nState          = nState_l[task]
        threshold_mult  = np.logspace(-1.0, 2.5, 30, endpoint=True) -2.0
        attr            = 'id'
        onoff_type      = 'online'
        check_methods   = ['change', 'global', 'globalChange', 'progress']
        check_dims      = [2]
        disp            = 'None'

        true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet \
          = dm.loadData(pkl_file, data_path, task_names[task], f_zero_size[task], f_thres[task], \
                        audio_thres[task], cross_data_path)

        if opt.bAllPlot is not True:
            fig_roc(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
                    task_names[task], nState, threshold_mult, \
                    opr='robot', attr='id', bPlot=opt.bPlot, cov_mult=cov_mult[task], renew=False, \
                    disp=disp, rm_run=opt.bRemoveRunning)
        else:
            fig_roc_all(cross_root_path, all_task_names, test_title, nState, threshold_mult, check_methods, \
                        check_dims)
                            

    #---------------------------------------------------------------------------           
    elif opt.bRocOfflineMethodCheck:
        
        print "ROC Online Robot with real anomalies"
        test_title      = 'offline_method_comp'
        cross_data_path = os.path.join(cross_root_path, 'multi_'+task_names[task], test_title)
        nState          = nState_l[task]
        threshold_mult  = np.logspace(-1.0, 2.5, 30, endpoint=True) -2.0
        attr            = 'id'
        onoff_type      = 'offline'
        check_methods   = ['global', 'progress']
        check_dims      = [2]
        disp            = 'None'

        true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet \
          = dm.loadData(pkl_file, data_path, task_names[task], f_zero_size[task], f_thres[task], \
                        audio_thres[task], cross_data_path)

        if opt.bAllPlot is not True:
            fig_roc(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
                    task_names[task], nState, threshold_mult, \
                    opr='robot', attr='id', bPlot=opt.bPlot, cov_mult=cov_mult[task], renew=False, \
                    disp=disp, rm_run=opt.bRemoveRunning)
        else:
            fig_roc_all(cross_root_path, all_task_names, test_title, nState, threshold_mult, check_methods, \
                        check_dims)


    #---------------------------------------------------------------------------           
    elif opt.bOnlineMethodCheck:
        
        print "Evaluation for Online Robot with real anomalies and automatic threshold decision"
        test_title      = 'online_method_check'
        cross_data_path = os.path.join(cross_root_path, 'multi_'+task_names[task], test_title)
        nState          = nState_l[task]
        attr            = 'id'
        onoff_type      = 'online'
        check_methods   = ['progress']
        check_dims      = [2]
        disp            = 'None'
        rFold           = 0.75 # ratio of training dataset in true dataset
        nDataSet        = 10

        true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet \
          = dm.loadData(pkl_file, data_path, task_names[task], f_zero_size[task], f_thres[task], \
                        audio_thres[task], cross_data_path, rFold=rFold, nDataSet=nDataSet)

        if opt.bAllPlot is not True:
            fig_eval(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
                     task_names[task], nState, \
                     opr='robot', attr='id', bPlot=opt.bPlot, cov_mult=cov_mult[task], renew=False, \
                     disp=disp, rm_run=opt.bRemoveRunning)
        else:
            fig_eval_all(cross_root_path, all_task_names, test_title, nState, check_methods, \
                         check_dims, nDataSet)
                        
    #---------------------------------------------------------------------------           
    ## elif opt.bRocOfflineSimMethodCheck:
        
    ##     print "ROC Online Robot with simulated anomalies"
    ##     cross_data_path = os.path.join(cross_root_path, 'multi_sim_'+task_names[task])
    ##     nState          = nState_l[task]
    ##     threshold_mult  = np.logspace(-1.0, 1.5, 20, endpoint=True) # np.arange(0.0, 30.001, 2.0) #
    ##     attr            = 'id'
    ##     onoff_type      = 'online'
    ##     check_methods   = ['global', 'progress']
    ##     check_dims      = [2]
    ##     test_title      = 'offline_method_comp'

    ##     fig_roc_sim(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
    ##                 task_names[task], nState, threshold_mult, \
    ##                 opr='robot', attr='id', bPlot=opt.bPlot, cov_mult=cov_mult[task], renew=False)

                    
    ## #---------------------------------------------------------------------------           
    ## elif opt.bRocOnlineRobot:

    ##     cross_data_path = os.path.join(cross_root_path, 'multi_'+task_names[task])
    ##     nState          = nState_l[task]
    ##     threshold_mult  = np.arange(0.0, 4.2, 0.1)    
    ##     attr            = 'id'

    ##     fig_roc(cross_data_path, aXData1, aXData2, chunks, labels, task_names[task], nState, threshold_mult, \
    ##             opr='robot', attr='id', bPlot=opt.bPlot, cov_mult=cov_mult[task])

    ##     if opt.bAllPlot:
    ##         if task ==1:
    ##             ## prefixes = ['microwave', 'microwave_black', 'microwave_white']
    ##             prefixes = ['microwave', 'microwave_black']
    ##         else:
    ##             prefixes = ['microwave', 'microwave_black']
                
    ##         cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/robot'                
    ##         fig_roc_all(cross_data_path, nState, threshold_mult, prefixes, opr='robot', attr='id')
            
            
    #---------------------------------------------------------------------------           
    elif opt.bRocOfflineRobot:
        
        print "ROC Offline Robot"
        cross_data_path = os.path.join(cross_root_path, 'multi_'+task_names[task])
        nState          = nState_l[task]
        threshold_mult  = np.arange(0.0, 24.2, 0.3)    
        attr            = 'id'

        fig_roc_offline(cross_data_path, \
                        true_aXData1, true_aXData2, true_chunks, \
                        false_aXData1, false_aXData2, false_chunks, \
                        task_names[task], nState, threshold_mult, \
                        opr='robot', attr='id', bPlot=opt.bPlot)

        if opt.bAllPlot:
            if task ==1:
                ## prefixes = ['microwave', 'microwave_black', 'microwave_white']
                prefixes = ['microwave', 'microwave_black']
            else:
                prefixes = ['microwave', 'microwave_black']
                
            cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/robot'                
            fig_roc_all(cross_data_path, nState, threshold_mult, prefixes, opr='robot', attr='id')
            

    #---------------------------------------------------------------------------
    elif opt.bOnePlot:

        true_aXData1_scaled, min_c1, max_c1 = dm.scaling(true_aXData1, scale=scale)
        true_aXData2_scaled, min_c2, max_c2 = dm.scaling(true_aXData2, scale=scale)    

        if opt.bAbnormal or opt.bSimAbnormal:
            for idx in xrange(len(false_aXData1)):
                # min max scaling
                false_aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=scale)
                false_aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=scale)

                plot_one(true_aXData1, true_aXData2, false_aXData1, false_aXData2, data_idx=idx, freq=freq)

        else:
            for idx in xrange(len(true_aXData1)):
                plot_one(true_aXData1, true_aXData2, data_idx=idx, freq=freq)            

            
    #---------------------------------------------------------------------------
    elif opt.bAllPlot:

        test_title      = 'plot_all'
        cross_data_path = os.path.join(cross_root_path, test_title)
        
        true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet \
          = dm.loadData(pkl_file, data_path, task_names[task], f_zero_size[task], f_thres[task], \
                        audio_thres[task])
        
        true_aXData1_scaled, min_c1, max_c1 = dm.scaling(true_aXData1, scale=scale)
        true_aXData2_scaled, min_c2, max_c2 = dm.scaling(true_aXData2, scale=scale)    

        if opt.bAbnormal or opt.bSimAbnormal:
            
            # min max scaling
            false_aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=scale)
            false_aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=scale)
                        
            ## plot_all(true_aXData1_scaled, true_aXData2_scaled, false_aXData1_scaled, false_aXData2_scaled)
            ## plot_all(true_aXData1, true_aXData2, false_aXData1, false_aXData2)
            plot_all(true_aXData1, true_aXData2, false_aXData1, false_aXData2, distribution=True)
            
        else:
            ## plot_all(true_aXData1_scaled, true_aXData2_scaled)            
            plot_all(true_aXData1, true_aXData2)            

        ## print min_c1, max_c1, np.min(aXData1_scaled), np.max(aXData1_scaled)
        ## print min_c2, max_c2, np.min(aXData2_scaled), np.max(aXData2_scaled)

        
    #---------------------------------------------------------------------------   
    elif opt.bFftDisp:
        d = dm.load_data(data_path, task_names[task], normal_only=False)
        
        audio_time_list = d['audio_time']
        audio_data_list = d['audio_data']

        plot_audio(audio_time_list[0], audio_data_list[0])
        
            
    #---------------------------------------------------------------------------   
    elif opt.bAnimation:

        nState   = 20
        trans_type= "left_right"
        ## nMaxStep = 36 # total step of data. It should be automatically assigned...

        aXData1_scaled, min_c1, max_c1 = dm.scaling(aXData1)
        aXData2_scaled, min_c2, max_c2 = dm.scaling(aXData2)    
        
        # Learning
        from hrl_anomaly_detection.HMM.learning_hmm_multi import learning_hmm_multi
        lhm = learning_hmm_multi(nState=nState, trans_type=trans_type)
        lhm.fit(aXData1_scaled, aXData2_scaled)

        
        lhm.simulation(aXData1_scaled[2], aXData2_scaled[2])

    #---------------------------------------------------------------------------   
    elif opt.bPathDisp:

        nState   = nState_l[task]
        trans_type= "left_right"
        check_dim = 2
        if check_dim == 0 or check_dim == 1: nEmissionDim=1
        else: nEmissionDim=2

        aXData1_scaled, min_c1, max_c1 = dm.scaling(true_aXData1)
        aXData2_scaled, min_c2, max_c2 = dm.scaling(true_aXData2)    
        true_labels = [True]*len(true_aXData1)

        true_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, true_chunks, true_labels)
            
        x_train1  = true_dataSet.samples[:,0,:]
        x_train2  = true_dataSet.samples[:,1,:]

        # Learning
        lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, nEmissionDim=nEmissionDim)

        if check_dim == 0: lhm.fit(x_train1, cov_mult=[cov_mult[task][0]]*4)
        elif check_dim == 1: lhm.fit(x_train2, cov_mult=[cov_mult[task][3]]*4)
        else: lhm.fit(x_train1, x_train2, cov_mult=cov_mult[task])

        x_test1 = x_train1
        x_test2 = x_train2
        lhm.path_disp(x_test1, x_test2, scale1=[min_c1, max_c1, scale], \
                                scale2=[min_c2, max_c2, scale])
                

    #---------------------------------------------------------------------------   
    elif opt.bProgressDiff:

        nState   = nState_l[task]
        trans_type= "left_right"
        check_dim = 2
        if check_dim == 0 or check_dim == 1: nEmissionDim=1
        else: nEmissionDim=2

        # Get train/test dataset
        aXData1_scaled, min_c1, max_c1 = dm.scaling(true_aXData1)
        aXData2_scaled, min_c2, max_c2 = dm.scaling(true_aXData2)    
        true_labels = [True]*len(true_aXData1)

        state_diff = None
            
        for i in xrange(len(true_labels)):
                
            true_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, true_chunks, true_labels)
            test_dataSet  = true_dataSet[i:i+1]
            train_ids = [val for val in true_dataSet.sa.id if val not in test_dataSet[0].sa.id] 
            train_ids = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
            train_dataSet = true_dataSet[train_ids]

            x_train1 = train_dataSet.samples[:,0,:]
            x_train2 = train_dataSet.samples[:,1,:]

            # Learning
            lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, nEmissionDim=nEmissionDim)

            if check_dim == 0: lhm.fit(x_train1, cov_mult=[cov_mult[task][0]]*4)
            elif check_dim == 1: lhm.fit(x_train2, cov_mult=[cov_mult[task][3]]*4)
            else: lhm.fit(x_train1, x_train2, cov_mult=cov_mult[task])

            x_test1  = test_dataSet.samples[:,0,:]
            x_test2  = test_dataSet.samples[:,1,:]

            off_progress, online_progress = lhm.progress_analysis(x_test1, x_test2, 
                                                                  scale1=[min_c1, max_c1, scale], 
                                                                  scale2=[min_c2, max_c2, scale])
            if state_diff is None:
                state_diff = off_progress-online_progress
            else:
                state_diff = np.vstack([state_diff, off_progress-online_progress])


        mu  = np.mean(state_diff,axis=0)
        sig = np.std(state_diff,axis=0)
        x   = np.arange(0., float(len(mu))) * (1./freq)

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        
        fig = plt.figure()
        plt.rc('text', usetex=True)
        
        ax1 = plt.subplot(111)
        ax1.plot(x, mu, '-g')
        ax1.fill_between(x, mu-sig, mu+sig, facecolor='green', edgecolor='1.0',
                         alpha=0.5, interpolate=True)
        
        ax1.set_ylabel('Estimation Error', fontsize=18)
        ax1.set_xlabel('Time [sec]', fontsize=18)

        mu_line = ax1.plot([], [], color='green', linewidth=2, 
                             label=r'$\mu$') #fake for legend
        bnd_line = ax1.plot([], [], color='green', alpha=0.5, linewidth=10, 
                             label=r'$\mu+\sigma$') #fake for legend
        
        ax1.legend(loc=1,prop={'size':18})       
        plt.show()

        fig.savefig('test.pdf')
        fig.savefig('test.png')
        
        
    #---------------------------------------------------------------------------           
    else:

        nState   = nState_l[task]
        trans_type= "left_right"
        check_dim = 2
        if check_dim == 0 or check_dim == 1: nEmissionDim=1
        else: nEmissionDim=2
        
        if opt.bAbnormal:
            test_title      = 'online_method_comp'
            cross_data_path = os.path.join(cross_root_path, 'multi_sim_'+task_names[task], test_title)
            false_data_flag = True
            true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet \
              = loadData(pkl_file, data_path, task_names[task], f_zero_size[task], f_thres[task], 
                         audio_thres[task], cross_data_path)            
        else:
            false_data_flag = False       
            true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet \
              = loadData(pkl_file, data_path, task_names[task], f_zero_size[task], f_thres[task], 
                         audio_thres[task])
                
        # Get train/test dataset
        aXData1_scaled, min_c1, max_c1 = dm.scaling(true_aXData1)
        aXData2_scaled, min_c2, max_c2 = dm.scaling(true_aXData2)    
        true_labels = [True]*len(true_aXData1)

        true_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, true_chunks, true_labels)
        test_dataSet  = true_dataSet[0:10]
        train_ids = [val for val in true_dataSet.sa.id if val not in test_dataSet[0].sa.id] 
        train_ids = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
        train_dataSet = true_dataSet[train_ids]

        x_train1 = train_dataSet.samples[:,0,:]
        x_train2 = train_dataSet.samples[:,1,:]

        # generate simulated data!!
        false_aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=scale)
        false_aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=scale)    
        false_labels = [False]*len(false_aXData1)
        false_dataSet = dm.create_mvpa_dataset(false_aXData1_scaled, false_aXData2_scaled, \
                                               false_chunks, false_labels)

        # If you want normal likelihood, class 0, data 1
        # testData 0
        # false data 0 (make it false)
        if false_data_flag:
            test_dataSet    = false_dataSet
                                                               
        for K in range(len(test_dataSet)):
            print "Test number : ", K
            
            if false_data_flag:
                x_test1 = np.array([test_dataSet.samples[K:K+1,0][0]])
                x_test2 = np.array([test_dataSet.samples[K:K+1,1][0]])
            else:
                x_test1  = test_dataSet.samples[:,0,:]
                x_test2  = test_dataSet.samples[:,1,:]
            
            # Learning
            lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, nEmissionDim=nEmissionDim)
            
            if check_dim == 0: lhm.fit(x_train1, cov_mult=[cov_mult[task][0]]*4)
            elif check_dim == 1: lhm.fit(x_train2, cov_mult=[cov_mult[task][3]]*4)
            else: lhm.fit(x_train1, x_train2, cov_mult=cov_mult[task], ml_pkl='likelihood.pkl', \
                          use_pkl=opt.bUseMLObspickle)


            ## # TEST
            ## ------------------------------------------------------------------
            ## nCurrentStep = 27
            ## ## X_test1 = aXData1_scaled[0:1,:nCurrentStep]
            ## ## X_test2 = aXData2_scaled[0:1,:nCurrentStep]
            ## X_test1 = aXData1_scaled[0:1]
            ## X_test2 = aXData2_scaled[0:1]
            
            ## #
            ## X_test2[0,nCurrentStep-3] = 10.7
            ## X_test2[0,nCurrentStep-2] = 12.7
            ## X_test2[0,nCurrentStep-1] = 11.7

            ## ------------------------------------------------------------------
            ## aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1)
            ## aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2)    

            ## idx = 0
            ## print "Chunk name: ", false_chunks[idx]
            ## X1 = np.array([aXData1_scaled[idx]])
            ## X2 = np.array([aXData2_scaled[idx]])


            lhm.likelihood_disp(x_test1, x_test2, 2.0, scale1=[min_c1, max_c1, scale], \
                                scale2=[min_c2, max_c2, scale])
            print "-------------------------------------------------------------------"


            ## lhm.data_plot(X_test1, X_test2, color = 'r')

            ## X_test2[0,nCurrentStep-2] = 12.7
            ## X_test2[0,nCurrentStep-1] = 11.7
            ## X_test = lhm.convert_sequence(X_test1[:nCurrentStep], X_test2[:nCurrentStep], emission=False)

            ## fp, err = lhm.anomaly_check(X_test1, X_test2, ths_mult=0.01)
            ## print fp, err
            
            ## ## print lhm.likelihood(X_test), lhm.likelihood_avg
            ## ## mu, cov = self.predict(X_test)

            ## lhm.data_plot(X_test1, X_test2, color = 'b')

    ## #---------------------------------------------------------------------------           
    ## if opt.bRocHuman:
    ##     # not used for a while

    ##     cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/human/multi_'+\
    ##                       task_names[task]
    ##     nState          = 20
    ##     threshold_mult  = np.arange(0.01, 4.0, 0.1)    

    ##     fig_roc(cross_data_path, aXData1, aXData2, chunks, labels, task_names[task], nState, threshold_mult, \
    ##             opr='human', bPlot=opt.bPlot)

    ##     if opt.bAllPlot:
    ##         prefixes = ['microwave', 'microwave_black', 'microwave_white']
    ##         cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/human'
    ##         fig_roc_all(cross_data_path, nState, threshold_mult, prefixes, opr='human', attr='chunks')



## def fig_roc_offline_sim(cross_data_path, \
##                         true_aXData1, true_aXData2, true_chunks, \
##                         false_aXData1, false_aXData2, false_chunks, \
##                         prefix, nState=20, \
##                         threshold_mult = np.arange(0.05, 1.2, 0.05), opr='robot', attr='id', bPlot=False, \
##                         cov_mult=[1.0, 1.0, 1.0, 1.0], renew=False):

##     # For parallel computing
##     strMachine = socket.gethostname()+"_"+str(os.getpid())    
##     trans_type = "left_right"
    
##     # Check the existance of workspace
##     cross_test_path = os.path.join(cross_data_path, str(nState))
##     if os.path.isdir(cross_test_path) == False:
##         os.system('mkdir -p '+cross_test_path)

##     # min max scaling for true data
##     aXData1_scaled, min_c1, max_c1 = dm.scaling(true_aXData1, scale=10.0)
##     aXData2_scaled, min_c2, max_c2 = dm.scaling(true_aXData2, scale=10.0)    
##     labels = [True]*len(true_aXData1)
##     true_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, true_chunks, labels)
##     ## print "Scaling data: ", np.shape(true_aXData1), " => ", np.shape(aXData1_scaled)
    
##     # generate simulated data!!
##     aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
##     aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)    
##     labels = [False]*len(false_aXData1)
##     false_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, false_chunks, labels)

##     # K random training-test set
##     K = len(true_aXData1)/4 # the number of test data
##     M = 30
##     splits = []
##     for i in xrange(M):
##     ## for i in xrange(len(true_aXData1)):
##         print "(",K,",",K,") pairs in ", M, "iterations"
        
##         if os.path.isfile(os.path.join(cross_data_path,"train_dataSet_"+str(i))) is False:

##             ## test_dataSet = true_dataSet[i]
##             ## train_ids = [val for val in true_dataSet.sa.id if val not in test_dataSet.sa.id] 
##             ## train_ids = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
##             ## train_dataSet = true_dataSet[train_ids]
##             ## test_false_dataSet = false_dataSet[K]
            
##             test_dataSet  = Dataset.random_samples(true_dataSet, K)
##             train_ids = [val for val in true_dataSet.sa.id if val not in test_dataSet.sa.id] 
##             train_ids = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
##             train_dataSet = true_dataSet[train_ids]
##             test_false_dataSet  = Dataset.random_samples(false_dataSet, K)        

##             Dataset.save(train_dataSet, os.path.join(cross_data_path,"train_dataSet_"+str(i)) )
##             Dataset.save(test_dataSet, os.path.join(cross_data_path,"test_dataSet_"+str(i)) )
##             Dataset.save(test_false_dataSet, os.path.join(cross_data_path,"test_false_dataSet_"+str(i)) )

##         else:
##             try:
##                 train_dataSet = Dataset.from_hdf5( os.path.join(cross_data_path,"train_dataSet_"+str(i)) )
##                 test_dataSet = Dataset.from_hdf5( os.path.join(cross_data_path,"test_dataSet_"+str(i)) )
##                 test_false_dataSet = Dataset.from_hdf5( os.path.join(cross_data_path,"test_false_dataSet_"+str(i)) )
##             except:
##                 print cross_data_path
##                 print "test_dataSet_"+str(i)
            
##         splits.append([train_dataSet, test_dataSet, test_false_dataSet])

            
##     ## Multi dimension
##     for i in xrange(3):        
##         count = 0
##         for ths in threshold_mult:

##             # save file name
##             res_file = prefix+'_roc_'+opr+'_dim_'+str(i)+'_ths_'+str(ths)+'.pkl'
##             res_file = os.path.join(cross_test_path, res_file)

##             mutex_file_part = 'running_dim_'+str(i)+'_ths_'+str(ths)
##             mutex_file_full = mutex_file_part+'_'+strMachine+'.txt'
##             mutex_file      = os.path.join(cross_test_path, mutex_file_full)

##             if os.path.isfile(res_file): 
##                 count += 1            
##                 continue
##             elif hcu.is_file(cross_test_path, mutex_file_part): continue
##             elif os.path.isfile(mutex_file): continue
##             os.system('touch '+mutex_file)

##             print "---------------------------------"
##             print "Total splits: ", len(splits)

##             ## # temp
##             ## fn_ll = []
##             ## tn_ll = []
##             ## fn_err_ll = []
##             ## tn_err_ll = []
##             ## for j, (l_wdata, l_vdata, l_zdata) in enumerate(splits):
##             ##     fn_ll, tn_ll, fn_err_ll, tn_err_ll = anomaly_check_offline(j, l_wdata, l_vdata, nState, \
##             ##                                                            trans_type, ths, l_zdata, \
##             ##                                                            cov_mult=cov_mult, check_dim=i)
##             ##     print np.mean(fn_ll), np.mean(tn_ll)
##             ## sys.exit()
                                  
##             n_jobs = -1
##             r = Parallel(n_jobs=n_jobs)(delayed(anomaly_check_offline)(j, l_wdata, l_vdata, nState, \
##                                                                        trans_type, ths, l_zdata, \
##                                                                        cov_mult=cov_mult, check_dim=i) \
##                                         for j, (l_wdata, l_vdata, l_zdata) in enumerate(splits))
##             fn_ll, tn_ll, fn_err_ll, tn_err_ll = zip(*r)

##             import operator
##             fn_l = reduce(operator.add, fn_ll)
##             tn_l = reduce(operator.add, tn_ll)
##             fn_err_l = reduce(operator.add, fn_err_ll)
##             tn_err_l = reduce(operator.add, tn_err_ll)

##             d = {}
##             d['fn']  = np.mean(fn_l)
##             d['tp']  = 1.0 - np.mean(fn_l)
##             d['tn']  = np.mean(tn_l)
##             d['fp']  = 1.0 - np.mean(tn_l)

##             if fn_err_l == []:         
##                 d['fn_err'] = 0.0
##             else:
##                 d['fn_err'] = np.mean(fn_err_l)

##             if tn_err_l == []:         
##                 d['tn_err'] = 0.0
##             else:
##                 d['tn_err'] = np.mean(tn_err_l)

##             ut.save_pickle(d,res_file)        
##             os.system('rm '+mutex_file)
##             print "-----------------------------------------------"

##         if count == len(threshold_mult):
##             print "#############################################################################"
##             print "All file exist ", count
##             print "#############################################################################"        

        
##     if count == len(threshold_mult) and bPlot:

##         import itertools
##         colors = itertools.cycle(['g', 'm', 'c', 'k'])
##         shapes = itertools.cycle(['x','v', 'o', '+'])
        
##         fig = pp.figure()
        
##         for i in xrange(3):
##             fp_l = []
##             tp_l = []
##             err_l = []
##             for ths in threshold_mult:
##                 res_file   = prefix+'_roc_'+opr+'_dim_'+str(i)+'_'+'ths_'+str(ths)+'.pkl'
##                 res_file   = os.path.join(cross_test_path, res_file)

##                 d = ut.load_pickle(res_file)
##                 tp  = d['tp'] 
##                 fn  = d['fn'] 
##                 fp  = d['fp'] 
##                 tn  = d['tn'] 
##                 fn_err = d['fn_err']         
##                 tn_err = d['tn_err']         

##                 fp_l.append([fp])
##                 tp_l.append([tp])
##                 err_l.append([fn_err])

##             fp_l  = np.array(fp_l)*100.0
##             tp_l  = np.array(tp_l)*100.0

##             idx_list = sorted(range(len(fp_l)), key=lambda k: fp_l[k])
##             sorted_fp_l = [fp_l[j] for j in idx_list]
##             sorted_tp_l = [tp_l[j] for j in idx_list]
            
##             color = colors.next()
##             shape = shapes.next()

##             if i==0: semantic_label='Force only'
##             elif i==1: semantic_label='Sound only'
##             else: semantic_label='Force and sound'
##             pp.plot(sorted_fp_l, sorted_tp_l, '-'+shape+color, label= semantic_label, mec=color, ms=8, mew=2)



##         ## fp_l = fp_l[:,0]
##         ## tp_l = tp_l[:,0]
        
##         ## from scipy.optimize import curve_fit
##         ## def sigma(e, k ,n, offset): return k*((e+offset)**n)
##         ## param, var = curve_fit(sigma, fp_l, tp_l)
##         ## new_fp_l = np.linspace(fp_l.min(), fp_l.max(), 50)        
##         ## pp.plot(new_fp_l, sigma(new_fp_l, *param))

        
##         pp.xlabel('False positive rate (percentage)', fontsize=16)
##         pp.ylabel('True positive rate (percentage)', fontsize=16)    
##         pp.xlim([-1, 100])
##         pp.ylim([-1, 101])
##         pp.legend(loc=4,prop={'size':16})
        
##         pp.show()

##         fig.savefig('test.pdf')
##         fig.savefig('test.png')
        
##     return


    
## def fig_roc_all(cross_data_path, nState, threshold_mult, prefixes, opr='robot', attr='id'):
        
##     import itertools
##     colors = itertools.cycle(['g', 'm', 'c', 'k'])
##     shapes = itertools.cycle(['x','v', 'o', '+'])
    
##     pp.figure()    
##     ## pp.title("ROC of anomaly detection ")
##     for i, prefix in enumerate(prefixes):

##         cross_test_path = os.path.join(cross_data_path, 'multi_'+prefix, str(nState))
        
##         fp_l = []
##         err_l = []
##         for ths in threshold_mult:
##             res_file   = prefix+'_roc_'+opr+'_'+'ths_'+str(ths)+'.pkl'
##             res_file   = os.path.join(cross_test_path, res_file)

##             d = ut.load_pickle(res_file)
##             fp  = d['fp'] 
##             err = d['err']         

##             fp_l.append([fp])
##             err_l.append([err])

##         fp_l  = np.array(fp_l)*100.0

##         color = colors.next()
##         shape = shapes.next()

##         if i==0:
##             semantic_label='Known mechanism \n class -'+prefix
##         else:
##             semantic_label='Known mechanism \n identity -'+prefix
            
##         pp.plot(fp_l, err_l, '--'+shape+color, label= semantic_label, mec=color, ms=8, mew=2)

##     pp.legend(loc=0,prop={'size':14})

##     pp.xlabel('False positive rate (percentage)')
##     pp.ylabel('Mean excess log likelihood')    
##     pp.show()
        
