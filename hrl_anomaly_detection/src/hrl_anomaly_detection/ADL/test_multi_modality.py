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
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import data_manager as dm
import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
from hrl_anomaly_detection.HMM.learning_hmm_multi import learning_hmm_multi


def fig_roc_offline_sim(cross_data_path, \
                        true_aXData1, true_aXData2, true_chunks, \
                        false_aXData1, false_aXData2, false_chunks, \
                        prefix, nState=20, \
                        threshold_mult = np.arange(0.05, 1.2, 0.05), opr='robot', attr='id', bPlot=False, \
                        cov_mult=[1.0, 1.0, 1.0, 1.0], renew=False):

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
    ## print "Scaling data: ", np.shape(true_aXData1), " => ", np.shape(aXData1_scaled)
    
    # generate simulated data!!
    aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
    aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)    
    labels = [False]*len(false_aXData1)
    false_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, false_chunks, labels)

    # K random training-test set
    K = len(true_aXData1)/4
    M = 30
    splits = []
    for i in xrange(M):
    ## for i in xrange(len(true_aXData1)):
        print "(",K,",",K,") pairs in ", M, "iterations"
        
        if os.path.isfile(os.path.join(cross_data_path,"train_dataSet_"+str(i))) is False:

            ## test_dataSet = true_dataSet[i]
            ## train_ids = [val for val in true_dataSet.sa.id if val not in test_dataSet.sa.id] 
            ## train_ids = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
            ## train_dataSet = true_dataSet[train_ids]
            ## test_false_dataSet = false_dataSet[K]
            
            test_dataSet  = Dataset.random_samples(true_dataSet, K)
            train_ids = [val for val in true_dataSet.sa.id if val not in test_dataSet.sa.id] 
            train_ids = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
            train_dataSet = true_dataSet[train_ids]
            test_false_dataSet  = Dataset.random_samples(false_dataSet, K)        

            Dataset.save(train_dataSet, os.path.join(cross_data_path,"train_dataSet_"+str(i)) )
            Dataset.save(test_dataSet, os.path.join(cross_data_path,"test_dataSet_"+str(i)) )
            Dataset.save(test_false_dataSet, os.path.join(cross_data_path,"test_false_dataSet_"+str(i)) )

        else:

            train_dataSet = Dataset.from_hdf5( os.path.join(cross_data_path,"train_dataSet_"+str(i)) )
            test_dataSet = Dataset.from_hdf5( os.path.join(cross_data_path,"test_dataSet_"+str(i)) )
            test_false_dataSet = Dataset.from_hdf5( os.path.join(cross_data_path,"test_false_dataSet_"+str(i)) )
            
        splits.append([train_dataSet, test_dataSet, test_false_dataSet])

            
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
                                  
            n_jobs = 4
            r = Parallel(n_jobs=n_jobs)(delayed(anomaly_check_offline)(j, l_wdata, l_vdata, nState, \
                                                                       trans_type, ths, l_zdata, \
                                                                       cov_mult=cov_mult, check_dim=i) \
                                        for j, (l_wdata, l_vdata, l_zdata) in enumerate(splits))
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
        
        pp.figure()
        
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

            idx_list = sorted(range(len(fp_l)), key=lambda k: fp_l[k])
            sorted_fp_l = [fp_l[j] for j in idx_list]
            sorted_tp_l = [tp_l[j] for j in idx_list]
            
            color = colors.next()
            shape = shapes.next()

            if i==0: semantic_label='Force only'
            elif i==1: semantic_label='Sound only'
            else: semantic_label='Force and sound'
            pp.plot(sorted_fp_l, sorted_tp_l, '-'+shape+color, label= semantic_label, mec=color, ms=8, mew=2)



        ## fp_l = fp_l[:,0]
        ## tp_l = tp_l[:,0]
        
        ## from scipy.optimize import curve_fit
        ## def sigma(e, k ,n, offset): return k*((e+offset)**n)
        ## param, var = curve_fit(sigma, fp_l, tp_l)
        ## new_fp_l = np.linspace(fp_l.min(), fp_l.max(), 50)        
        ## pp.plot(new_fp_l, sigma(new_fp_l, *param))

        
        pp.xlabel('False positive rate (percentage)', fontsize=16)
        pp.ylabel('True positive rate (percentage)', fontsize=16)    
        ## pp.xlim([0, 70])
        pp.ylim([0, 101])
        pp.legend(loc=4,prop={'size':16})
        
        pp.show()
                            
    return


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
                                  
            n_jobs = 4
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
        
        pp.figure()
        
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

def fig_roc(cross_data_path, aXData1, aXData2, chunks, labels, prefix, nState=20, \
            threshold_mult = np.arange(0.05, 1.2, 0.05), opr='robot', attr='id', bPlot=False):

    # For parallel computing
    strMachine = socket.gethostname()+"_"+str(os.getpid())    
    trans_type = "left_right"
    
    # Check the existance of workspace
    cross_test_path = os.path.join(cross_data_path, str(nState))
    if os.path.isdir(cross_test_path) == False:
        os.system('mkdir -p '+cross_test_path)

    # min max scaling
    aXData1_scaled, min_c1, max_c1 = dm.scaling(aXData1)
    aXData2_scaled, min_c2, max_c2 = dm.scaling(aXData2)    
    dataSet    = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, chunks, labels)

    # Cross validation   
    nfs    = NFoldPartitioner(cvtype=1,attr=attr) # 1-fold ?
    spl    = splitters.Splitter(attr='partitions')
    splits = [list(spl.generate(x)) for x in nfs.generate(dataSet)] # split by chunk

    count = 0
    for ths in threshold_mult:
    
        # save file name
        res_file = prefix+'_roc_'+opr+'_'+'ths_'+str(ths)+'.pkl'
        res_file = os.path.join(cross_test_path, res_file)
        
        mutex_file_part = 'running_ths_'+str(ths)
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

        n_jobs = 4
        r = Parallel(n_jobs=n_jobs)(delayed(anomaly_check)(i, l_wdata, l_vdata, nState, trans_type, ths) \
                                    for i, (l_wdata, l_vdata) in enumerate(splits)) 
        fp_ll, err_ll = zip(*r)

        
        import operator
        fp_l = reduce(operator.add, fp_ll)
        err_l = reduce(operator.add, err_ll)
        
        d = {}
        d['fp']  = np.mean(fp_l)
        if err_l == []:         
            d['err'] = 0.0
        else:
            d['err'] = np.mean(err_l)

        ut.save_pickle(d,res_file)        
        os.system('rm '+mutex_file)
        print "-----------------------------------------------"

    if count == len(threshold_mult):
        print "#############################################################################"
        print "All file exist ", count
        print "#############################################################################"        

        
    if count == len(threshold_mult) and bPlot:

        fp_l = []
        err_l = []
        for ths in threshold_mult:
            res_file   = prefix+'_roc_'+opr+'_'+'ths_'+str(ths)+'.pkl'
            res_file   = os.path.join(cross_test_path, res_file)

            d = ut.load_pickle(res_file)
            fp  = d['fp'] 
            err = d['err']         

            fp_l.append([fp])
            err_l.append([err])

        fp_l  = np.array(fp_l)*100.0
        sem_c = 'b'
        sem_m = '+'
        semantic_label='likelihood detection \n with known mechanism class'
        pp.figure()
        pp.plot(fp_l, err_l, '--'+sem_m+sem_c, label= semantic_label, mec=sem_c, ms=8, mew=2)
        pp.xlabel('False positive rate (percentage)')
        pp.ylabel('Mean excess log likelihood')    
        ## pp.xlim([0, 30])
        pp.show()
                            
    return

    
def anomaly_check_offline(i, l_wdata, l_vdata, nState, trans_type, ths, false_dataSet=None, 
                          cov_mult=[1.0, 1.0, 1.0, 1.0], check_dim=2):

    # Cross validation
    if check_dim is not 2:
        x_train1 = l_wdata.samples[:,check_dim,:]

        lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, nEmissionDim=1)
        if check_dim==0: lhm.fit(x_train1, cov_mult=[cov_mult[0]]*4)
        elif check_dim==1: lhm.fit(x_train1, cov_mult=[cov_mult[3]]*4)
    else:
        x_train1 = l_wdata.samples[:,0,:]
        x_train2 = l_wdata.samples[:,1,:]

        lhm = learning_hmm_multi(nState=nState, trans_type=trans_type)
        lhm.fit(x_train1, x_train2, cov_mult=cov_mult)
       
    fn_l  = []
    tn_l  = []
    fn_err_l = []
    tn_err_l = []

    # True data
    if check_dim == 2:
        x_test1 = l_vdata.samples[:,0]
        x_test2 = l_vdata.samples[:,1]
    else:
        x_test1 = l_vdata.samples[:,check_dim]

    n,_ = np.shape(x_test1)
    for i in range(n):
        if check_dim == 2:
            fn, err = lhm.anomaly_check(x_test1[i:i+1], x_test2[i:i+1], ths_mult=ths)           
        else:
            fn, err = lhm.anomaly_check(x_test1[i:i+1], ths_mult=ths)           

        fn_l.append(fn)
        if err != 0.0: fn_err_l.append(err)

    # False data
    if check_dim == 2:
        x_test1 = false_dataSet.samples[:,0]
        x_test2 = false_dataSet.samples[:,1]
    else:
        x_test1 = false_dataSet.samples[:,check_dim]
        
    n = len(x_test1)
    for i in range(n):
        if check_dim == 2:            
            tn, err = lhm.anomaly_check(np.array([x_test1[i]]), np.array([x_test2[i]]), ths_mult=ths)           
        else:
            tn, err = lhm.anomaly_check(np.array([x_test1[i]]), ths_mult=ths)           
            
        tn_l.append(tn)
        if err != 0.0: tn_err_l.append(err)

    return fn_l, tn_l, fn_err_l, tn_err_l
    

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
    

def fig_roc_all(cross_data_path, nState, threshold_mult, prefixes, opr='robot', attr='id'):
        
    import itertools
    colors = itertools.cycle(['g', 'm', 'c', 'k'])
    shapes = itertools.cycle(['x','v', 'o', '+'])
    
    pp.figure()    
    ## pp.title("ROC of anomaly detection ")
    for i, prefix in enumerate(prefixes):

        cross_test_path = os.path.join(cross_data_path, 'multi_'+prefix, str(nState))
        
        fp_l = []
        err_l = []
        for ths in threshold_mult:
            res_file   = prefix+'_roc_'+opr+'_'+'ths_'+str(ths)+'.pkl'
            res_file   = os.path.join(cross_test_path, res_file)

            d = ut.load_pickle(res_file)
            fp  = d['fp'] 
            err = d['err']         

            fp_l.append([fp])
            err_l.append([err])

        fp_l  = np.array(fp_l)*100.0

        color = colors.next()
        shape = shapes.next()

        if i==0:
            semantic_label='Known mechanism \n class -'+prefix
        else:
            semantic_label='Known mechanism \n identity -'+prefix
            
        pp.plot(fp_l, err_l, '--'+shape+color, label= semantic_label, mec=color, ms=8, mew=2)

    pp.legend(loc=0,prop={'size':14})

    pp.xlabel('False positive rate (percentage)')
    pp.ylabel('Mean excess log likelihood')    
    pp.show()
        
    

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
    ## ax = pp.subplot(412)
    ## mfcc_feat = librosa.feature.mfcc(data_seq, n_mfcc=4, sr=rate, n_fft=1024)
    ## ## mfcc_feat = feature.mfcc(data_list, sr=rate, n_mfcc=13, n_fft=1024, )
    ## pp.imshow(mfcc_feat, origin='down')
    ## ## ax.set_xlim([0, t[-1]*100])

    ## ax = pp.subplot(413)
    ## S = feature.melspectrogram(data_list, sr=rate, n_fft=1024, hop_length=1, n_mels=128)
    ## log_S = librosa.logamplitude(S, ref_power=np.max)        
    ## ## mfcc_feat = librosa.feature.mfcc(S=log_S, n_mfcc=20, sr=rate, n_fft=1024)
    ## mfcc_feat = librosa.feature.delta(mfcc_feat)
    ## pp.imshow(mfcc_feat, origin='down')

    ## ax = pp.subplot(414)
    ## mfcc_feat = librosa.feature.delta(mfcc_feat, order=2)
    ## pp.imshow(mfcc_feat, origin='down')

    
    
    
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

def plot_all(data1, data2, false_data1=None, false_data2=None, labels=None):

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

    pp.figure()
    plt.rc('text', usetex=True)
    ax1 = pp.subplot(211)
    for i, d in enumerate(data1):
        ## if i==22 or i==27 or i==17: continue
        ## if i<11: continue
        if labels is not None and labels[i] == False:
            pp.plot(d, label=str(i), color='k', linewidth=2.0)
        else:
            pp.plot(d, label=str(i))

    # for false data
    if false_data1 is not None:
        for i, d in enumerate(false_data1):
            pp.plot(d, label=str(i), color='k', linewidth=2.0)
            
    ## ax1.set_title("Force")
    ax1.set_ylabel("Force [L2]", fontsize=18)

        
    ax2 = pp.subplot(212)
    for i, d in enumerate(data2):
        if labels is not None and labels[i] == False:
            pp.plot(d, color='k', linewidth=2.0)
        else:
            pp.plot(d)

    # for false data
    if false_data2 is not None:
        for i, d in enumerate(false_data2):
            pp.plot(d, color='k', linewidth=2.0)
            
    ## ax2.set_title("Audio")
    ax2.set_ylabel("Audio [RMS]", fontsize=18)
    ax2.set_xlabel("Time step [43Hz]", fontsize=18)
    
    #ax1.legend()
    pp.show()
    
    


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--abnormal', '--an', action='store_true', dest='bAbnormal',
                 default=False, help='Renew pickle files.')
    p.add_option('--simulated_abnormal', '--sim_an', action='store_true', dest='bSimAbnormal',
                 default=False, help='.')
    p.add_option('--animation', '--ani', action='store_true', dest='bAnimation',
                 default=False, help='Plot by time using animation')
    p.add_option('--roc_human', '--rh', action='store_true', dest='bRocHuman',
                 default=False, help='Plot by a figure of ROC human')
    p.add_option('--roc_offline_simulated_robot', '--roffsim', action='store_true', dest='bRocOfflineSimRobot',
                 default=False, help='Plot by a figure of ROC robot')    
    p.add_option('--roc_online_robot', '--ron', action='store_true', dest='bRocOnlineRobot',
                 default=False, help='Plot by a figure of ROC robot')
    p.add_option('--roc_offline_robot', '--roff', action='store_true', dest='bRocOfflineRobot',
                 default=False, help='Plot by a figure of ROC robot')
    p.add_option('--all_plot', '--all', action='store_true', dest='bAllPlot',
                 default=False, help='Plot all data')
    p.add_option('--plot', '--p', action='store_true', dest='bPlot',
                 default=False, help='Plot')
    opt, args = p.parse_args()


    ## data_path = os.environ['HRLBASEPATH']+'/src/projects/anomaly/test_data/'
    cross_root_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/robot'
    
    class_num = 2
    task  = 0
    if class_num == 0:
        class_name = 'door'
        task_names = ['microwave_black', 'microwave_white', 'lab_cabinet']
        f_zero_size = [8, 5, 10]
        f_thres     = [1.0, 1.7, 3.0]
        audio_thres = [1.0, 1.0, 1.0]
        cov_mult = [[1.0, 1.5, 1.5, 1.5],[1.0, 1.0, 1.0, 1.0],[1.5, 5.5, 5.5, 5.5]]
        nState_l    = [20, 20, 20]
    elif class_num == 1: 
        class_name = 'switch'
        task_names = ['wallsw', 'switch_device', 'switch_outlet']
        f_zero_size = [3, 18, 10]
        f_thres     = [0.5, 0.5, 0.5]
        audio_thres = [1.0, 0.7, 1.0]
        cov_mult = [[1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0]]
        nState_l    = [20, 20, 20]
    elif class_num == 2:        
        class_name = 'lock'
        task_names = ['case', 'lock_wipes', 'lock_huggies']
        f_zero_size = [5, 5, 5]
        f_thres     = [1.0, 1.35, 1.35]
        audio_thres = [1.0, 1.0, 1.0]
        cov_mult = [[1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0]]
        nState_l    = [20, 20, 20]
    elif class_num == 3:        
        class_name = 'complex'
        task_names = ['toaster_white', 'glass_case']
        f_zero_size = [5, 3, 8]
        f_thres     = [1.0, 1.5, 1.35]
        audio_thres = [1.0, 1.0, 1.0]
        cov_mult    = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        nState_l    = [20, 10, 20]
    elif class_num == 4:        
        class_name = 'button'
        task_names = ['joystick', 'keyboard']
        f_zero_size = [5, 5, 8]
        f_thres     = [1.35, 1.35, 1.35]
        audio_thres = [1.0, 1.0, 1.0]
        cov_mult    = [[1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0],[1.0, 1.0, 1.0, 1.0]]
        nState_l    = [20, 20, 20]
    else:
        print "Please specify right task."
        sys.exit()

    dtw_flag = False
    
    # Load data
    pkl_file  = os.path.join(cross_root_path,task_names[task]+"_data.pkl")    
    data_path = os.environ['HRLBASEPATH']+'/src/projects/anomaly/test_data/robot_20150213/'+class_name+'/'
    
    if os.path.isfile(pkl_file) and opt.bRenew is False:
        d = ut.load_pickle(pkl_file)
    else:
        ## d = dm.load_data(data_path, task_names[task], normal_only=(not opt.bAbnormal))
        ## d = dm.cutting(d, dtw_flag=dtw_flag)        
        d = dm.load_data(data_path, task_names[task], normal_only=False)
        d = dm.cutting_for_robot(d, f_zero_size=f_zero_size[task], f_thres=f_thres[task], \
                                 audio_thres=audio_thres[task], dtw_flag=dtw_flag)        
        ut.save_pickle(d, pkl_file)
        
    #
    aXData1  = d['ft_force_mag_l']
    aXData2  = d['audio_rms_l'] 
    labels   = d['labels']
    ## chunks   = d['chunks'] 

    true_aXData1 = d['ft_force_mag_true_l']
    true_aXData2 = d['audio_rms_true_l'] 
    true_chunks  = d['true_chunks']

    # Load simulated anomaly
    if opt.bSimAbnormal or opt.bRocOfflineSimRobot:
        pkl_file = os.path.join(cross_root_path,task_names[task]+"_sim_an_data.pkl")
        if os.path.isfile(pkl_file) and opt.bRenew is False:
            dd = ut.load_pickle(pkl_file)
        else:
            n_false_data = 100
            dd = dm.generate_sim_anomaly(true_aXData1, true_aXData2, n_false_data)
            ut.save_pickle(dd, pkl_file)

        false_aXData1 = dd['ft_force_mag_sim_false_l']
        false_aXData2 = dd['audio_rms_sim_false_l'] 
        false_chunks  = dd['sim_false_chunks']
    else:
        false_aXData1 = d['ft_force_mag_false_l']
        false_aXData2 = d['audio_rms_false_l'] 
        false_chunks  = d['false_chunks']

    print "All: ", len(true_aXData1)+len(false_aXData1), \
      " Success: ", len(true_aXData1), \
      " Failure: ", len(false_aXData1)
      
    #---------------------------------------------------------------------------           
    if opt.bRocHuman:

        cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/human/multi_'+\
                          task_names[task]
        nState          = 20
        threshold_mult  = np.arange(0.01, 4.0, 0.1)    

        fig_roc(cross_data_path, aXData1, aXData2, chunks, labels, task_names[task], nState, threshold_mult, \
                opr='human', bPlot=opt.bPlot)

        if opt.bAllPlot:
            prefixes = ['microwave', 'microwave_black', 'microwave_white']
            cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/human'
            fig_roc_all(cross_data_path, nState, threshold_mult, prefixes, opr='human', attr='chunks')


    #---------------------------------------------------------------------------           
    elif opt.bRocOfflineSimRobot:
        
        print "ROC Offline Robot with simulated anomalies"
        cross_data_path = os.path.join(cross_root_path, 'multi_sim_'+task_names[task])
        nState          = nState_l[task]
        threshold_mult  = np.logspace(0.1, 2.0, 30, endpoint=True) - 5.0 #np.arange(0.0, 25.001, 0.5)    
        ## threshold_mult  = np.logspace(0.1, 2.0, 30, endpoint=True) - 1.0 #np.arange(0.0, 25.001, 0.5)    
        attr            = 'id'

        fig_roc_offline_sim(cross_data_path, \
                            true_aXData1, true_aXData2, true_chunks, \
                            false_aXData1, false_aXData2, false_chunks, \
                            task_names[task], nState, threshold_mult, \
                            opr='robot', attr='id', bPlot=opt.bPlot, renew=False)

            
    #---------------------------------------------------------------------------           
    elif opt.bRocOnlineRobot:

        cross_data_path = os.path.join(cross_root_path, 'multi_'+task_names[task])
        nState          = nState_l[task]
        threshold_mult  = np.arange(0.0, 4.2, 0.1)    
        attr            = 'id'

        fig_roc(cross_data_path, aXData1, aXData2, chunks, labels, task_names[task], nState, threshold_mult, \
                opr='robot', attr='id', bPlot=opt.bPlot, cov_mult=cov_mult[task])

        if opt.bAllPlot:
            if task ==1:
                ## prefixes = ['microwave', 'microwave_black', 'microwave_white']
                prefixes = ['microwave', 'microwave_black']
            else:
                prefixes = ['microwave', 'microwave_black']
                
            cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/robot'                
            fig_roc_all(cross_data_path, nState, threshold_mult, prefixes, opr='robot', attr='id')
            
            
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
    elif opt.bAllPlot:

        true_aXData1_scaled, min_c1, max_c1 = dm.scaling(true_aXData1, scale=10.0)
        true_aXData2_scaled, min_c2, max_c2 = dm.scaling(true_aXData2, scale=10.0)    

        if opt.bAbnormal or opt.bSimAbnormal:
            # min max scaling
            false_aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
            false_aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)
                        
            ## plot_all(true_aXData1_scaled, true_aXData2_scaled, false_aXData1_scaled, false_aXData2_scaled)
            plot_all(true_aXData1, true_aXData2, false_aXData1, false_aXData2)
            
        else:
            ## plot_all(true_aXData1_scaled, true_aXData2_scaled)            
            plot_all(true_aXData1, true_aXData2)            

        ## print min_c1, max_c1, np.min(aXData1_scaled), np.max(aXData1_scaled)
        ## print min_c2, max_c2, np.min(aXData2_scaled), np.max(aXData2_scaled)
       

            
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
    else:

        nState   = nState_l[task]
        trans_type= "left_right"
        check_dim = 2
        if check_dim == 0 or check_dim == 1: nEmissionDim=1
        else: nEmissionDim=2

        aXData1_scaled, min_c1, max_c1 = dm.scaling(true_aXData1)
        aXData2_scaled, min_c2, max_c2 = dm.scaling(true_aXData2)    
        true_labels = [True]*len(true_aXData1)

        # generate simulated data!!
        false_aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
        false_aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)    
        false_labels = [False]*len(false_aXData1)
        false_dataSet = dm.create_mvpa_dataset(false_aXData1_scaled, false_aXData2_scaled, \
                                               false_chunks, false_labels)
            

        true_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, true_chunks, true_labels)
        test_dataSet  = true_dataSet[0]
        train_ids = [val for val in true_dataSet.sa.id if val not in test_dataSet.sa.id] 
        train_ids = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
        train_dataSet = true_dataSet[train_ids]

        x_train1 = train_dataSet.samples[:,0,:]
        x_train2 = train_dataSet.samples[:,1,:]
        x_test1  = test_dataSet.samples[:,0,:]
        x_test2  = test_dataSet.samples[:,1,:]

        
        for K in range(len(false_labels)):
                
            test_dataSet  = false_dataSet[K]
            x_test1 = np.array([test_dataSet.samples[:,0][0]])
            x_test2 = np.array([test_dataSet.samples[:,1][0]])
                        
            print false_chunks[K]

            
            # Learning
            lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, nEmissionDim=nEmissionDim)
            
            if check_dim == 0: lhm.fit(x_train1, cov_mult=[cov_mult[task][0]]*4)
            elif check_dim == 1: lhm.fit(x_train2, cov_mult=[cov_mult[task][3]]*4)
            else: lhm.fit(x_train1, x_train2, cov_mult=cov_mult[task])


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


            lhm.likelihood_disp(x_test1, x_test2, 15.0)



            ## lhm.data_plot(X_test1, X_test2, color = 'r')

            ## X_test2[0,nCurrentStep-2] = 12.7
            ## X_test2[0,nCurrentStep-1] = 11.7
            ## X_test = lhm.convert_sequence(X_test1[:nCurrentStep], X_test2[:nCurrentStep], emission=False)

            ## fp, err = lhm.anomaly_check(X_test1, X_test2, ths_mult=0.01)
            ## print fp, err
            
            ## ## print lhm.likelihood(X_test), lhm.likelihood_avg
            ## ## mu, cov = self.predict(X_test)

            ## lhm.data_plot(X_test1, X_test2, color = 'b')

