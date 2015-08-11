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

# Util
import hrl_lib.util as ut
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl
from hrl_anomaly_detection.HMM.anomaly_checker import anomaly_checker

def load_data(data_path, prefix, normal_only=True):

    pkl_list = glob.glob(data_path+'*_'+prefix+'*.pkl')
    ## pkl_list = glob.glob(data_path+'*.pkl')

    ft_time_list   = []
    ft_force_list  = []
    ft_torque_list = []

    audio_time_list = []
    audio_data_list = []
    audio_amp_list = []
    audio_freq_list = []
    audio_chunk_list = []

    label_list = []
    name_list = []

    count = -1
    for i, pkl in enumerate(pkl_list):
                
        bNormal = True
        if pkl.find('success') < 0: bNormal = False
        if normal_only and bNormal is False: continue

        ## if bNormal is False:
        ##     if pkl.find('gatsbii_glass_case_robot_stickblock_1') < 0: continue
        
        ## if bNormal: count += 1        
        ## if bNormal and (count==17 or count == 22 or count == 27):                 
        ##     print "aaaaaa ", pkl
        ##     continue

        d = ut.load_pickle(pkl)

        ft_time   = d.get('ft_time',None)
        ft_force  = d.get('ft_force_raw',None)
        ft_torque = d.get('ft_torque_raw',None)

        if len(ft_force) == 0: 
            print "No FT data!!!"
            sys.exit()
        
        audio_time  = d['audio_time']
        audio_data  = d['audio_data']
        audio_amp   = d['audio_amp']
        audio_freq  = d['audio_freq']
        audio_chunk = d['audio_chunk']

        ft_force = np.array(ft_force).squeeze().T
        ft_torque = np.array(ft_torque).squeeze().T
        
        ft_time_list.append(ft_time)
        ft_force_list.append(ft_force)
        ft_torque_list.append(ft_torque)

        audio_time_list.append(audio_time)
        audio_data_list.append(audio_data)
        audio_amp_list.append(audio_amp)
        audio_freq_list.append(audio_freq)
        audio_chunk_list.append(audio_chunk)

        label_list.append(bNormal)

        head, tail = os.path.split(pkl)

        ## name = tail.split('_')[0] + '_' + tail.split('_')[1] + '_' + tail.split('_')[2]
        name = tail.split('.')[0] 
        name_list.append(name)


    d = {}
    d['ft_time']       = ft_time_list
    d['ft_force_raw']  = ft_force_list
    d['ft_torque_raw'] = ft_torque_list

    d['audio_time']  = audio_time_list
    d['audio_data']  = audio_data_list
    d['audio_amp']   = audio_amp_list
    d['audio_freq']  = audio_freq_list
    d['audio_chunk'] = audio_chunk_list    

    d['labels'] = label_list
    d['names'] = name_list
        
    return d


def cutting(d, dtw_flag=False):

    labels      = d['labels']
    names       = d['names']
    ft_time_l   = d['ft_time']
    ft_force_l  = d['ft_force_raw']
    ft_torque_l = d['ft_torque_raw']

    audio_time_l = d['audio_time']
    audio_data_l = d['audio_data']
    audio_amp_l  = d['audio_amp']
    audio_freq_l = d['audio_freq']
    audio_chunk_l= d['audio_chunk']

    ft_time_list     = []
    ft_force_list    = []
    ft_torque_list   = []
    ft_force_mag_list= []
    audio_time_list  = []
    audio_data_list  = []
    audio_amp_list   = []
    audio_freq_list  = []
    audio_chunk_list = []
    audio_rms_list   = []

    label_list = []
    name_list  = []

    MAX_INT = 32768.0
    CHUNK   = 1024 #frame per buffer
    RATE    = 44100 #sampling rate

    #------------------------------------------------------
    # Get reference data
    #------------------------------------------------------
    # Ref ID    
    max_f   = 0.0
    max_idx = 0
    for i, force in enumerate(ft_force_l):
        if labels[i] is False: continue
        else: 
            ft_force_mag = np.linalg.norm(force,axis=0)
            f = np.max(ft_force_mag)
            if max_f < f:
                ref_idx = i
                max_f = f

    # Ref force and audio data
    ft_time   = ft_time_l[ref_idx]
    ft_force  = ft_force_l[ref_idx]
    ft_torque = ft_torque_l[ref_idx]        
    ft_force_mag = np.linalg.norm(ft_force,axis=0)
    audio_time = audio_time_l[ref_idx]
    audio_data = audio_data_l[ref_idx]    

    # Force cut
    nZero = 5
    ft_zero = np.mean(ft_force_mag[:nZero]) * 1.5

    idx_start = None
    idx_end   = None        
    for j in xrange(len(ft_force_mag)-nZero):
        avg = np.mean(ft_force_mag[j:j+nZero])

        if avg > ft_zero and idx_start is None:
            idx_start = j #i+nZero
        if idx_start is not None:
            if avg < ft_zero and idx_end is None:
                idx_end = j+nZero

    if idx_end is None: idx_end = len(ft_force_mag)-nZero
    idx_length = idx_end - idx_start

    ft_time_cut      = np.array(ft_time[idx_start:idx_end])
    ft_force_cut     = ft_force[:,idx_start:idx_end]
    ft_torque_cut    = ft_torque[:,idx_start:idx_end]
    ft_force_mag_cut = ft_force_mag[idx_start:idx_end]
    
    # Audio cut
    start_time = ft_time[idx_start]
    end_time   = ft_time[idx_end]

    a_idx_start = None
    a_idx_end   = None                
    for j, t in enumerate(audio_time):
        if t > start_time and a_idx_start is None:
            if (audio_time[j+1] - audio_time[j]) >  float(CHUNK)/float(RATE) :
                a_idx_start = j
        if t > end_time and a_idx_end is None:
            a_idx_end = j
            
    idx_length = a_idx_end - a_idx_start
            
    audio_time_cut = np.array(audio_time[a_idx_start:a_idx_end])
    audio_data_cut = np.array(audio_data[a_idx_start:a_idx_end])

    # normalized rms
    audio_rms_ref = np.zeros(len(audio_time_cut))
    for j, data in enumerate(audio_data_cut):
        audio_rms_ref[j] = get_rms(data, MAX_INT)

    x   = np.linspace(0.0, 1.0, len(ft_time_cut))
    tck = interpolate.splrep(x, ft_force_mag_cut, s=0)

    xnew = np.linspace(0.0, 1.0, len(audio_rms_ref))
    ft_force_mag_ref = interpolate.splev(xnew, tck, der=0)

    print "==================================="
    print ft_force_mag_ref.shape,audio_rms_ref.shape 
    print "==================================="

    
    # DTW wrt the reference
    for i, force in enumerate(ft_force_l):

        if ref_idx == i:
            print "its reference"
            label_list.append(labels[ref_idx])
            name_list.append(names[ref_idx])            
            ft_force_mag_list.append(ft_force_mag_ref)                
            audio_rms_list.append(audio_rms_ref)
            continue

        ft_time   = ft_time_l[i]
        ft_force  = ft_force_l[i]
        ft_torque = ft_torque_l[i]        
        ft_force_mag = np.linalg.norm(ft_force,axis=0)

        start_time = ft_time[0]
        end_time   = ft_time[-1]
        ft_time_cut      = np.array(ft_time)
        ft_force_cut     = ft_force
        ft_torque_cut    = ft_torque
        ft_force_mag_cut = np.linalg.norm(ft_force_cut, axis=0)            
        
        audio_time = audio_time_l[i]
        audio_data = audio_data_l[i]
        a_idx_start = None
        a_idx_end   = None                
        for j, t in enumerate(audio_time):
            if t > start_time and a_idx_start is None:
                if (audio_time[j+1] - audio_time[j]) >  float(CHUNK)/float(RATE) :
                    a_idx_start = j
            if t > end_time and a_idx_end is None:
                a_idx_end = j
                break

        if dtw_flag == True:
            audio_time_cut = np.array(audio_time[a_idx_start:a_idx_end+1])
            audio_data_cut = np.array(audio_data[a_idx_start:a_idx_end+1])              
        else:
            audio_time_cut = np.array(audio_time[a_idx_start:a_idx_start+idx_length])
            audio_data_cut = np.array(audio_data[a_idx_start:a_idx_start+idx_length])              
        
        # normalized rms
        audio_rms_cut = np.zeros(len(audio_time_cut))
        for j, data in enumerate(audio_data_cut):
            audio_rms_cut[j] = get_rms(data, MAX_INT)


        ## print ft_time_cut.shape, ft_force_mag_cut.shape
        ## print len(audio_time_cut), audio_rms_cut.shape

        x   = np.linspace(0.0, 1.0, len(ft_time_cut))
        tck = interpolate.splrep(x, ft_force_mag_cut, s=0)

        xnew = np.linspace(0.0, 1.0, len(audio_rms_cut))
        ft_force_mag_cut = interpolate.splev(xnew, tck, der=0)

        

        ## pp.figure(1)
        ## ax = pp.subplot(311)
        ## pp.plot(ft_force_mag_cut)
        ## ## pp.plot(ft_time_cut, ft_force_mag_cut)
        ## ## ax.set_xlim([0, 6.0])
        ## ax = pp.subplot(312)
        ## pp.plot(audio_rms_cut)
        ## ## pp.plot(audio_time_cut, audio_rms_cut)
        ## ## ax.set_xlim([0, 6.0])
        ## ax = pp.subplot(313)
        ## pp.plot(audio_time_cut,'r')
        ## pp.plot(ft_time_cut,'b')
        ## ## pp.plot(audio_rms_cut)
        ## ## ax.set_xlim([0, 6.0])
        ## pp.show()
        

        #-------------------------------------------------------------
        if dtw_flag:
            from test_dtw2 import Dtw

            ref_seq    = np.vstack([ft_force_mag_ref, audio_rms_ref])
            tgt_seq = np.vstack([ft_force_mag_cut, audio_rms_cut])

            dtw = Dtw(ref_seq.T, tgt_seq.T, distance_weights=[1.0, 1.0])
            ## dtw = Dtw(ref_seq.T, tgt_seq.T, distance_weights=[1.0, 10000.0])
            ## dtw = Dtw(ref_seq.T, tgt_seq.T, distance_weights=[1.0, 10000000.0])

            dtw.calculate()
            path = dtw.get_path()
            path = np.array(path).T

            #-------------------------------------------------------------        
            ## dist, cost, path_1d = mlpy.dtw_std(ft_force_mag_ref, ft_force_mag_cut, dist_only=False)        
            ## fig = plt.figure(1)
            ## ax = fig.add_subplot(111)
            ## plot1 = plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
            ## plot2 = plt.plot(path_1d[0], path_1d[1], 'w')
            ## plot2 = plt.plot(path[0], path[1], 'r')
            ## xlim = ax.set_xlim((-0.5, cost.shape[0]-0.5))
            ## ylim = ax.set_ylim((-0.5, cost.shape[1]-0.5))
            ## plt.show()
            ## sys.exit()
            #-------------------------------------------------------------        

            ft_force_mag_cut_dtw = []        
            audio_rms_cut_dtw    = []        
            new_idx = []
            for idx in xrange(len(path[0])-1):
                if path[0][idx] == path[0][idx+1]: continue

                new_idx.append(path[1][idx])
                ft_force_mag_cut_dtw.append(ft_force_mag_cut[path[1][idx]])
                audio_rms_cut_dtw.append(audio_rms_cut[path[1][idx]])
            ft_force_mag_cut_dtw.append(ft_force_mag_cut[path[1][-1]])
            audio_rms_cut_dtw.append(audio_rms_cut[path[1][-1]])


            print "==================================="
            print names[i], len(ft_force_mag_cut_dtw), len(audio_rms_cut_dtw)
            print "==================================="

            label_list.append(labels[i])
            name_list.append(names[i])
            ft_force_mag_list.append(ft_force_mag_cut_dtw)                
            audio_rms_list.append(audio_rms_cut_dtw)
        else:
            label_list.append(labels[i])
            name_list.append(names[i])
            ft_force_mag_list.append(ft_force_mag_cut)                
            audio_rms_list.append(audio_rms_cut)
            
       
    d = {}
    d['labels'] = label_list
    d['chunks'] = name_list
    d['ft_force_mag_l'] = ft_force_mag_list
    d['audio_rms_l']    = audio_rms_list

    return d


def cutting_for_robot(d, f_zero_size=5, f_thres=1.25, audio_thres=1.0, dtw_flag=False):

    print "Cutting for Robot"
    
    labels      = d['labels']
    names       = d['names']
    ft_time_l   = d['ft_time']
    ft_force_l  = d['ft_force_raw']
    ft_torque_l = d['ft_torque_raw']

    audio_time_l = d['audio_time']
    audio_data_l = d['audio_data']
    audio_amp_l  = d['audio_amp']
    audio_freq_l = d['audio_freq']
    audio_chunk_l= d['audio_chunk']

    ft_time_list     = []
    ft_force_list    = []
    ft_torque_list   = []
    ft_force_mag_list= []
    audio_time_list  = []
    audio_data_list  = []
    audio_amp_list   = []
    audio_freq_list  = []
    audio_chunk_list = []
    audio_rms_list   = []

    label_list = []
    name_list  = []

    ft_force_mag_true_list= []
    audio_rms_true_list   = []
    true_name_list        = []

    ft_force_mag_false_list= []
    audio_rms_false_list   = []
    false_name_list        = []

    MAX_INT = 32768.0
    CHUNK   = 1024 #frame per buffer
    RATE    = 44100 #sampling rate

    #------------------------------------------------------
    # Get reference data
    #------------------------------------------------------
    # Ref ID    
    max_f   = 0.0
    max_idx = 0
    idx     = 1
    ref_idx = 0
    for i, force in enumerate(ft_force_l):
        if labels[i] is False: continue
        else: 
            ft_force_mag = np.linalg.norm(force,axis=0)

            # find end part starting to settle force
            for j, f_mag in enumerate(ft_force_mag[::-1]):
                if f_mag > f_thres: #ft_force_mag[-1]*2.0: 
                    idx = len(ft_force_mag)-j
                    break
                
            if max_idx < idx:
                max_idx = idx
                ref_idx = i                                       
                
    ## print len(ft_time_l), ref_idx

    # Ref force and audio data
    ft_time   = ft_time_l[ref_idx]
    ft_force  = ft_force_l[ref_idx]
    ft_torque = ft_torque_l[ref_idx]        
    ft_force_mag = np.linalg.norm(ft_force,axis=0)
    audio_time = audio_time_l[ref_idx]
    audio_data = audio_data_l[ref_idx]    

    # normalized rms
    audio_rms_ref = np.zeros(len(audio_time))
    for j, data in enumerate(audio_data):
        audio_rms_ref[j] = get_rms(data, MAX_INT)

    # time start & end
    if ft_time[0] < audio_time[0]: start_time = audio_time[0]
    else: start_time = ft_time[0]
        
    if ft_time[-1] < audio_time[-1]: end_time = ft_time[-1]
    else: end_time = audio_time[-1]

    ## print "Time: ", start_time, end_time
        
    # Cut sequence
    idx_start = None
    idx_end   = None                
    for i, t in enumerate(ft_time):
        if t > start_time and idx_start is None:
            idx_start = i
        if t > end_time and idx_end is None:
            idx_end = i            
    if idx_end is None: idx_end = len(ft_time)-1
    ## print "idx: ", idx_start, idx_end
    
    a_idx_start = None
    a_idx_end   = None                
    for j, t in enumerate(audio_time):
        if t > start_time and a_idx_start is None:
            if (audio_time[j+1] - audio_time[j]) >  float(CHUNK)/float(RATE) :
                a_idx_start = j
        if t > end_time and a_idx_end is None:
            a_idx_end = j            
    if a_idx_end is None: a_idx_end = len(audio_time)-1
    ## print "a_idx: ", a_idx_start, a_idx_end

    # Interpolated sequences
    ft_time_cut       = ft_time[idx_start:idx_end]
    ft_force_mag_cut  = ft_force_mag[idx_start:idx_end]
    audio_time_cut    = audio_time[a_idx_start:a_idx_end]
    audio_rms_ref_cut = audio_rms_ref[a_idx_start:a_idx_end]

    x   = np.linspace(0.0, 1.0, len(ft_time_cut))
    tck = interpolate.splrep(x, ft_force_mag_cut, s=0)

    xnew = np.linspace(0.0, 1.0, len(audio_rms_ref_cut))
    ft_force_mag_cut = interpolate.splev(xnew, tck, der=0)


    print "==================================="
    print "Reference size"
    print "-----------------------------------"
    print ft_force_mag_cut.shape,audio_rms_ref_cut.shape 
    print "==================================="

    
    # Cut wrt maximum length
    nZero = f_zero_size #for mix 2
    idx_start = None
    idx_end   = None                    
    for i in xrange(len(ft_force_mag_cut)):

        if i+2*nZero == len(ft_force_mag_cut): break
        ft_avg  = np.mean(ft_force_mag_cut[i:i+nZero])
        ft_avg2 = np.mean(ft_force_mag_cut[i+1*nZero:i+2*nZero])

        if idx_start == None:
            if ft_avg > f_thres and ft_avg < ft_avg2:
                idx_start = i-nZero
        else:
            if ft_avg < f_thres and idx_end is None:
                idx_end = i + nZero
            if idx_end is not None:
                if audio_rms_ref_cut[i] > audio_thres:
                    idx_end = i + nZero
                if ft_avg > 10.0*f_thres: idx_end = None
                ##     idx_end = i
                    
                        
    if idx_end is None: idx_end = len(ft_force_mag_cut)-nZero        
    if idx_end <= idx_start: idx_end += 3
    idx_length = idx_end - idx_start + nZero
    ## print idx_start, idx_end, idx_length, len(ft_force_mag), len(ft_time)
    
    #-------------------------------------------------------------------        
    
    # DTW wrt the reference
    for i, force in enumerate(ft_force_l):

        ft_time      = ft_time_l[i]
        ft_force     = ft_force_l[i]
        ft_torque    = ft_torque_l[i]        
        ft_force_mag = np.linalg.norm(ft_force,axis=0)
        audio_time   = audio_time_l[i]
        audio_data   = audio_data_l[i]    

        # normalized rms
        audio_rms_ref = np.zeros(len(audio_time))
        for j, data in enumerate(audio_data):
            audio_rms_ref[j] = get_rms(data, MAX_INT)

        # time start & end
        if ft_time[0] < audio_time[0]: start_time = audio_time[0]
        else: start_time = ft_time[0]

        if ft_time[-1] < audio_time[-1]: end_time = ft_time[-1]
        else: end_time = audio_time[-1]

        # Cut sequence
        idx_start = None
        idx_end   = None                
        for j, t in enumerate(ft_time):
            if t > start_time and idx_start is None:
                idx_start = j
            if t > end_time and idx_end is None:
                idx_end = j            
        if idx_end is None: idx_end = len(ft_time)-1

        a_idx_start = None
        a_idx_end   = None                
        for j, t in enumerate(audio_time):
            if t > start_time and a_idx_start is None:
                if (audio_time[j+1] - audio_time[j]) >  float(CHUNK)/float(RATE) :
                    a_idx_start = j
            if t > end_time and a_idx_end is None:
                a_idx_end = j            
        if a_idx_end is None: a_idx_end = len(audio_time)-1

        # Interpolated sequences
        ft_time_cut       = ft_time[idx_start:idx_end]
        ft_force_mag_cut  = ft_force_mag[idx_start:idx_end]
        audio_time_cut    = audio_time[a_idx_start:a_idx_end]
        audio_rms_ref_cut = audio_rms_ref[a_idx_start:a_idx_end]

        x   = np.linspace(0.0, 1.0, len(ft_time_cut))
        tck = interpolate.splrep(x, ft_force_mag_cut, s=0)

        xnew = np.linspace(0.0, 1.0, len(audio_rms_ref_cut))
        ft_force_mag_cut = interpolate.splev(xnew, tck, der=0)
            
        # Cut wrt maximum length
        nZero = f_zero_size #for mix 2
        idx_start = None
        idx_end   = None                    
        for j in xrange(len(ft_force_mag_cut)):
            if j+2*nZero == len(ft_force_mag_cut): break
            ft_avg  = np.mean(ft_force_mag_cut[j:j+nZero])
            ft_avg2 = np.mean(ft_force_mag_cut[j+1*nZero:j+2*nZero])

            if idx_start == None:
                if ft_avg > f_thres and ft_avg < ft_avg2 and j-nZero>=0:
                    idx_start = j-nZero
            else:
                if ft_avg < f_thres and idx_end is None:
                    idx_end = j + nZero
                if idx_end is not None:
                    if audio_rms_ref_cut[j] > audio_thres:
                        idx_end = j + nZero
                    if ft_avg > 10.0*f_thres: idx_end = None
                    
        if idx_start == None: idx_start = 0
        if idx_end is None: idx_end = len(ft_force_mag_cut)-nZero        
        

        if labels[i] is True:            
            ## while True:
            ##     if idx_start == 0: break            
            ##     elif idx_start+idx_length >= len(ft_time): idx_start -= 1                
            ##     else: break
            if idx_start+idx_length >= len(ft_force_mag_cut): 
                print "Wrong idx length size", idx_start+idx_length, len(ft_force_mag_cut)
                sys.exit()

            ft_force_mag_cut  = ft_force_mag_cut[idx_start:idx_start+idx_length]
            audio_rms_ref_cut = audio_rms_ref_cut[idx_start:idx_start+idx_length]
        else:
            print labels[i], " : ", idx_start, idx_end, " in ", np.shape(ft_force_mag_cut), \
              " ", np.shape(audio_rms_ref_cut)
            ft_force_mag_cut  = ft_force_mag_cut[idx_start:idx_end]
            audio_rms_ref_cut = audio_rms_ref_cut[idx_start:idx_end]

        label_list.append(labels[i])
        name_list.append(names[i])
        ft_force_mag_list.append(ft_force_mag_cut)                
        audio_rms_list.append(audio_rms_ref_cut)

        if labels[i] is True:
            ft_force_mag_true_list.append(ft_force_mag_cut)
            audio_rms_true_list.append(audio_rms_ref_cut)
            true_name_list.append(names[i])        
        else:
            ft_force_mag_false_list.append(ft_force_mag_cut)
            audio_rms_false_list.append(audio_rms_ref_cut)
            false_name_list.append(names[i])        
        
            
        ## pp.figure(1)
        ## ax = pp.subplot(311)
        ## pp.plot(ft_force_mag_cut)
        ## ## pp.plot(ft_time_cut, ft_force_mag_cut)
        ## ## ax.set_xlim([0, 6.0])
        ## ax = pp.subplot(312)
        ## pp.plot(audio_rms_cut)
        ## ## pp.plot(audio_time_cut, audio_rms_cut)
        ## ## ax.set_xlim([0, 6.0])
        ## ax = pp.subplot(313)
        ## pp.plot(audio_time_cut,'r')
        ## pp.plot(ft_time_cut,'b')
        ## ## pp.plot(audio_rms_cut)
        ## ## ax.set_xlim([0, 6.0])
        ## pp.show()

        #-------------------------------------------------------------
        ## if dtw_flag:
        ##     from test_dtw2 import Dtw

        ##     ref_seq    = np.vstack([ft_force_mag_ref, audio_rms_ref])
        ##     tgt_seq = np.vstack([ft_force_mag_cut, audio_rms_cut])

        ##     dtw = Dtw(ref_seq.T, tgt_seq.T, distance_weights=[1.0, 1.0])
        ##     ## dtw = Dtw(ref_seq.T, tgt_seq.T, distance_weights=[1.0, 10000.0])
        ##     ## dtw = Dtw(ref_seq.T, tgt_seq.T, distance_weights=[1.0, 10000000.0])

        ##     dtw.calculate()
        ##     path = dtw.get_path()
        ##     path = np.array(path).T

        ##     #-------------------------------------------------------------        
        ##     ## dist, cost, path_1d = mlpy.dtw_std(ft_force_mag_ref, ft_force_mag_cut, dist_only=False)        
        ##     ## fig = plt.figure(1)
        ##     ## ax = fig.add_subplot(111)
        ##     ## plot1 = plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
        ##     ## plot2 = plt.plot(path_1d[0], path_1d[1], 'w')
        ##     ## plot2 = plt.plot(path[0], path[1], 'r')
        ##     ## xlim = ax.set_xlim((-0.5, cost.shape[0]-0.5))
        ##     ## ylim = ax.set_ylim((-0.5, cost.shape[1]-0.5))
        ##     ## plt.show()
        ##     ## sys.exit()
        ##     #-------------------------------------------------------------        

        ##     ft_force_mag_cut_dtw = []        
        ##     audio_rms_cut_dtw    = []        
        ##     new_idx = []
        ##     for idx in xrange(len(path[0])-1):
        ##         if path[0][idx] == path[0][idx+1]: continue

        ##         new_idx.append(path[1][idx])
        ##         ft_force_mag_cut_dtw.append(ft_force_mag_cut[path[1][idx]])
        ##         audio_rms_cut_dtw.append(audio_rms_cut[path[1][idx]])
        ##     ft_force_mag_cut_dtw.append(ft_force_mag_cut[path[1][-1]])
        ##     audio_rms_cut_dtw.append(audio_rms_cut[path[1][-1]])


        ##     print "==================================="
        ##     print names[i], len(ft_force_mag_cut_dtw), len(audio_rms_cut_dtw)
        ##     print "==================================="

        ##     label_list.append(labels[i])
        ##     name_list.append(names[i])
        ##     ft_force_mag_list.append(ft_force_mag_cut_dtw)                
        ##     audio_rms_list.append(audio_rms_cut_dtw)
        ## else:
        ##     label_list.append(labels[i])
        ##     name_list.append(names[i])
        ##     ft_force_mag_list.append(ft_force_mag_cut)                
        ##     audio_rms_list.append(audio_rms_cut)
            
       
    d = {}
    d['labels'] = label_list
    d['chunks'] = name_list
    d['ft_force_mag_l'] = ft_force_mag_list
    d['audio_rms_l']    = audio_rms_list

    d['ft_force_mag_true_l'] = ft_force_mag_true_list
    d['audio_rms_true_l']    = audio_rms_true_list
    d['true_chunks']         = true_name_list

    d['ft_force_mag_false_l'] = ft_force_mag_false_list
    d['audio_rms_false_l']    = audio_rms_false_list
    d['false_chunks']         = false_name_list

    return d

    

def create_mvpa_dataset(aXData1, aXData2, chunks, labels):

    feat_list = []
    for x1, x2, chunk in zip(aXData1, aXData2, chunks):
        feat_list.append([x1, x2])

    data = Dataset(samples=feat_list)
    data.sa['id']      = range(0,len(labels))
    data.sa['chunks']  = chunks
    data.sa['targets'] = labels

    return data


def get_rms(frame, MAX_INT=32768.0):
    
    count = len(frame)
    return  np.linalg.norm(frame/MAX_INT) / np.sqrt(float(count))


def scaling(X, min_c=None, max_c=None, scale=10.0, verbose=False):
    '''        
    scale should be over than 10.0(?) to avoid floating number problem in ghmm.
    Return list type
    '''
    ## X_scaled = preprocessing.scale(np.array(X))

    if min_c is None or max_c is None:
        min_c = np.min(X)
        max_c = np.max(X)

    X_scaled = []
    for x in X:
        if verbose is True: print min_c, max_c, " : ", np.min(x), np.max(x)
        X_scaled.append(((x-min_c) / (max_c-min_c) * scale))

    ## X_scaled = (X-min_c) / (max_c-min_c) * scale

    return X_scaled, min_c, max_c


def scaling_reverse(X, min_c, max_c, scale=10.0):

    X_scaled = []
    for x in X:
        X_scaled.append( (x/scale)*(max_c-min_c)+min_c )

    return X_scaled
    

def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    #including valid will REQUIRE there to be enough datapoints.
    #for example, if you take out valid, it will start @ point one,
    #not having any prior points, so itll be 1+0+0 = 1 /3 = .3333

    new_values = []
    for value in values:
        new_value = np.convolve(value, weigths, 'valid')
        new_values.append(new_value)
    return np.array(new_values)
    
    
def simulated_anomaly(true_aXData1, true_aXData2, num, min_c1, max_c1, min_c2, max_c2, an_type,
                      force_an, sound_an):
    '''
    num : number of anomaly data
    '''
    
    ## an_types = ['force', 'sound', 'both']
    ## force_an = ['normal', 'inelastic', 'inelastic_continue', 'elastic', 'elastic_continue']
    ## force_an = ['inelastic', 'inelastic_continue', 'elastic', 'elastic_continue']
    ## force_an = ['normal', 'magnified', 'shrinked', 'amplified', 'weaken']
    
    ## sound_an = ['normal', 'rndsharp', 'rnddull']
    ## sound_an = ['rndsharp', 'rnddull']
    ## sound_an = ['normal', 'weaken', 'rndimpulse']

    #force_anomaly for roc graph was 0.1~0.5?
    
    length = len(true_aXData1[0])

    # mean, var of true data    
    x = range(len(true_aXData1[0]))
    y1 = np.sum(true_aXData1, axis=0)
    y1 = y1 / np.sum(y1)
    y2 = np.sum(true_aXData2, axis=0).tolist()
    ## y2 = y2 / np.sum(y2)    
    max_y2_idx = y2.index(max(y2))

    ## mu1, var1 = hdl.gaussian_param_estimation(x, y1)
    ## mu2, var2 = hdl.gaussian_param_estimation(x, y2)
    new_X1 = []
    new_X2 = []
    chunks = []
    an_start =[]
    an_peak = []
    an_width = []
    sampling_complete = False
                
    for i in xrange(num):

        # randomly select 
        x_idx = random.randint(0,len(true_aXData1)-1)

        # three categories of anomalies
        # 1) only force
        # 2) only sound
        # 3) both
        ## an_type = random.choice(an_types)            
        ## an_type = 'both'

        while sampling_complete == False:

            x1_anomaly = copy.copy(true_aXData1[x_idx])
            x2_anomaly = copy.copy(true_aXData2[x_idx])
            an1 = 'normal'
            an2 = 'normal'
            an_idx = None # anomaly start location
            
            # random anomaly type
            if (an_type == 'force' or an_type == 'both'):

                # Select a type of anomaly
                while True:
                    an1 = random.choice(force_an)
                    if an_type == 'force' and an1 == 'normal': continue
                    else: break
                       
                if an1 == 'normal':
                    print "normal force"

                elif an1 == 'magnified':
                    print "magnified force"

                    mag = random.uniform(1.1, 1.5)

                    x   = np.linspace(0.0, 1.0, length)
                    tck = interpolate.splrep(x, true_aXData1[x_idx], s=0)

                    xnew = np.linspace(0.0, 1.0, length*mag)
                    x1_anomaly = interpolate.splev(xnew, tck, der=0)*mag
                    an_idx = 0                                        

                elif an1 == 'shrinked':
                    print "shrinked force"

                    mag = random.uniform(0.4, 0.9)

                    x   = np.linspace(0.0, 1.0, length)
                    tck = interpolate.splrep(x, true_aXData1[x_idx], s=0)

                    xnew = np.linspace(0.0, 1.0, length*mag)
                    x1_anomaly = interpolate.splev(xnew, tck, der=0)*mag
                    an_idx = 0                                        

                elif an1 == 'amplified':
                    print "Amplied force"
                    mag = random.uniform(1.1, 1.5)                
                    x1_anomaly = x1_anomaly*mag
                    an_idx = 0                    

                elif an1 == 'weaken':
                    print "Weaken force"
                    mag = random.uniform(0.1, 0.9)                
                    x1_anomaly = x1_anomaly*mag
                    an_idx = 0                    

                elif an1 == 'rndimpulse' or an1 == 'inelastic' or an1 == 'inelastic_continue':
                    print "Random impulse force which is inelastic collision force: ", an1
                    #http://www.vernier.com/innovate/impulse-comparison-for-elastic-and-inelastic-collisions/
                    peak   = max_c1 * random.uniform(0.01, 0.3)
                    width  = random.randint(3,10)                    

                    ## peak   = max_c1 * random.uniform(0.1, 0.5)                    
                    ## width  = random.randint(3,6)                    
                    an_idx = loc = random.randint(1,length-1-width)

                    xnew    = range(width)

                    if an1 == 'inelastic_continue':                    
                        for i in xrange(width/2):
                            x1_anomaly[loc+i] += (i+1)*peak/(float(width)/2.0)
                        x1_anomaly[loc+width/2:] += peak
                    else:
                        for i in xrange(width):
                            if i < width/2:
                                x1_anomaly[loc+i] += (i+1)*peak/(float(width)/2.0)
                            else:
                                x1_anomaly[loc+i] += -(i-(float(width)/2.0))*peak/(float(width)/2.0) + peak

                elif an1 == 'elastic' or an1 == 'elastic_continue':
                    print "elastic collision with continuous force"

                    peak  = max_c1 * random.uniform(0.01, 0.3)
                    if len(x1_anomaly) <= 25: 
                        width = random.randint(5,10)
                    else:
                        width = random.randint(5,20)
                        
                    ## peak  = max_c1 * random.uniform(0.1, 0.5)
                    ## if len(x1_anomaly) <= 25: 
                    ##     width = random.randint(5,10)
                    ## else:
                    ##     width = random.randint(5,20)

                    an_idx  = loc = random.randint(1,length-1-width)
                    xnew    = range(width)

                    if an1 == 'elastic_continue':                    
                        for i in xrange(width/2):
                            x1_anomaly[loc+i] += peak * (1.0 - ( ((i+1)-(float(width)/2.0))/(float(width)/2.0) )**2)
                        x1_anomaly[loc+width/2:] += peak
                    else:                
                        for i in xrange(width):
                            x1_anomaly[loc+i] += peak * (1.0 - ( ((i+1)-(float(width)/2.0))/(float(width)/2.0) )**2)
                else:
                    print "Not implemented type of force anomaly : ", an1

            if an_type == 'sound' or an_type == 'both':

                # Select a type of anomaly
                while True:
                    an2 = random.choice(sound_an)
                    if an_type == 'both' and an1 == 'normal' and an2 == 'normal': continue
                    elif an_type == 'sound' and an2 == 'normal': continue
                    else: break

                if an2 == 'normal':
                    print "normal sound"
                    
                elif an2 == 'weaken':
                    print "Weaken sound"

                    mag = random.uniform(0.05, 0.3)                
                    x2_anomaly = x2_anomaly*mag

                elif an2 == 'rndimpulse' or an2 == 'rndsharp':
                    print "Random impulse sound"
                    
                    # sample sound shape
                    peak  = max_c2 * random.uniform(0.2, 0.5)
                    width = random.randint(3,4)
                    if len(x1_anomaly) <= 20+width: block_size = 4
                    else: block_size = 10

                    # sampling location
                    loc = random.randint(1+(width)/2,len(x1_anomaly)-1-(width)/2)                         
                    if an_idx is None: an_idx = loc                    
                    if loc < an_idx: an_idx = loc

                    if loc+width > length-1:
                        continue            
                    if loc+width < max_y2_idx - block_size or loc > max_y2_idx + block_size:                     
                        continue

                    xnew    = range(width)
                    impulse = np.zeros(width)
                    ## x2_anomaly = np.ones(np.shape(x1_anomaly)) * x2_anomaly[0]

                    for i in xrange(width):
                        if i < width/2:
                            impulse[i] = (i+1)*peak/(float(width)/2.)
                        else:
                            impulse[i] = -(i-(float(width)/2.0))*peak/(float(width)/2.0) + peak
                        x2_anomaly[loc+i] += impulse[i] 

                elif an2 == 'rnddull':
                    print "Random dull sound"

                    # sampling shape
                    peak  = max_c2 * random.uniform(0.1, 0.5)
                    if len(x1_anomaly) <= 25: 
                        width = random.randint(10,20)
                        block_size = 4
                    else: 
                        width = random.randint(5,20)
                        block_size = 10

                    # sampling location
                    loc   = random.randint(1,len(x1_anomaly)-width-1)                        
                    if an_idx is None: an_idx = loc                                        
                    if loc < an_idx: an_idx = loc

                    if loc+width > length-1:
                        continue                                
                    if loc+width < max_y2_idx - block_size or loc > max_y2_idx + block_size:
                        continue

                    xnew    = range(width)
                    impulse = np.zeros(width)
                    ## x2_anomaly = np.ones(np.shape(x1_anomaly)) * x2_anomaly[0]

                    for i in xrange(width):
                        impulse[i] = peak * (1.0 - ( (i+1-(float(width)/2.0))/(float(width)/2.0) )**2)
                        x2_anomaly[loc+i] += impulse[i] 

                else:
                    print "Not implemented type of sound anomaly : ", an2

            else:
                print "Undefined options"
                continue


            # Zero padding to meet same length
            n1 = len(x1_anomaly)
            n2 = len(x2_anomaly)
            if n1 >= n2:
                x2_anomaly = np.hstack([x2_anomaly, [x2_anomaly[-1]]*(n1-n2)])
            else:
                x1_anomaly = np.hstack([x1_anomaly, [x1_anomaly[-1]]*(n2-n1)])

            #print "New anomaly data size: ", len(x1_anomaly), " ", len(x2_anomaly)

            new_X1.append(x1_anomaly)
            new_X2.append(x2_anomaly)
            chunks.append(an1+"_"+an2)
            an_start.append(an_idx)

            an_peak.append(peak)
            an_width.append(width)
            
            sampling_complete = True
        
    return new_X1, new_X2, chunks, an_start, an_peak, an_width


def generate_sim_anomaly(true_aXData1, true_aXData2, n_false_data, an_type, force_an, sound_an):

    _, min_c1, max_c1 = scaling(true_aXData1, scale=10.0)
    _, min_c2, max_c2 = scaling(true_aXData2, scale=10.0)    
    
    # generate simulated data!!
    aXData1, aXData2, chunks, an_start, an_peak, an_width = simulated_anomaly(true_aXData1, true_aXData2, 
                                                                              n_false_data, \
                                                                              min_c1, max_c1, min_c2, max_c2, 
                                                                              an_type, force_an, sound_an)

    d = {}
    d['ft_force_mag_sim_false_l'] = aXData1
    d['audio_rms_sim_false_l'] = aXData2
    d['sim_false_chunks'] = chunks
    d['anomaly_start_idx'] = an_start
    d['anomaly_peak'] = an_peak
    d['anomaly_width'] = an_width
    return d


def loadData(pkl_file, data_path, task_name, f_zero_size, f_thres, audio_thres, cross_data_path=None, 
             an_type=None, force_an=None, sound_an=None, bRenew=False, rFold=None, nDataSet=None):

    if os.path.isfile(pkl_file) and bRenew is False:
        d = ut.load_pickle(pkl_file)
    else:
        d = load_data(data_path, task_name, normal_only=False)
        d = cutting_for_robot(d, f_zero_size=f_zero_size, f_thres=f_thres, \
                                 audio_thres=audio_thres, dtw_flag=False)        
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
    if an_type is not None:

        nDataSet = len(true_aXData1)
        
        if rFold is None:
            # leave-one-out
            nDataSet = len(true_aXData1)            
            n_false_data = 1 # the number of test data
        else:
            nTest = int(len(true_chunks)*(1.0-rFold))            
            n_false_data = 1 # if not 1, there will be error
            nDataSet = len(true_aXData1)            

        if os.path.isdir(cross_data_path) == False:
            os.system('mkdir -p '+cross_data_path)
            
        for i in xrange(nDataSet):

            pkl_file = os.path.join(cross_data_path, "dataSet_"+str(i))
            dd = None
            if os.path.isfile(pkl_file) is False:

                labels        = [True]*len(true_aXData1)
                true_dataSet  = create_mvpa_dataset(true_aXData1, true_aXData2, true_chunks, labels)

                if rFold is None:
                    test_ids      = Dataset.get_samples_by_attr(true_dataSet, 'id', i)
                else:
                    test_ids      = Dataset.get_samples_by_attr(true_dataSet, 'id', 
                                                                random.sample(range(len(labels)), nTest))
                    
                test_dataSet  = true_dataSet[test_ids]
                train_ids     = [val for val in true_dataSet.sa.id if val not in test_dataSet.sa.id] 
                train_ids     = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
                train_dataSet = true_dataSet[train_ids]

                train_aXData1 = train_dataSet.samples[:,0,:]
                train_aXData2 = train_dataSet.samples[:,1,:]
                train_chunks  = train_dataSet.sa.chunks 

                test_aXData1 = test_dataSet.samples[:,0,:]
                test_aXData2 = test_dataSet.samples[:,1,:]
                test_chunks  = test_dataSet.sa.chunks

                dd = generate_sim_anomaly(test_aXData1, test_aXData2, n_false_data, an_type, force_an, 
                                             sound_an)
                dd['ft_force_mag_train_l'] = train_aXData1 
                dd['audio_rms_train_l']    = train_aXData2 
                dd['train_chunks']         = train_chunks
                dd['ft_force_mag_test_l']  = test_aXData1 
                dd['audio_rms_test_l']     = test_aXData2 
                dd['test_chunks']          = test_chunks

                ut.save_pickle(dd, pkl_file)
            else:
                dd = ut.load_pickle(pkl_file)

            false_aXData1       = dd['ft_force_mag_sim_false_l']
            false_aXData2       = dd['audio_rms_sim_false_l'] 
            false_chunks        = dd['sim_false_chunks']
            false_anomaly_start = dd['anomaly_start_idx']
 
    ## elif opt.bSimAbnormal or opt.bRocOfflineSimAnomaly:
    ##     pkl_file = os.path.join(cross_root_path,task_names[task]+"_sim_an_data.pkl")
    ##     if os.path.isfile(pkl_file) and opt.bRenew is False:
    ##         dd = ut.load_pickle(pkl_file)
    else:
        print "Load real anomaly data"
        
        false_aXData1 = d['ft_force_mag_false_l']
        false_aXData2 = d['audio_rms_false_l'] 
        false_chunks  = d['false_chunks']
        false_anomaly_start = np.zeros(len(false_chunks))

        # need a way to assign anomaly index by human
        if False: #True:
            false_anomaly_start = set_anomaly_start(true_aXData1,true_aXData2,
                                                    false_aXData1,false_aXData2,false_chunks)

        # For non-roc test, nTest is used to count threshold-test set     
        if rFold is None:
            nTest = len(false_chunks)            
        else:
            nTest = int(len(true_chunks)*(1.0-rFold))            

        if nDataSet is None:
            nDataSet = 1
            n_false_data = len(false_aXData1)
        else:            
            # leave-one-out
            nDataSet = len(true_aXData1)            
            n_false_data = 1 # if not 1, there will be error

        if cross_data_path is not None:
        
            if os.path.isdir(cross_data_path) == False:
                os.system('mkdir -p '+cross_data_path)        

            for i in xrange(nDataSet):
                
                pkl_file = os.path.join(cross_data_path, "dataSet_"+str(i))
                dd = {}
                if os.path.isfile(pkl_file) is False:

                    labels        = [True]*len(true_aXData1)
                    true_dataSet  = create_mvpa_dataset(true_aXData1, true_aXData2, true_chunks, labels)

                    if nDataSet == len(true_aXData1):
                        test_ids      = Dataset.get_samples_by_attr(true_dataSet, 'id', 
                                                                    random.sample(range(len(labels)), nTest))
                    else:
                        test_ids      = Dataset.get_samples_by_attr(true_dataSet, 'id', i)
                            
                    test_dataSet  = true_dataSet[test_ids]
                    train_ids     = [val for val in true_dataSet.sa.id if val not in test_dataSet.sa.id] 
                    train_ids     = Dataset.get_samples_by_attr(true_dataSet, 'id', train_ids)
                    train_dataSet = true_dataSet[train_ids]

                    train_aXData1 = train_dataSet.samples[:,0,:]
                    train_aXData2 = train_dataSet.samples[:,1,:]
                    train_chunks  = train_dataSet.sa.chunks 

                    test_aXData1 = test_dataSet.samples[:,0,:]
                    test_aXData2 = test_dataSet.samples[:,1,:]
                    test_chunks  = test_dataSet.sa.chunks

                    dd['ft_force_mag_train_l'] = train_aXData1 
                    dd['audio_rms_train_l']    = train_aXData2 
                    dd['train_chunks']         = train_chunks
                    dd['ft_force_mag_test_l']  = test_aXData1 
                    dd['audio_rms_test_l']     = test_aXData2 
                    dd['test_chunks']          = test_chunks

                    if n_false_data == len(false_aXData1):
                        dd['ft_force_mag_false_l'] = false_aXData1
                        dd['audio_rms_false_l']    = false_aXData2
                        dd['false_chunks']         = false_chunks                 
                        dd['anomaly_start_idx']    = false_anomaly_start
                    else:
                        labels        = [False]*len(false_aXData1)
                        false_dataSet = create_mvpa_dataset(false_aXData1, false_aXData2, false_chunks, labels)
                        test_ids      = Dataset.get_samples_by_attr(false_dataSet, 'id', 
                                                                    random.sample(range(len(labels)), 
                                                                                  n_false_data))

                        test_dataSet  = false_dataSet[test_ids]
                        test_aXData1  = np.array([test_dataSet.samples[0,0]])
                        test_aXData2  = np.array([test_dataSet.samples[0,1]])
                        test_chunks   = test_dataSet.sa.chunks
                        dd['ft_force_mag_false_l'] = test_aXData1
                        dd['audio_rms_false_l']    = test_aXData2
                        dd['false_chunks']         = test_chunks                 
                        dd['anomaly_start_idx']    = np.zeros(len(test_chunks))

                    ut.save_pickle(dd, pkl_file)


    print "All: ", len(true_aXData1)+len(false_aXData1), \
      " Success: ", len(true_aXData1), \
      " Failure: ", len(false_aXData1)
        
    return true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet


def set_anomaly_start(true_aXData1, true_aXData2, false_aXData1, false_aXData2, false_chunks):
    
    anomaly_start = np.zeros(len(false_chunks))
    
    for i in xrange(len(false_chunks)):
        print "false_chunks: ", false_chunks[i]

        data = false_aXData1[i]        
        x   = np.arange(0., float(len(data)))
        
        fig = pp.figure()
        plt.rc('text', usetex=True)

        ax1 = pp.subplot(411)
        for ii, d in enumerate(true_aXData1):
            r = len(x) - len(d)
            if r>0: data = np.hstack([d, np.zeros(r)])
            else: data = d                
            true_line, = pp.plot(x, data)
                
        ax2 = pp.subplot(412)
        for ii, d in enumerate(true_aXData2):
            r = len(x) - len(d)
            if r>0: data = np.hstack([d, np.zeros(r)])
            else: data = d
            true_line, = pp.plot(x, data)


        ax3 = pp.subplot(413)
        data = false_aXData1[i]        

        pp.plot(x, data, 'b', linewidth=1.5, label='Force')
        #ax1.set_xlim([0, len(data)])
        ax3.set_ylim([0, np.amax(data)*1.1])
        pp.grid()
        ax3.set_ylabel("Magnitude [N]", fontsize=18)


        ax4 = pp.subplot(414)
        data = false_aXData2[i]

        pp.plot(x, data, 'b', linewidth=1.5, label='Sound')
        #ax2.set_xlim([0, len(data)])
        pp.grid()
        ax4.set_ylabel("Energy", fontsize=18)
        ax4.set_xlabel("Time [sec]", fontsize=18)

        ax3.legend(prop={'size':18})
        ax4.legend(prop={'size':18})

        pp.show()

        anomaly_start[i] = int(raw_input("Enter anomaly location: "))

    return anomaly_start
    
