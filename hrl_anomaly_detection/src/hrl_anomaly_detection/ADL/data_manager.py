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

    for pkl in pkl_list:
        
        bNormal = True
        if pkl.find('success') < 0: bNormal = False
        if normal_only and bNormal is False: continue

        d = ut.load_pickle(pkl)

        ft_time  = d.get('ft_time',None)
        ft_force  = d.get('ft_force_raw',None)
        ft_torque = d.get('ft_torque_raw',None)
        
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

        name = tail.split('_')[0] + '_' + tail.split('_')[1] + '_' + tail.split('_')[2]
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


def cutting(d):

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

    ## print "==================================="
    ## print ft_force_mag_ref.shape,audio_rms_ref.shape 
    ## print "==================================="

    
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

        audio_time_cut = np.array(audio_time[a_idx_start:a_idx_end+1])
        audio_data_cut = np.array(audio_data[a_idx_start:a_idx_end+1])              
        
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
       
    d = {}
    d['labels'] = label_list
    d['chunks'] = name_list
    d['ft_force_mag_l'] = ft_force_mag_list
    d['audio_rms_l']    = audio_rms_list

    return d


def create_mvpa_dataset(aXData1, aXData2, chunks, labels):

    feat_list = []
    for x1, x2, chunk in zip(aXData1, aXData2, chunks):
        feat_list.append([x1, x2])
    
    data = Dataset(samples=feat_list)
    data.sa['chunks'] = chunks
    data.sa['targets'] = labels

    return data


def get_rms(frame, MAX_INT=32768.0):
    
    count = len(frame)
    return  np.linalg.norm(frame/MAX_INT) / np.sqrt(float(count))


def scaling(X):
    '''        
    '''
    ## X_scaled = preprocessing.scale(np.array(X))

    min_c = np.min(X)
    max_c = np.max(X)
    X_scaled = (X-min_c) / (max_c-min_c) * 10.0

    return X_scaled, min_c, max_c


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
    
    
