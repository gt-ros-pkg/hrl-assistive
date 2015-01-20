#!/usr/bin/python

import sys, os, copy
import numpy as np, math
import glob
import socket
import time
import random 

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

# Util
import hrl_lib.util as ut
import matplotlib.pyplot as pp
import matplotlib as mpl


import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
from hrl_anomaly_detection.HMM.anomaly_checker import anomaly_checker

def load_data(data_path):

    pkl_list = glob.glob(data_path+'s_*.pkl')

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
        if '_b' in pkl: bNormal = False

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
        name_list.append(tail)


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


def scaling(data):

    labels = data['labels']
    names = data['names']
    ft_time_l   = data['ft_time']
    ft_force_l  = data['ft_force_raw']
    ft_torque_l = data['ft_torque_raw']

    audio_time_l = d['audio_time']
    audio_data_l = d['audio_data']
    audio_amp_l  = d['audio_amp']
    audio_freq_l = d['audio_freq']
    audio_chunk_l= d['audio_chunk']

    ft_time_list   = []
    ft_force_list  = []
    ft_torque_list = []
    audio_time_list = []
    audio_data_list = []
    audio_amp_list = []
    audio_freq_list = []
    audio_chunk_list = []
    
    # Cut force data
    for i, force in enumerate(ft_force_l):

        f = np.linalg.norm(force,axis=0)
                
        nZero = 5
        ft_zero = np.mean(f[:nZero]) * 1.5

        idx_start = None
        idx_end   = None        
        for j in xrange(len(f)-nZero):
            avg = np.mean(f[j:j+nZero])
            
            if avg > ft_zero and idx_start is None:
                idx_start = j #i+nZero

            if idx_start is not None:
                if avg < ft_zero and idx_end is None:
                    idx_end = j+nZero

        ft_time_list.append(ft_time_l[i][idx_start:idx_end])
        ft_force_list.append(ft_force_l[i][:,idx_start:idx_end])
        ft_torque_list.append(ft_torque_l[i][:,idx_start:idx_end])

        ## # find init
        ## pp.figure()
        ## pp.subplot(211)
        ## pp.plot(f)
        ## pp.stem([idx_start, idx_end], [f[idx_start], f[idx_end]], 'k-*', bottom=0)
        ## pp.title(names[i])
        ## pp.subplot(212)
        ## pp.plot(force[2,:])
        ## pp.show()
        
        #----------------------------------------------------

        start_time = ft_time_l[i][idx_start]
        end_time   = ft_time_l[i][idx_end]
        audio_time = audio_time_l[i]

        print ft_time_l[i]
        print audio_time
        
        idx_start = None
        idx_end   = None                
        for j, t in enumerate(audio_time):
            
            if t > start_time and idx_start is None:
                idx_start = j
            if t > end_time and idx_end is None:
                idx_end = j

        print idx_start, idx_end

        audio_time_list.append(audio_time[idx_start:idx_end])
        audio_data_list.append(audio_data_l[i][idx_start:idx_end])
                        
        # find init
        pp.figure()
        pp.subplot(211)
        pp.plot(audio_data_l[i])
        pp.stem([idx_start, idx_end], [audio_data_l[i][idx_start], audio_data_l[i][idx_end]], 'k-*', bottom=0)
        pp.title(names[i])
        pp.subplot(212)
        pp.plot(audio_freq_l[i], audio_amp_l[i])
        pp.show()

        
        
                    
        

        
        ## if block: 
        ##     if '_b3' in pkl:
        ##         pp.plot(force[0], 'b-')            
        ## elif block is False: pp.plot(force[0], 'r-')

        ## if block: 
        ##     if '_b3' in pkl:
        ##         pp.plot(audio_data, 'b-')            
        ## elif block is False: pp.plot(audio_data, 'r-')


        

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='renew',
                 default=False, help='Renew pickle files.')


    data_path = os.environ['HRLBASEPATH']+'/src/projects/anomaly/test_data/'
    nMaxStep  = 36 # total step of data. It should be automatically assigned...

    # Load data
    pkl_file = "./all_data.pkl"
    if os.path.isfile(pkl_file):
        d = ut.load_pickle(pkl_file)
    else:
        d = load_data(data_path)
        ut.save_pickle(d, pkl_file)
    

    # Cutting
    scaling(d)
    

    # Learning


    # TEST
