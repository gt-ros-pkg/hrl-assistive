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

def load_data(data_path, prefix, normal_only=True):

    pkl_list = glob.glob(data_path+prefix+'*.pkl')

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


def cutting(data):

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
    ft_force_mag_list = []
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
                    idx_end = j+nZero*2

        ft_time_list.append(ft_time_l[i][idx_start:idx_end])
        ft_force_list.append(ft_force_l[i][:,idx_start:idx_end])
        ft_torque_list.append(ft_torque_l[i][:,idx_start:idx_end])

        ft_force_mag_list.append(np.linalg.norm(ft_force_l[i][:,idx_start:idx_end], axis=0))
        
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
        MAX_INT = 32768.0
        CHUNK   = 1024 #frame per buffer
        RATE    = 44100 #sampling rate

        def downSample(fftx,ffty,degree=10):
            x,y=[],[]
            for i in range(len(ffty)/degree-1):
                x.append(fftx[i*degree+degree/2])
                y.append(sum(ffty[i*degreei+1)*degree])/degree)
        return [x,y]
        #----------------------------------------------------
        
        start_time = ft_time_l[i][idx_start]
        end_time   = ft_time_l[i][idx_end]
        audio_time = audio_time_l[i]

        a_idx_start = None
        a_idx_end   = None                
        for j, t in enumerate(audio_time):
            
            if t > start_time and a_idx_start is None:
                a_idx_start = j
            if t > end_time and a_idx_end is None:
                a_idx_end = j

        audio_time_cut = np.array(audio_time[a_idx_start:a_idx_end])
        audio_data_cut = audio_data_l[i][a_idx_start:a_idx_end]

        cut_coff = int(float(len(audio_time_cut))/float(len(ft_time_list[i])))
        for j, sample in audio_data_cut:

            audio_freq = np.fft.fftfreq(self.CHUNK, self.UNIT_SAMPLE_TIME) 
            audio_amp = np.fft.fft(audio_data / float(self.MAX_INT)) 
            
            downSample(sample)

        

        import scipy as scp
        new_audio_data = scp.signal.resample(np.array(audio_data_cut).flatten(), 1000)

        pp.figure()
        pp.plot(new_audio_data)
        pp.show()

        
        ## # resample? down sample
        ## for j in xrange(len(ft_time_list[i])-1):
        ##     start_time = ft_time_list[i][j]
        ##     end_time   = ft_time_list[i][j+1]

        ##     audio_data_set = []            
        ##     for k, t in enumerate(audio_time):
        ##         if t >= start_time and t < end_time:                                    
        ##             audio_data_set.

        
        ## print np.array(ft_time_list).shape
        ## print len(audio_time_cut)
        ## sys.exit()



        
        audio_time_list.append(audio_time_cut)
        audio_data_list.append(audio_data_cut)



        

        ## time_range = np.arange(0.0, 1024.0, 1.0)/44100.0               
                                
        ## # find init
        ## pp.figure()
        ## pp.subplot(411)
        ## pp.plot(ft_time_l[i][idx_start:idx_end], f[idx_start:idx_end])
        ## ## pp.stem([idx_start, idx_end], [f[idx_start], f[idx_end]], 'k-*', bottom=0)
        ## pp.title(names[i])
        
        ## pp.subplot(412)
        ## for k in xrange(len(audio_data_cut)):            
        ##     cur_time = time_range + audio_time_cut[0] + float(k)*1024.0/44100.0
        ##     pp.plot(cur_time, audio_data_cut[k], 'b.')


        ## pp.subplot(413)
        ## pp.plot(audio_time_cut, np.mean((np.array(audio_data_cut)),axis=1))
            
        ## pp.subplot(414)
        ## pp.plot(audio_time_cut, np.std((np.array(audio_data_cut)),axis=1))
        ## ## pp.stem([idx_start, idx_end], [max(audio_data_l[i][idx_start]), max(audio_data_l[i][idx_end])], 'k-*', bottom=0)
        ## ## pp.title(names[i])
        ## ## pp.plot(audio_freq_l[i], audio_amp_l[i])
        ## pp.show()

               
    # find minimum length data
    ft_min_idx = -1        
    for i, ft_time in enumerate(ft_time_list):
        
        if labels[i] is False:
            print i, len(ft_time_list[i])
            if ft_min_idx == -1: 
                ft_min_idx = i
            else:
                if len(ft_time_list[ft_min_idx]) > len(ft_time_list[i]):
                    ft_min_idx = i
            
    print "'''''''''''''''''''''''''''''''''''''''''''"
    print "Minimum data index is ", ft_min_idx
    print "Minimum data length is ", len(ft_time_list[ft_min_idx])
    ## ft_data_min = int(ft_data_min/10.0)*10
    ## print "We manually fix the length into ", ft_data_min
    print "'''''''''''''''''''''''''''''''''''''''''''"
    

    ## Scaling or resample
    import mlpy
    
    dist, cost, path = mlpy.dtw_std(ft_force_list[ft_min_idx][2], ft_force_list[1][2], dist_only=False)

    print path

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plot1 = plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    plot2 = plt.plot(path[0], path[1], 'w')
    xlim = ax.set_xlim((-0.5, cost.shape[0]-0.5))
    ylim = ax.set_ylim((-0.5, cost.shape[1]-0.5))
    plt.show()
    

    ## pp.figure()
    ## pp.plot(ft_force_list[0][2,:],'r')
    ## pp.plot(ft_force_mag_list[0])
    ## pp.show()
    
            
        
        
        
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

    task = 2
    if task == 1:
        prefix = 'microwave'
    else:
        prefix = 'close'
    
    # Load data
    pkl_file = "./all_data.pkl"
    if os.path.isfile(pkl_file):
        d = ut.load_pickle(pkl_file)
    else:
        d = load_data(data_path, prefix)
        ut.save_pickle(d, pkl_file)
    

    # Cutting
    d = cutting(d)
    ## scaling(d)
    

    # Learning
    


    # TEST
