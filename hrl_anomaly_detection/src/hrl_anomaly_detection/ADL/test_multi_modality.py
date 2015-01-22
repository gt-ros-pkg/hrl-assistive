#!/usr/bin/python

import sys, os, copy
import numpy as np, math
import glob
import socket
import time
import random 
import scipy as scp

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

# Util
import hrl_lib.util as ut
import matplotlib.pyplot as pp
import matplotlib as plt


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

        if labels[i] == True:
            idx_start = None
            idx_end   = None        
            for j in xrange(len(f)-nZero):
                avg = np.mean(f[j:j+nZero])

                if avg > ft_zero and idx_start is None:
                    idx_start = j #i+nZero

                if idx_start is not None:
                    if avg < ft_zero and idx_end is None:
                        idx_end = j+nZero*2

            ft_time_cut  = np.array(ft_time_l[i][idx_start:idx_end])
            ft_force_cut = ft_force_l[i][:,idx_start:idx_end]
            ft_torque_cut= ft_torque_l[i][:,idx_start:idx_end]
        else:
            idx_start = 0
            idx_end = len(ft_time_l[i])-1
            ft_time_cut  = np.array(ft_time_l[i])
            ft_force_cut = ft_force_l[i]
            ft_torque_cut= ft_torque_l[i]
            
                    
        ft_time_list.append(ft_time_cut)
        ft_force_list.append(ft_force_cut)
        ft_torque_list.append(ft_torque_cut)

        ## ft_force_mag_list.append(np.linalg.norm(ft_force_l[i][:,idx_start:idx_end], axis=0))
        
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

        ## def downSample(fftx,ffty,degree=10):
        ##     x,y=[],[]
        ##     for i in range(len(ffty)/degree-1):
        ##         x.append(fftx[i*degree+degree/2])
        ##         y.append(sum(ffty[i*degreei+1)*degree])/degree)
        ## return [x,y]
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
        audio_data_cut = np.array(audio_data_l[i][a_idx_start:a_idx_end])


        ## for j, data in enumerate(audio_data_cut):
        ##     data = np.hstack([data, np.zeros(len(data))])
        ##     F = np.fft.fft(data / float(MAX_INT))  #normalization & FFT          
        ##     #F = np.fft.fft(data)  #normalization & FFT          
        ##     print np.sum(np.abs(data))
        ##     ## if np.sum(F) == 0.0:
        ##     ##     print audio_data_cut[j-1], audio_data_cut[j], audio_data_cut[j+1]

        ## sys.exit()


        print "============================"
        print audio_time_cut.shape
        print audio_data_cut.shape
        print "============================"
        plot_audio(audio_time_cut, audio_data_cut, chunk=CHUNK, rate=RATE, title=names[i])
        
        ## cut_coff = int(float(len(audio_time_cut))/float(len(ft_time_list[i])))
        ## for j, sample in audio_data_cut:

        ##     audio_freq = np.fft.fftfreq(self.CHUNK, self.UNIT_SAMPLE_TIME) 
        ##     audio_amp = np.fft.fft(audio_data / float(self.MAX_INT)) 
            
        ##     downSample(sample)

        

        ## import scipy as scp
        ## new_audio_data = scp.signal.resample(np.array(audio_data_cut).flatten(), 1000)

        ## pp.figure()
        ## pp.plot(new_audio_data)
        ## pp.show()

        
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
    

    ## ## Scaling or resample
    ## import mlpy
    
    ## dist, cost, path = mlpy.dtw_std(ft_force_list[ft_min_idx][2], ft_force_list[1][2], dist_only=False)

    ## print path

    ## import matplotlib.pyplot as plt
    ## import matplotlib.cm as cm
    ## fig = plt.figure(1)
    ## ax = fig.add_subplot(111)
    ## plot1 = plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    ## plot2 = plt.plot(path[0], path[1], 'w')
    ## xlim = ax.set_xlim((-0.5, cost.shape[0]-0.5))
    ## ylim = ax.set_ylim((-0.5, cost.shape[1]-0.5))
    ## plt.show()
    

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

    ax = pp.subplot(414)            
    f = np.arange(1, 10) * 1000
    for i, data in enumerate(data_list):        

        new_data = np.hstack([data/max_int, np.zeros(len(data))]) # zero padding
        fft = np.fft.fft(new_data)  # FFT          
        fftr=10*np.log10(abs(fft.real))[:len(new_data)/2]
        freq=np.fft.fftfreq(np.arange(len(new_data)).shape[-1])[:len(new_data)/2]
        
        print fftr.shape, freq.shape

        #count bin
        
        

        
    #========== RMS =========================
    ## ax2 = pp.subplot(412)    
    ## rms_list = []
    ## for i, data in enumerate(data_list):
    ##     rms_list.append(get_rms(data))
    ## t = np.arange(0.0, len(data_list), 1.0)*chunk/rate    
    ## pp.plot(t, rms_list) 

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


def get_rms(frame, MAX_INT=32768.0):
    
    count = len(frame)
    return  np.linalg.norm(frame/MAX_INT) / np.sqrt(float(count))

    

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--abnormal', '--an', action='store_true', dest='bAbnormal',
                 default=False, help='Renew pickle files.')
    opt, args = p.parse_args()


    data_path = os.environ['HRLBASEPATH']+'/src/projects/anomaly/test_data/'
    nMaxStep  = 36 # total step of data. It should be automatically assigned...

    task = 0
    if task == 1:
        prefix = 'microwave'
    elif task == 2:        
        prefix = 'down'
    elif task == 3:        
        prefix = 'lock'
    else:
        prefix = 'close'
    
    # Load data
    pkl_file = "./all_data.pkl"
    
    if os.path.isfile(pkl_file) and opt.bRenew is False:
        d = ut.load_pickle(pkl_file)
    else:
        d = load_data(data_path, prefix, normal_only=(not opt.bAbnormal))
        ut.save_pickle(d, pkl_file)
    

    # Cutting
    d = cutting(d)
    ## scaling(d)
    

    # Learning
    


    # TEST
