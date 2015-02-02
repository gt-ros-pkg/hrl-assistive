#!/usr/bin/python

import sys, os, copy
import numpy as np, math
import glob
import socket
import time
import random 
import scipy as scp
from scipy import interpolate       
import mlpy
from sklearn import preprocessing

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

# Util
import hrl_lib.util as ut
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
from hrl_anomaly_detection.HMM.anomaly_checker import anomaly_checker

def load_data(data_path, prefix, normal_only=True):

    pkl_list = glob.glob(data_path+prefix+'*.pkl')
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

    ft_time_list   = []
    ft_force_list  = []
    ft_torque_list = []
    ft_force_mag_list = []
    audio_time_list = []
    audio_data_list = []
    audio_amp_list = []
    audio_freq_list = []
    audio_chunk_list = []
    audio_rms_list = []

    hmm_input_l = []    

    MAX_INT = 32768.0
    CHUNK   = 1024 #frame per buffer
    RATE    = 44100 #sampling rate

    
    #------------------------------------------
    # Get reference data
    for i, force in enumerate(ft_force_l):
        if labels[i] is False: continue
        else: 
            ref_idx = i
            break
        
    ft_time   = ft_time_l[ref_idx]
    ft_force  = ft_force_l[ref_idx]
    ft_torque = ft_torque_l[ref_idx]        
    ft_force_mag = np.linalg.norm(ft_force,axis=0)
                
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

    ft_time_cut      = np.array(ft_time[idx_start:idx_end])
    ft_force_cut     = ft_force[:,idx_start:idx_end]
    ft_torque_cut    = ft_torque[:,idx_start:idx_end]
    ft_force_mag_cut = np.linalg.norm(ft_force_cut, axis=0)            

    #----------------------------------------------------
    start_time = ft_time[idx_start]
    end_time   = ft_time[idx_end]
    audio_time = audio_time_l[ref_idx]
    audio_data = audio_data_l[ref_idx]

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

    print "==================================="
    print ft_force_mag_ref.shape,audio_rms_ref.shape 
    print "==================================="

    
    # DTW wrt the reference
    for i, force in enumerate(ft_force_l):

        if ref_idx == i:
            print "its reference"
            ft_force_mag_list.append(ft_force_mag_ref)                
            audio_rms_list.append(audio_rms_ref)
            hmm_input_l.append(np.vstack([ft_force_mag_ref, audio_rms_ref]))            
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
             
        
        # Compare with reference
        dist, cost, path = mlpy.dtw_std(ft_force_mag_ref, ft_force_mag_cut, dist_only=False)
        ## fig = plt.figure(1)
        ## ax = fig.add_subplot(111)
        ## plot1 = plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
        ## plot2 = plt.plot(path[0], path[1], 'w')
        ## xlim = ax.set_xlim((-0.5, cost.shape[0]-0.5))
        ## ylim = ax.set_ylim((-0.5, cost.shape[1]-0.5))
        ## plt.show()


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


            
        ## dist, cost, path = mlpy.dtw_std(ft_force_mag_ref, ft_force_mag_cut_dtw, dist_only=False)
        ## fig = plt.figure(1)
        ## ax = fig.add_subplot(111)
        ## plot1 = plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
        ## plot2 = plt.plot(path[0], path[1], 'w')
        ## xlim = ax.set_xlim((-0.5, cost.shape[0]-0.5))
        ## ylim = ax.set_ylim((-0.5, cost.shape[1]-0.5))
        ## plt.show()
                    
        print "==================================="
        print len(ft_force_mag_cut_dtw), len(audio_rms_cut_dtw)
        print "==================================="
        
        ft_force_mag_list.append(ft_force_mag_cut_dtw)                
        audio_rms_list.append(audio_rms_cut_dtw)
        hmm_input_l.append(np.vstack([ft_force_mag_cut_dtw, audio_rms_cut_dtw]))
       
    d = {}
    d['ft_force_mag_l'] = np.array(ft_force_mag_list)
    d['audio_rms_l']    = np.array(audio_rms_list)
    d['hmm_input_l']    = np.array(hmm_input_l)

    return d
    


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

def plot_all(data):

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
    pp.subplot(211)
    for i, d in enumerate(data):
        pp.plot(d[0])
        
    pp.subplot(212)
    for i, d in enumerate(data):
        pp.plot(d[1])
    pp.show()
    
    

def get_rms(frame, MAX_INT=32768.0):
    
    count = len(frame)
    return  np.linalg.norm(frame/MAX_INT) / np.sqrt(float(count))


def scaling(X):
    '''        
    '''

    ## X_scaled = preprocessing.scale(np.array(X))

    min_c = np.min(X)
    max_c = np.max(X)
    X_scaled = (X-min_c) / (max_c-min_c) * 5.0

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
    

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--abnormal', '--an', action='store_true', dest='bAbnormal',
                 default=False, help='Renew pickle files.')
    p.add_option('--animation', '--ani', action='store_true', dest='bAnimation',
                 default=False, help='Plot by time using animation')
    opt, args = p.parse_args()


    data_path = os.environ['HRLBASEPATH']+'/src/projects/anomaly/test_data/'

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
    pkl_file = "./"+prefix+"_data.pkl"
    
    if os.path.isfile(pkl_file) and opt.bRenew is False:
        d = ut.load_pickle(pkl_file)
    else:
        d = load_data(data_path, prefix, normal_only=(not opt.bAbnormal))
        d = cutting(d)        
        ut.save_pickle(d, pkl_file)

    ## plot_all(d['hmm_input_l'])        
    ## aXData   = d['hmm_input_l']
    aXData1  = d['ft_force_mag_l']
    aXData2  = d['audio_rms_l'] 

    # Mvg filtering
    ## aXData1_avg = movingaverage(aXData1, 5)
    ## aXData2_avg = movingaverage(aXData2, 5)    
    aXData1_avg = aXData1
    aXData2_avg = aXData2
    
    # min max scaling
    aXData1_scaled, min_c1, max_c1 = scaling(aXData1_avg)
    aXData2_scaled, min_c2, max_c2 = scaling(aXData2_avg)    
    ## print min_c1, max_c1, np.min(aXData1_scaled), np.max(aXData1_scaled)
    ## print min_c2, max_c2, np.min(aXData2_scaled), np.max(aXData2_scaled)
    
    nState   = 20
    trans_type= "left_right"
    ## nMaxStep = 36 # total step of data. It should be automatically assigned...
            
    # Learning
    from hrl_anomaly_detection.HMM.learning_hmm_multi import learning_hmm_multi
    lhm = learning_hmm_multi(nState=nState, trans_type=trans_type)
    lhm.fit(aXData1_scaled, aXData2_scaled)
    print "----------------------------"

    if opt.bAnimation:

        lhm.simulation(aXData1_scaled[0], aXData2_scaled[0])
        
    else:
        # TEST
        nCurrentStep = 37
        ## X_test1 = aXData1_scaled[0:1,:nCurrentStep]
        ## X_test2 = aXData2_scaled[0:1,:nCurrentStep]
        X_test1 = aXData1_scaled[0:1]
        X_test2 = aXData2_scaled[0:1]

        #
        lhm.init_plot()
        lhm.data_plot(X_test1, X_test2, color = 'r')

        X_test1[0,50] = 2.7
        X_test1[0,51] = 2.7
        X_test = lhm.convert_sequence(X_test1, X_test2, emission=False)
        print lhm.likelihood(X_test), lhm.likelihood_avg
        ## mu, cov = self.predict(X_test)
        lhm.data_plot(X_test1, X_test2, color = 'b')

        
        lhm.final_plot()
