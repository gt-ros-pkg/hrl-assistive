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


def fig_roc_human(cross_data_path, aXData1, aXData2, chunks, labels, prefix, bPlot=False):

    # For parallel computing
    strMachine = socket.gethostname()+"_"+str(os.getpid())    
    nState     = 20
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
    nfs    = NFoldPartitioner(cvtype=1) # 1-fold ?
    spl    = splitters.Splitter(attr='partitions')
    splits = [list(spl.generate(x)) for x in nfs.generate(dataSet)] # split by chunk

    threshold_mult = np.arange(0.05, 1.2, 0.05)    
    count = 0
    for ths in threshold_mult:
    
        # save file name
        res_file = prefix+'_roc_human_'+'ths_'+str(ths)+'.pkl'
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

        for l_wdata, l_vdata in splits:
            fp, err = anomaly_check(l_wdata, l_vdata, nState, trans_type, ths)
            print fp, err

        print "aaaaaaaaaaaaaaa"
        
        n_jobs = 4
        r = Parallel(n_jobs=n_jobs)(delayed(anomaly_check)(l_wdata, l_vdata, nState, trans_type, ths) \
                                    for l_wdata, l_vdata in splits) 
        fp_ll, err_ll = zip(*r)

        print fp_ll, err_ll, ths        
        import operator
        fp_l = reduce(operator.add, fp_ll)
        err_l = reduce(operator.add, err_ll)
        print fp_l, err_l
        sys.exit()
        
        d = {}
        d['fp']  = np.mean(fp_l)
        if err_l == []:         
            d['err'] = 0.0
        else:
            d['err'] = np.mean(err_l)
        
        ut.save_pickle(d,res_file)        
        os.system('rm '+mutex_file)
        print "-----------------------------------------------"

    if count == len(threshold_mult) and bPlot:
        print "#############################################################################"
        print "All file exist "
        print "#############################################################################"        

        fp_l = []
        err_l = []
        for ths in threshold_mult:
            res_file   = prefix+'_roc_human_'+'ths_'+str(ths)+'.pkl'
            res_file   = os.path.join(cross_test_path, res_file)

            d = ut.load_pickle(pkl_file)
            fp  = d['fp'] 
            err = d['err']         

            fp_l.append(fp)
            err_l.append(err)
        
            fp_l  = np.array(fp_l)*100.0
            sem_c = 'b'
            sem_m = '+'
            semantic_label='likelihood detection \n with known mechanism class'

            pp.figure()
            pp.plot(fp_l, err_l, '--'+sem_m+sem_c, label= semantic_label, mec=sem_c, ms=8, mew=2)
            pp.xlabel('False positive rate (percentage)')
            pp.ylabel('Mean excess likelihood')    
            pp.xlim([0, 30])
            pp.show()
                            
    return

def anomaly_check(l_wdata, l_vdata, nState, trans_type, ths):

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
        for j in range(2,4,1):
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

def plot_all(data1, data2):

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
    for i, d in enumerate(data1):
        pp.plot(d)
        
    pp.subplot(212)
    for i, d in enumerate(data2):
        pp.plot(d)
    pp.show()
    
    


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--abnormal', '--an', action='store_true', dest='bAbnormal',
                 default=False, help='Renew pickle files.')
    p.add_option('--animation', '--ani', action='store_true', dest='bAnimation',
                 default=False, help='Plot by time using animation')
    p.add_option('--roc_human', '--rh', action='store_true', dest='bRocHuman',
                 default=False, help='Plot by a figure of ROC human')
    p.add_option('--all_plot', '--all', action='store_true', dest='bAllPlot',
                 default=False, help='Plot all data')
    opt, args = p.parse_args()


    data_path = os.environ['HRLBASEPATH']+'/src/projects/anomaly/test_data/'

    task = 1
    if task == 1:
        prefix = 'microwave_black'
    elif task == 2:        
        prefix = 'door'
    elif task == 3:        
        prefix = 'lock'
    else:
        prefix = 'close'
    
    # Load data
    pkl_file = "./"+prefix+"_data.pkl"
    
    if os.path.isfile(pkl_file) and opt.bRenew is False:
        d = ut.load_pickle(pkl_file)
    else:
        d = dm.load_data(data_path, prefix, normal_only=(not opt.bAbnormal))
        d = dm.cutting(d)        
        ut.save_pickle(d, pkl_file)

    #
    aXData1  = d['ft_force_mag_l']
    aXData2  = d['audio_rms_l'] 
    labels   = d['labels']
    chunks   = d['chunks'] 

    
    #---------------------------------------------------------------------------
    if opt.bAllPlot:

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
       
        plot_all(aXData1_scaled, aXData2_scaled)

    #---------------------------------------------------------------------------           
    elif opt.bRocHuman:

        cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/multi_'+prefix
        
        fig_roc_human(cross_data_path, aXData1, aXData2, chunks, labels, prefix)

    
    #---------------------------------------------------------------------------   
    elif opt.bAnimation:

        nState   = 20
        trans_type= "left_right"
        ## nMaxStep = 36 # total step of data. It should be automatically assigned...

        # Learning
        from hrl_anomaly_detection.HMM.learning_hmm_multi import learning_hmm_multi
        lhm = learning_hmm_multi(nState=nState, trans_type=trans_type)
        lhm.fit(aXData1_scaled, aXData2_scaled)

        
        lhm.simulation(aXData1_scaled[2], aXData2_scaled[2])

    #---------------------------------------------------------------------------           
    else:

        nState   = 20
        trans_type= "left_right"
        ## nMaxStep = 36 # total step of data. It should be automatically assigned...

        # Learning
        lhm = learning_hmm_multi(nState=nState, trans_type=trans_type)
        lhm.fit(aXData1_scaled, aXData2_scaled)
        
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
