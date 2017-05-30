#!/usr/bin/python

import sys, os, copy
import numpy as np, math
# Machine learning library
## import mlpy
import scipy as scp
from scipy import interpolate       
from sklearn import preprocessing
# Util
import hrl_lib.util as ut
import matplotlib
import matplotlib.pyplot as pp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from itertools import product

from mvpa2.datasets.base import Dataset


tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
tableau20 = np.array(tableau20)/255.0

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



def loadData(pkl_file, data_path, task_name, f_zero_size, f_thres, audio_thres, cross_data_path=None, 
             an_type=None, force_an=None, sound_an=None, bRenew=False, rFold=None, nDataSet=None, 
             delete=False):

    if os.path.isfile(pkl_file) and bRenew is False:
        d = ut.load_pickle(pkl_file)
    else:
        print "Not connected!!!!!!!"
        sys.exit()
        ## d = load_data(data_path, task_name, normal_only=False)
        ## d = cutting_for_robot(d, f_zero_size=f_zero_size, f_thres=f_thres, \
        ##                          audio_thres=audio_thres, dtw_flag=False)        
        ## ut.save_pickle(d, pkl_file)

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

        if delete==True and False:
            os.system('rm -rf '+cross_data_path)            
            
        if os.path.isdir(cross_data_path) == False and False:
            os.system('mkdir -p '+cross_data_path)
            
        for i in xrange(nDataSet):

            pkl_file = os.path.join(cross_data_path, "dataSet_"+str(i))
            dd = None
            if os.path.isfile(pkl_file) is False:
                print "aaaaaaaaaaaaa"
                sys.exit()

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

def create_mvpa_dataset(aXData1, aXData2, chunks, labels):

    feat_list = []
    for x1, x2, chunk in zip(aXData1, aXData2, chunks):
        feat_list.append([x1, x2])

    data = Dataset(samples=feat_list)
    data.sa['id']      = range(0,len(labels))
    data.sa['chunks']  = chunks
    data.sa['targets'] = labels

    return data


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



def plot_all(data1, data2, false_data1=None, false_data2=None, labels=None, distribution=False, freq=43.0,
             skip_idx=[]):


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
    for i, d in enumerate(data1):
        if i in skip_idx: continue
        if not(labels is not None and labels[i] == False):
            true_line, = pp.plot(x, d, label='Normal data')

    ax1.set_ylabel("Force Magnitude [N]", fontsize=18)

    #-----------------------------------------------------------------
    ax2 = pp.subplot(212)
    for i, d in enumerate(data2):
        if i in skip_idx: continue
        
        if not(labels is not None and labels[i] == False):
            pp.plot(x, d)

    ax2.set_ylabel("Sound Energy", fontsize=18)
    ax2.set_xlabel("Time step [sec]", fontsize=18)


    ax1.set_xlim([0,3.5])
    ax2.set_xlim([0,3.5])
    fig.savefig('test.pdf')
    fig.savefig('test.png')    
    fig.savefig('test.eps')    
    os.system('cp test.p* ~/Dropbox/HRL/')
    os.system('cp test.eps ~/Dropbox/HRL/')
    pp.show()
    



if __name__ == '__main__':

    cross_root_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/Humanoids2015/robot'
    all_task_names  = ['microwave_black', 'microwave_white', 'lab_cabinet', 'wallsw', 'switch_device', \
                       'switch_outlet', 'lock_wipes', 'lock_huggies', 'toaster_white', 'glass_case']
                       
    class_num = 0 #int(opt.nClass)
    task  = 0 #int(opt.nTask)


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
        f_zero_size = [5, 10, 7]
        f_thres     = [0.7, 0.8, 1.0]
        audio_thres = [1.0, 0.7, 0.0015]
        cov_mult = [[10.0, 10.0, 10.0, 10.0],[50.0, 50.0, 50.0, 50.0],[10.0, 10.0, 10.0, 10.0]]
        nState_l    = [10, 20, 20]
    elif class_num == 2:        
        class_name = 'lock'
        task_names = ['case', 'lock_wipes', 'lock_huggies']
        f_zero_size = [5, 5, 5]
        f_thres     = [0.7, 1.0, 1.35]
        audio_thres = [1.0, 1.0, 1.0]
        cov_mult = [[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0],[10.0, 10.0, 10.0, 10.0]]
        nState_l    = [20, 10, 20]
    elif class_num == 3:        
        class_name = 'complex'
        task_names = ['toaster_white', 'glass_case']
        f_zero_size = [5, 3, 8]
        f_thres     = [0.8, 1.5, 1.35]
        audio_thres = [1., 1.0, 1.0]
        cov_mult    = [[10.0, 10.0, 10.0, 10.0],[20.0, 20.0, 20.0, 20.0],[10.0, 10.0, 10.0, 10.0]]
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
    data_path = '/home/dpark/svn/robot1/src/projects/anomaly/test_data/robot_20150213/'+class_name+'/'
    ## data_path = os.environ['HRLBASEPATH']+'/src/projects/anomaly/test_data/robot_20150213/'+class_name+'/'



    if True: #opt.bAllPlot:

        print "Visualization of all sequence"
        true_aXData1, true_aXData2, true_chunks, false_aXData1, false_aXData2, false_chunks, nDataSet \
          = loadData(pkl_file, data_path, task_names[task], f_zero_size[task], f_thres[task], \
                        audio_thres[task])
        
        true_aXData1_scaled, min_c1, max_c1 = scaling(true_aXData1, scale=scale)
        true_aXData2_scaled, min_c2, max_c2 = scaling(true_aXData2, scale=scale)    

        ## skip_idx = [0,1,2,3,4,5,6]
        skip_idx = [2,12, 29,19,24] # +range(24,26)
        plot_all(true_aXData1, true_aXData2, skip_idx=skip_idx)            

