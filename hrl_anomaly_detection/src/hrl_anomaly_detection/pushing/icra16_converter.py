#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system
import os, sys, copy
import glob

# util
import numpy as np
import scipy
import hrl_lib.util as ut

from scipy import interpolate       

from mvpa2.datasets.base import Dataset
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters


def convertData(data_path, task_name, new_data_path, f_zero, f_thres, audio_thres):

    # load_data
    d = load_data(data_path, task_name, normal_only=False)

    # cutting
    d = cutting_for_robot(d, f_zero_size=f_zero, f_thres=f_thres, \
                          audio_thres=audio_thres, dtw_flag=False)        

    true_aXData1 = d['ft_force_mag_true_l']
    true_aXData2 = d['audio_rms_true_l'] 
    true_chunks  = d['true_chunks']

    false_aXData1 = d['ft_force_mag_false_l']
    false_aXData2 = d['audio_rms_false_l'] 
    false_chunks  = d['false_chunks']

    print "All: ", len(true_aXData1)+len(false_aXData1), \
      " Success: ", len(true_aXData1), \
      " Failure: ", len(false_aXData1)

    if os.path.isdir(new_data_path) is False:
        os.system('mkdir -p '+new_data_path)

    ## target_data_path = os.path.join(new_data_path, task_name)
    ## pkl_list = glob.glob(data_path+'*_'+task_name+'*.pkl')
    ## pkl_list = glob.glob(data_path+'*.pkl')

    for idx in xrange(2):
        if idx == 0:
            aXData1 = true_aXData1
            aXData2 = true_aXData2
            chunks  = true_chunks
        else:
            aXData1 = false_aXData1
            aXData2 = false_aXData2
            chunks  = false_chunks

        count = 0
        for force_mag, audio_rms, chunk in zip(aXData1, aXData2, chunks):
            count += 1

            ft_time  = range( 0, len(force_mag) )
            audio_time = range( 0, len(audio_rms) )

            d = {}
            d['init_time']         = 0.0
            d['kinematics_time']   = audio_time
            d['kinematics_ee_pos'] = np.zeros( (3, len(d['kinematics_time'])) ) 
            d['kinematics_ee_quat']= np.zeros( (4, len(d['kinematics_time'])) )
            d['kinematics_target_pos'] = np.zeros( (3, len(d['kinematics_time'])) ) 
            d['kinematics_target_quat']= np.zeros( (4, len(d['kinematics_time'])) )
            d['kinematics_jnt_pos']= np.zeros( (7, len(d['kinematics_time'])) )

            d['audio_time'] = audio_time
            d['audio_azimuth']  = [0.0] * len(d['audio_time'])
            d['audio_power']    = audio_rms

            d['ft_time']  = ft_time
            d['ft_force'] = force_mag

            if idx == 0:
                # success
                file_name = os.path.join(new_data_path, 'gatsbii_'+task_name+'_robot_success_'+str(count)+\
                                         '.pkl' )
            else:
                # failure
                file_name = os.path.join(new_data_path, 'gatsbii_'+task_name+'_robot_failure_'+str(count)+\
                                         '.pkl' )

            print file_name, len(ft_time)
            ut.save_pickle(d,file_name)
                    
    return 


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
    audio_time   = audio_time_l[ref_idx]
    audio_data   = audio_data_l[ref_idx]    

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
    print ft_force_mag_cut.shape, audio_rms_ref_cut.shape 
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

    
def get_rms(frame, MAX_INT=32768.0):
    
    count = len(frame)
    return  np.linalg.norm(frame/MAX_INT) / np.sqrt(float(count))

    

if __name__ == '__main__':


    all_task_names  = ['microwave_black', 'microwave_white', 'lab_cabinet', 'wallsw', 'switch_device', \
                       'switch_outlet', 'lock_wipes', 'lock_huggies', 'toaster_white', 'glass_case']

    class_num = 0
    task  = 2

    if class_num == 0:
        class_name = 'door'
        task_names = ['microwave_black', 'microwave_white', 'lab_cabinet']
        f_zero_size = [8, 5, 10]
        f_thres     = [1.0, 1.7, 3.0]
        audio_thres = [1.0, 1.0, 1.0]
    elif class_num == 1: 
        class_name = 'switch'
        task_names = ['wallsw', 'switch_device', 'switch_outlet']
        f_zero_size = [5, 10, 7]
        f_thres     = [0.7, 0.8, 1.0]
        audio_thres = [1.0, 0.7, 0.0015]
    elif class_num == 2:        
        class_name = 'lock'
        task_names = ['case', 'lock_wipes', 'lock_huggies']
        f_zero_size = [5, 5, 5]
        f_thres     = [0.7, 1.0, 1.35]
        audio_thres = [1.0, 1.0, 1.0]
    elif class_num == 3:        
        class_name = 'complex'
        task_names = ['toaster_white', 'glass_case']
        f_zero_size = [5, 3, 8]
        f_thres     = [0.8, 1.5, 1.35]
        audio_thres = [1., 1.0, 1.0]
    elif class_num == 4:        
        class_name = 'button'
        task_names = ['joystick', 'keyboard']
        f_zero_size = [5, 5, 8]
        f_thres     = [1.35, 1.35, 1.35]
        audio_thres = [1.0, 1.0, 1.0]
    else:
        print "Please specify right task."
        sys.exit()

    
    # Load data
    data_path = '/home/dpark/svn/robot1/src/projects/anomaly/test_data/robot_20150213/'+class_name+'/'
    new_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/TRO2016/gatsbii_pushing_'+task_names[task]+'/'

    convertData(data_path, task_names[task], new_data_path, \
                f_zero=f_zero_size[task], \
                f_thres=f_thres[task],\
                audio_thres=audio_thres[task])

