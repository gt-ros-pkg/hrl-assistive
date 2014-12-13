#!/usr/local/bin/python

import sys, os, copy
import numpy as np, math
import glob
import socket
import time

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters

# Util
import hrl_lib.util as ut
import matplotlib.pyplot as pp
import matplotlib as mpl

import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad
import hrl_anomaly_detection.advait.mechanism_analyse_RAM as mar
from hrl_anomaly_detection.HMM.learning_hmm import learning_hmm
from hrl_anomaly_detection.HMM.anomaly_checker import anomaly_checker
import hrl_anomaly_detection.door_opening.door_open_common as doc
import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
import hrl_lib.matplotlib_util as mpu

roc_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/roc_sig_0_3/door_roc_data'
    
def get_interp_data(x,y):

    # Cubic-spline interpolation
    from scipy import interpolate
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(x[0], x[-1], 0.25)
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew   

def generate_roc_curve(mech_vec_list, mech_nm_list,                        
                       nFutureStep, fObsrvResol,
                       semantic_range = np.arange(0.2, 2.7, 0.3),
                       target_class=['Freezer','Fridge','Office Cabinet'],
                       bPlot=False, roc_root_path=roc_data_path,
                       semantic_label='PHMM anomaly detection w/ known mechanisum class', 
                       sem_l='-',sem_c='r',sem_m='*', trans_type="left_right"):

    start_step = 2
    
    t_nm_list, t_mech_vec_list = [], []
    for i, nm in enumerate(mech_nm_list):
        ## print 'nm:', nm
        if 'known' in nm:
            continue
        t_nm_list.append(nm)
        t_mech_vec_list.append(mech_vec_list[i])

    data, _ = mar.create_blocked_dataset_semantic_classes(t_mech_vec_list, t_nm_list, append_robot = False)

 
    #---------------- semantic class prior -------------
    # init containers
    fp_l_l = []
    mn_l_l = []
    err_l_l = []
    mech_fp_l_l = []
    mech_mn_l_l = []
    mech_err_l_l = []

    # splitter
    nfs = NFoldPartitioner(cvtype=1, attr='targets') # 1-fold ?
    label_splitter = splitters.Splitter(attr='partitions')            
    splits = [list(label_splitter.generate(x)) for x in nfs.generate(data)]            

    X_test = np.arange(0.0, 36.0, 1.0)

    # Run by class
    for l_wdata, l_vdata in splits: #label_splitter(data):

        mech_class = l_vdata.targets[0]
        trials = l_vdata.samples # all data
    
        # check existence of computed result
        idx = doc.class_list.index(mech_class)        
        if mech_class not in target_class: continue
        ## elif os.path.isfile('roc_'+doc.class_dir_list[idx]+'.pkl'): continue
        ## elif os.path.isfile('roc_'+doc.class_dir_list[idx]+'_complete'): continue
        
        # cutting into the same length
        trials = trials[:,:36]

        pkl_file  = "mech_class_"+doc.class_dir_list[idx]+".pkl"        
        data_vecs, data_mech, data_chunks = mad.get_data(pkl_file, mech_class=mech_class, renew=opt.renew) # human data
        A, B, pi, nState = doc.get_hmm_init_param(mech_class=mech_class)        

        print "-------------------------------"
        print "Mech class: ", mech_class
        print "Data size: ", np.array(data_vecs).shape
        print "-------------------------------"
        
        # Training 
        lh = learning_hmm(data_path=os.getcwd(), aXData=data_vecs[0], nState=nState, 
                          nMaxStep=nMaxStep, nFutureStep=nFutureStep, 
                          fObsrvResol=fObsrvResol, nCurrentStep=nCurrentStep, trans_type=trans_type)    

        lh.fit(lh.aXData, A=A, B=B, pi=pi, verbose=opt.bVerbose)                
        
        mn_list = []
        fp_list, err_list = [], []        
        
        for n in semantic_range:
            print "n: ", n

            if os.path.isdir(roc_root_path) == False:
                os.system('mkdir -p '+roc_root_path)

            # check saved file
            target_pkl = roc_root_path+'/'+'fp_'+doc.class_dir_list[idx]+'_n_'+str(n)+'.pkl'

            # mutex file
            host_name = socket.gethostname()
            mutex_file = roc_root_path+'/'+host_name+'_mutex_'+doc.class_dir_list[idx]+'_'+str(n)
                        
            if os.path.isfile(target_pkl) == False \
                and hcu.is_file(roc_root_path, 'mutex_'+doc.class_dir_list[idx]+'_'+str(n)) == False: 
            
                os.system('touch '+mutex_file)

                # Init variables
                false_pos = np.zeros((len(trials), len(trials[0])-start_step))
                tot = trials.shape[0] * trials.shape[1]
                err_l = []
                    
                # Gives all profiles
                for i, trial in enumerate(trials):

                    # Init checker
                    ac = anomaly_checker(lh, sig_mult=n)

                    # Simulate each profile
                    for j in xrange(len(trial)):
                        # Update buffer
                        ac.update_buffer(X_test[:j+1], trial[:j+1])

                        if j>= start_step:                    
                            # check anomaly score
                            bFlag, fScore, max_err = ac.check_anomaly(trial[j])
                            if bFlag: 
                                false_pos[i, j-start_step] = 1.0 
                            else:
                                err_l.append(max_err)

                            print "(",i,"/",len(trials)," ",j, ") : ", false_pos[i, j-start_step], max_err

                # save data & remove mutex file
                d = {}
                d['false_pos'] = false_pos
                d['tot'] = tot
                d['err_l'] = err_l
                d['n'] = n
                ut.save_pickle(d, target_pkl)
                os.system('rm '+mutex_file)

            elif os.path.isfile(target_pkl) == True:
                
                d = ut.load_pickle(target_pkl)
                false_pos = d['false_pos']
                tot   = d['tot']  
                err_l = d['err_l']  
                n     = d['n']  
                
                fp_list.append(np.sum(false_pos)/(tot*0.01))
                err_list.append(err_l)
                ## mn_list.append(np.mean(np.array(err_l)))

                tot_e = 0.0
                tot_e_cnt = 0.0
                for e in err_l:
                    if np.isnan(e) == False:
                        tot_e += e 
                        tot_e_cnt += 1.0
                mn_list.append(tot_e/tot_e_cnt)
                
            else:
                print "Mutex exists"
                continue
                
        fp_l_l.append(fp_list)
        err_l_l.append(err_list)
        mn_l_l.append(mn_list)
        
    ## ll = [[] for i in err_l_l[0]]  # why 0?
    ## for i,e in enumerate(err_l_l): # labels
    ##     for j,l in enumerate(ll):  # multiplier range
    ##         l.append(e[j])

    if bPlot:
        mn_list = np.mean(np.row_stack(mn_l_l), 0).tolist() # means into a row
        fp_list = np.mean(np.row_stack(fp_l_l), 0).tolist()                
        pp.plot(fp_list, mn_list, sem_l+sem_m+sem_c, label= semantic_label,
                mec=sem_c, ms=6, mew=2)
        ## pp.plot(fp_list, mn_list, '--'+sem_m, label= semantic_label,
        ##         ms=6, mew=2)
        ## pp.legend(loc='best',prop={'size':16})
        pp.legend(loc=1,prop={'size':14})

def genCrossValData(data_path, cross_data_path):

    save_file = os.path.join(cross_data_path, 'data_1.pkl')        
    if os.path.isfile(save_file) is True: return
    
    # human and robot data
    pkl_list = glob.glob(data_path+'RAM_db/*_new.pkl') + glob.glob(data_path+'RAM_db/robot_trials/perfect_perception/*_new.pkl') + glob.glob(data_path+'RAM_db/robot_trials/simulate_perception/*_new.pkl')

    r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
    mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36) #get vec_list, name_list

    # data consists of (mech_vec_matrix?, label_string(Freezer...), mech_name)
    data, _ = mar.create_blocked_dataset_semantic_classes(mech_vec_list,
                                                      mech_nm_list, append_robot = True)    

    # create the generator
    #label_splitter = NFoldSplitter(cvtype=1, attr='labels')
    nfs = NFoldPartitioner(cvtype=1) # 1-fold ?
    spl = splitters.Splitter(attr='partitions')
    splits = [list(spl.generate(x)) for x in nfs.generate(data)] # split by chunk

    cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_human_cross_data'
    os.system('mkdir -p '+cross_data_path)        
    d = {}
    count = 0
    for l_wdata, l_vdata in splits:

        if 'robot' in l_vdata.chunks[0]: 
            print "Pass a robot chunk"
            continue

        count += 1
        non_robot_idxs = np.where(['robot' not in i for i in l_wdata.chunks])[0] # if there is no robot, true 
        idxs = np.where(l_wdata.targets[non_robot_idxs] == l_vdata.targets[0])[0] # find same target samples in non_robot target samples

        train_trials = (l_wdata.samples[non_robot_idxs])[idxs]
        test_trials  = l_vdata.samples
        chunk = l_vdata.chunks[0]
        target = l_vdata.targets[0]

        #SAVE!!
        d['train_trials'] = train_trials
        d['test_trials'] = test_trials
        d['chunk'] = chunk
        d['target'] = target
        save_file = os.path.join(cross_data_path, 'data_'+str(count)+'.pkl')
        ut.save_pickle(d, save_file)

        
def tuneCrossValHMM(cross_data_path, cross_test_path, nState, nMaxStep, fObsrvResol=0.1, trans_type="left_right"):

    if not(os.path.isdir(cross_test_path+'/'+str(nState))):
        os.system('mkdir -p '+cross_test_path+'/'+str(nState)) 
        time.sleep(0.5)
    elif hcu.is_file(cross_test_path+'/'+str(nState), 'complete'):
        print "#############################################################################"
        print "All file exist "
        print "#############################################################################"        
        return

    ## Load data pickle
    train_data = []
    test_data = []
    test_idx_list = []        
    for f in os.listdir(cross_data_path):
        if f.endswith(".pkl"):
            test_num = f.split('_')[-1].split('.')[0]
            
            # Load data
            d = ut.load_pickle( os.path.join(cross_data_path,f) )
            train_trials = d['train_trials']
            test_trials  = d['test_trials']
            chunk        = d['chunk'] 
            target       = d['target']

            test_idx_list.append(test_num)
            train_data.append(train_trials)
            test_data.append(test_trials)

            
    ####################################################################

    count = 0
    for i in xrange(len(train_data)):

        B_tune_pkl = cross_test_path+'/'+str(nState)+'/B_tune_data_'+str(test_idx_list[i])+'.pkl'
        mutex_file = cross_test_path+'/'+str(nState)+'/running_'+str(test_idx_list[i])+'_'+strMachine+'.txt'                 
        if os.path.isfile(B_tune_pkl): 
            count += 1
            continue
        elif hcu.is_file(cross_test_path+'/'+str(nState), 'running_'+str(test_idx_list[i])): 
            print "#############################################################################"
            print "Another machine Is Running already, ignore this : " , nState
            print "#############################################################################"
            continue
        else:
            os.system('touch '+mutex_file)
        
        lh = learning_hmm(data_path=os.getcwd(), aXData=train_data[i], nState=nState, 
                          nMaxStep=nMaxStep, fObsrvResol=fObsrvResol, trans_type=trans_type)            
        lh.param_optimization(save_file=B_tune_pkl)

        os.system('rm '+mutex_file)

    if count == len(train_data):
        print "#############################################################################"
        print "All file exist "
        print "#############################################################################"        
        os.system('touch complete.txt')
        

def get_threshold_by_cost(cross_data_path, cross_test_path, alpha, beta, nMaxStep, fObsrvResol, trans_type):


    return [idx_l, a_l, b_l]

        
## last_x = 0    
## last_score = 0
## def cross_val_score(self, x, *args):

##     train_data = args[0]
##     test_data  = args[1]
##     nState     = args[2]
##     nMaxStep   = args[3]
##     fObsrvResol= args[4]
##     trans_type = args[5]
##     B_upper    = args[6]
##     B_lower    = args[7]

##     global last_x
##     global last_score
    
##     # check limit
##     if last_x is None or np.linalg.norm(last_x-x) > 0.05:
##         tmax = bool(np.all(x <= B_upper))
##         tmin = bool(np.all(x >= B_lower))
##         if tmax and tmin == False: return 5            
##         last_x = x
##     else:
##         return last_score
            
##     B=x.reshape((nState,2))

    
##     total_score = 0.0
##     for i in xrange(len(train_data)):
        
##         # Training 
##         lh = learning_hmm(data_path=os.getcwd(), aXData=train_data[i], nState=nState, 
##                           nMaxStep=nMaxStep, 
##                           fObsrvResol=fObsrvResol, trans_type=trans_type)    

##         A, _, pi, _ = doc.get_hmm_init_param(mech_class=cls)               
##         lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    

        
##         total_score += lh.score(test_data[i])
    

##     return total_score
    

    
    
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='renew',
                 default=False, help='Renew pickle files.')
    p.add_option('--cross_val', '--cv', action='store_true', dest='bCrossVal',
                 default=False, help='N-fold cross validation for parameter')
    p.add_option('--optimize_mv', '--mv', action='store_true', dest='bOptMeanVar',
                 default=False, help='Optimize mean and vars for B matrix')
    p.add_option('--approx_pred', '--ap', action='store_true', dest='bApproxObsrv',
                 default=True, help='Approximately compute the distribution of multi-step observations')
    p.add_option('--block', '--b', action='store_true', dest='bUseBlockData',
                 default=False, help='Use blocked data')
    p.add_option('--animation', '--ani', action='store_true', dest='bAnimation',
                 default=False, help='Plot by time using animation')
    p.add_option('--fig_roc_human', action='store_true', dest='bROCHuman',
                 help='generate ROC like curve from the BIOROB dataset.')
    p.add_option('--fig_roc_robot', action='store_true', dest='bROCRobot',
                 default=False, help='Plot roc curve wrt robot data')
    p.add_option('--fig_roc_plot', '--plot', action='store_true', dest='bROCPlot',
                 default=False, help='Plot roc curve wrt robot data')
    p.add_option('--fig_roc_phmm_comp_plot', '--pc_plot', action='store_true', dest='bROCPHMMPlot',
                 default=False, help='Plot phmm comparison roc curve wrt robot data')
    p.add_option('--all_path_plot', '--all', action='store_true', dest='bAllPlot',
                 default=False, help='Plot all paths')
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out everything')
    opt, args = p.parse_args()

    ## Init variables
    ## data_path = os.environ['HRLBASEPATH']+'_data/usr/advait/ram_www/data_from_robot_trials/'
    data_path = os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook/'
    root_path = os.environ['HRLBASEPATH']+'/'
    ## nState    = 19
    nMaxStep  = 36 # total step of data. It should be automatically assigned...
    nFutureStep = 8
    ## data_column_idx = 1
    fObsrvResol = 0.1
    nCurrentStep = 14  #14
    trans_type = "left_right"


    # for block test
    if opt.bUseBlockData:    
        nClass = 2
        cls = doc.class_list[nClass]
        mech = 'kitchen_cabinet_pr2'
        ## mech = 'kitchen_cabinet_cody'
        ## mech = 'ikea_cabinet_pr2'
    else:
        nClass = 2
        cls = doc.class_list[nClass]

    
    pkl_file  = "mech_class_"+doc.class_dir_list[nClass]+".pkl"      
    data_vecs, _, _ = mad.get_data(pkl_file, mech_class=cls, renew=opt.renew) # human data       
    A, B, pi, nState = doc.get_hmm_init_param(mech_class=cls)        
    
    ######################################################    
    # Training 
    lh = learning_hmm(data_path=os.getcwd(), aXData=data_vecs[0], nState=nState, 
                      nMaxStep=nMaxStep, nFutureStep=nFutureStep, 
                      fObsrvResol=fObsrvResol, nCurrentStep=nCurrentStep, trans_type=trans_type)    


    if opt.bCrossVal and False:
        print "------------- Cross Validation -------------"

        # Save file name
        import socket, time
        host_name = socket.gethostname()
        t=time.gmtime()                
        save_file = os.path.join('/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune',
                                 host_name+'_'+str(t[0])+str(t[1])+str(t[2])+'_'
                                 +str(t[3])+str(t[4])+'.pkl')

        # Random step size
        step_size_list_set = []
        for i in xrange(300):
            step_size_list = [1] * lh.nState
            while sum(step_size_list)!=lh.nMaxStep:
                ## idx = int(random.gauss(float(lh.nState)/2.0,float(lh.nState)/2.0/2.0))
                idx = int(random.randrange(0, lh.nState, 1))
                
                if idx < 0 or idx >= lh.nState: 
                    continue
                else:
                    step_size_list[idx] += 1                
            step_size_list_set.append(step_size_list)                    

        
        ## tuned_parameters = [{'nState': [20,25,30,35], 'nFutureStep': [1], 
        ##                      'fObsrvResol': [0.05,0.1,0.15,0.2,0.25], 'nCurrentStep': [5,10,15,20,25]}]
        tuned_parameters = [{'nState': [lh.nState], 'nFutureStep': [1], 
                             'fObsrvResol': [0.05,0.1,0.15,0.2], 'step_size_list': step_size_list_set}]        

        ## tuned_parameters = [{'nState': [20,30], 'nFutureStep': [1], 'fObsrvResol': [0.1]}]
        lh.param_estimation(tuned_parameters, 10, save_file=save_file)

    elif opt.bCrossVal:
        print "------------- Cross Validation -------------"
        # 1) Get data (Human)
        # 2) Split by class
        # 3) k-fold
        # 4) Find a set of parameters (B,n) for HMM about training set
        # 5) Find a d1 for AD about training set
        # 6) AD Test about test data
        
        cross_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_human_cross_data'
        cross_test_path = os.path.join(cross_data_path,'human_'+trans_type)        
        strMachine = socket.gethostname()
        
        genCrossValData(data_path, cross_data_path)

        # optimization                
        for nState in xrange(10,35,1):        
            tuneCrossValHMM(cross_data_path, cross_test_path, nState, nMaxStep, fObsrvResol, trans_type)

        # Search best a and b + Get ROC data
        alphas = np.arange(0.0, 0.6+0.00001, 0.2)
        betas = np.arange(0.2, 0.6+0.00001, 0.2)

        ## fp_list = []
        ## mn_list = []
        ## err_list = []
        ## for alpha in alphas:
        ##     for beta in betas:

        ##         [idx_l, a_l, b_l] = get_threshold_by_cost(cross_data_path, cross_test_path, alpha, beta, nMaxStep, fObsrvResol, trans_type)
                               
        ##         [err, fp] = generate_roc_data_by_cost(cross_data_path, cross_test_path, idx_l, a_l, b_l, nMaxStep, fObsrvResol, trans_type)
                
        ##         fp_list.append(fp)
        ##         err_list.append(err)
        ##         ## mn_list.append(mn_list)

        ## # save data?
            
                
            
        
        
    elif opt.bOptMeanVar:
        print "------------- Optimize B matrix -------------"

        # Save file name
        import socket, time
        host_name = socket.gethostname()
        t=time.gmtime()                
        os.system('mkdir -p /home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune_'+doc.class_dir_list[nClass])
        save_file = os.path.join('/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune_'+doc.class_dir_list[nClass],
                                 host_name+'_'+str(t[0])+str(t[1])+str(t[2])+'_'
                                 +str(t[3])+str(t[4])+str(t[5])+'.pkl')

        lh.param_optimization(save_file=save_file)
        
    elif opt.bUseBlockData:

        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    

        ######################################################    
        # Test data
        h_config, h_ftan = mad.get_a_blocked_detection(mech, ang_interval=0.25) # robot blocked test data
        h_config =  np.array(h_config)*180.0/3.14

        # Training data            
        h_ftan   = data_vecs[0][12,:].tolist() # ikea cabinet door openning data
        h_config = np.arange(0,float(len(h_ftan)), 1.0)

        x_test = h_ftan[:nCurrentStep]
        x_test_next = h_ftan[nCurrentStep:nCurrentStep+lh.nFutureStep]
        x_test_all  = h_ftan
                
        if opt.bAnimation:

            ## x,y = get_interp_data(h_config, h_ftan)
            x,y = h_config, h_ftan
            ac = anomaly_checker(lh, sig_mult=1.0)
            ac.simulation(x,y)
            
            ## lh.animated_path_plot(x_test_all, opt.bAniReload)
        
        elif opt.bApproxObsrv:
            import time
            start_time = time.clock()
            lh.init_plot(bAni=opt.bAnimation)            

            x_pred, x_pred_prob = lh.multi_step_approximated_predict(x_test,
                                                                     full_step=True, 
                                                                     verbose=opt.bVerbose)

            elapsed = []
            elapsed.append(time.clock() - start_time)

            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), 
                                    x_pred_prob, np.array(x_test_next), 
                                    X_test_all=x_test_all)

            elapsed.append(time.clock() - elapsed[-1])        
            print elapsed
            
            lh.final_plot()
        else:               
            lh.init_plot(bAni=opt.bAnimation)            
            x_pred, x_pred_prob = lh.multi_step_predict(x_test, verbose=opt.bVerbose)
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), 
                                    x_pred_prob, np.array(x_test_next), 
                                    X_test_all=x_test_all)
            lh.final_plot()
        
            
        
    elif opt.bApproxObsrv:
        if lh.nFutureStep <= 1: print "Error: nFutureStep should be over than 2."
        
        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    
        
        for i in xrange(1,31,2):
            
            x_test      = data_vecs[0][i,:nCurrentStep].tolist()
            x_test_next = data_vecs[0][i,nCurrentStep:nCurrentStep+lh.nFutureStep].tolist()
            x_test_all  = data_vecs[0][i,:].tolist()

            x_pred, x_pred_prob = lh.multi_step_approximated_predict(x_test, 
                                                                     full_step=True, 
                                                                     verbose=opt.bVerbose)
            
            lh.init_plot()            
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), x_pred_prob, np.array(x_test_next))
            lh.final_plot()

    elif opt.bROCHuman:
        pkl_list = glob.glob(data_path+'RAM_db/*_new.pkl')
        s_range = np.arange(0.05, 5.0, 0.3) 
        m_range = np.arange(0.1, 3.8, 0.6)
        
        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)

        ## mpu.set_figure_size(10, 7.)
        nFutureStep = 8
        cross_val_roc_curve
        generate_roc_curve(mech_vec_list, mech_nm_list, \
                           nFutureStep=nFutureStep,fObsrvResol=fObsrvResol,
                           semantic_range = np.arange(0.2, 2.7, 0.3), bPlot=opt.bROCPlot,
                           roc_root_path=roc_root_path, semantic_label=str(nFutureStep)+ \
                           ' step PHMM with \n accurate state estimation', 
                           sem_c=color,sem_m=shape)

        
        ## pp.figure()
        ## generate_roc_curve_no_prior(mech_vec_list, mech_nm_list)
        ## generate_roc_curve(mech_vec_list, mech_nm_list)
        ## f = pp.gcf()
        ## f.subplots_adjust(bottom=.15, top=.96, right=.98, left=0.15)
        ## pp.savefig('roc_compare.pdf')
        ## pp.show()
                
            
    elif opt.bROCRobot:
        pkl_list = glob.glob(data_path+'RAM_db/robot_trials/simulate_perception/*_new.pkl')
        s_range = np.arange(0.05, 5.0, 0.3) 
        m_range = np.arange(0.1, 3.8, 0.6)

        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)        
        
        ## mpu.set_figure_size(13, 7.)
        if opt.bROCPlot and False:        
            pp.figure()        
            mar.generate_roc_curve(mech_vec_list, mech_nm_list,
                               s_range, m_range, sem_c='c', sem_m='^',
                               ## semantic_label = 'operating 1st time with \n uncertainty in state estimation', plot_prev=False)
                               semantic_label = 'probabilistic model with \n uncertainty in state estimation', plot_prev=False)

        #--------------------------------------------------------------------------------
        
        pkl_list = glob.glob(data_path+'RAM_db/robot_trials/perfect_perception/*_new.pkl')
        s_range = np.arange(0.05, 3.8, 0.2) 
        m_range = np.arange(0.1, 3.8, 0.6)        
        
        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)
        
        # advait
        if opt.bROCPlot:
            mad.generate_roc_curve(mech_vec_list, mech_nm_list,
                                    s_range, m_range, sem_c='b',
                                    ## semantic_label = 'operating 1st time with \n accurate state estimation',
                                    semantic_label = 'probabilistic model with \n accurate state estimation',
                                    plot_prev=False)

        #--------------------------------------------------------------------------------
        # Set the default color cycle
        import itertools
        colors = itertools.cycle(['g', 'm', 'c', 'k'])
        shapes = itertools.cycle(['x','v', 'o', '+'])
        ## mpl.rcParams['axes.color_cycle'] = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
        ## pp.gca().set_color_cycle(['r', 'g', 'b', 'y', 'm', 'c', 'k'])
        
        ## for i in xrange(1,9,3):
        for i in [1,8]:
            color = colors.next()
            shape = shapes.next()
            roc_root_path = roc_data_path+'_'+str(i)
            generate_roc_curve(mech_vec_list, mech_nm_list, \
                               nFutureStep=i,fObsrvResol=fObsrvResol,
                               semantic_range = np.arange(0.2, 2.7, 0.3), bPlot=opt.bROCPlot,
                               roc_root_path=roc_root_path, semantic_label=str(i)+ \
                               ' step PHMM with \n accurate state estimation', 
                               sem_c=color,sem_m=shape)
        ## mad.generate_roc_curve(mech_vec_list, mech_nm_list)
        
        if opt.bROCPlot: 
            ## pp.xlim(-0.5,27)
            pp.xlim(-0.5,5)
            pp.ylim(0.,5)
            pp.savefig('robot_roc_sig_0_3.pdf')
            pp.show()
                

    elif opt.bROCPHMMPlot:
        pkl_list = glob.glob(data_path+'RAM_db/robot_trials/perfect_perception/*_new.pkl')
        s_range = np.arange(0.05, 3.8, 0.2) 
        m_range = np.arange(0.1, 3.8, 0.6)        
        
        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)

        ## mpu.set_figure_size(26, 14.)        
        pp.figure()        

        # Set the default color cycle
        import itertools
        colors = itertools.cycle(['r', 'g', 'b'])
        shapes = itertools.cycle(['x','v', 'o'])
        ## colors = itertools.cycle(['r', 'g', 'b', 'y'])
        ## shapes = itertools.cycle(['x','v', 'o', '+'])
        lines  = itertools.cycle(['--','-'])
        ## mpl.rcParams['axes.color_cycle'] = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
        ## pp.gca().set_color_cycle(['r', 'g', 'b', 'y', 'm', 'c', 'k'])

        sig_offs = [0, 3]
        for sig_off in sig_offs:
            roc_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/roc_sig_0_'+str(sig_off)+'/door_roc_data'
            line = lines.next()

            ## for i in xrange(1,9,3):
            for i in [1,4,8]:
                color = colors.next()
                shape = shapes.next()
                roc_root_path = roc_data_path+'_'+str(i)
                generate_roc_curve(mech_vec_list, mech_nm_list, \
                                   nFutureStep=i,fObsrvResol=fObsrvResol,
                                   semantic_range = np.arange(0.2, 2.7, 0.3), bPlot=True,
                                   roc_root_path=roc_root_path, semantic_label=str(i)+ \
                                   ' step PHMM with 0.'+str(sig_off)+' offset', 
                                   sem_l=line,sem_c=color,sem_m=shape)
            
        ## pp.xlim(-0.5,27)
        pp.xlim(-0.5,5)
        pp.ylim(0.,5)
        pp.legend(loc='best',prop={'size':14})
        pp.xlabel('False positive rate (percentage)', fontsize=22)
        pp.ylabel('Mean excess force (Newtons)', fontsize=22)
        pp.savefig('robot_roc_phmm_comp.pdf')
        pp.show()
            

    elif opt.bAllPlot:

        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    
        lh.all_path_plot(lh.aXData)
        lh.final_plot()
                
    else:
        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    
        ## lh.path_plot(data_vecs[0], data_vecs[0,:,3])

        for i in xrange(18,31,2):
            
            x_test      = data_vecs[0][i,:nCurrentStep].tolist()
            x_test_next = data_vecs[0][i,nCurrentStep:nCurrentStep+lh.nFutureStep].tolist()
            x_test_all  = data_vecs[0][i,:].tolist()
                
            x_pred, x_pred_prob = lh.multi_step_predict(x_test, verbose=opt.bVerbose)
            lh.init_plot()            
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), x_pred_prob, np.array(x_test_next))
            lh.final_plot()



    ## # Compute mean and std
    ## mu    = np.zeros((nMaxStep,1))
    ## sigma = np.zeros((nMaxStep,1))
    ## index = 0
    ## m_init = 0
    ## while (index < nMaxStep):
    ##     temp_vec = lh.aXData[:,(m_init):(m_init + 1)] 
    ##     m_init = m_init + 1

    ##     mu[index] = np.mean(temp_vec)
    ##     sigma[index] = np.std(temp_vec)
    ##     index = index+1

    ## for i in xrange(len(mu)):
    ##     print mu[i],sigma[i]

            
    ## print lh.A
    ## print lh.B
    ## print lh.pi
            
    ## print lh.mean_path_plot(lh.mu, lh.sigma)
        
    ## print x_test
    ## print x_test[-4:]

    ## fig = plt.figure(1)
    ## ax = fig.add_subplot(111)

    ## ax.plot(obsrv_range, future_prob)
    ## plt.show()










