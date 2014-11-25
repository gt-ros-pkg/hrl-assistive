#!/usr/local/bin/python

import sys, os, copy
import numpy as np, math
import glob
import socket

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters

# Util
import hrl_lib.util as ut
import matplotlib.pyplot as pp
import matplotlib as mpl

import hrl_anomaly_detection.mechanism_analyse_daehyung as mad
import hrl_anomaly_detection.advait.mechanism_analyse_RAM as mar
from learning_hmm import learning_hmm
from anomaly_checker import anomaly_checker
import door_open_common as doc
import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
import hrl_lib.matplotlib_util as mpu

roc_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_roc_data'

def get_data(pkl_file, mech_class='Office Cabinet', verbose=False, renew=False):

    ######################################################    
    # Get Training Data
    if os.path.isfile(pkl_file) and renew==False:
        print "Saved pickle found"
        data = ut.load_pickle(pkl_file)
        data_vecs = data['data_vecs']
        data_mech = data['data_mech']
        data_chunks = data['data_chunks']
    else:        
        print "No saved pickle found"        
        data_vecs, data_mech, data_chunks = mad.get_all_blocked_detection()
        data = {}
        data['data_vecs'] = data_vecs
        data['data_mech'] = data_mech
        data['data_chunks'] = data_chunks
        ut.save_pickle(data,pkl_file)

    ## from collections import OrderedDict
    ## print list(OrderedDict.fromkeys(data_mech)), len(list(OrderedDict.fromkeys(data_mech)))
    ## print list(OrderedDict.fromkeys(data_chunks)), len(list(OrderedDict.fromkeys(data_chunks)))
    ## print len(data_mech), data_vecs.shape
    ## sys.exit()
    ## print data_mech
    
    # Filtering
    idxs = np.where([mech_class in i for i in data_mech])[0].tolist()
    print "Load ", mech_class
    
    ## print data_mech
    ## print data_vecs.shape, np.array(data_mech).shape, np.array(data_chunks).shape
    data_vecs = data_vecs[:,idxs]
    data_mech = [data_mech[i] for i in idxs]
    data_chunks = [data_chunks[i] for i in idxs]

    ## X data
    data_vecs = np.array([data_vecs.T]) # category x number_of_data x profile_length
    data_vecs[0] = mad.approx_missing_value(data_vecs[0])    

    ## ## time step data
    ## m, n = data_vecs[0].shape
    ## aXData = np.array([np.arange(0.0,float(n)-0.0001,1.0).tolist()] * m)

    if verbose==True:
        print data_vecs.shape, np.array(data_mech).shape, np.array(data_chunks).shape
    
    return data_vecs, data_mech, data_chunks

    
def get_interp_data(x,y):

    # Cubic-spline interpolation
    from scipy import interpolate
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(x[0], x[-1], 0.25)
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew


def get_init_param(mech_class='Office Cabinet'):

    A=None        
    B=None
    pi=None    
    nState=None
    
    if mech_class=='Office Cabinet':
        print "load Office Cabinet"
        nState = 21
        B= np.array([[6.74608512, 3.35549131],  
                     [12.31993548,  2.38471423],   
                     [5.88693723,   2.46635188],  
                     [12.14891174,  2.93535919],   
                     [9.26237564,   0.822744],
                     [9.75125194,   3.97297654],   
                     [5.77729007,   0.7507263], 
                     [11.43058112,  2.87005899],  
                     [18.06107565,  2.39662467],   
                     [6.19268309,   2.26957599],
                     [15.82898597,  2.63266457],  
                     [12.31750394,  1.7914311],    
                     [5.16760403,   3.05843564],  
                     [15.19103332,  2.0700275],   
                     [14.46836571,  2.45861045],
                     [16.9665216,   2.26536577],   
                     [4.53682151,   2.34544588],   
                     [3.72332427,   3.42523012],   
                     [2.49712096,   2.04864712],   
                     [1.92725332,   1.45080684],
                     [19.14819314,  3.03628879]])
        
    ## elif mech_class=='Kitchen Cabinet':
    ##     nState = 19
    ##     B= np.array([[18.58312854,   3.97148752],
    ##                  [8.02421244,   2.53093271],   
    ##                  [4.22877542,   2.43305639],   
    ##                  [3.06682069,   1.22338961],   
    ##                  [0.8969525,    0.90515595],
    ##                  [1.96895872,   0.40562971],   
    ##                  [0.7455034,    0.40224727],   
    ##                  [5.85727348,   0.14408786],   
    ##                  [1.38835362,   0.98968569],   
    ##                  [2.97388944,   0.56633029],
    ##                  [0.88473839,   1.07680861],   
    ##                  [0.72355969,   1.07902071],  
    ##                  [1.06439733,   1.55401679],   
    ##                  [0.31953582,   1.09563166],   
    ##                  [3.63555268,   0.42373409],
    ##                  [2.01966014,   0.71553896],   
    ##                  [0.78481239,   0.17447195],   
    ##                  [2.15617073,   0.78075969],   
    ##                  [2.97842764,   1.05199494]])

    elif mech_class=='Freezer':
        nState = 19
        B= np.array([[18.58312854,   3.97148752],   
                     [8.02421244,   2.53093271],   
                     [4.22877542,   2.43305639],   
                     [3.06682069,   1.22338961],   
                     [0.8969525,    0.90515595],
                     [1.96895872,   0.40562971],   
                     [0.7455034,    0.40224727],   
                     [5.85727348,   0.14408786],   
                     [1.38835362,   0.98968569],   
                     [2.97388944,   0.56633029],
                     [0.88473839,   1.07680861],   
                     [0.72355969,   1.07902071],   
                     [1.06439733,   1.55401679],  
                     [0.31953582,   1.09563166],   
                     [3.63555268,   0.42373409],
                     [2.01966014,   0.71553896],   
                     [0.78481239,   0.17447195],   
                     [2.15617073,   0.78075969],   
                     [2.97842764,   1.05199494]])

    elif mech_class=='Fridge':
        nState = 31
        B= np.array([[17.55486023,  3.65713984],
                     [18.22998446,  3.64747577],
                     [15.86584935,  3.98602125],
                     [10.07188122,  3.25049983], 
                     [8.0921412,   3.40126467],    
                     [4.96405642,  3.46917105], 
                     [3.61047634,  3.24121805], 
                     [1.74283831,   2.75573126], 
                     [2.55587326,  2.80451476], 
                     [0.25480083,  2.3808652],
                     [4.04989768,  1.18629104], 
                     [2.48367472,  1.90527102], 
                     [0.37727605,  1.40430661], 
                     [2.66604068,  0.61319149], 
                     [0.76780143,  1.74494431],
                     [2.17628331,  2.22347328], 
                     [1.76746579,  1.66254031], 
                     [1.68953164,  1.14549621], 
                     [1.648831,    1.36946679], 
                     [3.8295474,   1.23186153],
                     [3.85246291,  1.3042204],  
                     [0.38940239,  1.62819136], 
                     [1.87882352,  1.06820492], 
                     [0.89935806,  1.03536701], 
                     [2.7160048,   1.96402042],
                     [1.31125561,  1.03713852], 
                     [1.36717773,  1.46653763], 
                     [1.95822406,  1.99598714], 
                     [1.63074714,  2.22753852], 
                     [1.83764532,  1.32454358],
                     [3.07627057,  1.64749328]])                        

    else:
        print "No initial parameters"
            
    return A, B, pi, nState
    

def generate_roc_curve(mech_vec_list, mech_nm_list,                        
                       nFutureStep, fObsrvResol,
                       semantic_range = np.arange(0.2, 2.7, 0.3),
                       target_class=['Freezer','Fridge','Office Cabinet'],
                       bPlot=False, roc_root_path=roc_data_path,
                       semantic_label='PHMM anomaly detection w/ known mechanisum class', 
                       sem_c='r',sem_m='*'):

    start_step = 2
    
    t_nm_list, t_mech_vec_list = [], []
    for i, nm in enumerate(mech_nm_list):
        ## print 'nm:', nm
        if 'known' in nm:
            continue
        t_nm_list.append(nm)
        t_mech_vec_list.append(mech_vec_list[i])

    data, _ = mar.create_blocked_dataset_semantic_classes(t_mech_vec_list, t_nm_list, append_robot = False)

    ## thresh_dict = ut.load_pickle('blocked_thresh_dict.pkl')
    ## mean_charlie_dict = thresh_dict['mean_charlie']
    ## mean_known_mech_dict = thresh_dict['mean_known_mech']
 
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
        data_vecs, _, _ = get_data(pkl_file, mech_class=mech_class, renew=opt.renew) # human data
        A, B, pi, nState = get_init_param(mech_class=mech_class)        

        print "-------------------------------"
        print "Mech class: ", mech_class
        print "Data size: ", np.array(data_vecs).shape
        print "-------------------------------"
                
        # Training 
        lh = learning_hmm(data_path=os.getcwd(), aXData=data_vecs[0], nState=nState, 
                          nMaxStep=nMaxStep, nFutureStep=nFutureStep, 
                          fObsrvResol=fObsrvResol, nCurrentStep=nCurrentStep)    

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
        pp.plot(fp_list, mn_list, '-'+sem_m+sem_c, label= semantic_label,
                mec=sem_c, ms=6, mew=2)
        ## pp.plot(fp_list, mn_list, '--'+sem_m, label= semantic_label,
        ##         ms=6, mew=2)
        mpu.legend()
        
    
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
                 default=False, help='Approximately compute the distribution of multi-step observations')
    p.add_option('--block', '--b', action='store_true', dest='bUseBlockData',
                 default=False, help='Use blocked data')
    p.add_option('--animation', '--ani', action='store_true', dest='bAnimation',
                 default=False, help='Plot by time using animation')
    p.add_option('--fig_roc_robot', '--roc', action='store_true', dest='bROCRobot',
                 default=False, help='Plot roc curve wrt robot data')
    p.add_option('--fig_roc_plot', '--plot', action='store_true', dest='bROCPlot',
                 default=False, help='Plot roc curve wrt robot data')
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
    nCurrentStep = 4  #14


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
    data_vecs, _, _ = get_data(pkl_file, mech_class=cls, renew=opt.renew) # human data       
    A, B, pi, nState = get_init_param(mech_class=cls)        

    ######################################################    
    # Training 
    lh = learning_hmm(data_path=os.getcwd(), aXData=data_vecs[0], nState=nState, 
                      nMaxStep=nMaxStep, nFutureStep=nFutureStep, 
                      fObsrvResol=fObsrvResol, nCurrentStep=nCurrentStep)    

    if opt.bCrossVal:
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
        h_ftan   = data_vecs[0][12,:].tolist()
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

            elapsed.append(time.clock() - start_time)        
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

    elif opt.bROCRobot:
        pkl_list = glob.glob(data_path+'RAM_db/robot_trials/simulate_perception/*.pkl')
        s_range = np.arange(0.05, 5.0, 0.3) 
        m_range = np.arange(0.1, 3.8, 0.6)

        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)

        ## mpu.set_figure_size(13, 7.)
        if opt.bROCPlot:        
            pp.figure()        
            mar.generate_roc_curve(mech_vec_list, mech_nm_list,
                               s_range, m_range, sem_c='c', sem_m='^',
                               semantic_label = 'operating 1st time with \n uncertainty in state estimation', plot_prev=False)

        #--------------------------------------------------------------------------------
        
        pkl_list = glob.glob(data_path+'RAM_db/robot_trials/perfect_perception/*.pkl')
        s_range = np.arange(0.05, 3.8, 0.2) 
        m_range = np.arange(0.1, 3.8, 0.6)        
        
        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)

        # advait
        if opt.bROCPlot:
            mad.generate_roc_curve(mech_vec_list, mech_nm_list,
                                    s_range, m_range, sem_c='b',
                                    semantic_label = 'operating 1st time with \n accurate state estimation',
                                    plot_prev=True)

        #--------------------------------------------------------------------------------
        # Set the default color cycle
        import itertools
        colors = itertools.cycle(['g', 'm', 'c', 'k'])
        shapes = itertools.cycle(['x','v', 'o', '+'])
        ## mpl.rcParams['axes.color_cycle'] = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
        ## pp.gca().set_color_cycle(['r', 'g', 'b', 'y', 'm', 'c', 'k'])
        
        ## for i in xrange(1,9,3):
        for i in [1,2,4,8]:
            color = colors.next()
            shape = shapes.next()
            roc_root_path = roc_data_path+'_'+str(i)
            generate_roc_curve(mech_vec_list, mech_nm_list, \
                               nFutureStep=i,fObsrvResol=fObsrvResol,
                               semantic_range = np.arange(0.2, 2.7, 0.3), bPlot=opt.bROCPlot,
                               roc_root_path=roc_root_path, semantic_label=str(i)+' step PHMM', 
                               sem_c=color,sem_m=shape)
        ## mad.generate_roc_curve(mech_vec_list, mech_nm_list)
        
        if opt.bROCPlot: 
            ## pp.ylim(0.,13)
            ## pp.xlim(-0.5,45)
            pp.xlim(-0.5,27)
            pp.ylim(0.,5)
            pp.savefig('robot_roc.pdf')
            pp.show()
                
            

    elif opt.bAllPlot:

        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    
        lh.init_plot()            
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










