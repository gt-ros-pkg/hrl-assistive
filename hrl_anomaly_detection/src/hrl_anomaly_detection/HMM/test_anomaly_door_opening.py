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

import hrl_anomaly_detection.mechanism_analyse_daehyung as mad
import hrl_anomaly_detection.advait.mechanism_analyse_RAM as mar
from learning_hmm import learning_hmm
from anomaly_checker import anomaly_checker
import door_open_common as doc

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
    print "----------------"
    print idxs
    print "----------------"
    
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


def get_init_param(nState, mech_class='Office Cabinet'):

    A=None        
    B=None
    pi=None    
    
    if mech_class=='Office Cabinet':
        if nState == 27:
            B= np.array([[  0.65460858,  3.75559027],
                [ 19.14088894,  3.00439256],
                [ 16.67292067,  2.59510363],
                [  9.30149164,  3.35532136],
                [ 10.20360709,  0.82194885],
                [ 11.57887639,  2.50314713],
                [ 11.97284535,  1.83956526],
                [ 10.54898096,  0.12454761],
                [  3.85245744,  2.39909837],
                [ 15.07665305,  3.68764635],
                [  7.88338746,  2.09772464],
                [  5.50403149,  0.34692303],
                [ 15.99616779,  1.2888493 ],
                [  8.889266,    3.65985848],
                [ 14.66218247,  3.66735491],
                [  7.37560428,  3.09516763],
                [  6.67233779,  1.6771348 ],
                [  7.21410401,  1.58022427],
                [ 16.52255113,  3.90849502],
                [ 15.64859841,  1.03275153],
                [  5.43240626,  2.9589057 ],
                [ 18.13599286,  2.89700896],
                [  4.0560603,   0.49477993],
                [ 12.90839232,  1.90842489],
                [  4.39049378,  3.83603433],
                [ 15.00135479,  0.55290679],
                [ 10.97389832,  2.40767146]])
        elif nState == 21:
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
        
    elif mech_class=='Kitchen Cabinet':
        if nState == 33:
            B= np.array([[5.82609791,  0.22797547], 
                         [1.35068022,  0.38903556], 
                         [5.88140016,  1.05072438], 
                         [2.75491195,  0.63561952], 
                         [6.95285334,  0.34320559],
                         [7.59079115,  1.91095634], 
                         [0.44844863,  0.38284611], 
                         [4.91412235,  1.83300524],
                         [4.49126044,  2.06662935], 
                         [0.99333182,  1.4167013],
                         [0.9026897 ,  0.82453133], 
                         [2.89052113,  1.05436995], 
                         [6.58255477,  1.13670999], 
                         [6.13579192,  1.28408747],
                         [3.29649674,  0.9556620],
                         [2.79467913,  0.77709517], 
                         [5.23910711,  0.39687226], 
                         [0.74639724,  1.3998018 ],
                         [1.07876437,  2.5017018],  
                         [5.43190763,  0.62575458],
                         [3.03965904,  0.28233228], 
                         [7.75936811,  0.2907158],  
                         [9.67230225,  3.11645626], 
                         [0.14301478,  1.76283024], 
                         [3.64895918,  0.1262195],
                         [0.28216467,  0.25779701], 
                         [5.454912 ,   1.01494105], 
                         [1.80125584,  2.67710306],
                         [13.96819518, 0.55123172], 
                         [1.04622689,  0.3311333],
                         [4.19770017,  0.62637557], 
                         [6.44035373,  0.66963928], 
                         [4.01059659,  1.22099346]])

            
    return A, B, pi
    

def generate_roc_curve(mech_vec_list, mech_nm_list,                        
                       nState, nFutureStep, fObsrvResol,
                       semantic_range = np.arange(0.2, 2.7, 0.3),
                       target_class=['Freezer','Fridge','Kitchen Cabinet','Office Cabinet'],
                       bPlot=False):

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
        trials = trials[0:1,:36]

        pkl_file  = "mech_class_"+doc.class_dir_list[idx]+".pkl"        
        data_vecs, _, _ = get_data(pkl_file, mech_class=mech_class, renew=opt.renew)        
        A, B, pi = get_init_param(nState, mech_class=mech_class)        

        print "-------------------------------"
        print "Mech class: ", mech_class
        print "Data size: ", np.array(data_vecs).shape
        print "-------------------------------"
                
        # Training 
        lh = learning_hmm(data_path=os.getcwd(), aXData=data_vecs[0], nState=nState, 
                          nMaxStep=nMaxStep, nFutureStep=nFutureStep, 
                          fObsrvResol=fObsrvResol, nCurrentStep=nCurrentStep, 
                          step_size_list=step_size_list)    

        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)                
        
        mn_list = []
        fp_list, err_list = [], []        
        
        for n in semantic_range:
            print "n: ", n

            # check saved file
            target_pkl = roc_data_path+'/'+'fp_'+doc.class_dir_list[idx]+'_n_'+str(n)+'.pkl'

            # mutex file
            host_name = socket.gethostname()
            mutex_file = roc_data_path+'/'+host_name+'_mutex_'+doc.class_dir_list[idx]+'_'+str(n)
            
            if os.path.isfile(target_pkl) == False and os.path.isfile(mutex_file) == False: 
            
                os.system('touch '+mutex_file)

                # Init variables
                false_pos = np.zeros((len(trials), len(trials[0])-start_step))
                tot = trials.shape[0] * trials.shape[1]
                err_l = []

                # Gives all profiles
                for i, trial in enumerate(trials):

                    # Init checker
                    ac = anomaly_checker(lh, sig_coff=n)

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

                            print "(",i,j, ") : ", false_pos[i, j-start_step], max_err

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
                mn_list.append(np.mean(np.array(err_l)))
                
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
            
    mn_list = np.mean(np.row_stack(mn_l_l), 0).tolist() # means into a row
    fp_list = np.mean(np.row_stack(fp_l_l), 0).tolist()

    if bPlot:
        sem_c = 'b'
        sem_m = '+'
        semantic_label='PHMM anomaly detection w/ known mechanisum class'
        pp.plot(fp_list, mn_list, '--'+sem_m+sem_c, label= semantic_label,
                mec=sem_c, ms=8, mew=2)
    
    
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
    nState    = 19
    nMaxStep  = 36 # total step of data. It should be automatically assigned...
    nFutureStep = 8
    ## data_column_idx = 1
    fObsrvResol = 0.1
    nCurrentStep = 8  #14


    # for block test
    nClass = 2
    cls = doc.class_list[nClass]
    ## mech = 'kitchen_cabinet_pr2'
    ## mech = 'kitchen_cabinet_cody'
    ## mech = 'ikea_cabinet_pr2'
    
    pkl_file  = "mech_class_"+doc.class_dir_list[nClass]+".pkl"    
    step_size_list = None


    
    if step_size_list != None and (len(step_size_list) !=nState 
                                   or sum(step_size_list) != nMaxStep):
        print len(step_size_list), " : ", sum(step_size_list)
        sys.exit()

    data_vecs, _, _ = get_data(pkl_file, mech_class=cls, renew=opt.renew)        
    A, B, pi = get_init_param(nState, mech_class=cls)        

    ######################################################    
    # Training 
    lh = learning_hmm(data_path=os.getcwd(), aXData=data_vecs[0], nState=nState, 
                      nMaxStep=nMaxStep, nFutureStep=nFutureStep, 
                      fObsrvResol=fObsrvResol, nCurrentStep=nCurrentStep, 
                      step_size_list=step_size_list)    

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
        h_config, h_ftan = mad.get_a_blocked_detection(mech, ang_interval=0.25)
        h_config =  np.array(h_config)*180.0/3.14

        x_test = h_ftan[:nCurrentStep]
        x_test_next = h_ftan[nCurrentStep:nCurrentStep+lh.nFutureStep]
        x_test_all  = h_ftan
                
        if opt.bAnimation:

            print type(h_config), type(h_ftan)
            ## x,y = get_interp_data(h_config, h_ftan)
            x,y = h_config, h_ftan
            ac = anomaly_checker(lh)
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
        r_pkls = mar.filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = mar.pkls_to_mech_vec_list(r_pkls, 36)

        if opt.bROCPlot: pp.figure()        
        generate_roc_curve(mech_vec_list, mech_nm_list, \
                           nState=nState, nFutureStep=nFutureStep,fObsrvResol=fObsrvResol,
                           semantic_range = np.arange(2.6, 2.7, 0.3), bPlot=opt.bROCPlot)
        mad.generate_roc_curve()
        if opt.bROCPlot: pp.show()
            

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










