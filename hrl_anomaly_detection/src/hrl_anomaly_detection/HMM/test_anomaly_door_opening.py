#!/usr/local/bin/python

import sys, os, copy
import numpy as np, math

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

# Util
import hrl_lib.util as ut

import hrl_anomaly_detection.mechanism_analyse_daehyung as mad
from learning_hmm import learning_hmm

def get_data(pkl_file, verbose=False):

    ######################################################    
    # Get Training Data
    if os.path.isfile(pkl_file):
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
        
    # Filtering
    idxs = np.where(['Office Cabinet' in i for i in data_mech])[0].tolist()

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
    
def get_init_param(nState):

    if nState == 25:
        A=[[ 0.590,  0.180,  0.000,  0.194,  0.010,  0.000,  0.000,  0.012,  0.013,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.372,  0.628,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.720,  0.004,  0.000,  0.160,  0.115,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.584,  0.416,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.879,  0.121,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.684,  0.316,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.526,  0.464,  0.010,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.441,  0.554,  0.005,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.446,  0.542,  0.012,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.228,  0.738,  0.029,  0.000,  0.004,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.199,  0.705,  0.088,  0.000,  0.008,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.277,  0.622,  0.101,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.232,  0.557,  0.161,  0.051,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.149,  0.693,  0.136,  0.016,  0.000,  0.006,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.413,  0.480,  0.107,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.233,  0.600,  0.045,  0.090,  0.033,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.365,  0.540,  0.017,  0.078,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.458,  0.492,  0.050,  0.000,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.451,  0.525,  0.024,  0.000,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.801,  0.198,  0.000,  0.002,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.620,  0.380,  0.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.258,  0.207,  0.477,  0.057 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.000,  0.000 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.707,  0.293 ],
           [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000 ]]
    else:
        A=None

        
    B=None
    pi=None    

    return A, B, pi
    
    
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
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out everything')
    opt, args = p.parse_args()

    ## Init variables    
    data_path = os.getcwd()
    nState    = 11
    nMaxStep  = 36 # total step of data. It should be automatically assigned...
    pkl_file  = "door_opening_data.pkl"    
    nFutureStep = 1
    ## data_column_idx = 1
    fObsrvResol = 0.1
    nCurrentStep = 14  #14

    step_size_list = None

    if step_size_list != None and (len(step_size_list) !=nState 
                                   or sum(step_size_list) != nMaxStep):
        print len(step_size_list), " : ", sum(step_size_list)
        sys.exit()
        
    data_vecs, _, _ = get_data(pkl_file)        
    A, B, pi = get_init_param(nState)        


    ## # TEMP
    ## data_vecs = data_vecs[:][:40,:]

    
    ######################################################    
    # Training 
    lh = learning_hmm(data_path=data_path, aXData=data_vecs[0], nState=nState, 
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
        save_file = os.path.join('/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune',
                                 host_name+'_'+str(t[0])+str(t[1])+str(t[2])+'_'
                                 +str(t[3])+str(t[4])+'.pkl')
        
        lh.param_optimization(save_file=save_file)
        
    elif opt.bUseBlockData:
        
        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    

        ######################################################    
        # Test data
        h_config, h_ftan = mad.get_a_blocked_detection()
        print np.array(h_config)*180.0/3.14
        print len(h_ftan)

        x_test = h_ftan[:nCurrentStep]
        x_test_next = h_ftan[nCurrentStep:nCurrentStep+lh.nFutureStep]
        x_test_all  = h_ftan

        lh.init_plot()            
                
        if opt.bAnimation:
            lh.animated_path_plot(x_test_all)
        
        elif opt.bApproxObsrv:
            x_pred, x_pred_prob = lh.multi_step_approximated_predict(x_test,
                                                                     full_step=True, 
                                                                     verbose=opt.bVerbose)
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), 
                                    x_pred_prob, np.array(x_test_next), 
                                    X_test_all=x_test_all)
            lh.final_plot()
        else:               
            x_pred, x_pred_prob = lh.multi_step_predict(x_test, verbose=opt.bVerbose)
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), 
                                    x_pred_prob, np.array(x_test_next), 
                                    X_test_all=x_test_all)
            lh.final_plot()
        
            
        
    elif opt.bApproxObsrv:
        if lh.nFutureStep <= 1: print "Error: nFutureStep should be over than 2."
        
        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    

        for i in xrange(18,31,2):
            
            x_test      = data_vecs[0][i,:nCurrentStep].tolist()
            x_test_next = data_vecs[0][i,nCurrentStep:nCurrentStep+lh.nFutureStep].tolist()
            x_test_all  = data_vecs[0][i,:].tolist()

            x_pred, x_pred_prob = lh.multi_step_approximated_predict(x_test, 
                                                                     full_step=True, 
                                                                     verbose=opt.bVerbose)
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), x_pred_prob, np.array(x_test_next))
            lh.final_plot()
        
    else:
        lh.fit(lh.aXData, A=A, B=B, verbose=opt.bVerbose)    
        ## lh.path_plot(data_vecs[0], data_vecs[0,:,3])


        for i in xrange(18,31,2):
            
            x_test      = data_vecs[0][i,:nCurrentStep].tolist()
            x_test_next = data_vecs[0][i,nCurrentStep:nCurrentStep+lh.nFutureStep].tolist()
            x_test_all  = data_vecs[0][i,:].tolist()
                
            x_pred, x_pred_prob = lh.multi_step_predict(x_test, verbose=opt.bVerbose)
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










