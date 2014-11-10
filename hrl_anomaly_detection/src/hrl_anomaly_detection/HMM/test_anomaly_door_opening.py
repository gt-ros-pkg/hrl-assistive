#!/usr/local/bin/python

import sys, os, copy
import numpy as np, math

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

# Util
import hrl_lib.util as ut

import hrl_anomaly_detection.mechanism_analyse_daehyung as mad
from learning_hmm import learning_hmm
from anomaly_checker import anomaly_checker

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

    
def get_interp_data(x,y):

    # Cubic-spline interpolation
    from scipy import interpolate
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(x[0], x[-1], 0.25)
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew


def get_init_param(nState):

    if nState == 27:
        A=None
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
    p.add_option('--ani_reload', '--ar', action='store_true', dest='bAniReload',
                 default=False, help='Plot by time using animation')
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out everything')
    opt, args = p.parse_args()

    ## Init variables    
    data_path = os.getcwd()
    nState    = 27
    nMaxStep  = 36 # total step of data. It should be automatically assigned...
    pkl_file  = "door_opening_data.pkl"    
    nFutureStep = 8
    ## data_column_idx = 1
    fObsrvResol = 0.1
    nCurrentStep = 5  #14

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
        h_config =  np.array(h_config)*180.0/3.14

        x_test = h_ftan[:nCurrentStep]
        x_test_next = h_ftan[nCurrentStep:nCurrentStep+lh.nFutureStep]
        x_test_all  = h_ftan

        lh.init_plot(bAni=opt.bAnimation)            
                
        if opt.bAnimation:
            
            x,y = get_interp_data(h_config, h_ftan)
            ac = anomaly_checker(lh)
            ac.simulation(x,y, opt.bAniReload)
            
            ## lh.animated_path_plot(x_test_all, opt.bAniReload)
        
        elif opt.bApproxObsrv:
            import time
            start_time = time.clock()

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










