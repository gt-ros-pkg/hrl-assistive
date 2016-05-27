# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection.classifiers import classifier as cb
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.params import *
from hrl_anomaly_detection.hmm import learning_hmm as hmm

from sklearn import preprocessing
from joblib import Parallel, delayed
import random, copy
from sklearn import metrics


import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 

def onlineEvaluationSingle(task, raw_data_path, save_data_path, param_dict, renew=False ):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # SVM
    SVM_dict = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']

    fit_methods = ['single_fit','single_incremental_fit','full_fit','full_incremental_fit']
    fit_renew_methods = []
    ## fit_renew_methods = ['single_fit','single_incremental_fit']
    
    #------------------------------------------
    # get subject1 - task1 's hmm & classifier data
    nFolds = data_dict['nNormalFold'] * data_dict['nAbnormalFold']
    method = 'sgd'
    ROC_dict['nPoints'] = nPoints = 10

    for fit_method in fit_methods:
        roc_data_pkl = os.path.join(save_data_path, 'roc_sgd_'+task+'_'+fit_method+'.pkl')

        if os.path.isfile(roc_data_pkl) is False or renew is True or fit_method in fit_renew_methods:

            ROC_data = {}
            ROC_data[method] = {}
            ROC_data[method]['complete'] = False 
            ROC_data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
            ## ROC_data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['result'] = [ [] for j in xrange(nPoints) ]

            # parallelization 
            r = Parallel(n_jobs=-1, verbose=50)(delayed(run_classifiers)( idx, save_data_path, task, \
                                                                         method, ROC_data, ROC_dict, AE_dict, \
                                                                         SVM_dict, fit_method=fit_method ) \
                                                                         for idx in xrange(nFolds) )
            l_data = r
            for i in xrange(len(l_data)):
                for j in xrange(nPoints):
                    try:
                        method = l_data[i].keys()[0]
                    except:
                        print l_data[i]
                        sys.exit()
                    ## if ROC_data[method]['complete'] == True: continue
                    ROC_data[method]['tp_l'][j] += l_data[i][method]['tp_l'][j]
                    ROC_data[method]['fp_l'][j] += l_data[i][method]['fp_l'][j]
                    ROC_data[method]['tn_l'][j] += l_data[i][method]['tn_l'][j]
                    ROC_data[method]['fn_l'][j] += l_data[i][method]['fn_l'][j]
                    ## ROC_data[method]['delay_l'][j] += l_data[i][method]['delay_l'][j]
                    ROC_data[method]['result'][j].append(l_data[i][method]['result'][j])

            ROC_data[method]['complete'] = True
            ut.save_pickle(ROC_data, roc_data_pkl)


    fig = plt.figure(1)        
    for fit_method in fit_methods:
        roc_data_pkl = os.path.join(save_data_path, 'roc_sgd_'+task+'_'+fit_method+'.pkl')
        ROC_data = ut.load_pickle(roc_data_pkl)
        plotROC(method, nPoints, ROC_data, fit_method=fit_method, fig=fig)
    plt.legend(loc=4,prop={'size':16})                    
    plt.show()
        
    return


def onlineEvaluationDouble(task, raw_data_path, save_data_path, param_dict, \
                           task2, raw_data_path2, save_data_path2, param_dict2, renew=False,\
                           bUpdateHMM=False ):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict    = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # SVM
    SVM_dict = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']
    
    #------------------------------------------
    # get subject1 - task1 's hmm & classifier data
    nFolds = 2 #data_dict['nNormalFold'] * data_dict['nAbnormalFold']
    method = 'sgd'
    ROC_dict['nPoints'] = nPoints = 10
    roc_data_pkl = os.path.join(save_data_path, 'roc_sgd_'+task+'_'+task2+'.pkl')

    if os.path.isfile(roc_data_pkl) is False or renew is True:

        ROC_data = {}
        ROC_data[method] = {}
        ROC_data[method]['complete'] = False 
        ROC_data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
        ## ROC_data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['result'] = [ [] for j in xrange(nPoints) ]

        # parallelization 
        r = Parallel(n_jobs=1, verbose=50)(delayed(run_classifiers_diff)( idx,
                                                                          task, raw_data_path, \
                                                                          save_data_path,\
                                                                          param_dict, \
                                                                          task2, raw_data_path2, \
                                                                          save_data_path2, \
                                                                          param_dict2, \
                                                                          method, ROC_data,\
                                                                          bUpdateHMM=bUpdateHMM) \
                                                                          for idx in xrange(nFolds) )
        l_data = r
        for i in xrange(len(l_data)):
            for j in xrange(nPoints):
                try:
                    method = l_data[i].keys()[0]
                except:
                    print l_data[i]
                    sys.exit()
                ## if ROC_data[method]['complete'] == True: continue
                ROC_data[method]['tp_l'][j] += l_data[i][method]['tp_l'][j]
                ROC_data[method]['fp_l'][j] += l_data[i][method]['fp_l'][j]
                ROC_data[method]['tn_l'][j] += l_data[i][method]['tn_l'][j]
                ROC_data[method]['fn_l'][j] += l_data[i][method]['fn_l'][j]
                ## ROC_data[method]['delay_l'][j] += l_data[i][method]['delay_l'][j]
                ROC_data[method]['result'][j].append(l_data[i][method]['result'][j])

        ROC_data[method]['complete'] = True
        ut.save_pickle(ROC_data, roc_data_pkl)
    else:
        ROC_data = ut.load_pickle(roc_data_pkl)

    plotROC(method, nPoints, ROC_data)

    return


def onlineEvaluationSingleIncremental(task, raw_data_path, save_data_path, param_dict, renew=False ):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # SVM
    SVM_dict = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']

    fit_methods = ['single_incremental_fit']
    ## fit_renew_methods = ['single_fit','single_incremental_fit','full_fit','full_incremental_fit']
    fit_renew_methods = []
    
    #------------------------------------------
    # get subject1 - task1 's hmm & classifier data
    nFolds = data_dict['nNormalFold'] * data_dict['nAbnormalFold']
    method = 'sgd'
    ROC_dict['nPoints'] = nPoints = 5
    nValidData   = 10

    for fit_method in fit_methods:
        roc_data_pkl = os.path.join(save_data_path, 'plr_sgd_'+task+'_'+fit_method+'.pkl')

        if os.path.isfile(roc_data_pkl) is False or renew is True or fit_method in fit_renew_methods:

            ROC_data = {}
            ROC_data[method] = {}
            ROC_data[method]['complete'] = False 
            ROC_data[method]['tp_l'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]
            ROC_data[method]['fp_l'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]
            ROC_data[method]['tn_l'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]
            ROC_data[method]['fn_l'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]
            ROC_data[method]['result'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]

            # parallelization 
            r = Parallel(n_jobs=1, verbose=50)(delayed(run_classifiers_incremental)\
                                               ( idx, save_data_path, task, \
                                                 method, ROC_data, ROC_dict, AE_dict, \
                                                 SVM_dict, fit_method=fit_method,\
                                                nValidData=nValidData) \
                                                for idx in xrange(nFolds) )
            l_data = r
            for i in xrange(len(l_data)): # each fold
                for j in xrange(nPoints):
                    try:
                        method = l_data[i].keys()[0]
                    except:
                        print l_data[i]
                        sys.exit()
                    for k in xrange(nValidData):
                        ROC_data[method]['tp_l'][j][k] += l_data[i][method]['tp_l'][j][k]
                        ROC_data[method]['fp_l'][j][k] += l_data[i][method]['fp_l'][j][k]
                        ROC_data[method]['tn_l'][j][k] += l_data[i][method]['tn_l'][j][k]
                        ROC_data[method]['fn_l'][j][k] += l_data[i][method]['fn_l'][j][k]
                        ROC_data[method]['result'][j][k].append(l_data[i][method]['result'][j][k])

            ROC_data[method]['complete'] = True
            ut.save_pickle(ROC_data, roc_data_pkl)

            print ROC_data[method]['tp_l'][0], ROC_data[method]['fp_l'][0], ROC_data[method]['fn_l'][0],ROC_data[method]['tn_l'][0]



    fig = plt.figure(1)        
    for fit_method in fit_methods:
        roc_data_pkl = os.path.join(save_data_path, 'plr_sgd_'+task+'_'+fit_method+'.pkl')
        ROC_data = ut.load_pickle(roc_data_pkl)
        plotPLR(method, nPoints, nValidData, ROC_data, fit_method=fit_method, fig=fig)
    plt.legend(loc=4,prop={'size':16})                    
    plt.show()
        
    return



def getParams(task, bDataRenew, bAERenew, bHMMRenew, bAESwitch, dim):
    
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    local_range   = 10.0
    
    #---------------------------------------------------------------------------
    if task == 'scooping':
        subjects = ['Wonyoung', 'Tom', 'lin', 'Ashwin', 'Song', 'Henry2'] #'Henry', 
        raw_data_path, save_data_path, param_dict = getScooping(task, bDataRenew, \
                                                                bAERenew, bHMMRenew,\
                                                                rf_center, local_range,\
                                                                ae_swtch=bAESwitch, dim=dim)
        
    #---------------------------------------------------------------------------
    elif task == 'feeding':
        subjects = ['Tom', 'lin', 'Ashwin', 'Song'] #'Wonyoung']
        raw_data_path, save_data_path, param_dict = getFeeding(task, bDataRenew, \
                                                               bAERenew, bHMMRenew,\
                                                               rf_center, local_range,\
                                                               ae_swtch=bAESwitch, dim=dim)
        
    #---------------------------------------------------------------------------           
    elif task == 'pushing_microwhite':
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingMicroWhite(task, bDataRenew, \
                                                                         bAERenew, bHMMRenew,\
                                                                         rf_center, local_range, \
                                                                         ae_swtch=bAESwitch, dim=dim)
                                                                         
    #---------------------------------------------------------------------------           
    elif task == 'pushing_microblack':
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingMicroBlack(task, bDataRenew, \
                                                                         bAERenew, bHMMRenew,\
                                                                         rf_center, local_range, \
                                                                         ae_swtch=bAESwitch, dim=dim)
        
    #---------------------------------------------------------------------------           
    elif task == 'pushing_toolcase':
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingToolCase(task, bDataRenew, \
                                                                       bAERenew, bHMMRenew,\
                                                                       rf_center, local_range, \
                                                                       ae_swtch=bAESwitch, dim=dim)
    else:
        print "Selected task name is not available."
        sys.exit()

    param_dict['subject_list'] = subjects
    return raw_data_path, save_data_path, param_dict


def plotROC(method, nPoints, ROC_data, fit_method=None, fig=None):
    #-------------------------------------------------------------------------------------
    if method == 'svm': label='HMM-SVM'
    elif method == 'progress_time_cluster': label='HMMs with a dynamic threshold'
    elif method == 'fixed': label='HMMs with a fixed threshold'
    elif method == 'cssvm': label='HMM-CSSVM'
    elif method == 'sgd': label='SGD'

    if fit_method is not None: label = fit_method


    if False:
        fig = plt.figure(1)

        tpr_ll = []
        fpr_ll = []
        score_ll = []
        for i in xrange(1,nPoints):
            result_list = ROC_data[method]['result'][i]
            print "Points: ", i, " nFold: ", len(result_list) 
            if len(result_list) == 0:
                print "No data for point: ", i
                continue

            min_len = 10000
            for ii in xrange(len(result_list)):
                length = len(result_list[ii])
                print " nData: ", length
                if length < min_len: min_len = length
            if min_len == 1000: continue

            tp_ll = [ [] for ii in xrange(min_len) ]
            fn_ll = [ [] for ii in xrange(min_len) ]
            fp_ll = [ [] for ii in xrange(min_len) ]
            tn_ll = [ [] for ii in xrange(min_len) ]
            for jj in xrange(len(result_list)):
                if len(result_list[jj]) == 0: continue                
                for ii in xrange(min_len):
                    ## print i, jj, ii, np.shape(result_list), np.shape(result_list[jj]), min_len
                    tp_ll[ii].append(result_list[jj][ii][0])
                    fn_ll[ii].append(result_list[jj][ii][1])
                    fp_ll[ii].append(result_list[jj][ii][2])
                    tn_ll[ii].append(result_list[jj][ii][3])

            tpr_l = []
            fpr_l = []
            for ii in xrange(min_len):
                print min_len, ii, " : ", float(np.sum(tp_ll[ii])), float(np.sum(fn_ll[ii])), \
                  float(np.sum(fp_ll[ii])), float(np.sum(tn_ll[ii]))
                if np.sum(tp_ll[ii])==0: tpr_l.append(0)
                else:
                    tpr_l.append( float(np.sum(tp_ll[ii]))/float(np.sum(tp_ll[ii])+np.sum(fn_ll[ii]))*100.0 )
                if np.sum(fp_ll[ii]) == 0: fpr_l.append(0)
                else:
                    fpr_l.append( float(np.sum(fp_ll[ii]))/float(np.sum(fp_ll[ii])+np.sum(tn_ll[ii]))*100.0 )
            

            score_list = np.array(tpr_l)/(1.0+np.array(fpr_l))
            print score_list
            print tpr_l
            print fpr_l

            # visualization
            color = colors.next()
            shape = shapes.next()
            ax1 = fig.add_subplot(111)            
            plt.plot(score_list, '-'+shape+color, label=str(i), mec=color, ms=6, mew=2)
            print "------------------------------------------------------"

        plt.show()


    else:
        if fig is None:
            fig = plt.figure(1)
            show_flag = True
        else:
            show_flag = False
        
        tp_ll = ROC_data[method]['tp_l']
        fp_ll = ROC_data[method]['fp_l']
        tn_ll = ROC_data[method]['tn_l']
        fn_ll = ROC_data[method]['fn_l']

        tpr_l = []
        fpr_l = []
        fnr_l = []

        for i in xrange(nPoints):
            print i, " : ", np.sum(tp_ll[i]), np.sum(fp_ll[i]), np.sum(tn_ll[i]), np.sum(fn_ll[i])
            tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
            fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )
            fnr_l.append( 100.0 - tpr_l[-1] )

        print "--------------------------------"
        print method
        print tpr_l
        print fpr_l
        print metrics.auc(fpr_l, tpr_l, True)
        print "--------------------------------"

        # visualization
        color = colors.next()
        shape = shapes.next()
        ax1 = fig.add_subplot(111)            
        plt.plot(fpr_l, tpr_l, '-'+shape+color, label=label, mec=color, ms=6, mew=2)
        plt.xlim([-1, 101])
        plt.ylim([-1, 101])
        plt.ylabel('True positive rate (percentage)', fontsize=22)
        plt.xlabel('False positive rate (percentage)', fontsize=22)

        ## font = {'family' : 'normal',
        ##         'weight' : 'bold',
        ##         'size'   : 22}
        ## matplotlib.rc('font', **font)
        ## plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks([0, 50, 100], fontsize=22)
        plt.yticks([0, 50, 100], fontsize=22)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if show_flag: plt.show()
    
    return


def plotPLR(method, nPoints, nValidData, ROC_data, fit_method=None, fig=None):
    #----------------------------------------------------------------------------
    # Plot a positive likelihood ratio graph
    #----------------------------------------------------------------------------
    if method == 'svm': label='HMM-SVM'
    elif method == 'progress_time_cluster': label='HMMs with a dynamic threshold'
    elif method == 'fixed': label='HMMs with a fixed threshold'
    elif method == 'cssvm': label='HMM-CSSVM'
    elif method == 'sgd': label='SGD'

    if fit_method is not None: label = fit_method


    if fig is None:
        fig = plt.figure(1)
        show_flag = True
    else:
        show_flag = False

    tp_ll = ROC_data[method]['tp_l']
    fp_ll = ROC_data[method]['fp_l']
    tn_ll = ROC_data[method]['tn_l']
    fn_ll = ROC_data[method]['fn_l']

    tpr_ll = []
    fpr_ll = []
    fnr_ll = []
    plr_ll = []


    print "--------------------------------"
    print method
    for j in xrange(nValidData):
        tpr_l = []
        fpr_l = []
        fnr_l = []
        plr_l = []

        for i in xrange(nPoints):
            print i,j, " : ", np.sum(tp_ll[i][j]), np.sum(fp_ll[i][j]), np.sum(tn_ll[i][j]), np.sum(fn_ll[i][j])
            tpr_l.append( float(np.sum(tp_ll[i][j]))/float(np.sum(tp_ll[i][j])+np.sum(fn_ll[i][j]))*100.0 )
            fpr_l.append( float(np.sum(fp_ll[i][j]))/float(np.sum(fp_ll[i][j])+np.sum(tn_ll[i][j]))*100.0 )
            fnr_l.append( 100.0 - tpr_l[-1] )
            ## plr_l.append( tpr_l[-1]/fpr_l[-1] )
            ## print plr_l

        print metrics.auc(fpr_l, tpr_l, True)
        # visualization
        color = colors.next()
        shape = shapes.next()
        ax1 = fig.add_subplot(111)            
        plt.plot(fpr_l, tpr_l, '-'+shape+color, label=label+str(j), mec=color, ms=6, mew=2)

    print "--------------------------------"

    plt.xlim([-1, 101])
    plt.ylim([-1, 101])
    plt.ylabel('True positive rate (percentage)', fontsize=22)
    plt.xlabel('False positive rate (percentage)', fontsize=22)
    plt.xticks([0, 50, 100], fontsize=22)
    plt.yticks([0, 50, 100], fontsize=22)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if show_flag: plt.show()

    return



def run_classifiers(idx, save_data_path, task, method, ROC_data, ROC_dict, AE_dict, SVM_dict,
                    fit_method='full_fit'):

    modeling_pkl = os.path.join(save_data_path, 'hmm_'+task+'_'+str(idx)+'.pkl')

    print "start to load hmm data, ", modeling_pkl
    d            = ut.load_pickle(modeling_pkl)
    nState       = d['nState']        
    ll_classifier_train_X   = d['ll_classifier_train_X']
    ll_classifier_train_Y   = d['ll_classifier_train_Y']         
    ll_classifier_train_idx = d['ll_classifier_train_idx']
    ll_classifier_test_X    = d['ll_classifier_test_X']  
    ll_classifier_test_Y    = d['ll_classifier_test_Y']
    ll_classifier_test_idx  = d['ll_classifier_test_idx']
    nLength = d['nLength']
    nPoints = ROC_dict['nPoints']

    #-----------------------------------------------------------------------------------------
    X_train, Y_train, idx_train = flattenSample(ll_classifier_train_X, \
                                                ll_classifier_train_Y, \
                                                ll_classifier_train_idx)

    data = {}
    # pass method if there is existing result
    data[method] = {}
    data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
    ## data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['result'] = [ [] for j in xrange(nPoints) ]
    
    if ROC_data[method]['complete'] == True: return data

    #-----------------------------------------------------------------------------------------
    if method.find('svm')>=0 or method.find('sgd')>=0:
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = X_train
    
    # training set
    if fit_method.find('full') >= 0:
        print method, " : Before classification : ", np.shape(X_train_scaled), np.shape(Y_train)   
        initial_train_X = X_train_scaled
        initial_train_Y = Y_train
    else:
        train_X = []
        train_Y = []
        print method, " : Before classification : ", np.shape(ll_classifier_train_X), \
          np.shape(ll_classifier_train_Y)
        
        # get single data
        for j in xrange(len(ll_classifier_train_X)):
            if method.find('svm')>=0 or method.find('sgd')>=0:
                X = scaler.transform(ll_classifier_train_X[j])                                
            elif method == 'progress_time_cluster' or method == 'fixed':
                X = ll_classifier_train_X[j]

            train_X.append(X)
            train_Y.append(ll_classifier_train_Y[j])
        
        train_idx_list = range(len(train_Y))
        random.shuffle(train_idx_list)
        initial_train_X  = []
        initial_train_Y  = []
        initial_idx_list = []
        abnormal_data=False
        for idx in train_idx_list:
            if train_Y[idx][0] == -1:
                initial_train_X.append(train_X[idx])
                initial_train_Y.append([-1]*len(train_X[idx]))
                initial_idx_list.append(idx)
            if train_Y[idx][0] == 1 and abnormal_data is False:
                initial_train_X.append(train_X[idx])        
                initial_train_Y.append([1]*len(train_X[idx]))
                initial_idx_list.append(idx)
                abnormal_data = True
            ## else:
            ##     initial_eval_X.append(train_X[idx])
            ##     initial_eval_Y.append([1]*len(train_X[idx]))

        initial_train_X, initial_train_Y, _ = flattenSample(initial_train_X, initial_train_Y)
        print np.shape(initial_train_X), np.shape(initial_train_Y)

    # test set
    X_test = []
    Y_test = [] 
    for j in xrange(len(ll_classifier_test_X)):
        if len(ll_classifier_test_X[j])==0: continue

        try:
            if method.find('svm')>=0 or method.find('sgd')>=0:
                X = scaler.transform(ll_classifier_test_X[j])                                
            elif method == 'progress_time_cluster' or method == 'fixed':
                X = ll_classifier_test_X[j]
        except:
            print ll_classifier_test_X[j]
            continue
            
        X_test.append(X)
        Y_test.append(ll_classifier_test_Y[j])

    idx_list = range(len(X_test))
    new_idx_list = randomList(idx_list)

    X_test = [X_test[idx] for idx in new_idx_list]
    Y_test = [Y_test[idx] for idx in new_idx_list]

    #-----------------------------------------------------------------------------------------
    dtc = cb.classifier( method=method, nPosteriors=nState, nLength=nLength )
    for j in xrange(nPoints): 
        dtc.set_params(**SVM_dict)        
        if method == 'sgd' and fit_method.find('full')>=0:
            weights = np.logspace(-2, 1.2, nPoints) #ROC_dict['sgd_param_range']
            dtc.set_params( class_weight=weights[j] )
        elif method == 'sgd' and fit_method.find('single')>=0:
            weights = np.logspace(-1.5, 2.0, nPoints) #ROC_dict['sgd_param_range']
            dtc.set_params( class_weight=weights[j] )
            dtc.set_params( class_weight=weights[j], sgd_n_iter=20 )
        else:
            print "Not available method"
            return "Not available method", -1, params

        ## print "Start to train a classifier: ", idx, j, np.shape(initial_train_X), np.shape(initial_train_Y)
        ret = dtc.fit(initial_train_X, initial_train_Y)
        if fit_method.find('single') >= 0 and False:
            for k in range(10):
                for idx in train_idx_list:
                    if idx not in initial_idx_list:
                        X_ptrain, Y_ptrain = train_X[idx], train_Y[idx]
                        #sample_weight = [10.0]*nLength
                        if Y_ptrain == -1:
                            sample_weight = [1.0]*nLength
                        else:
                            sample_weight = np.logspace(1,2.0,nLength )
                            sample_weight /= np.amax(sample_weight)
                        sample_weight /= 10.0

                        ret = dtc.partial_fit(X_ptrain, Y_ptrain, classes=[-1,1]) #, sample_weight=sample_weight)
                        
        if ret is False: return 'fit failed', -1

        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        result_list = []
        nSamples = len(initial_train_Y)

        # incremental learning and classification
        for i in xrange(len(X_test)):
            if len(Y_test[i])==0: continue
            
            # 1) update classifier
            # Get partial fitting data
            if i is not 0 and fit_method.find('incremental')>=0:
                X_ptrain, Y_ptrain = X_test[i-1], Y_test[i-1]

                if fit_method == 'single_fit' and False:
                    sample_weight = [1.0]*len(X_ptrain) #np.linspace(1.,2.,len(X_ptrain))**3 #good
                elif fit_method.find('single') >= 0:
                    ## sample_weight = np.array([1.0]*len(X_ptrain)) #np.linspace(1.,2.,len(X_ptrain))**3 #good
                    if Y_ptrain == -1:
                        sample_weight = np.array([1.0]*nLength)
                    else:
                        sample_weight = np.logspace(1,2.0,nLength )
                        sample_weight /= np.amax(sample_weight)
                    
                    sample_weight *= 20.0                
                    sample_weight /= (float(nSamples + i))
                else:
                    ## sample_weight = np.log10( np.linspace(1.,10.,len(X_ptrain)) )
                    ## sample_weight = np.linspace(1.,2.,len(X_ptrain))**3 #good
                    ## sample_weight = np.linspace(1.,8.,len(X_ptrain))
                    ## sample_weight = np.ones(len(X_ptrain))
                    sample_weight = np.logspace(1,20,len(X_ptrain) )
                    sample_weight = np.logspace(1,2.0,len(X_ptrain) )

                    # normalize and scaling
                    sample_weight /= np.amax(sample_weight)
                    sample_weight *= 10.0                
                    sample_weight /= float(nSamples + i)

                dtc.partial_fit(X_ptrain, Y_ptrain, classes=[-1,1], sample_weight=sample_weight)

            # 2) test classifier
            X_ptest = X_test[i]
            Y_ptest = Y_test[i]
            Y_est   = dtc.predict(X_ptest, y=Y_ptest)
            for k, y_est in enumerate(Y_est):
                if y_est > 0:
                    break

            tp=0; fp=0; tn=0; fn=0
            if Y_ptest[0] > 0:
                if y_est > 0:
                    tp = 1
                    tp_l.append(1)
                else:
                    fn = 1
                    fn_l.append(1)
            elif Y_ptest[0] <= 0:
                if y_est > 0:
                    fp = 1
                    fp_l.append(1)
                else:
                    tn = 1
                    tn_l.append(1)

            result_list.append([tp,fn,fp,tn])

        data[method]['tp_l'][j] += tp_l
        data[method]['fp_l'][j] += fp_l
        data[method]['fn_l'][j] += fn_l
        data[method]['tn_l'][j] += tn_l
        ## ROC_data[method]['delay_l'][j] += delay_l
        ROC_data[method]['result'][j].append(result_list)

    return data

def run_classifiers_incremental(idx, save_data_path, task, method, ROC_data, ROC_dict, AE_dict, SVM_dict,
                                fit_method='full_fit', nValidData=10):

    modeling_pkl = os.path.join(save_data_path, 'hmm_'+task+'_'+str(idx)+'.pkl')
    nPoints = ROC_dict['nPoints']

    if method == 'sgd' and fit_method.find('full')>=0:
        weights = np.logspace(-2, 1.2, nPoints) #ROC_dict['sgd_param_range']
    elif method == 'sgd' and fit_method.find('single')>=0:
        weights = np.logspace(-1.5, 2.0, nPoints) #ROC_dict['sgd_param_range']
    

    print "start to load hmm data, ", modeling_pkl
    d            = ut.load_pickle(modeling_pkl)
    nState       = d['nState']        
    ll_classifier_train_X   = d['ll_classifier_train_X']
    ll_classifier_train_Y   = d['ll_classifier_train_Y']         
    ll_classifier_train_idx = d['ll_classifier_train_idx']
    ll_classifier_test_X    = d['ll_classifier_test_X']  
    ll_classifier_test_Y    = d['ll_classifier_test_Y']
    ll_classifier_test_idx  = d['ll_classifier_test_idx']
    nLength = d['nLength']

    #-----------------------------------------------------------------------------------------
    X_train, Y_train, idx_train = flattenSample(ll_classifier_train_X, \
                                                ll_classifier_train_Y, \
                                                ll_classifier_train_idx)

    data = {}
    # pass method if there is existing result
    data[method] = {}
    data[method]['tp_l'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]
    data[method]['fp_l'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]
    data[method]['tn_l'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]
    data[method]['fn_l'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]
    data[method]['result'] = [ [ [] for i in xrange(nValidData) ] for j in xrange(nPoints) ]
    
    if ROC_data[method]['complete'] == True: return data

    #-----------------------------------------------------------------------------------------
    if method.find('svm')>=0 or method.find('sgd')>=0:
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = X_train
    
    # training set
    if fit_method.find('full') >= 0:
        print method, " : Before classification : ", np.shape(X_train_scaled), np.shape(Y_train)   
        initial_train_X = X_train_scaled
        initial_train_Y = Y_train
    else:
        train_X = []
        train_Y = []
        print method, " : Before classification : ", np.shape(ll_classifier_train_X), \
          np.shape(ll_classifier_train_Y)
        
        # get single data
        for j in xrange(len(ll_classifier_train_X)):
            if method.find('svm')>=0 or method.find('sgd')>=0:
                X = scaler.transform(ll_classifier_train_X[j])                                
            elif method == 'progress_time_cluster' or method == 'fixed':
                X = ll_classifier_train_X[j]

            train_X.append(X)
            train_Y.append(ll_classifier_train_Y[j])
        
        train_idx_list = range(len(train_Y))
        random.shuffle(train_idx_list)
        initial_train_X  = []
        initial_train_Y  = []
        initial_idx_list = []
        normal_data=False
        abnormal_data=False
        for idx in train_idx_list:
            if train_Y[idx][0] == -1 and normal_data is False:
                initial_train_X.append(train_X[idx])
                initial_train_Y.append([-1]*len(train_X[idx]))
                initial_idx_list.append(idx)
                normal_data = True
            if train_Y[idx][0] == 1 and abnormal_data is False:
                initial_train_X.append(train_X[idx])        
                initial_train_Y.append([1]*len(train_X[idx]))
                initial_idx_list.append(idx)
                abnormal_data = True

        initial_train_X, initial_train_Y, _ = flattenSample(initial_train_X, initial_train_Y)
        ## print np.shape(initial_train_X), np.shape(initial_train_Y)

    ## validation set (incremental fitting) and test set
    X_test = []
    Y_test = [] 
    for j in xrange(len(ll_classifier_test_X)):
        if len(ll_classifier_test_X[j])==0: continue

        try:
            if method.find('svm')>=0 or method.find('sgd')>=0:
                X = scaler.transform(ll_classifier_test_X[j])                                
            elif method == 'progress_time_cluster' or method == 'fixed':
                X = ll_classifier_test_X[j]
        except:
            print ll_classifier_test_X[j]
            continue
            
        X_test.append(X)
        Y_test.append(ll_classifier_test_Y[j])

    idx_list     = range(len(X_test))
    new_idx_list = randomList(idx_list)
    X_valid_test = [X_test[idx] for idx in new_idx_list[:nValidData]]
    Y_valid_test = [Y_test[idx] for idx in new_idx_list[:nValidData]]
    X_eval_test  = [X_test[idx] for idx in new_idx_list[nValidData:]]
    Y_eval_test  = [Y_test[idx] for idx in new_idx_list[nValidData:]]
    ## print np.shape(X_test), np.shape(X_valid_test), np.shape(X_eval_test)

    #-----------------------------------------------------------------------------------------
    nSamples = len(initial_train_Y)
    dtc = cb.classifier( method=method, nPosteriors=nState, nLength=nLength )
    for j in xrange(nPoints): 
        dtc.set_params(**SVM_dict)        
        if method == 'sgd' and fit_method.find('full')>=0:
            dtc.set_params( class_weight=weights[j] )
        elif method == 'sgd' and fit_method.find('single')>=0:
            dtc.set_params( class_weight=weights[j] )
            dtc.set_params( class_weight=weights[j], sgd_n_iter=20 )
        else:
            print "Not available method"
            return "Not available method", -1, params

        # Training fit
        ret = dtc.fit(initial_train_X, initial_train_Y)
        if ret is False: return 'fit failed', -1

        # Incremental fit
        if fit_method.find('incremental') >= 0:
            for idx in xrange(len(X_valid_test)):
                X_ptrain, Y_ptrain = X_valid_test[idx], Y_valid_test[idx]
                sample_weight = [1.0]*nLength
                ## if Y_ptrain == -1:
                ##     sample_weight = [1.0]*nLength
                ## else:
                ##     sample_weight = np.logspace(1,2.0,nLength )
                ##     sample_weight /= np.amax(sample_weight)
                ## sample_weight *= 20.0
                ## sample_weight /= (float(nSamples + idx+1))

                ## ret = dtc.partial_fit(X_ptrain, Y_ptrain, classes=[-1,1], sample_weight=sample_weight)

                tp_l = []
                fp_l = []
                tn_l = []
                fn_l = []
                result_list = []

                # incremental learning and classification
                for i in xrange(len(X_eval_test)):
                    if len(Y_eval_test[i])==0: continue

                    # 2) test classifier
                    X_ptest = X_eval_test[i]
                    Y_ptest = Y_eval_test[i]
                    Y_est   = dtc.predict(X_ptest, y=Y_ptest)
                    for k, y_est in enumerate(Y_est):
                        if y_est > 0:
                            break

                    tp=0; fp=0; tn=0; fn=0
                    if Y_ptest[0] > 0:
                        if y_est > 0:
                            tp = 1
                            tp_l.append(1)
                        else:
                            fn = 1
                            fn_l.append(1)
                    elif Y_ptest[0] <= 0:
                        if y_est > 0:
                            fp = 1
                            fp_l.append(1)
                        else:
                            tn = 1
                            tn_l.append(1)

                    result_list.append([tp,fn,fp,tn])

                data[method]['tp_l'][j][idx] += tp_l
                data[method]['fp_l'][j][idx] += fp_l
                data[method]['fn_l'][j][idx] += fn_l
                data[method]['tn_l'][j][idx] += tn_l
                data[method]['result'][j][idx].append(result_list)

                print j, idx, " = ",  np.sum(tp_l),  np.sum(fp_l),  np.sum(fn_l),  np.sum(tn_l)

        else:
            idx  = 0
            tp_l = []
            fp_l = []
            tn_l = []
            fn_l = []
            result_list = []
            nSamples = len(initial_train_Y)

            # incremental learning and classification
            for i in xrange(len(X_eval_test)):
                if len(Y_eval_test[i])==0: continue

                # 2) test classifier
                X_ptest = X_eval_test[i]
                Y_ptest = Y_eval_test[i]
                Y_est   = dtc.predict(X_ptest, y=Y_ptest)
                for k, y_est in enumerate(Y_est):
                    if y_est > 0:
                        break

                tp=0; fp=0; tn=0; fn=0
                if Y_ptest[0] > 0:
                    if y_est > 0:
                        tp = 1
                        tp_l.append(1)
                    else:
                        fn = 1
                        fn_l.append(1)
                elif Y_ptest[0] <= 0:
                    if y_est > 0:
                        fp = 1
                        fp_l.append(1)
                    else:
                        tn = 1
                        tn_l.append(1)

                result_list.append([tp,fn,fp,tn])

            data[method]['tp_l'][j][idx] += tp_l
            data[method]['fp_l'][j][idx] += fp_l
            data[method]['fn_l'][j][idx] += fn_l
            data[method]['tn_l'][j][idx] += tn_l
            data[method]['result'][j][idx].append(result_list)

    return data



def run_classifiers_diff( idx, task, raw_data_path, save_data_path, param_dict, \
                          task2, raw_data_path2, save_data_path2, param_dict2, \
                          method, ROC_data, bUpdateHMM=False ):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # SVM
    SVM_dict = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']


    ### Training HMM and SVM using task 1 data ------------------------------------------------
    crossVal_pkl = os.path.join(save_data_path, 'cv_'+task+'.pkl')
    modeling_pkl = os.path.join(save_data_path, 'hmm_'+task+'_'+str(idx)+'.pkl')

    # Scaling data
    d = ut.load_pickle(crossVal_pkl)
    init_param_dict = d['param_dict']
    

    print "start to load hmm data, ", modeling_pkl
    d            = ut.load_pickle(modeling_pkl)
    nState       = d['nState']
    nEmissionDim = d['nEmissionDim']
    ll_classifier_train_X   = d['ll_classifier_train_X']
    ll_classifier_train_Y   = d['ll_classifier_train_Y']         
    ll_classifier_train_idx = d['ll_classifier_train_idx']
    ll_classifier_test_X    = d['ll_classifier_test_X']  
    ll_classifier_test_Y    = d['ll_classifier_test_Y']
    ll_classifier_test_idx  = d['ll_classifier_test_idx']
    nLength = d['nLength']
    nPoints = param_dict['ROC']['nPoints']

    nNormalTrain = 0
    for i in xrange(len(ll_classifier_train_Y)):
        if ll_classifier_train_Y[i][0]<0: nNormalTrain += 1


    X_train, Y_train, idx_train = flattenSample(ll_classifier_train_X, \
                                                ll_classifier_train_Y, \
                                                ll_classifier_train_idx)

    # data preparation
    if method.find('svm')>=0 or method.find('sgd')>=0:
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = X_train
    print method, " : Before classification : ", np.shape(X_train_scaled), np.shape(Y_train)
    
    ## # pre-trained HMM from training data
    ## ml = hmm.learning_hmm(nState, d['nEmissionDim'], verbose=False)
    ## ml.set_hmm_object(d['A'], d['B'], d['pi'])
    
    ## Getting Test data from task 2 ------------------------------------------------------
    d2 = dm.getDataSet(param_dict2['subject_list'], task2, raw_data_path2, save_data_path2, \
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       handFeatures=param_dict['data_param']['handFeatures'], \
                       cut_data=param_dict['data_param']['cut_data'] )
                       
    successData = d2['successData'] 
    failureData = d2['failureData']
    init_param_dict2 = d2['param_dict']


    target_max = init_param_dict['feature_max']
    target_min = init_param_dict['feature_min']
    cur_max    = init_param_dict2['feature_max']
    cur_min    = init_param_dict2['feature_min']


    testDataX = []
    testDataY = []
    for i in xrange(nEmissionDim):
        temp = np.vstack([successData[i], failureData[i]])
        testDataX.append( temp )

    testDataY = np.hstack([ -np.ones(len(successData[0])), \
                            np.ones(len(failureData[0])) ])

    # rescaling by two param dicts
    new_testDataX = []
    for i, feature in enumerate(testDataX):
        # recover
        new_feature = feature *(cur_max[i] - cur_min[i]) + cur_min[i]
        # rescaling
        new_feature = (new_feature-target_min[i])/(target_max[i] - target_min[i])
        new_testDataX.append(new_feature)

    print np.shape(testDataX), np.shape(new_testDataX), np.shape(feature), np.shape(new_feature)
    testDataX = np.array(new_testDataX)*HMM_dict['scale']

    A,B,pi = d['A'], d['B'], d['pi']

    # update hmm
    if bUpdateHMM is True:
        ml = hmm.learning_hmm(d['nState'], d['nEmissionDim'], verbose=False)
        for i in xrange(len(testDataX[0])):
            if testDataY[i] > 0: continue

            ml.set_hmm_object(A,B,pi)            
            A,B,pi = ml.partial_fit( testDataX[:,i:i+1,:], nNormalTrain+i, HMM_dict['scale'], \
                                     weight=4.0)

            


    #### Run HMM with the test data from task 2 ----------------------------------------------

    startIdx = 4
    r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)\
                            (i, A, B, pi, d['F'], \
                             [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                             nEmissionDim, nState,\
                             startIdx=startIdx, \
                            bPosterior=True)
                            for i in xrange(len(testDataX[0])))
    _, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)

    # nSample x nLength
    ll_classifier_test_X = []
    ll_classifier_test_Y = []
    for i in xrange(len(ll_logp)):
        l_X = []
        l_Y = []
        for j in xrange(len(ll_logp[i])):        
            l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

            if testDataY[i] > 0.0: l_Y.append(1)
            else: l_Y.append(-1)

            if np.isnan(ll_logp[i][j]):
                print "nan values in ", i, j
                print testDataX[0][i]
                print ll_logp[i][j], ll_post[i][j]
                sys.exit()

        ll_classifier_test_X.append(l_X)
        ll_classifier_test_Y.append(l_Y)


    X_test = []
    Y_test = [] 
    for j in xrange(len(ll_classifier_test_X)):
        if len(ll_classifier_test_X[j])==0: continue

        try:
            if method.find('svm')>=0 or method.find('sgd')>=0:
                X = scaler.transform(ll_classifier_test_X[j])                                
            elif method == 'progress_time_cluster' or method == 'fixed':
                X = ll_classifier_test_X[j]
        except:
            print ll_classifier_test_X[j]
            continue
            
        X_test.append(X)
        Y_test.append(ll_classifier_test_Y[j])

    idx_list = range(len(X_test))
    new_idx_list = randomList(idx_list)
    X_test = [X_test[idx] for idx in new_idx_list]
    Y_test = [Y_test[idx] for idx in new_idx_list]
   
    data = {}
    # pass method if there is existing result
    data[method] = {}
    data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
    ## data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['result'] = [ [] for j in xrange(nPoints) ]
    
    if ROC_data[method]['complete'] == True: return data

    #-----------------------------------------------------------------------------------------
    dtc = cb.classifier( method=method, nPosteriors=nState, nLength=nLength )
    for j in xrange(nPoints): 
        dtc.set_params(**SVM_dict)        
        if method == 'sgd':
            weights = np.logspace(-2, 1.2, nPoints) #ROC_dict['sgd_param_range']
            dtc.set_params( class_weight=weights[j] )
        else:
            print "Not available method"
            return "Not available method", -1, params

        print "Start to train a classifier: ", idx, j, np.shape(X_train), np.shape(Y_train)
        ret = dtc.fit(X_train_scaled, Y_train)
        if ret is False: return 'fit failed', -1

        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        result_list = []

        # incremental learning and classification
        for i in xrange(len(X_test)):
            if len(Y_test[i])==0: continue

            # 1) update model

            
            # 2) update classifier
            # Get partial fitting data
            if i is not 0:
                X_ptrain, Y_ptrain = X_test[i-1], Y_test[i-1]
                ## sample_weight = np.logspace(-4.,0,len(X_ptrain))
                ## sample_weight/=np.sum(sample_weight)
                ## sample_weight = np.logspace(1,2.0,len(X_ptrain) )
                sample_weight = np.linspace(0,1.0,len(X_ptrain) )
                sample_weight /= np.amax(sample_weight)
                sample_weight *= 10.0                
                sample_weight /= float(len(ll_classifier_train_X) + i)                
                dtc.partial_fit(X_ptrain, Y_ptrain, sample_weight=sample_weight)

            # 2) test classifier
            X_ptest = X_test[i]
            Y_ptest = Y_test[i]
            Y_est   = dtc.predict(X_ptest, y=Y_ptest)
            for k, y_est in enumerate(Y_est):
                if y_est > 0:
                    break

            tp=0; fp=0; tn=0; fn=0
            if Y_ptest[0] > 0:
                if y_est > 0:
                    tp = 1
                    tp_l.append(1)
                else:
                    fn = 1
                    fn_l.append(1)
            elif Y_ptest[0] <= 0:
                if y_est > 0:
                    fp = 1
                    fp_l.append(1)
                else:
                    tn = 1
                    tn_l.append(1)

            result_list.append([tp,fn,fp,tn])

        data[method]['tp_l'][j] += tp_l
        data[method]['fp_l'][j] += fp_l
        data[method]['fn_l'][j] += fn_l
        data[method]['tn_l'][j] += tn_l
        ## ROC_data[method]['delay_l'][j] += delay_l
        ROC_data[method]['result'][j].append(result_list)
        ## print "length: ", np.shape(ROC_data[method]['result'][j]), np.shape(result_list)

    return data


def multiDataPlot(task, raw_data_path, save_data_path, param_dict, \
                  task2, raw_data_path2, save_data_path2, param_dict2, renew=False):

    #### Getting Test data from task 2 ------------------------------------------------------
    d1 = dm.getDataSet(param_dict['subject_list'], task, raw_data_path, save_data_path, \
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       handFeatures=param_dict['data_param']['handFeatures'], \
                       cut_data=param_dict['data_param']['cut_data'] )
    successData1     = d1['successData'] 
    failureData1     = d1['failureData']
    init_param_dict1 = d1['param_dict']

                       
    #### Getting Test data from task 2 ------------------------------------------------------
    d2 = dm.getDataSet(param_dict2['subject_list'], task2, raw_data_path2, save_data_path2, \
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       handFeatures=param_dict['data_param']['handFeatures'], \
                       cut_data=param_dict['data_param']['cut_data'] )
    successData2     = d2['successData'] 
    failureData2     = d2['failureData']
    init_param_dict2 = d2['param_dict']

    #----------------------------------------------------------------------------------------

    print np.shape(successData1), np.shape(successData2)
    ## sys.exit()
    
    # plot data
    fig = plt.figure(1)
    for i in xrange(len(successData1)):
        # each dim
        print i, np.shape(successData1)
        ax = fig.add_subplot(400+10+i+1)
        for j in xrange(len(successData1[i])):
            if j>5: break
            plt.plot(successData1[i][j], 'b-')

    for i in xrange(len(successData2)):
        # each dim
        print i, np.shape(successData2)
        ax = fig.add_subplot(400+10+i+1)
        for j in xrange(len(successData2[i])):
            if j>5: break
            plt.plot(successData2[i][j], 'r-')


    plt.show()            
    sys.exit()



def likelihoodPlot(task, raw_data_path, save_data_path, param_dict, \
                   task2, raw_data_path2, save_data_path2, param_dict2, renew=False, \
                   bUpdateHMM=False ):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # SVM
    SVM_dict = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']

    #### Getting Test data from task 1 ------------------------------------------------------
    idx = 1
    crossVal_pkl = os.path.join(save_data_path, 'cv_'+task+'.pkl')
    modeling_pkl = os.path.join(save_data_path, 'hmm_'+task+'_'+str(idx)+'.pkl')

    # Scaling data
    d = ut.load_pickle(crossVal_pkl)
    init_param_dict = d['param_dict']
    
    ## d1 = dm.getDataSet(param_dict['subject_list'], task, raw_data_path, save_data_path, \
    ##                    param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
    ##                    downSampleSize=param_dict['data_param']['downSampleSize'], \
    ##                    handFeatures=param_dict['data_param']['handFeatures'], \
    ##                    cut_data=param_dict['data_param']['cut_data'] )
    ## successData1     = d1['successData'] 
    ## failureData1     = d1['failureData']
    ## init_param_dict = d1['param_dict']

    print "start to load hmm data, ", modeling_pkl
    d                       = ut.load_pickle(modeling_pkl)
    nState                  = d['nState']        
    nEmissionDim            = d['nEmissionDim']
    ll_classifier_train_X   = d['ll_classifier_train_X']
    ll_classifier_train_Y   = d['ll_classifier_train_Y']         
    ll_classifier_train_idx = d['ll_classifier_train_idx']
    ## ll_classifier_test_X    = d['ll_classifier_test_X']  
    ## ll_classifier_test_Y    = d['ll_classifier_test_Y']
    ## ll_classifier_test_idx  = d['ll_classifier_test_idx']
    nLength                 = d['nLength']
    print "train_X: ", np.shape(ll_classifier_train_X), np.shape(ll_classifier_train_Y)
    nNormalTrain = 0
    for i in xrange(len(ll_classifier_train_Y)):
        if ll_classifier_train_Y[i][0]<0: nNormalTrain += 1
    
    #### Getting Test data from task 2 ------------------------------------------------------
    d2 = dm.getDataSet(param_dict2['subject_list'], task2, raw_data_path2, save_data_path2, \
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       handFeatures=param_dict['data_param']['handFeatures'], \
                       cut_data=param_dict['data_param']['cut_data'] )
    successData2     = d2['successData'] 
    failureData2     = d2['failureData']
    init_param_dict2 = d2['param_dict']

    target_max = init_param_dict['feature_max']
    target_min = init_param_dict['feature_min']
    cur_max    = init_param_dict2['feature_max']
    cur_min    = init_param_dict2['feature_min']


    testDataX = []
    testDataY = []
    for i in xrange(nEmissionDim):
        temp = np.vstack([successData2[i], failureData2[i]])
        testDataX.append( temp )

    testDataY = np.hstack([ -np.ones(len(successData2[0])), \
                            np.ones(len(failureData2[0])) ])
    ## testDataX = successData2
    ## testDataY = -np.ones(len(successData2[0]))

    # rescaling by two param dicts
    new_testDataX = []
    for i, feature in enumerate(testDataX):
        # recover
        new_feature = feature *(cur_max[i] - cur_min[i]) + cur_min[i]
        # rescaling
        new_feature = (new_feature-target_min[i])/(target_max[i] - target_min[i])
        new_testDataX.append(new_feature)

    testDataX = np.array(new_testDataX)*HMM_dict['scale']
    print "testDataX: ", np.shape(testDataX)

    modeling_pkl = os.path.join(save_data_path2, 'hmm_'+task2+'_'+str(idx)+'.pkl')
    d2           = ut.load_pickle(modeling_pkl)
    A2,B2,pi2    = d2['A'], d2['B'], d2['pi'] 


    A,B,pi = d['A'], d['B'], d['pi']
    if bUpdateHMM:
        ml = hmm.learning_hmm(d['nState'], d['nEmissionDim'], verbose=False)

        ## plt.ion() -----------------------------------------------
        fig = plt.figure()
        org_mu_l  = []
        org_cov_l = []
        org_mu2_l  = []
        org_cov2_l = []
        for j in xrange(len(d['B'])):
            org_mu_l.append(d['B'][j,0]) # mu
            org_cov_l.append( np.reshape(d['B'][j,1], (nEmissionDim, nEmissionDim)) ) # cov
            org_mu2_l.append(d2['B'][j,0]) # mu
            org_cov2_l.append( np.reshape(d2['B'][j,1], (nEmissionDim, nEmissionDim)) ) # cov
        org_mu_l  = np.array(org_mu_l)
        org_cov_l = np.array(org_cov_l)
        org_mu2_l  = np.array(org_mu2_l)
        org_cov2_l = np.array(org_cov2_l)

        for j in xrange(nEmissionDim):
            ax = fig.add_subplot(100*nEmissionDim+10+j+1)                            
            plt.plot(org_mu_l[:,j], label='Task1' )
            plt.plot(org_mu2_l[:,j], linewidth=3.0, label='Task2' )
        # ----------------------------------------------------------
        
        for i in xrange(10): #xrange(len(testDataX[0])):
            if testDataY[i] > 0: continue

            ml.set_hmm_object(A,B,pi)            
            A,B,pi = ml.partial_fit( testDataX[:,i:i+1,:], nNormalTrain+i, HMM_dict['scale'], \
                                     weight=4.0)
            
            # ----------------------------------------------------------
            mu_l = []
            cov_l= []
            for j in xrange(len(B)):
                mu_l.append(B[j,0])
                cov_l.append( np.reshape(B[j,1], (nEmissionDim, nEmissionDim)) )
            mu_l  = np.array(mu_l)
            cov_l = np.array(cov_l)

            for j in xrange(nEmissionDim):
                plt.subplot(100*nEmissionDim+10+j+1)                            
                if j == 0: plt.plot(mu_l[:,j], label=str(i) )
                else:      plt.plot(mu_l[:,j] )


        
        plt.legend(loc=3,prop={'size':16})            
        plt.show()
        sys.exit()
        # ----------------------------------------------------------


        
    print "----------------------------------------------------------"
        
    #### Run HMM with the test data from task 2 ----------------------------------------------

    startIdx = 4
    r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)\
                            (i, A, B, pi, d['F'], \
                             [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                             nEmissionDim, nState,\
                             startIdx=startIdx, \
                            bPosterior=True)
                            for i in xrange(15))
                            ## for i in xrange(len(testDataX[0])))
    _, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)

    # nSample x nLength
    ll_classifier_test_X = []
    ll_classifier_test_Y = []
    for i in xrange(len(ll_logp)):
        l_X = []
        l_Y = []
        for j in xrange(len(ll_logp[i])):        
            l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

            if testDataY[i] > 0.0: l_Y.append(1)
            else: l_Y.append(-1)

            if np.isnan(ll_logp[i][j]):
                print "nan values in ", i, j
                print testDataX[0][i]
                print ll_logp[i][j], ll_post[i][j]
                continue
                ## sys.exit()

        ll_classifier_test_X.append(l_X)
        ll_classifier_test_Y.append(l_Y)

    #----------------------------------------------------------------------------------------

    # training data
    print np.shape(ll_classifier_test_X), np.shape(ll_classifier_test_Y)

    ll_logp_train = []
    for i in xrange(10):
        ll_logp_train.append(np.swapaxes(ll_classifier_train_X[i], 0, 1)[0])

    ll_logp_test_normal = []
    for i in xrange(10):
        ll_logp_test_normal.append(np.swapaxes(ll_classifier_test_X[i], 0, 1)[0])

    ## ll_logp_test_abnormal = []
    ## for i in xrange(15):
    ##     ll_logp_test_abnormal.append(np.swapaxes(ll_classifier_test_X[-i], 0, 1)[0])
        
    fig = plt.figure()
    plt.plot( np.swapaxes(ll_logp_train,0,1), 'b-' )
    plt.plot( np.swapaxes(ll_logp_test_normal,0,1), 'g-' )
    ## plt.plot( np.swapaxes(ll_logp_test_abnormal,0,1), 'r-' )
    plt.show()        

                          



## def getAUC(fpr_l, tpr_l):
##     area = 0.0
##     for i in range(len(fpr_l)-1):        
##         area += (fpr_l[i+1]-fpr_l[i])*(tpr_l[i]+tpr_l[i+1])*0.5
##     return area


# random order
def randomList(a):
    b = []
    for i in range(len(a)):
        element = random.choice(a)
        a.remove(element)
        b.append(element)
    return b


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='pushing_microwhite',
                 help='type the desired task name')
    p.add_option('--task2', action='store', dest='task2', type='string', default='pushing_microblack',
                 help='type the desired task name')

    p.add_option('--same_class', '--sc', action='store_true', dest='bSameClass', \
                 default=False,help='Run online evaluation with the same class data')
    p.add_option('--diff_class', '--dc', action='store_true', dest='bDiffClass', \
                 default=False,help='Run online evaluation with the different class data')
    p.add_option('--multi_plot', '--mp', action='store_true', dest='bMultiDataPlot', \
                 default=False,help='plot multiple classes')
    p.add_option('--likelihood_plot', '--lp', action='store_true', dest='bLikelihoodPlot', \
                 default=False,help='plot likelihood given multiple classes')

    
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--AERenew', '--ar', action='store_true', dest='bAERenew',
                 default=False, help='Renew AE data.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew result.')

    p.add_option('--update_hmm', '--uh', action='store_true', dest='bUpdateHMM',
                 default=False, help='Update HMM.')    

    p.add_option('--dim', action='store', dest='dim', type=int, default=4,
                 help='type the desired dimension')
    p.add_option('--aeswtch', '--aesw', action='store_true', dest='bAESwitch',
                 default=False, help='Enable AE data.')
    
    opt, args = p.parse_args()



    #---------------------------------------------------------------------------           
    #---------------------------------------------------------------------------           
    #---------------------------------------------------------------------------
    if opt.bMultiDataPlot:
        raw_data_path1, save_data_path1, param_dict1 = \
          getParams(opt.task, opt.bDataRenew, opt.bAERenew, opt.bHMMRenew, opt.bAESwitch, opt.dim)
        raw_data_path2, save_data_path2, param_dict2 = \
          getParams(opt.task2, opt.bDataRenew, opt.bAERenew, opt.bHMMRenew, opt.bAESwitch, opt.dim)
        multiPlot(opt.task, raw_data_path1, save_data_path1, param_dict1, \
                  opt.task2, raw_data_path2, save_data_path2, param_dict2, \
                  renew=opt.bRenew )
    elif opt.bLikelihoodPlot:
        raw_data_path1, save_data_path1, param_dict1 = \
          getParams(opt.task, opt.bDataRenew, opt.bAERenew, opt.bHMMRenew, opt.bAESwitch, opt.dim)
        raw_data_path2, save_data_path2, param_dict2 = \
          getParams(opt.task2, opt.bDataRenew, opt.bAERenew, opt.bHMMRenew, opt.bAESwitch, opt.dim)
        likelihoodPlot(opt.task, raw_data_path1, save_data_path1, param_dict1, \
                       opt.task2, raw_data_path2, save_data_path2, param_dict2, \
                       renew=opt.bRenew, bUpdateHMM=opt.bUpdateHMM )
        
    elif opt.bDiffClass:
        raw_data_path1, save_data_path1, param_dict1 = \
          getParams(opt.task, opt.bDataRenew, opt.bAERenew, opt.bHMMRenew, opt.bAESwitch, opt.dim)
        raw_data_path2, save_data_path2, param_dict2 = \
          getParams(opt.task2, opt.bDataRenew, opt.bAERenew, opt.bHMMRenew, opt.bAESwitch, opt.dim)
        ## opt.task2 = opt.task
        ## raw_data_path2, save_data_path2, param_dict2 = raw_data_path1, save_data_path1, param_dict1
        onlineEvaluationDouble(opt.task, raw_data_path1, save_data_path1, param_dict1, \
                               opt.task2, raw_data_path2, save_data_path2, param_dict2, \
                               renew=opt.bRenew, bUpdateHMM=opt.bUpdateHMM )
                               
    else:
        raw_data_path, save_data_path, param_dict = \
          getParams(opt.task, opt.bDataRenew, opt.bAERenew, opt.bHMMRenew, opt.bAESwitch, opt.dim)
        onlineEvaluationSingleIncremental(opt.task, raw_data_path, save_data_path, param_dict, renew=opt.bRenew)
        ## onlineEvaluationSingle(opt.task, raw_data_path, save_data_path, param_dict, renew=opt.bRenew)
            


    

    
