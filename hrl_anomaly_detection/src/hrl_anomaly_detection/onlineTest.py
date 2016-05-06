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

from sklearn import preprocessing
from joblib import Parallel, delayed


import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 

def onlineEvaluation(task, raw_data_path, save_data_path, param_dict, renew=False ):

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
    
    #------------------------------------------
    # get subject1 - task1 's hmm & classifier data
    nFolds = data_dict['nNormalFold'] * data_dict['nAbnormalFold']
    method = 'sgd'
    ROC_dict['nPoints'] = nPoints = 10
    roc_data_pkl = os.path.join(save_data_path, 'roc_sgd_'+task+'.pkl')

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
        r = Parallel(n_jobs=-1, verbose=50)(delayed(run_classifiers)( idx, save_data_path, task, \
                                                                     method, ROC_data, ROC_dict, AE_dict, \
                                                                     SVM_dict ) \
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
        
    #-------------------------------------------------------------------------------------
    if method == 'svm': label='HMM-SVM'
    elif method == 'progress_time_cluster': label='HMMs with a dynamic threshold'
    elif method == 'fixed': label='HMMs with a fixed threshold'
    elif method == 'cssvm': label='HMM-CSSVM'
    elif method == 'sgd': label='SGD'


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
        fig = plt.figure(1)
        
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
        print getAUC(fpr_l, tpr_l)
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
        plt.show()
    
    return ROC_data


def run_classifiers(idx, save_data_path, task, method, ROC_data, ROC_dict, AE_dict, SVM_dict ):

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
    # data preparation
    if method.find('svm')>=0 or method.find('sgd')>=0:
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = X_train
    print method, " : Before classification : ", np.shape(X_train_scaled), np.shape(Y_train)

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

    # random order
    import random, copy
    def randomList(a):
        b = []
        for i in range(len(a)):
            element = random.choice(a)
            a.remove(element)
            b.append(element)
        return b

    idx_list = range(len(X_test))
    new_idx_list = randomList(idx_list)
    X_test = [X_test[idx] for idx in new_idx_list]
    Y_test = [Y_test[idx] for idx in new_idx_list]
    
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
            
            # 1) update classifier
            # Get partial fitting data
            if i is not 0:
                X_ptrain, Y_ptrain = X_test[i-1], Y_test[i-1]
                sample_weight = np.logspace(-4.,0,len(X_ptrain))
                sample_weight/=np.sum(sample_weight)
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


def getAUC(fpr_l, tpr_l):
    area = 0.0
    for i in range(len(fpr_l)-1):        
        area += (fpr_l[i+1]-fpr_l[i])*(tpr_l[i]+tpr_l[i+1])*0.5
    return area


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='pushing_microwhite',
                 help='type the desired task name')
    p.add_option('--task2', action='store', dest='task2', type='string', default='pushing_microblack',
                 help='type the desired task name')

    
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--AERenew', '--ar', action='store_true', dest='bAERenew',
                 default=False, help='Renew AE data.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew result.')

    p.add_option('--dim', action='store', dest='dim', type=int, default=4,
                 help='type the desired dimension')
    p.add_option('--aeswtch', '--aesw', action='store_true', dest='bAESwitch',
                 default=False, help='Enable AE data.')
    
    opt, args = p.parse_args()

    rf_center     = 'kinEEPos'        
    scale         = 1.0
    local_range   = 10.0
    
    #---------------------------------------------------------------------------
    if opt.task == 'scooping':
        subjects = ['Wonyoung', 'Tom', 'lin', 'Ashwin', 'Song', 'Henry2'] #'Henry', 
        raw_data_path, save_data_path, param_dict = getScooping(opt.task, opt.bDataRenew, \
                                                                opt.bAERenew, opt.bHMMRenew,\
                                                                rf_center, local_range,\
                                                                ae_swtch=opt.bAESwitch, dim=opt.dim)
        
    #---------------------------------------------------------------------------
    elif opt.task == 'feeding':
        subjects = ['Tom', 'lin', 'Ashwin', 'Song'] #'Wonyoung']
        raw_data_path, save_data_path, param_dict = getFeeding(opt.task, opt.bDataRenew, \
                                                               opt.bAERenew, opt.bHMMRenew,\
                                                               rf_center, local_range,\
                                                               ae_swtch=opt.bAESwitch, dim=opt.dim)
        
    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing_microwhite':
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingMicroWhite(opt.task, opt.bDataRenew, \
                                                                         opt.bAERenew, opt.bHMMRenew,\
                                                                         rf_center, local_range, \
                                                                         ae_swtch=opt.bAESwitch, dim=opt.dim)
                                                                         
    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing_microblack':
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingMicroBlack(opt.task, opt.bDataRenew, \
                                                                         opt.bAERenew, opt.bHMMRenew,\
                                                                         rf_center, local_range, \
                                                                         ae_swtch=opt.bAESwitch, dim=opt.dim)
        
    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing_toolcase':
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingToolCase(opt.task, opt.bDataRenew, \
                                                                       opt.bAERenew, opt.bHMMRenew,\
                                                                       rf_center, local_range, \
                                                                       ae_swtch=opt.bAESwitch, dim=opt.dim)
        
    else:
        print "Selected task name is not available."
        sys.exit()


    #---------------------------------------------------------------------------           
    #---------------------------------------------------------------------------           
    #---------------------------------------------------------------------------                  
    onlineEvaluation(opt.task, raw_data_path, save_data_path, param_dict, renew=opt.bRenew )
            


    

    
