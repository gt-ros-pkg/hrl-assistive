
import commands
import numpy as np, math
import scipy.optimize as so
import os, sys, time
import matplotlib.pyplot as pp

from mvpa2.datasets.base import Dataset
from mvpa2.clfs.knn import kNN
from mvpa2.clfs.libsvmc.svm import SVM
## from mvpa2.clfs.transerror import TransferError
## from mvpa2.algorithms.cvtranserror import CrossValidatedTransferError
## from mvpa2.datasets.splitters import HalfSplitter

from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters

import mechanism_analyse_advait as maa
sys.path.append(os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook')

## import roslib; roslib.load_manifest('modeling_forces')
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import ram_db as rd
import hrl_lib.util as ut
import hrl_lib.matplotlib_util as mpu



def filter_pkl_list(pkl_list, typ):
    filt_list = []
    len_list = []
    for pkl in pkl_list:
        d = ut.load_pickle(pkl)
        if d['typ'] != typ:
            continue
        filt_list.append(pkl)
    return filt_list

def pkls_to_mech_vec_list(pkl_list, reject_len):
    all_d_list = [ut.load_pickle(pkl) for pkl in pkl_list]
    len_list = []
    d_list = []
    ang_vel_list = []
    linear_vel_list = []

    for d in all_d_list:
        l = 100
        for v in d['vec_list']:
            l = min(l, len(v))
        if l < reject_len:
            ## print 'mechanism:', d['name'], 'len:', l
            continue
        len_list.append(l)
        d_list.append(d)
        
    ## min_max_config = np.min(len_list) # necessary?
    min_max_config = reject_len

    mech_vec_list = []
    mech_nm_list = []
    for d in d_list:
        mech_vec_list.append(np.matrix(d['vec_list'])[:,:min_max_config].T)
        mech_nm_list.append(d['name'])

        if 'vel_list' in d:
            for av in d['vel_list']:
                if av < 0:
                    continue
                if av > math.radians(50):
                    continue

                ang_vel_list.append(av)
                linear_vel_list.append(av * d['rad'])
    
#    print 'max linear vel:', np.max(linear_vel_list)
#    print 'min linear vel:', np.min(linear_vel_list)
#    print 'mean linear vel:', np.mean(linear_vel_list)
#    print 'median linear vel:', np.median(linear_vel_list)
#
#    print 'max ang vel:', math.degrees(np.max(ang_vel_list))
#    print 'min ang vel:', math.degrees(np.min(ang_vel_list))
#    print 'mean ang vel:', math.degrees(np.mean(ang_vel_list))
#    print 'median ang vel:', math.degrees(np.median(ang_vel_list))

    return mech_vec_list, mech_nm_list

def dimen_reduction_mechanisms(pkl_list, dimen):
    mech_vec_list, mech_nm_list = pkls_to_mech_vec_list(pkl_list, 36)
    all_vecs = np.column_stack(mech_vec_list)
    U, s, _ = np.linalg.svd(np.cov((all_vecs)))

    proj_mat = U[:,0:dimen]
    return proj_mat, s, mech_vec_list, mech_nm_list

def viz_pca(proj_mat, s, mech_vec_list):
    perc_account = np.cumsum(s) / np.sum(s)
    pp.plot([0]+list(perc_account))

    mpu.set_figure_size(5.,5.)
    mpu.figure()
    all_vecs = np.column_stack(mech_vec_list)
    mn = np.mean(all_vecs, 1)
    pp.plot(mn/np.linalg.norm(mn), color = '#FF3300',
            label = 'mean (normalized)', ms=5)

    c_list = ['#%02x%02x%02x'%(r,g,b) for (r,g,b) in [(152, 32, 176), (23,94,16)]]
    c_list = ['#00CCFF', '#643DFF']
    for i in range(2):
        pp.plot(proj_mat[:,i].flatten(), color = c_list[i],
                    label = 'Eigenvector %d'%(i+1), ms=5)

    mpu.pl.axhline(y=0., color = 'k')
    mpu.legend(display_mode='less_space')

    fig = mpu.figure()
    proj_mat = proj_mat[:, 0:2]
    for v_mat in mech_vec_list:
        color = mpu.random_color()
        for v in v_mat.T:
            v = v.T
            p = proj_mat.T * (v - mn)
            pp.plot(p[1,:].A1, p[0,:].A1, color = color,
                        linewidth = 0, marker='o',
                        picker=0.5)
            pp.xlabel('\huge{First Principal Component}')
            pp.ylabel('\huge{Second Principal Component}')
            pp.axis('equal')

    mpu.pl.axhline(y=0., color = 'k', ls='--')
    mpu.pl.axvline(x=0., color = 'k', ls='--')
    #mpu.legend()
    pp.show()

def plot_reference_trajectory(config, mean, std, typ, title, c1=None,
                              c2=None, label=None, linestyle='-'):
    if typ == 'rotary':
        config = np.degrees(config)

    if c1 == None and c2 == None:
        c1 = '#%02X%02X%02X'%(95, 132, 53)
        c2 = '#%02X%02X%02X'%(196, 251, 100)
    
    if label == None:
        label = 'Human'

    non_nans = ~np.isnan(mean)
    mean = mean[non_nans]
    config = config[non_nans]
    std = std[non_nans]

    lw = 2.
    if linestyle == '-':
        lw = 1.
    pp.plot(config, mean, color=c1, label=label, linewidth=lw,
            linestyle=linestyle)
    pp.fill_between(config, np.array(mean)+np.array(std),
            np.array(mean)-np.array(std), color=c2, alpha=0.1)
    pp.title(title)
    pp.xlabel('Angle (degrees)')
    #pp.ylabel('Tangential Force (N)')
    pp.ylabel('Opening Force (N)')

def compute_mean_std_force_traj(dir_name, plot = False):
    d = maa.extract_pkls(dir_name, True)

    mechx_l_l = d['mechx_l_l']
    ftan_l_l = d['ftan_l_l']

    max_config_l = []
    for mech_x_l in mechx_l_l:
        max_config_l.append(np.max(mech_x_l[1:]))

    if ('typ' not in d) or (d['typ'] == 'rotary'):
        typ = 'rotary'
        bin_size = math.radians(1.)
    else:
        typ = 'prismatic'
        bin_size = 0.005

    if typ == 'rotary':
        print 'max_config_l:', np.degrees(max_config_l)
    else:
        print 'max_config_l:', max_config_l


    lim = np.min(max_config_l)

    vec_list = []
    vel_list = []
    for i in range(len(mechx_l_l)):
        mechx_l = mechx_l_l[i]
        ftan_l = ftan_l_l[i]
        # In case you want to filter by avg vel.
        print 'lim:', lim
        vel = maa.compute_average_velocity(mechx_l, d['time_l_l'][i],
                                           lim, typ)
        vel_list.append(vel)
        #print 'vel:', np.degrees(abs(vel))
        #if vel > math.radians(50):
        #    continue
        vec_list.append(maa.make_vector(mechx_l[1:], ftan_l[1:], lim, bin_size).A1.tolist())
    print 'Number of trials:', i+1

    print 'Before velocity filtering'
    if typ == 'rotary':
        print 'vel_list:', np.degrees(vel_list)
    else:
        print 'vel_list:', vel_list

#    vv = zip(vel_list, vec_list)
#    vv.sort()
#    vel_list, vec_list = zip(*vv[:3]) # take least three velocities
#
#    print 'After velocity filtering'
#    if typ == 'rotary':
#        print 'vel_list:', np.degrees(vel_list)
#    else:
#        print 'vel_list:', vel_list


    vec_mat = np.matrix(vec_list)
    mean_force_traj = vec_mat.mean(0).A1
    std_force_traj = vec_mat.std(0).A1
    config_traj = bin_size * np.array(range(len(mean_force_traj)))

    d2 = {}
    d2['vel_list'] = vel_list
    d2['vec_list'] = vec_list
    d2['mean'] = mean_force_traj
    d2['std'] = std_force_traj
    d2['config'] = config_traj
    nm = dir_name.split('/')[-2]
    d2['name'] = nm
    d2['typ'] = typ
    d2['rad'] = d['rad']

    ut.save_pickle(d2, nm+'.pkl')

    if plot:
        maa.plot_tangential_force(dir_name, 'aloha')
        #plot_reference_trajectory(d2['config'], d2['mean'], d2['std'],
        #                          d2['typ'], d2['name'])

def create_mvpa_dataset(proj_mat, mech_vec_list, mech_nm_list):
    selected_vec_list = []
    labels = []
    lab = 0
    for i, v_mat in enumerate(mech_vec_list):
        nm = mech_nm_list[i]

        if 'microwave' in nm:
            continue

        selected_vec_list.append(v_mat)

        if nm not in rd.tags_dict:
            ## print nm + ' is not in tags_dict'
            #raw_input('Hit ENTER to continue')
            continue
        tags = rd.tags_dict[nm]
        for v in v_mat.T:
            #labels.append(lab)
            #labels.append(mech_nm_list[i])
            labels.append(tags[-1])
        lab = lab+1

    all_vecs = np.column_stack(selected_vec_list)
    feats = proj_mat.T * all_vecs

    data = Dataset(samples=feats.A.T, labels=labels)
    return data

def create_mvpa_dataset_semantic_classes(proj_mat, mech_vec_list, mech_nm_list):
    all_vecs = np.column_stack(mech_vec_list)
    labels = []
    lab_num = 0
    chunk_num = 0
    feat_list = []
    chunks = []
    for i, v_mat in enumerate(mech_vec_list):
        nm = mech_nm_list[i]
        if nm not in rd.tags_dict:
            ## print nm + ' is not in tags_dict'
            raw_input('Hit ENTER to continue')
            continue
        tags = rd.tags_dict[nm]
        if rd.ig in tags:
            continue
        if rd.k in tags and rd.r in tags:
            lab_str = 'Refrigerator'
            lab_num = 0
        elif rd.k in tags and rd.f in tags:
            lab_str = 'Freezer'
            lab_num = 1
        elif rd.k in tags and rd.c in tags:
            lab_str = 'Kitchen Cabinet'
            lab_num = 2
        elif rd.o in tags and rd.c in tags:
            lab_str = 'Office Cabinet'
            lab_num = 3
        elif rd.do in tags and rd.s in tags:
            lab_str = 'Springloaded Door'
            lab_num = 4
        else:
            continue
        for v in v_mat.T:
            labels.append(lab_str)
            chunks.append(mech_nm_list[i])
            #labels.append(lab_num)
            #chunks.append(chunk_num)
        feat_list.append(proj_mat.T * v_mat)
        chunk_num = chunk_num+1
    feats = np.column_stack(feat_list)
    #chunks=None
    data = Dataset(samples=feats.A.T, labels=labels, chunks=chunks)
    return data

def compute_cv_error(data, clf, plot_confusion = False):
    print 'before training'
    clf.train(data)
    print 'after training'

    terr = TransferError(clf)
    cvterr = CrossValidatedTransferError(terr,
                            NFoldSplitter(cvtype=1),
                            enable_states=['confusion', 'samples_error'])
    error = cvterr(data)
    #look_at_confusion_in_detail(data, cvterr)
    if plot_confusion:
        #cvterr.confusion.plot(cmap='gray_r', numbers=True)
        cvterr.confusion.plot(cmap='gray_r', numbers=False)
    return error

def look_at_confusion_in_detail(data, cvterr):
    di = cvterr.samples_error
    for k in di:
        err = di[k][0]
        if err != 0:
            print 'Semantic Class:', data.labels[k]
            print 'Mechanism Name:', data.chunks[k]

def variation_with_dimensions(proj_mat, mech_vec_list, mech):
    mpu.set_figure_size(10, 6.)
    pp.figure()
    n_dim = proj_mat.shape[1]
    color_list = ['#FF6633', '#66FF33', '#3366FF']
    clf1 = SVM('poly', svm_impl='C_SVC', C=2) # 4.8% error
    clf2 = kNN(1)
    clf3 = kNN(3)
    lab_list = ['k=3', 'k=1', 'SVM']
    marker_list = ['x', '+', 'o']
    clf_list = [clf3, clf2, clf1]
    for j, clf in enumerate(clf_list):
        err_list = []
        dim_list = []
        for i in range(1,n_dim+1):
            if j == 2 and i == 1:
                continue
            print 'dimen:', i
            data = create_mvpa_dataset(proj_mat[:,:i], mech_vec_list,
                                       mech_nm_list)
            err_list.append(compute_cv_error(data, clf)*100)
            dim_list.append(i)
        #pp.bar(dim_list, err_list, align='center', linewidth=0)
        if j == 2:
            ms = 4
        else:
            ms = 8
        pp.plot(dim_list, err_list, color=color_list[j],
                label=lab_list[j], ms=ms, marker=marker_list[j],
                mew=2, mec=color_list[j])
    pp.xlabel('Dimensionality of sub-space')
    pp.ylabel('Crossvalidation Error (\%)')
    pp.xlim(0,21)
    pp.ylim(0, 80)
    pp.legend()
    f = pp.gcf()
    f.subplots_adjust(bottom=.18, top=.94, right=.98, left=0.15)
    pp.savefig('parameter_effect.pdf')

def test_mvpa(proj_mat, mech_vec_list, mech_nm_list):
    data = create_mvpa_dataset(proj_mat, mech_vec_list, mech_nm_list)
    #data = create_mvpa_dataset_semantic_classes(proj_mat, mech_vec_list, mech_nm_list)

    mpu.figure()
    clf = kNN(1)
    #clf = SVM(svm_impl='C_SVC', C=200) # 6.2% error
    #clf = SVM('rbf', svm_impl='C_SVC', C=200, gamma=0.001) # 6.2% error
    #clf = SVM('poly', svm_impl='C_SVC', C=200, degree=3) # 4.8% error
    error = compute_cv_error(data, clf, plot_confusion = True)
    print 'Error:', error
    pp.show()

def create_raw_profiles_dataset_semantic(mech_vec_list, mech_nm_list):
    proj_mat = np.matrix(np.eye(mech_vec_list[0].shape[0]))
    data = create_mvpa_dataset_semantic_classes(proj_mat, mech_vec_list, mech_nm_list)
    return data

def initial_force_histogram(mech_vec_list, mech_nm_list):
    mpu.set_figure_size(11., 5.)
    data = create_raw_profiles_dataset_semantic(mech_vec_list, mech_nm_list)
    labels = data.uniquelabels.tolist()
    mn_list, std_list = [], []
    angle = 10. # angle over which to consider the max force.
    for l in labels:
        force_trajs = data.samples[np.where(data.labels == l)]
        max_force = np.max(force_trajs[:, 0:10], 1)
        mn_list.append(np.mean(max_force))
        std_list.append(np.std(max_force))

    fig = pp.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(labels))
    rect1 = ax.barh(ind, mn_list, xerr=std_list, align='center',
                    color='y', linewidth=1, ecolor='g', capsize=5)
    ax.set_yticks(ind)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Force (Newtons)')
    ax.set_ylabel('Mechanism class')
    f = pp.gcf()
    f.subplots_adjust(bottom=.2, top=.97, right=.96, left=0.35)
    pp.savefig('locked_force.pdf')
    pp.show()

#---------------- blocked analysis --------------------
#
# lets perform this analysis for freezer, fridge, and office cabinet
# class. I have maximum data for these classes.
#
#-----------------------------------------------------

def create_blocked_dataset_semantic_classes(mech_vec_list,
                                            mech_nm_list, append_robot):
    all_vecs = np.column_stack(mech_vec_list)
    lab_num = 0
    chunk_num = 0
    labels = []
    feat_list = []
    chunks = []
    labels_test = []
    feat_list_test = []
    chunks_test = []
    for i, v_mat in enumerate(mech_vec_list):
        nm = mech_nm_list[i]
        if nm not in rd.tags_dict: #name filtering
            ## print nm + ' is not in tags_dict'
            #raw_input('Hit ENTER to continue')
            continue
        tags = rd.tags_dict[nm]
        if 'recessed' in nm:
            continue
        if 'HSI_Executive_Board_Room_Cabinet_Left' in nm:
            continue

        if rd.ig in tags or rd.te in tags:
            continue

        if rd.k in tags and rd.r in tags:
            #lab_str = 'Refrigerator'
            lab_str = 'Fridge'
            lab_num = 0
        elif rd.k in tags and rd.f in tags:
            lab_str = 'Freezer'
            lab_num = 1
        elif rd.k in tags and rd.c in tags:
            lab_str = 'Kitchen Cabinet'
            lab_num = 2
        elif rd.o in tags and rd.c in tags:
            lab_str = 'Office Cabinet'
            lab_num = 3
            if 'HSI_kitchen_cabinet_left' in nm:
                v_mat = 1.0*v_mat + 0.
        elif rd.do in tags and rd.s in tags:
            lab_str = 'Springloaded Door'
            lab_num = 4
        else:
            continue
        for v in v_mat.T:
            if rd.te in tags:
                labels_test.append(lab_str)
                chunks_test.append(mech_nm_list[i])
            else:
                labels.append(lab_str)
                if rd.ro in tags and append_robot:
                    chunks.append(mech_nm_list[i]+'_robot')
                else:
                    chunks.append(mech_nm_list[i])
        if rd.te in tags:
            feat_list_test.append(v_mat)
        else:
            feat_list.append(v_mat)
        ## print '-------------------------'
        ## print 'nm:', nm
        if nm == 'HSI_kitchen_cabinet_right':
            print '####################33'
            print '####################33'
            print '####################33'

    ## print labels # mechanism
    ## print len(chunks) # mechanism + actor?
            
    #chunks=None
    feats = np.column_stack(feat_list)
    data = Dataset.from_wizard(samples=feats.A.T, targets=labels, chunks=chunks) # make samples with labels, chunks is name of sample

    if feat_list_test == []:
        ## print '############################3'
        ## print 'feat_list_test is empty'
        ## print '############################3'
        data_test = data
    else:
        feats = np.column_stack(feat_list_test)
        data_test = Dataset(samples=feats.A.T, labels=labels_test, chunks=chunks_test)

    return data, data_test


def blocked_detection_n_equals_1(mech_vec_list, mech_nm_list):
    data, _ = create_blocked_dataset_semantic_classes(mech_vec_list, mech_nm_list, append_robot = False)
    nfs = NFoldPartitioner(cvtype=1, attr='targets') # 1-fold ?
    spl = splitters.Splitter(attr='partitions')
    splits = [list(spl.generate(x)) for x in nfs.generate(data)]
    
    ## splitter = NFoldSplitter(cvtype=1)
    ## label_splitter = NFoldSplitter(cvtype=1, attr='labels')
    mean_thresh_known_mech_dict = {}
    for l_wdata, l_vdata in splits:
        mean_thresh_known_mech_list = []
        Ms = compute_Ms(data, l_vdata.targets[0], plot=False)

        mechs = l_vdata.uniquechunks
        for m in mechs:                            
            n_std = 0.
            all_trials = l_vdata.samples[np.where(l_vdata.chunks == m)]
            le = all_trials.shape[1]
            for i in range(all_trials.shape[0]):
                one_trial = all_trials[i,:].reshape(1,le)
                mn_list, std_list = estimate_theta(one_trial, Ms, plot=False)
                mn_arr, std_arr = np.array(mn_list), np.array(std_list)

                n_std = max(n_std, np.max(np.abs(all_trials - mn_arr) / std_arr))
                
                ## if np.isnan(np.max(std_arr)):
                ##     for i,std in enumerate(std_arr):
                ##         if np.isnan(std): continue
                ##         new_n_std = np.max(np.abs(all_trials[:,i] - mn_arr[i]) / std_arr[i])
                ##         if n_std < new_n_std:
                ##             n_std = new_n_std
                ## else:
                ##     n_std = max(n_std, np.max(np.abs(all_trials - mn_arr) / std_arr))


            mean_thresh_known_mech_dict[m] = (Ms, n_std) # store on a per mechanism granularity
            print 'n_std for', m, ':', n_std
            print 'max error force for', m, ':', np.max(n_std*std_arr[2:]) #, ' ', std_arr[2:]
            

            ## if m=='Jason_refrigerator':
            ##     print Ms
            ##     print n_std, np.max(np.abs(all_trials - mn_arr) / std_arr)
            ##     print all_trials - mn_arr
            ##     sys.exit()
            

    d = ut.load_pickle('blocked_thresh_dict.pkl')
    d['mean_known_mech'] = mean_thresh_known_mech_dict
    ut.save_pickle(d, 'blocked_thresh_dict.pkl')

    
def blocked_detection(mech_vec_list, mech_nm_list):

    # data consists of (mech_vec_matrix?, label_string(Freezer...), mech_name)
    data, _ = create_blocked_dataset_semantic_classes(mech_vec_list,
                                    mech_nm_list, append_robot = True)    

    ## # there can be multiple chunks with a target, chunk is unique...
    ## mean_thresh_charlie_dict = {}    
    ## for chunk in data.uniquechunks:
    ##     non_robot_idxs = np.where(['robot' not in i for i in data.chunks])[0] # if there is no robot, true 
    ##     idxs = np.where(data.chunks[non_robot_idxs] == chunk)[0] # find same target samples in non_robot target samples
    ##     train_trials = (data.samples[non_robot_idxs])[idxs]

    ##     # skip empty set
    ##     if (train_trials.shape)[0] == 0: continue

    ##     mean_force_profile = np.mean(train_trials, 0)
    ##     std_force_profile = np.std(train_trials, 0)
            
    ##     if 'robot' in chunk:
    ##         # remove the trailing _robot
    ##         key = chunk[0:-6]
    ##     else:
    ##         key = chunk
    ##     mean_thresh_charlie_dict[key] = (mean_force_profile * 0.,
    ##                                      mean_force_profile, std_force_profile)

        
    # create the generator
    #label_splitter = NFoldSplitter(cvtype=1, attr='labels')
    nfs = NFoldPartitioner(cvtype=1) # 1-fold ?
    spl = splitters.Splitter(attr='partitions')
    splits = [list(spl.generate(x)) for x in nfs.generate(data)]

    # 1) Split by a chunk
    # 2) Select a chunk
    # 3) Find non-robot data in the same class() with the chunk
    # Pick each chunk once in l_vdata, where the set of chunks are unique
    # NOTE: l_wdata does not include the sample of l_vdata. See autonomous robots paper
    mean_thresh_charlie_dict = {}
    #for l_wdata, l_vdata in label_splitter(data):
    for l_wdata, l_vdata in splits:
        print l_vdata.chunks
        sys.exit()
        
        non_robot_idxs = np.where(['robot' not in i for i in l_wdata.chunks])[0] # if there is no robot, true 
        idxs = np.where(l_wdata.targets[non_robot_idxs] == l_vdata.targets[0])[0] # find same target samples in non_robot target samples
        train_trials = (l_wdata.samples[non_robot_idxs])[idxs]

        #idxs = np.where(l_wdata.labels == l_vdata.labels[0])[0]
        #train_trials = (l_wdata.samples)[idxs]

#        print 'train_trials.shape', train_trials.shape
#        print 'train_trials.labels', l_wdata.labels[idxs]
#        print 'train_trials.chunks', l_wdata.chunks[idxs]

        mean_force_profile = np.mean(train_trials, 0)
        std_force_profile = np.std(train_trials, 0)
        
#        print '#######################'
#        print 'l_vdata.chunks[0]', l_vdata.chunks[0]
#        mean_thresh_charlie_dict[l_vdata.labels[0]] = (mean_force_profile * 0.,
#                                                       mean_force_profile,
#                                                       std_force_profile)
        if 'robot' in l_vdata.chunks[0]:
            # remove the trailing _robot
            key = l_vdata.chunks[0][0:-6]
        else:
            key = l_vdata.chunks[0]

        mean_thresh_charlie_dict[key] = (mean_force_profile * 0.,
                                         mean_force_profile, std_force_profile)
        
    d = {'mean_charlie': mean_thresh_charlie_dict}
    ut.save_pickle(d, 'blocked_thresh_dict.pkl')
    pp.show()
    

def generate_roc_curve(mech_vec_list, mech_nm_list,
                       semantic_range = np.arange(0.2, 6.4, 0.3),
                       mech_range = np.arange(0.2, 6.5, 0.7),
                       n_prev_trials = 1, prev_c = 'r',
                       plot_prev=True, sem_c = 'b', sem_m = '+',
                       plot_semantic=True, semantic_label='operating 1st time and \n known mechanism class'):

    t_nm_list, t_mech_vec_list = [], []
    for i, nm in enumerate(mech_nm_list):
        ## print 'nm:', nm
        if 'known' in nm:
            continue
        t_nm_list.append(nm)
        t_mech_vec_list.append(mech_vec_list[i])

    # test data
    data, _ = create_blocked_dataset_semantic_classes(t_mech_vec_list, t_nm_list, append_robot = False)

    # To decide mean and var
    ## label_splitter = NFoldSplitter(cvtype=1, attr='labels')
    thresh_dict = ut.load_pickle('blocked_thresh_dict.pkl')
    mean_charlie_dict = thresh_dict['mean_charlie']
    mean_known_mech_dict = thresh_dict['mean_known_mech']

    #---------------- semantic class prior -------------
    if plot_semantic:
        fp_l_l = []
        mn_l_l = []
        err_l_l = []
        mech_fp_l_l = []
        mech_mn_l_l = []
        mech_err_l_l = []

        ## nfs = NFoldPartitioner(cvtype=1, attr='targets') # 1-fold ?
        nfs = NFoldPartitioner(cvtype=1, attr='chunks') # 1-fold ?
        label_splitter = splitters.Splitter(attr='partitions')            
        splits = [list(label_splitter.generate(x)) for x in nfs.generate(data)]            
            
        for l_wdata, l_vdata in splits: #label_splitter(data):

            # Why zero??? who is really mean?  -> changed into 10
            lab = l_vdata.targets[0]
            chunk = l_vdata.chunks[0]
            trials = l_vdata.samples # all data
            
            if lab == 'Refrigerator':
                lab = 'Fridge'

            # mean and std of data except chunk
            #_, mean, std =  mean_charlie_dict[lab]
            _, mean, std =  mean_charlie_dict[chunk]

            # cutting into the same length
            min_len = min(len(mean), trials.shape[1])
            trials = trials[:,:min_len]
            mean = mean[:min_len]
            std = std[:min_len]

            mn_list = []
            fp_list, err_list = [], []
            for n in semantic_range:
                err = (mean + n*std) - trials                    
                #false_pos = np.sum(np.any(err<0, 1))
                #tot = trials.shape[0]
                false_pos = np.sum(err<0)
                tot = trials.shape[0] * trials.shape[1]
                fp_list.append(false_pos/(tot*0.01))
                err = err[np.where(err>0)]
                err_list.append(err.flatten())
                mn_list.append(np.mean(err))
            err_l_l.append(err_list)
            fp_l_l.append(fp_list)
            mn_l_l.append(mn_list)

        ll = [[] for i in err_l_l[0]]
        for i,e in enumerate(err_l_l):
            for j,l in enumerate(ll):
                l.append(e[j])

        std_list = []
        for l in ll:
            std_list.append(np.std(np.concatenate(l).flatten()))

        mn_list = np.mean(np.row_stack(mn_l_l), 0).tolist()
        fp_list = np.mean(np.row_stack(fp_l_l), 0).tolist()
        #pp.errorbar(fp_list, mn_list, std_list)

        ## mn_list = np.array(mn_l_l).flatten()
        ## fp_list = np.array(fp_l_l).flatten()
        
        pp.plot(fp_list, mn_list, '--'+sem_m+sem_c, label= semantic_label,
                mec=sem_c, ms=8, mew=2)
        #pp.plot(fp_list, mn_list, '-ob', label='with prior')

    #---------------- mechanism knowledge prior -------------
    if plot_prev:
        
        t_nm_list, t_mech_vec_list = [], []
        for i, nm in enumerate(mech_nm_list):
            ## print 'nm:', nm
            if 'known' in nm:
                t_nm_list.append(nm)
                t_mech_vec_list.append(mech_vec_list[i])
        if t_nm_list == []:
            t_mech_vec_list = mech_vec_list
            t_nm_list = mech_nm_list

        data, _ = create_blocked_dataset_semantic_classes(t_mech_vec_list, t_nm_list, append_robot = False)
        
        ## chunk_splitter = NFoldSplitter(cvtype=1, attr='chunks')        
        nfs = NFoldPartitioner(cvtype=1, attr='chunks') # 1-fold ?
        chunk_splitter = splitters.Splitter(attr='partitions')            
        ## splits = [list(label_splitter.generate(x)) for x in nfs.generate(data)]            
        splits = [list(chunk_splitter.generate(x)) for x in nfs.generate(data)]            
        
        err_mean_list = []
        err_std_list = []
        fp_list = []
        for n in mech_range:
            false_pos = 0
            n_trials = 0
            err_list = []
            for _, l_vdata in splits: #chunk_splitter(data):
                lab = l_vdata.targets[0]
                trials = l_vdata.samples
                m = l_vdata.chunks[0]
                #one_trial = trials[0].reshape(1, len(trials[0]))
                one_trial = trials[0:n_prev_trials]

                Ms, n_std = mean_known_mech_dict[m]
                mn_list, std_list = estimate_theta(one_trial, Ms, plot=False, add_var = 0.0)
                mn_mech_arr = np.array(mn_list)
                std_mech_arr = np.array(std_list)

    #            trials = trials[:,:len(mn_mech_arr)]
                min_len = min(len(mn_mech_arr), trials.shape[1])
                trials = trials[:,:min_len]
                mn_mech_arr = mn_mech_arr[:min_len]
                std_mech_arr = std_mech_arr[:min_len]

                for t in trials:
                    err = (mn_mech_arr + n*std_mech_arr) - t
                    #false_pos += np.any(err<0)
                    #n_trials += 1
                    false_pos += np.sum(err<0)
                    n_trials += len(err)
                    err = err[np.where(err>0)]
                    err_list.append(err)

            e_all = np.concatenate(err_list)
            err_mean_list.append(np.mean(e_all))
            err_std_list.append(np.std(e_all))
            fp_list.append(false_pos/(n_trials*0.01))

        #pp.plot(fp_list, err_mean_list, '-o'+prev_c, label='knowledge of mechanism and \n opened earlier %d times'%n_prev_trials)
        pp.plot(fp_list, err_mean_list, '-o'+prev_c, mec=prev_c,
                ms=5, label='operating 2nd time and \n known mechanism identity')
        #pp.plot(fp_list, err_mean_list, '-or', label='with prior')


    pp.xlabel('False positive rate (percentage)')
    pp.ylabel('Mean excess force (Newtons)')
    pp.xlim(-0.5,45)
    mpu.legend()

def generate_roc_curve_no_prior(mech_vec_list, mech_nm_list):
    #pp.figure()
    data, _ = create_blocked_dataset_semantic_classes(mech_vec_list, mech_nm_list, append_robot = False)
    ## chunk_splitter = NFoldSplitter(cvtype=1, attr='chunks')

    err_mean_list = []
    err_std_list = []
    fp_list = []
    all_trials = data.samples
    n_trials = all_trials.shape[0] * all_trials.shape[1]
    le = all_trials.shape[1]
    for n in np.arange(0.1, 1.7, 0.15):
        err = (all_trials[:,0]*n).T - all_trials.T + 2.
        false_pos = np.sum(err<0)
        err = err[np.where(err>0)]
        err_mean_list.append(np.mean(err))
        err_std_list.append(np.std(err))
        fp_list.append(false_pos/(n_trials*0.01))

    pp.plot(fp_list, err_mean_list, ':xy', mew=2, ms=8, label='No prior (ratio of \n initial force)', mec='y')

    err_mean_list = []
    err_std_list = []
    fp_list = []
    for f in np.arange(2.5, 45, 5.):
        err = f - all_trials
        false_pos = np.sum(err<0)
        err = err[np.where(err>0)]
        err_mean_list.append(np.mean(err))
        err_std_list.append(np.std(err))
        fp_list.append(false_pos/(n_trials*0.01))

    pp.plot(fp_list, err_mean_list, '-.^g', ms=8, label='No prior (constant)', mec='g')

    pp.xlabel('False positive rate (percentage)')
    pp.ylabel('Mean excess force (Newtons)')
    #mpu.legend(display_mode='less_space', draw_frame=False)
    mpu.legend()
    pp.xlim(-0.5,45)

def compute_Ms(data, semantic_class, plot):

    nfs = NFoldPartitioner(cvtype=1) # 1-fold ?
    spl = splitters.Splitter(attr='targets',attr_values=[semantic_class])
    a   = [list(spl.generate(x)) for x in nfs.generate(data)]
    semantic_data = a[0][0]

    ## label_splitter = NFoldSplitter(cvtype=1, attr='labels')
    ## a = label_splitter.splitDataset(data, [semantic_class])
    ## semantic_data = a[0]

    mechs = semantic_data.uniquechunks
    mn_list, std_list = [], []
    if plot:
        pp.figure()
    for m in mechs:
        trials = semantic_data.samples[np.where(semantic_data.chunks == m)]
        mn = np.mean(trials, 0)
        std = np.std(trials, 0)
        mn_list.append(mn)
        std_list.append(std)
        if plot:
            pp.errorbar(range(len(mn)), mn, std, color=mpu.random_color(),
                        label=m)
    if plot:
        pp.legend()

    mn_mn_arr = np.row_stack(mn_list)
    mn_mn = np.mean(mn_mn_arr, 0)
    mn_std = np.std(mn_mn_arr, 0)

    std_mn_arr = np.row_stack(std_list)
    var_arr = std_mn_arr * std_mn_arr
    var_mn = np.mean(var_arr, 0)
    var_std = np.std(var_arr, 0)

    if plot:
        fig = pp.figure()

        ax1 = fig.add_subplot(111)
        c1 = 'r'
        ax1.errorbar(range(len(mn)), mn_mn, mn_std, color=c1)
        ax1.set_xlabel('angle (degrees)')
        ax1.set_xlim(-1, len(mn)+0.1)
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('mean', color=c1)
        for tl in ax1.get_yticklabels():
            tl.set_color(c1)

        ax2 = ax1.twinx()
        c2 = 'b'
        ax2.errorbar(range(len(mn)), mn_std, var_std, color=c2)
        ax2.set_ylabel('std', color=c2)
        for tl in ax2.get_yticklabels():
            tl.set_color(c2)
        pp.legend()
        pp.title(semantic_class)
        fig.savefig('/home/dpark/Dropbox/HRL/mech.pdf', format='pdf')

    #return mn_mn, std_mn, mn_std, std_std
    return mn_mn, var_mn, mn_std, var_std


##
# return the mean and std that we should expect at every configuration
# by mixing the prior and the data from the few trials that we have.
def estimate_theta(trials, Ms, plot, add_var = 0.):
    mn_mn, var_mn, mn_std, var_std = Ms
#    nm_l = [
#            'kitchen_cabinet_noisy_cody',
#            'ikea_cabinet_noisy_cody',
#            'kitchen_cabinet_noisy_pr2',
#            'ikea_cabinet_noisy_pr2',
#            #'lab_spring_door_noisy_cody',
#            ]
#
#    v_l = []
#    len_l = []
#    for nm in nm_l:
#        d = ut.load_pickle('RAM_db/robot_trials/simulate_perception/'+nm+'.pkl')
#        v = np.array(d['std'])*np.array(d['std'])
#        v_l.append(v)
#        len_l.append(len(v))
#
#    min_l = np.min(len_l)
#    v_arr = np.row_stack([v[:min_l] for v in v_l])
#    var_mn = np.mean(v_arr, 0).tolist()

    le = trials.shape[1]
    mn_mn = mn_mn[:le]
    var_mn = var_mn[:le]
    mn_std = mn_std[:le]
    var_std = var_std[:le] + add_var

    pred_mn_list, pred_std_list = [], []
    for config in range(len(mn_mn)):
        f_config = trials[:, config].flatten()
        mu_s = mn_mn[config]
        var_s = var_mn[config]
        sigma_s = math.sqrt(var_s)
        sigma_mu_s = mn_std[config]
        sigma_var_s = var_std[config]

        def error_function(params):
            mu, sigma_sq = params[0], params[1]
            sigma_sq = abs(sigma_sq)
            sigma = math.sqrt(sigma_sq)
            t1 = (f_config - mu)/sigma
            res = len(t1) * np.log(sigma)
            res += np.sum(t1*t1) + ((mu-mu_s)/sigma_mu_s)**2 + ((sigma_sq-var_s)/sigma_var_s)**2
            return res

        params = [np.mean(f_config), var_s]
        r = so.fmin_bfgs(error_function, params, full_output=1,
                         disp = False, gtol=1e-5)
        result = r[0]
        pred_mn_list.append(result[0])
        var_opt = abs(result[1])
        pred_std_list.append(math.sqrt(var_s))
        
    if plot:
        pp.figure()
        pp.plot(mn_mn, 'g', label='Mean mean')
        trial_mn = np.mean(trials, 0)
        trial_std = np.std(trials, 0)
        pp.errorbar(range(len(pred_mn_list)), trial_mn, trial_std,
                    color='b', label='Trial Mean (std)')
        pp.errorbar(range(len(pred_mn_list)), pred_mn_list, pred_std_list,
                    color='r', label='Prediction')
        pp.legend()
        #pp.show()
    
    return pred_mn_list, pred_std_list

def test_bayesian(mech_vec_list, mech_nm_list):
    data, _ = create_blocked_dataset_semantic_classes(mech_vec_list, mech_nm_list)
    #compute_Ms(data, 'Office Cabinet', plot=True)
    #compute_Ms(data, 'Freezer', plot=True)
    #compute_Ms(data, 'Refrigerator', plot=True)
    #compute_Ms(data, 'Springloaded Door', plot=True)
    #pp.show()

    #test_trials = data.samples[np.where(data.chunks=='advait_refrigerator')]
    test_trials = data.samples[np.where(data.chunks=='hai_refrigerator')]
    Ms = compute_Ms(data, 'Refrigerator', plot=False)
    #test_trials = data.samples[np.where(data.chunks=='HSI_kitchen_cabinet_left')]
    #Ms = compute_Ms(data, 'Office Cabinet', plot=False)
    estimate_theta(test_trials[6:7,:], Ms, plot=True)
    for t in test_trials:
        pp.plot(range(len(t)), t, color='g', alpha=0.7, label='all trials')
    pp.legend()
    pp.show()


#--------------- sharing haptic experience ----------

def plot_different_mechanisms(nm_l, lab_l, c_l, max_len, st_list = None,
                              linestyle_l = None):
    if st_list == None:
        st_list = len(c_l) * [0]

    if linestyle_l == None:
        linestyle_l = len(c_l) * ['-']

    for i in range(len(nm_l)):
        c, nm, lab = c_l[i], nm_l[i], lab_l[i]
        ls = linestyle_l[i]
        st = st_list[i]
        ## print 'nm:', nm
        d = ut.load_pickle('RAM_db/'+nm+'.pkl')
        plot_reference_trajectory(d['config'][:max_len-st],
                d['mean'][st:max_len], d['std'][st:max_len], d['typ'],
                d['name'], c, c, lab, ls)
    mpu.legend()


def compare_cody_ikea_cabinet():
    pp.figure()
    max_len = 38

    nm_l = [
            'ikea_cabinet_pos1_10cm_cody',
            'ikea_cabinet_pos1_5cm_cody',
            'ikea_cabinet_pos2_10cm_cody',
            'ikea_cabinet_pos2_5cm_cody',
            'ikea_cabinet_move_pos1_5cm_cody',
            ]
    c_l = ['r', 'g',
            'b', 'c',
            'y'
            ]
    lab_l = nm_l
#    lab_l = ['Cody (10cm/s)', 'Cody (5cm/s)',
#             #'2.5cm/s',
#             'PR2']

    plot_different_mechanisms(nm_l, lab_l, c_l, max_len)

def pr2_cody_error_histogram():
    pp.figure()
    max_len = 40

    cody_pkl = 'ikea_cabinet_cody_5cm_cody'
    d = ut.load_pickle('RAM_db/'+cody_pkl+'.pkl')
    cody_mn = d['mean'][:max_len]

    pr2_pkl = 'ikea_cabinet_pr2'
    d = ut.load_pickle('RAM_db/'+pr2_pkl+'.pkl')
    pr2_mn = d['mean'][:max_len]

    cody_pkl = 'ikea_cabinet_cody_10cm_cody'
    d = ut.load_pickle('RAM_db/'+cody_pkl+'.pkl')
    cody_fast_mn = d['mean'][:max_len]

    bins = np.arange(-0.75, 1.5, 0.5)
    dat = np.column_stack([cody_mn-pr2_mn, cody_fast_mn-cody_mn])
    lab_l = ['Cody(5cm/s)-PR2', 'Cody(10cm/s)-Cody(5cm/s)']
    pp.hist(dat, bins, rwidth=0.6, label=lab_l)
    mpu.legend()
    pp.xlabel('Force difference (Newtons)')
    pp.ylabel('Count')

def compare_pr2_cody_ikea_cabinet():
    pp.figure()
    max_len = 38
#    nm_l = ['ikea_cabinet_thick_10cm_cody',
#            'ikea_cabinet_thick_5cm_cody',
#            'ikea_cabinet_thick_2.5cm_cody',
#            'ikea_cabinet_pr2']

    nm_l = [#'ikea_cabinet_cody_10cm_cody',
            #'ikea_cabinet_pos1_cody',
            'ikea_cabinet_pos1_pr2',
            'ikea_cabinet_pos2_pr2',
            'ikea_cabinet_pos3_pr2',
#            'ikea_cabinet_cody_5cm_cody',
#            'ikea_cabinet_cody_2.5cm_cody',
            #'ikea_cabinet_pr2'
            ]
    c_l = ['r', 'g', 'b', 'y']
    lab_l = nm_l

    plot_different_mechanisms(nm_l, lab_l, c_l, max_len)

def compare_pr2_different_positions():
    mpu.set_figure_size(10., 7.)
    pp.figure()
    max_len = 35
    nm_l = [
            'diff_pos/ikea_cabinet_pos1_pr2',
            'diff_pos/ikea_cabinet_pos2_pr2',
            'diff_pos/ikea_cabinet_pos3_pr2',
            ]
    c_l = ['r', 'b', 'y']
    lab_l = ['Position 1', 'Position 2', 'Position 3']
    linestyle_l = [':', '--', '-']

    plot_different_mechanisms(nm_l, lab_l, c_l, max_len,
                              linestyle_l=linestyle_l)
    f = pp.gcf()
    pp.title('')
    f.subplots_adjust(bottom=.15, top=.97, right=.98)
    pp.xlim(0., 35.)
    pp.savefig('different_pos_forces.pdf')
    pp.show()

def human_pr2_error_histogram():
    pp.figure()
    max_len = 45

    one_pkl = 'HSI_kitchen_cabinet_right'
    d = ut.load_pickle('RAM_db/'+one_pkl+'.pkl')
    one_mn = d['mean'][:max_len]

    two_pkl = 'kitchen_cabinet_dec7_10hz_separate_ft_pr2'
    d = ut.load_pickle('RAM_db/'+two_pkl+'.pkl')
    two_mn = d['mean'][:max_len]

    bins = np.arange(-0.75, 2.0, 0.5)
    dat = np.column_stack([two_mn-one_mn])
    dat = dat[~np.isnan(dat)]
    lab_l = ['PR2 - Human 1']
    pp.hist(dat, bins, rwidth=0.6, label=lab_l, color='y')
    mpu.legend()
    pp.xlabel('Force difference (Newtons)')
    pp.ylabel('Count')

def human_error_histogram():
    pp.figure()
    max_len = 40

    one_pkl = 'HSI_kitchen_cabinet_right_tiffany'
    d = ut.load_pickle('RAM_db/'+one_pkl+'.pkl')
    one_mn = d['mean'][:max_len]

    two_pkl = 'HSI_kitchen_cabinet_right_advait'
    d = ut.load_pickle('RAM_db/'+two_pkl+'.pkl')
    two_mn = d['mean'][:max_len]

    bins = np.arange(-0.75, 1.5, 0.5)
    dat = np.column_stack([one_mn-two_mn])
    lab_l = ['Human 2 - Human 1']
    pp.hist(dat, bins, rwidth=0.6, label=lab_l)
    mpu.legend()
    pp.xlabel('Force difference (Newtons)')
    pp.ylabel('Count')

def compare_human_trials():
    pp.figure()
    max_len = 40
    nm_l = ['HSI_kitchen_cabinet_right_advait',
            'HSI_kitchen_cabinet_right_tiffany']
    c_l = ['b', 'g']
    lab_l = ['Human 1', 'Human 2']

    plot_different_mechanisms(nm_l, lab_l, c_l, max_len)

def compare_human_pr2_cody():
    mpu.set_figure_size(10., 7.)
    pp.figure()
    max_len = 46
    nm_l = [
            'HSI_kitchen_cabinet_right',
            'HSI_kitchen_cabinet_right_tiffany',
            'kitchen_cabinet_dec6_cody',
            'kitchen_cabinet_dec7_10hz_separate_ft_pr2',
            ]
    c_l = ['r', 'c', 'b', 'y']
    lab_l = ['Human 1 (6 trials)', 'Human 2 (3 trials)', 
            'Cody (5 trials)', 'PR2 (5 trials)',
            ]
    linestyle_l = [':', '--', '-.', '-']

    st_list = [0, 0, 1, 0]
    plot_different_mechanisms(nm_l, lab_l, c_l, max_len, st_list, linestyle_l)
    pp.title('')
    f = pp.gcf()
    f.subplots_adjust(bottom=.15, top=.97, right=.98)
    pp.savefig('cody_human_pr2.pdf')

def intra_semantic_class():
    pp.figure()
    max_len = 38
    nm_l = [
            'HSI_kitchen_cabinet_right',
            'ikea_cabinet_pos1_cody',
            'HSI_Suite_210_brown_cabinet_right',
            'HSI_Executive_Board_Room_Cabinet_Right',
            ]
    c_l = ['r', 'g', 'b', 'y']
    lab_l = ['Cabinet 1', 'Cabinet 2', 
            'Cabinet 3', 'Cabinet 4',
            ]
    plot_different_mechanisms(nm_l, lab_l, c_l, max_len)

    nm_l = [
            'advait_refrigerator',
            'hai_refrigerator',
            'Jason_refrigerator',
            'naveen_refrigerator',
            ]
    lab_l = ['Refrigerator 1', 'Refrigerator 2', 
            'Refrigerator 3', 'Refrigerator 4',
            ]
    pp.figure()
    plot_different_mechanisms(nm_l, lab_l, c_l, max_len)

def semantic_class_confused_mechanism():
    pp.figure()
    max_len = 38
    nm_l = [
            'HSI_lab_cabinet_recessed_left',
            'advait_freezer',
            ]
    c_l = ['y', 'g', 'b', 'y']
    lab_l = ['Cabinet', 'Freezer', 
            ]
    plot_different_mechanisms(nm_l, lab_l, c_l, max_len)


if __name__ == '__main__':
    import optparse
    import glob
    p = optparse.OptionParser()
    p.add_option('-d', '--dir', action='store', default='',
                 type='string', dest='dir', help='directory with logged data')
    p.add_option('-f', '--fname', action='store', default='',
                 type='string', dest='fname', help='pkl with reference dict')
    p.add_option('--pca', action='store_true', dest='pca', help='do pca')
    p.add_option('--locked', action='store_true', dest='locked',
                 help='graphs for event detection: locked')
    p.add_option('--blocked', action='store_true', dest='blocked',
                 help='graphs for event detection: blocked')
    p.add_option('--bayes', action='store_true', dest='bayes',
                 help='estimate new model params in a bayesian way')
    p.add_option('--robot_haptic_id', action='store_true', dest='robot_haptic_id',
                 help='test haptic ID using data from the robot.')
    p.add_option('--sharing', action='store_true', dest='sharing',
                 help='plots etc. to figure out how to share haptic experience')
    p.add_option('--fig_sharing', action='store_true', dest='fig_sharing',
                 help='figure for the RAM paper. Cody, PR2 and Human data.')
    p.add_option('--fig_diff_pos', action='store_true', dest='fig_diff_pos',
                 help='figure for the RAM paper. PR2: different positions, same mechanism.')
    p.add_option('--fig_roc_human', action='store_true',
                 dest='fig_roc_human',
                 help='generate ROC like curve from the BIOROB dataset.')
    p.add_option('--kine', action='store_true', dest='kine',
                 help='testing what happens with online kinematic estimation.')
    p.add_option('--semantic_confusion', action='store_true',
                 dest='semantic_confusion',
                 help='semantic ID related plot.')
    p.add_option('--robot_roc', action='store_true', dest='robot_roc',
                 help='generating ROC like curves from the robot data.')

    opt, args = p.parse_args()

    root_path = os.environ['HRLBASEPATH']+'/'
    data_path = root_path+'src/projects/modeling_forces/handheld_hook/'

    if opt.fig_roc_human:
        pkl_list = glob.glob(data_path+'RAM_db/*.pkl')
        r_pkls = filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = pkls_to_mech_vec_list(r_pkls, 36)
        mpu.set_figure_size(10, 7.)

#        generate_roc_curve(mech_vec_list, mech_nm_list)
#        #generate_roc_curve(mech_vec_list, mech_nm_list,
#        #                   plot_semantic=False)
#        f = pp.gcf()
#        f.subplots_adjust(bottom=.15, top=.96, right=.98, left=0.15)
#        pp.savefig('roc_priors.pdf')

        pp.figure()
        generate_roc_curve_no_prior(mech_vec_list, mech_nm_list)
        generate_roc_curve(mech_vec_list, mech_nm_list)
        f = pp.gcf()
        f.subplots_adjust(bottom=.15, top=.96, right=.98, left=0.15)
        pp.savefig('roc_compare.pdf')
        pp.show()


    if opt.robot_roc:
        pkl_list = glob.glob(data_path+'RAM_db/robot_trials/simulate_perception/*.pkl')
        s_range = np.arange(0.05, 3.0, 0.2) 
        m_range = np.arange(0.1, 3.8, 0.6)

        r_pkls = filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = pkls_to_mech_vec_list(r_pkls, 36)

        mpu.set_figure_size(13, 7.)
        pp.figure()
        generate_roc_curve(mech_vec_list, mech_nm_list,
                           s_range, m_range, sem_c='c', sem_m='^',
                           semantic_label = 'operating 1st time with \n uncertainty in state estimation', plot_prev=False)

        pkl_list = glob.glob(data_path+'RAM_db/robot_trials/perfect_perception/*.pkl')
        s_range = np.arange(0.05, 1.8, 0.2) 
        m_range = np.arange(0.1, 3.8, 0.6)
        r_pkls = filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = pkls_to_mech_vec_list(r_pkls, 36)
        generate_roc_curve(mech_vec_list, mech_nm_list,
                           s_range, m_range, sem_c='b',
semantic_label = 'operating 1st time with \n accurate state estimation',
                           plot_prev=True)
#        generate_roc_curve(mech_vec_list, mech_nm_list,
#                           s_range, m_range, plot_semantic = False,
#                           n_prev_trials = 10, prev_c='y',
#semantic_label = 'operating 1st time with \n accurate state estimation',
#                           plot_prev=True)
        f = pp.gcf()
        pp.xlim(-0.1, 26)
        pp.ylim(0, 13.)
        f.subplots_adjust(bottom=.15, top=.99, right=.98, left=0.13)

        #generate_roc_curve(mech_vec_list, mech_nm_list,
        #                   s_range, m_range, n_prev_trials = 20,
        #                   prev_c = 'y', plot_semantic=False)
        #generate_roc_curve_no_prior(mech_vec_list, mech_nm_list)
        pp.savefig('roc_noisy_kinematics.pdf')
        pp.show()

    if opt.dir:
        dir_list = commands.getoutput('ls -d %s/*/'%(opt.dir)).splitlines()
        d_list = maa.input_mechanism_list(dir_list)
        for n, dir_nm in enumerate(d_list):
            compute_mean_std_force_traj(dir_nm, plot=True)

        pp.show()

    if opt.fname:
        d = ut.load_pickle(opt.fname)
        plot_reference_trajectory(d['config'], d['mean'], d['std'],
                                  d['typ'], d['name'])
        pp.show()

    if opt.pca:
        pkl_list = glob.glob('RAM_db_r6100/*.pkl')
        r_pkls = filter_pkl_list(pkl_list, typ = 'rotary')
        #proj_mat, s, mech_vec_list, mech_nm_list = dimen_reduction_mechanisms(r_pkls, dimen=2)
        #viz_pca(proj_mat, s, mech_vec_list)

        proj_mat, s, mech_vec_list, mech_nm_list = dimen_reduction_mechanisms(r_pkls, dimen=5)
        test_mvpa(proj_mat, mech_vec_list, mech_nm_list)

        #pp.figure()
        #proj_mat, s, mech_vec_list, mech_nm_list = dimen_reduction_mechanisms(r_pkls, dimen=20)
        #variation_with_dimensions(proj_mat, mech_vec_list, mech_nm_list)
        pp.show()

    if opt.robot_haptic_id:
        pkl_list = glob.glob('RAM_db/*.pkl')
        r_pkls = filter_pkl_list(pkl_list, typ = 'rotary')
        proj_mat, s, mech_vec_list, mech_nm_list = dimen_reduction_mechanisms(r_pkls, dimen=5)
        data = create_mvpa_dataset(proj_mat, mech_vec_list, mech_nm_list)
        splitter = NFoldSplitter(cvtype=1, attr='labels')
        test_data = splitter.splitDataset(data, ['kitchen_cabinet_pr2'])[0]
        label_l = data.uniquelabels.flatten().tolist()
        label_l.remove('kitchen_cabinet_pr2')
#        label_l.remove('HSI_kitchen_cabinet_left')
        train_data = splitter.splitDataset(data, [label_l])[0]

        clf = kNN(1)
        clf.train(train_data)
        print 'Test Result:'
        print clf.predict(test_data.samples)

        nm_l = ['HSI_kitchen_cabinet_left',
                'HSI_kitchen_cabinet_right', 'kitchen_cabinet_pr2',
                'HSI_Executive_Board_Room_Cabinet_Right']
        c_l = ['r', 'g', 'b']
        lab_l = ['Human Left', 'Human Right', 'PR2',
                'HSI_Executive_Board_Room_Cabinet_Right']
        for i in range(len(c_l)):
            c, nm, lab = c_l[i], nm_l[i], lab_l[i]
            d = ut.load_pickle('RAM_db/'+nm+'.pkl')
            plot_reference_trajectory(d['config'], d['mean'], d['std'],
                                      d['typ'], d['name'], c, c, lab)
        mpu.legend()
        pp.show()

    if opt.locked:
        pkl_list = glob.glob('RAM_db/*.pkl')
        r_pkls = filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = pkls_to_mech_vec_list(r_pkls, 36)
        #proj_mat, s, mech_vec_list, mech_nm_list = dimen_reduction_mechanisms(r_pkls, dimen=5)
        initial_force_histogram(mech_vec_list, mech_nm_list)

    if opt.blocked:
        ## human data only
        #pkl_list = glob.glob('RAM_db/*.pkl')

        # human and robot data
        pkl_list = glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/*_new.pkl') + glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/perfect_perception/*_new.pkl') + glob.glob(root_path+'src/projects/modeling_forces/handheld_hook/RAM_db/robot_trials/simulate_perception/*_new.pkl')
        
        r_pkls = filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = pkls_to_mech_vec_list(r_pkls, 36) #get vec_list, name_list

        blocked_detection(mech_vec_list, mech_nm_list)
        blocked_detection_n_equals_1(mech_vec_list, mech_nm_list)

        #test_blocked_detection(mech_vec_list, mech_nm_list)
        #test_blocked_detection_new(mech_vec_list, mech_nm_list)

        ## generate_roc_curve(mech_vec_list, mech_nm_list)
        #generate_roc_curve_no_prior(mech_vec_list, mech_nm_list)
        #pp.show()

    if opt.bayes:
        pkl_list = glob.glob('RAM_db/*.pkl')
        r_pkls = filter_pkl_list(pkl_list, typ = 'rotary')
        mech_vec_list, mech_nm_list = pkls_to_mech_vec_list(r_pkls, 36)
        test_bayesian(mech_vec_list, mech_nm_list)

    if opt.sharing:
        #compare_pr2_cody_ikea_cabinet()
        #pr2_cody_error_histogram()

        #compare_cody_ikea_cabinet()

        #compare_human_trials()
        #human_error_histogram()
        compare_human_pr2_cody()

        #intra_semantic_class()

        pp.show()

    # figure in the paper.
    if opt.fig_sharing:
        compare_human_pr2_cody()
        human_pr2_error_histogram()
        pp.show()

    if opt.fig_diff_pos:
        compare_pr2_different_positions()
        pp.show()

    if opt.kine:
        nm_l = [
                #'kitchen_cabinet_dec6_cody',
                #'/robot_trials/simulate_perception/kitchen_cabinet_noisy_cody',
                #'/robot_trials/simulate_perception/kitchen_cabinet_noisy_pr2',
                #'kitchen_cabinet_dec7_10hz_separate_ft_pr2',

                'ikea_cabinet_cody_5cm_cody',
                '/robot_trials/simulate_perception/ikea_cabinet_noisy_cody',
                '/robot_trials/simulate_perception/ikea_cabinet_noisy_pr2',
                'ikea_cabinet_pr2',

                #'HSI_Glass_Door',
                #'/robot_trials/simulate_perception/lab_spring_door_noisy_cody',

                #'/robot_trials/perfect_perception/not_using/lab_fridge_cody',
                #'/robot_trials/simulate_perception/lab_fridge_noisy_cody',
                #'naveen_refrigerator',

                ]
        c_l = ['b', 'r', 'g', 'y', 'c', 'k',]
        max_len = 30
        plot_different_mechanisms(nm_l, nm_l, c_l, max_len)

#        pp.figure()
#        nm_l = [
#                'kitchen_cabinet_dec6_cody',
#                'kitchen_cabinet_online_no_prior_cody',
#                'kitchen_cabinet_rad_fixed_30cm_cody',
#                'kitchen_cabinet_rad_fixed_35cm_cody',
#                'kitchen_cabinet_rad_fixed_40cm_cody',
#                'kitchen_cabinet_rad_fixed_45cm_cody',
#                ]
#        plot_different_mechanisms(nm_l, nm_l, c_l, max_len)

        pp.show()

    if opt.semantic_confusion:
        # confusion matrix - on marvin, using old version.
        semantic_class_confused_mechanism()
        pp.show()


