
import scipy.optimize as so
import matplotlib.pyplot as pp
import numpy as np, math
import sys, os, time
import roslib; roslib.load_manifest('hrl_anomaly_detection')

## import ram_db as rd
import mechanism_analyse_RAM as mar
import mechanism_analyse_advait as maa
## sys.path.append(os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook')

## import roslib; roslib.load_manifest('modeling_forces')
import hrl_lib.util as ut, hrl_lib.transforms as tr
import hrl_lib.matplotlib_util as mpu

import arm_trajectories_ram as atr


def test_known_mechanism(detect_tuple, one_trial):
    # refrigerator = 20.
    # office cabinet = 7.
    Ms, n_std = detect_tuple
    print 'before'
    mn, std = mar.estimate_theta(one_trial, Ms, plot=False)
    print 'after'

    mn = np.array(mn)
    std = np.array(std)
    non_nans = ~np.isnan(mn)
    mn = mn[non_nans]
    std = std[non_nans]
    n_std = 5.8

    #lab = 'Expected force for \n collision free trial'
    lab = 'Expected force \n (no collisions)'
    pp.plot(mn, 'b-', linewidth=1.5, label=lab)
    #lab = 'Min force to detect collision \n using mechanism identity'
    #lab = 'Mechanism identity'
    lab = 'Operating 2nd time'
    pp.plot(mn+n_std*std, 'b--', linewidth=1.5, label=lab)

def plot_trial(pkl_nm, max_ang, start_idx=None, mech_idx=None,
               class_idx=None, start=None, mech=None, sem=None):

    pull_dict = ut.load_pickle(pkl_nm)
    typ = 'rotary'
    pr2_log =  'pr2' in pkl_nm
    h_config, h_ftan = atr.force_trajectory_in_hindsight(pull_dict,
                                                   typ, pr2_log)
    h_config = np.array(h_config)
    h_ftan = np.array(h_ftan)
    h_ftan = h_ftan[h_config < max_ang]
    h_config = h_config[h_config < max_ang] # cut
    bin_size = math.radians(1.)
    h_config_degrees = np.degrees(h_config)
    ftan_raw = h_ftan

    # resampling with specific interval
    h_config, h_ftan = maa.bin(h_config, h_ftan, bin_size, np.mean, True) 
    pp.plot(np.degrees(h_config), h_ftan, 'yo-', mew=0, ms=0,
            label='applied force', linewidth=2)
    ## pp.xlabel('Angle (degrees)')
    ## pp.ylabel('Opening Force (N)')

    if start != None:
        x = start[0]
        y = start[1]
        pp.plot([x], [y], 'ko', mew=0, ms=5)
        #pp.text(x, y-1.5, '1', withdash=True) # cody_kitchen_box
        #pp.text(x-1, y+1.0, '1', withdash=True) # cody_fridge_box
        #pp.text(x, y-4., '1', withdash=True) # cody_fridge_chair
        #pp.text(x+1.0, y-0.5, '1', withdash=True) # locked_pr2
        #pp.text(x+1.0, y-0.5, '1', withdash=True) # locked_cody

        x = mech[0]
        y = mech[1]
        pp.plot([x], [y], 'ko', mew=0, ms=5)

        x = sem[0]
        y = sem[1]
        pp.plot([x], [y], 'ko', mew=0, ms=5)

    if start_idx != None:
        pp.plot([h_config_degrees[start_idx]], [ftan_raw[start_idx]],
                'ko', mew=0, ms=5)
        pp.plot([h_config_degrees[mech_idx]], [ftan_raw[mech_idx]],
                'ko', mew=0, ms=5)
        pp.plot([h_config_degrees[class_idx]], [ftan_raw[class_idx]],
                'ko', mew=0, ms=5)
        print 'Time with mechanism known:', (mech_idx-start_idx) * 100.
        print 'Time with class known:', (class_idx-start_idx) * 100.
        print ''
        print 'Force increase with known mechanism:', ftan_raw[mech_idx] - ftan_raw[start_idx]
        print 'Force increase with known class:', ftan_raw[class_idx] - ftan_raw[start_idx]
        print ''
        print 'Angle increase with known mechanism:', h_config_degrees[mech_idx] - h_config_degrees[start_idx]
        print 'Angle increase with known class:', h_config_degrees[class_idx] - h_config_degrees[start_idx]

def test_known_semantic_class(detect_tuple):
    # only sematic class is known.
    n_std, mn, std = detect_tuple
    n_std = 3.2 # anomaly detection threshold for 1st operation force curve 
    #    pp.plot(mn, 'g-', linewidth=1.5, label='Mean (Semantic Class)')
    #lab = 'Min force to detect collision \n using semantic class'
    #lab = 'Semantic class'
    lab = 'Operating 1st time'
    pp.plot(mn+n_std*std, 'g:', linewidth=2.5, label=lab)

def robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, start_idx=None,
                     mech_idx=None, class_idx=None, plt_st=None,
                     plt_mech=None, plt_sem=None):

    # pkl_nm    : collision pickle
    # one_pkl_nm: perfect_perception

    one_d = ut.load_pickle(one_pkl_nm) 
    one_trial = np.array(one_d['vec_list'][0:1]) # ee force_profile ? 4xN
    #one_trial = one_trial.reshape(1,len(one_trial))
    dt = second_time[mech] 

    # Applied force (collision)
    plot_trial(pkl_nm, math.radians(len(dt[0][0])), start_idx,
               mech_idx, class_idx, plt_st, plt_mech, plt_sem) 

    # Operating 1st time
    # semantic: human and robot data in where each category has (n_std, mn, std) <= force profile
    # 'RAM_db/*.pkl' 'RAM_db/robot_trials/perfect_perception/*.pkl' 'RAM_db/robot_trials/simulate_perception/*.pkl'
    # mechanism anlyse RAM with blocked option generates semantic data
    test_known_semantic_class(semantic[cls])
    
    # Expected force and operating 2nd time 
    test_known_mechanism(dt, one_trial)

    ## pp.title(mech)
    mpu.legend(display_mode='normal')

def fridge_box_collision():
    cls = 'Fridge'
    cls = mech = 'lab_fridge_cody'
    pkl_nm = data_path+'robot_trials/lab_fridge_collision_box/pull_trajectories_lab_refrigerator_2010Dec10_044022_new.pkl'
    one_pkl_nm = pth + 'RAM_db/robot_trials/perfect_perception/lab_fridge_cody_new.pkl'
    #robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, 150, 177, 173)

    mpu.set_figure_size(10, 7.0)
    f = pp.figure()
    f.set_facecolor('w')
    x1, y1 = 13., 2.21
    x2, y2 = 14.534, 5.75
    x3, y3 = 14.2, 4.98

    robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, None, None, None,
                     (x1, y1), (x2, y2), (x3, y3))

    pp.text(x1-1., y1+1., '1', withdash=True)
    pp.text(x2-0., y2+2., '2', withdash=True)
    pp.text(x3+1.5, y3-0.5, '3', withdash=True)

    pp.ylim(-0., 55.)
#    pp.xlim(0., 34.)
    f.subplots_adjust(bottom=.15, top=.99, right=.98)
#    mpu.legend(loc='lower left')
    pp.savefig('collision_detection_fridge_box.pdf')
    pp.show()

def fridge_chair_collision():
    cls = 'Fridge'
    cls = mech = 'lab_fridge_cody'
    pkl_nm = data_path+'robot_trials/lab_fridge_collision_chair/pull_trajectories_lab_refrigerator_2010Dec10_042926_new.pkl'
    one_pkl_nm = pth + 'RAM_db/robot_trials/perfect_perception/lab_fridge_cody_new.pkl'
    #robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, 290, 295, 295)

    mpu.set_figure_size(10, 7.0)
    f = pp.figure()
    f.set_facecolor('w')
    x1, y1 = 25., 2.
    x2, y2 = 25.14, 4.68
    x3, y3 = 25.17, 5.31
    robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, None, None, None,
                     (x1, y1), (x2, y2), (x3, y3))
    pp.text(x1-0., y1-4., '1', withdash=True)
    pp.text(x2+1., y2-2., '2', withdash=True)
    pp.text(x3-1., y3+1.0, '3', withdash=True)
    pp.ylim(-4., 55.)
    f.subplots_adjust(bottom=.15, top=.99, right=.98)
    pp.savefig('collision_detection_fridge_chair.pdf')
    pp.show()


def ikea_cabinet_no_collision():
    cls = 'Office Cabinet'
    cls = mech = 'ikea_cabinet_pr2'
    pkl_nm = data_path+'robot_trials/ikea_cabinet/pr2_pull_2010Dec08_204324_new.pkl'
    one_pkl_nm = pth + 'RAM_db/robot_trials/perfect_perception/ikea_cabinet_pr2_new.pkl'
    robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm)

def kitchen_cabinet_chair_cody():
    cls = 'Office Cabinet'
    cls = mech = 'kitchen_cabinet_cody'
    pkl_nm = data_path+'robot_trials/hsi_kitchen_collision_chair/pull_trajectories_kitchen_cabinet_2010Dec10_060852_new.pkl'
    one_pkl_nm = pth + 'RAM_db/robot_trials/perfect_perception/kitchen_cabinet_cody_new.pkl'
    robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, 113, 115, 119)

def kitchen_cabinet_box_cody():
    ## cls = 'Office Cabinet'
    cls = mech = 'kitchen_cabinet_cody'
    pkl_nm = data_path+'robot_trials/hsi_kitchen_collision_box/pull_trajectories_kitchen_cabinet_2010Dec10_060454_new.pkl'
    one_pkl_nm = pth + 'RAM_db/robot_trials/perfect_perception/kitchen_cabinet_cody_new.pkl'
    #robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, 68, 77, 104)

    mpu.set_figure_size(10, 7.0)
    f = pp.figure()
    f.set_facecolor('w')
    x1,y1 = 6, 10.35
    x2, y2 = 6.95, 12.39
    x3, y3 = 10.1314, 13.3785
       
    robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, None, None, None,
                     (x1, y1), (x2, y2), (x3, y3))
    pp.text(x1-0., y1-1.5, '1', withdash=True)
    pp.text(x2-1., y2-0., '2', withdash=True)
    pp.text(x3-0., y3+0.7, '3', withdash=True)

    pp.ylim(-0., 16.)
    pp.xlim(0., 34.)
    f.subplots_adjust(bottom=.15, top=.99, right=.99)
    mpu.legend(loc='lower left')
    pp.savefig('collision_detection_hsi_kitchen_cody.pdf')
    pp.show()

def kitchen_cabinet_box_pr2():
    #cls = 'Office Cabinet'
    cls = mech = 'kitchen_cabinet_pr2'
    pkl_nm = data_path + 'robot_trials/hsi_kitchen_collision_pr2/pr2_pull_2010Dec10_071602_new.pkl'
    one_pkl_nm = data_path + 'robot_trials/perfect_perception/kitchen_cabinet_pr2.pkl'

    # robot_trial
    # 1) collision: ['ftan_list', 'ee_list', 'config_list', 'f_list', 'frad_list', 'cep_list']
    # 2) perfect  : ['std', 'rad', 'name', 'typ', 'vec_list', 'config', 'mean'] # vec_list gives ee_pos?       
    
    ## one_pkl_nm = pth + 'RAM_db/robot_trials/perfect_perception/kitchen_cabinet_pr2.pkl'
    #robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, 125, 128, 136)
    
    mpu.set_figure_size(10, 7.0)
    f = pp.figure()
    f.set_facecolor('w')
    x1,y1 = 14., 7.44
    x2,y2 = 14.47, 8.96
    x3,y3 = 14.84, 10.11

    # Plot operating, applied, and expected forces
    robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, None, None, None,
                     (x1, y1), (x2, y2), (x3, y3))
    pp.text(x1-0.5, y1-1.5, '1', withdash=True) # pr2_box_kitchen
    pp.text(x2-1.5, y2-0.7, '2', withdash=True) # pr2_box_kitchen
    pp.text(x3-0.5, y3+1, '3', withdash=True) # pr2_box_kitchen

    pp.ylim(-3., 16.)
    pp.xlim(0., 34.)
    f.subplots_adjust(bottom=.15, top=.99, right=.99)
    mpu.legend(loc='lower left')
    pp.savefig('collision_detection_hsi_kitchen_pr2.pdf')
    pp.show()

def kitchen_cabinet_locked_cody():
    cls = 'Office Cabinet'
    cls = mech = 'kitchen_cabinet_cody'
    pkl_nm = data_path+'robot_trials/kitchen_cabinet_locked/pull_trajectories_kitchen_cabinet_2010Dec11_233625_new.pkl'
    one_pkl_nm = pth + 'RAM_db/robot_trials/perfect_perception/kitchen_cabinet_cody_new.pkl'
    #robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, 14, 24, 35)

    mpu.set_figure_size(10, 7)
    f = pp.figure()
    f.set_facecolor('w')
    x1, y1 = 0.24, 7.46
    x2, y2 = 0.48, 10.66
    x3, y3 = 0.66, 13.52
    robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, None, None, None,
                     (x1, y1), (x2, y2), (x3, y3))
    pp.text(x1+1., y1-.5, '1', withdash=True)
    pp.text(x2+1., y2-0., '2', withdash=True)
    pp.text(x3+1., y3-0.5, '3', withdash=True)
    pp.ylim(0., 16.)
    pp.xlim(-0.5, 35.)
    f.subplots_adjust(bottom=.16, top=.99, right=.98)
    pp.savefig('locked_cody.pdf')
    pp.show()

def kitchen_cabinet_locked_pr2():
    cls = 'Office Cabinet'
    cls = mech = 'kitchen_cabinet_pr2'
    #pkl_nm = 'robot_trials/kitchen_cabinet_locked/pr2_pull_2010Dec11_215502_new.pkl'
    pkl_nm = data_path + 'robot_trials/kitchen_cabinet_locked/pr2_pull_2010Dec12_005340_new.pkl'
    one_pkl_nm = pth + 'RAM_db/robot_trials/perfect_perception/kitchen_cabinet_pr2_new.pkl'
    #robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, 49, 116, 330)

    mpu.set_figure_size(10, 7)
    f = pp.figure()
    f.set_facecolor('w')
    x1, y1 = 0.2, 7.64
    x2, y2 = 0.43, 11.02
    x3, y3 = 0.59, 13.42
    robot_trial_plot(cls, mech, pkl_nm, one_pkl_nm, None, None, None,
                     (x1, y1), (x2, y2), (x3, y3))
    pp.text(x1+1., y1-.5, '1', withdash=True)
    pp.text(x2+1., y2-1., '2', withdash=True)
    pp.text(x3+1., y3-0.5, '3', withdash=True)
    pp.ylim(0., 16.)
    pp.xlim(-0.5, 35.)
    f.subplots_adjust(bottom=.16, top=.99, right=.98)
    pp.savefig('locked_pr2.pdf')
    pp.show()



if __name__ == '__main__':

    pth       = os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook/'
    data_path = os.environ['HRLBASEPATH']+'_data/usr/advait/ram_www/data_from_robot_trials/'
    
    blocked_thresh_dict = ut.load_pickle(pth+'blocked_thresh_dict.pkl') # ['mean_charlie', 'mean_known_mech']
    #blocked_thresh_dict = ut.load_pickle('./blocked_thresh_dict.pkl') # ['mean_charlie', 'mean_known_mech']

    semantic = blocked_thresh_dict['mean_charlie'] # each category has (n_std, mn, std)  <= force profiles
    second_time = blocked_thresh_dict['mean_known_mech'] # (Ms(mn_mn, var_mn, mn_std, var_std), n_std)=(tuple(4),float)
    
    ikea_cabinet_no_collision()
    ## fridge_chair_collision()
    ## fridge_box_collision()
    ## kitchen_cabinet_chair_cody()
    ## kitchen_cabinet_box_pr2()
    ## kitchen_cabinet_box_cody()
    ## kitchen_cabinet_locked_cody()
    ## kitchen_cabinet_locked_pr2()
    pp.show()


    
    #1: When sematic (category, 'kitchen_cabinet_pr2') is known, 
    #   it gives 'operating 1st time' opening force boundary.
    #   Force profile (+std*std) is listed by door angle.
    #   ex) blocked_thresh_dict['mean_charlie']
    #2: When mechanism (category+????) is known, it gives 'operating 2nd time' opening force boundary.
    #   Force profile (+std*std) is listed by door angle.
    #   ex) robot_trials/perfect_perception/kitchen_cabinet_pr2.pkl
    #3: Real collision profile.
    #   ex) robot_trials/hsi_kitchen_collision_pr2/pr2_pull_2010Dec10_071602_new.pkl



