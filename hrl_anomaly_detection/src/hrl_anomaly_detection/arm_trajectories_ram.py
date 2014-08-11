
import scipy.optimize as so
import matplotlib.pyplot as pp
import numpy as np, math
import sys, os, time
sys.path.append(os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook')
import ram_db as rd
import mechanism_analyse_RAM as mar
import mechanism_analyse_advait as maa

import arm_trajectories as at
#import tangential_force_monitor as tfm

import roslib; roslib.load_manifest('equilibrium_point_control')
import hrl_lib.util as ut, hrl_lib.transforms as tr
import hrl_lib.matplotlib_util as mpu


## assuming that the mechnism is rotary.
# @return r, cx, cy
def estimate_mechanism_kinematics(pull_dict, pr2_log):
    if not pr2_log:
        act_tl = at.joint_to_cartesian(pull_dict['actual'], pull_dict['arm'])
        force_tl = pull_dict['force']
        actual_cartesian, force_ts = act_tl, force_tl
        #actual_cartesian, force_ts = at.account_segway_motion(act_tl,
        #                                    force_tl, pull_dict['segway'])
        cartesian_force_clean, _ = at.filter_trajectory_force(actual_cartesian,
                                                              pull_dict['force'])
        pts_list = actual_cartesian.p_list
        pts_2d, reject_idx = at.filter_cartesian_trajectory(cartesian_force_clean)
    else:
        # not performing force filtering for PR2 trajectories.
        # might have to add that in later.
        p_list, f_list = pull_dict['ee_list'], pull_dict['f_list']
        p_list = p_list[::2]
        f_list = f_list[::2]
        pts = np.matrix(p_list).T
        px = pts[0,:].A1
        py = pts[1,:].A1
        pts_2d = np.matrix(np.row_stack((px, py)))

    #rad = 0.4
    rad = 1.1
    x_guess = pts_2d[0,0]
    y_guess = pts_2d[1,0] - rad
    rad_guess = rad
    rad, cx, cy = at.fit_circle(rad_guess,x_guess,y_guess,pts_2d,
                                method='fmin_bfgs',verbose=False,
                                rad_fix=False)
    #rad, cx, cy = rad_guess, x_guess, y_guess
    return rad, cx, cy

def force_trajectory_in_hindsight(pull_dict, mechanism_type, pr2_log):
    print '_________________________________________________'
    print 'starting force_trajectory_in_hindsight'
    print '_________________________________________________'

    if not pr2_log:
        arm = pull_dict['arm']
        print 'arm:', arm
        act_tl = at.joint_to_cartesian(pull_dict['actual'], arm)
        force_tl = pull_dict['force']
        actual_cartesian, force_ts = act_tl, force_tl
        #actual_cartesian, force_ts = at.account_segway_motion(act_tl,
        #                                    force_tl, pull_dict['segway'])
        p_list = actual_cartesian.p_list
        f_list = force_ts.f_list
    else:
        p_list, f_list = pull_dict['ee_list'], pull_dict['f_list']
        p_list = p_list[::2]
        f_list = f_list[::2]

    if mechanism_type == 'rotary':
        r, cx, cy = estimate_mechanism_kinematics(pull_dict, pr2_log)
        print 'rad, cx, cy:', r, cx, cy
        frad_list,ftan_list,_ = at.compute_radial_tangential_forces(f_list,
                                                            p_list,cx,cy)
        p0 = p_list[0]
        rad_vec_init = np.matrix((p0[0]-cx, p0[1]-cy)).T
        rad_vec_init = rad_vec_init / np.linalg.norm(rad_vec_init)
        config_list = []
        for p in p_list:
            rvec = np.matrix((p[0]-cx, p[1]-cy)).T
            rvec = rvec / np.linalg.norm(rvec)
            ang = np.arccos((rvec.T*rad_vec_init)[0,0])
            if np.isnan(ang):
                ang = 0
            config_list.append(ang)
    else:
        p0 = p_list[0]
        ftan_list, config_list = [], []
        for f, p in zip(f_list, p_list):
            config_list.append(p0[0] - p[0])
            ftan_list.append(abs(f[0]))

    return config_list, ftan_list

def online_force_with_radius(pull_dict, pr2_log, radius_err = 0.,
        with_prior = True):
    if not pr2_log:
        act_tl = at.joint_to_cartesian(pull_dict['actual'], pull_dict['arm'])
        force_tl = pull_dict['force']
        actual_cartesian, force_ts = at.account_segway_motion(act_tl,
                                            force_tl, pull_dict['segway'])
        p_list = actual_cartesian.p_list
        f_list = force_ts.f_list
    else:
        p_list, f_list = pull_dict['ee_list'], pull_dict['f_list']
        p_list = p_list[::2]
        f_list = f_list[::2]

    radius, _, _ = estimate_mechanism_kinematics(pull_dict, pr2_log)
    radius += radius_err
    print '_________________________________________________'
    print 'using radius:', radius
    print '_________________________________________________'

    pts_list = []
    ftan_list = []
    config_list = []
    for f,p in zip(f_list, p_list):
        pts_list.append(p)
        pts_2d = (np.matrix(pts_list).T)[0:2,:]

        x_guess = pts_list[0][0]
        y_guess = pts_list[0][1] - radius
        rad_guess = radius
        if with_prior:
            rad, cx, cy = at.fit_circle_priors(rad_guess, x_guess,
                    y_guess, pts_2d, sigma_r = 0.2, sigma_xy = 0.2,
                    sigma_pts = 0.01, verbose=False)
        else:
            rad, cx, cy = at.fit_circle(rad_guess,x_guess,y_guess,pts_2d,
                                        method='fmin_bfgs',verbose=False,
                                        rad_fix=True)
        print 'rad, cx, cy:', rad, cx, cy

        p0 = p_list[0]
        rad_vec_init = np.matrix((p0[0]-cx, p0[1]-cy)).T
        rad_vec_init = rad_vec_init / np.linalg.norm(rad_vec_init)

        rad_vec = np.array([p[0]-cx,p[1]-cy])
        rad_vec = rad_vec/np.linalg.norm(rad_vec)

        ang = np.arccos((rad_vec.T*rad_vec_init)[0,0])
        config_list.append(ang)

        tan_vec = (np.matrix([[0,-1],[1,0]]) * np.matrix(rad_vec).T).A1
        f_vec = np.array([f[0],f[1]])

        f_tan_mag = abs(np.dot(f_vec, tan_vec))
        ftan_list.append(f_tan_mag)

    return config_list, ftan_list

def load_ref_traj(nm):
    if 'kitchen_cabinet' in nm:
        #has_tags = ['HSI_kitchen_cabinet_left']
        has_tags = ['HSI_kitchen_cabinet_right']
    elif 'lab_cabinet_recessed_left' in nm:
        has_tags = ['HSI_lab_cabinet_recessed_left']
    elif 'lab_cabinet_recessed_right' in nm:
        has_tags = ['HRL_lab_cabinet_recessed_right']
    elif 'spring_loaded_door' in nm:
        has_tags = ['HSI_Glass_Door']
    elif 'hrl_toolchest' in nm:
        has_tags = ['HRL_toolchest_drawer_empty']
    elif 'ikea_cabinet' in nm:
        has_tags = ['HSI_kitchen_cabinet_right']
        #has_tags[rd.o, rd.c]
    elif 'refrigerator' in nm:
        has_tags = ['naveen_refrigerator']
        #has_tags = [rd.r]
    else:
        #has_tags = [rd.r]
        return None
    mn, std, config, typ = rd.get_mean_std_config(has_tags)
    return mn, std, config, typ

def error_lists(ftan_l, config_l, ref_dict):
    # matching the dict expected in tangential_force_monitor.py
    ref_force_dict = {}
    ref_force_dict['tangential'] = ref_dict['mean']
    ref_force_dict['configuration'] = ref_dict['config']
    ref_force_dict['type'] = ref_dict['typ']
    ref_force_dict['name'] = ref_dict['name']

    rel_err_list = []
    abs_err_list = []
    for f,c in zip(ftan_l, config_l):
        hi, lo, ref_hi, ref_lo = tfm.error_from_reference(ref_force_dict, f, c)
#        if ref_hi < 5.:
#            # only consider configs where ref force is greater than 3N
#            continue
        if lo > hi:
            err = -lo
            ref = ref_lo
        else:
            err = hi
            ref = ref_hi
        if ref == 0:
            continue
#        if abs(err) < 3.:
#            continue
        rel_err_list.append(err/ref)
        abs_err_list.append(err)

    return rel_err_list, abs_err_list

def plot_err_histogram(rel_list, abs_list, title):
    # plot relative error histogram.
    max_err = 1.0
    bin_width = 0.05 # relative err.
    bins = np.arange(0.-bin_width/2.-max_err, max_err+2*bin_width, bin_width)
    hist, bin_edges = np.histogram(np.array(rel_list), bins)
    mpu.figure()
    mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                       width=bin_width*0.8, xlabel='Relative Error',
                       plot_title=title)
    # plot relative error histogram.
    max_err = 20
    bin_width = 2 # relative err.
    bins = np.arange(0.-bin_width/2.-max_err, max_err+2*bin_width, bin_width)
    hist, bin_edges = np.histogram(np.array(abs_list), bins)
    mpu.figure()
    mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                       width=bin_width*0.8, xlabel='Absolute Error',
                       plot_title=title)

def truncate_to_config(ftan_l, config_l, config):
    idxs = np.where(np.array(config_l)<config)[0]
    idx = idxs[-1]
    return ftan_l[:idx+1], config_l[:idx+1]

def plot_pkl(pkl_nm):
    pull_dict = ut.load_pickle(pkl_nm)

    if 'pr2' in pkl_nm:
        pr2_log = True
        h_color = 'y'
    else:
        pr2_log = False
        h_color = 'r'

    t = load_ref_traj(pkl_nm)
    if t !=None:
        ref_mean, ref_std, ref_config, typ = t
        mar.plot_reference_trajectory(ref_config, ref_mean, ref_std, typ, 'Hello')
        ref_config = np.degrees(ref_config)
        max_config = np.max(ref_config)
    else:
        typ = 'rotary'
        max_config = 60.

    if pr2_log:
        o_ftan = pull_dict['ftan_list']
        o_config = pull_dict['config_list']
    else:
        o_ftan = pull_dict['online_ftan']
        o_config = pull_dict['online_ang']

    h_config, h_ftan = force_trajectory_in_hindsight(pull_dict,
                                                   typ, pr2_log)
    if typ == 'rotary':
        if opt.prior:
            r_config, r_ftan = online_force_with_radius(pull_dict,
                                                        pr2_log)
            r_config = np.degrees(r_config)
        o_config = np.degrees(o_config)
        h_config = np.degrees(h_config)

    o_ftan, o_config = truncate_to_config(o_ftan, o_config, max_config)
    h_ftan, h_config = truncate_to_config(h_ftan, h_config, max_config)

    if typ == 'rotary':
        if opt.prior:
            r_ftan, r_config = truncate_to_config(r_ftan, r_config, max_config)
        bin_size = 1.
    else:
        bin_size = 0.01

    #o_config, o_ftan = maa.bin(o_config, o_ftan, bin_size, np.mean, True)
    #h_config, h_ftan = maa.bin(h_config, h_ftan, bin_size, np.mean, True)

#    non_nans = ~np.isnan(h_ftan)
#    h_ftan = np.array(h_ftan)[non_nans]
#    h_config = np.array(h_config)[non_nans]
#    
#    non_nans = ~np.isnan(o_ftan)
#    o_ftan = np.array(o_ftan)[non_nans]
#    o_config = np.array(o_config)[non_nans]
    
#    h_config = h_config[:-1]
#    h_ftan = h_ftan[1:]

    if not pr2_log:
        m,c = get_cody_calibration()
        o_ftan = (np.array(o_ftan) - c) / m
        h_ftan = (np.array(h_ftan) - c) / m

    pp.plot(o_config, o_ftan, 'bo-', ms=5, label='online')
    pp.plot(h_config, h_ftan, h_color+'o-', ms=5, label='hindsight')
    if typ == 'rotary':
        if opt.prior:
            r_config, r_ftan = maa.bin(r_config, r_ftan, bin_size, max, True)
            pp.plot(r_config, r_ftan, 'go-', ms=5, label='online with priors')
    pp.xlabel('Configuration')
    pp.ylabel('Tangential Force')

    if pr2_log:
        pp.figure()
        p_list, f_list = pull_dict['ee_list'], pull_dict['f_list']
        p_list = p_list[::2]
        f_list = f_list[::2]
        x_l, y_l, z_l = zip(*p_list)
        pp.plot(x_l, y_l)
        r, cx, cy = estimate_mechanism_kinematics(pull_dict, pr2_log)
        mpu.plot_circle(cx, cy, r, 0., math.pi/2,
                        label='Actual\_opt', color='r')
        pp.axis('equal')


def compute_mean_std(pkls, bin_size):
    c_list = []
    f_list = []
    max_config = math.radians(100.)
    typ = 'rotary'
    for pkl_nm in pkls:
        pull_dict = ut.load_pickle(pkl_nm)
        pr2_log =  'pr2' in pkl_nm
        h_config, h_ftan = force_trajectory_in_hindsight(pull_dict,
                                                       typ, pr2_log)
        #h_config, h_ftan = online_force_with_radius(pull_dict, pr2_log)
        c_list.append(h_config)
        f_list.append(h_ftan)
        max_config = min(max_config, np.max(h_config))

    leng = int (max_config / bin_size) - 1
    ff = []
    for c, f in zip(c_list, f_list):
        #c, f = maa.bin(c, f, bin_size, max, True)
        c, f = maa.bin(c, f, bin_size, np.mean, False, empty_value = np.nan)
        f, c = truncate_to_config(f, c, max_config)
        f = np.ma.masked_array(f, np.isnan(f))
        f = f[:leng]
        c = c[:leng]
        ff.append(f)
    arr = np.array(ff)
    mean = arr.mean(0)
    std = arr.std(0)
    return mean, std, c, arr

def calibrate_least_squares(ref_mean, sensor_mean):
    ref_mean = np.array(ref_mean)
    length = min(ref_mean.shape[0], sensor_mean.shape[0])
    ref_mean = ref_mean[:length]
    sensor_mean = sensor_mean[:length]
    def error_function(params):
        m, c = params[0], params[1]
        sensor_predict = m * ref_mean + c
        err = (sensor_predict - sensor_mean)
        return np.sum((err * err) * np.abs(ref_mean))
        #return np.sum(err * err)
    
    params = [1., 0.]
    r = so.fmin_bfgs(error_function, params, full_output=1,
                     disp = False, gtol=1e-5)
    print 'Optimization result:', r[0]

def get_cody_calibration():
    m = 1.13769405
    c = 2.22946475
    # sensor = m * ref + c
    m, c = 1., 0.
    return m, c

def convert_to_ram_db(pkls, name):
    if pkls == []:
        return
    bin_size = math.radians(1.)
    mean, std, c, arr = compute_mean_std(pkls, bin_size)

    d = {}
    d['std'] = std
    d['mean'] = mean
    d['rad'] = -3.141592
    d['name'] = name
    d['config'] = c
    d['vec_list'] = arr.tolist()
    d['typ'] = 'rotary'
    ut.save_pickle(d, name+'.pkl')


def simulate_perception(pkls, percep_std, name):
    c_list = []
    f_list = []
    trials_per_pkl = 5
    bin_size = math.radians(1.)
    max_config = math.radians(100.)
    for pkl_nm in pkls:
        pull_dict = ut.load_pickle(pkl_nm)
        pr2_log =  'pr2' in pkl_nm

        for t in range(trials_per_pkl):
            radius_err = np.random.normal(scale=percep_std)
            #radius_err = np.random.uniform(-percep_std, percep_std)
            h_config, h_ftan = online_force_with_radius(pull_dict, pr2_log, radius_err)
            c_list.append(h_config)
            f_list.append(h_ftan)
            max_config = min(max_config, np.max(h_config))

    leng = int (max_config / bin_size) - 1
    ff = []
    for c, f in zip(c_list, f_list):
        c, f = maa.bin(c, f, bin_size, np.mean, False, empty_value = np.nan)
        f, c = truncate_to_config(f, c, max_config)
        f = np.ma.masked_array(f, np.isnan(f))
        f = f[:leng]
        c = c[:leng]
        ff.append(f)
    arr = np.array(ff)
    mean = arr.mean(0)
    std = arr.std(0)

    d = {}
    d['std'] = std
    d['mean'] = mean
    d['rad'] = -3.141592
    d['name'] = name
    d['config'] = c
    d['vec_list'] = arr.tolist()
    d['typ'] = 'rotary'
    ut.save_pickle(d, name+'.pkl')

def known_radius(pkls, name):
    c_list = []
    f_list = []
    trials_per_pkl = 1
    bin_size = math.radians(1.)
    max_config = math.radians(100.)
    for pkl_nm in pkls:
        pull_dict = ut.load_pickle(pkl_nm)
        pr2_log =  'pr2' in pkl_nm

        for t in range(trials_per_pkl):
            h_config, h_ftan = online_force_with_radius(pull_dict,
                                        pr2_log, with_prior = False)
            c_list.append(h_config)
            f_list.append(h_ftan)
            max_config = min(max_config, np.max(h_config))

    leng = int (max_config / bin_size) - 1
    ff = []
    for c, f in zip(c_list, f_list):
        c, f = maa.bin(c, f, bin_size, np.mean, False, empty_value = np.nan)
        f, c = truncate_to_config(f, c, max_config)
        f = np.ma.masked_array(f, np.isnan(f))
        f = f[:leng]
        c = c[:leng]
        ff.append(f)
    arr = np.array(ff)
    mean = arr.mean(0)
    std = arr.std(0)

    d = {}
    d['std'] = std
    d['mean'] = mean
    d['rad'] = -3.141592
    d['name'] = name
    d['config'] = c
    d['vec_list'] = arr.tolist()
    d['typ'] = 'rotary'
    ut.save_pickle(d, name+'.pkl')


if __name__ == '__main__':
    import optparse
    import glob
    p = optparse.OptionParser()
    p.add_option('-d', action='store', type='string', dest='dir_nm',
                 help='plot all the pkls in the directory.', default='')
    p.add_option('-f', action='store', type='string', dest='fname',
                 help='pkl file to use.', default='')
    p.add_option('--prior', action='store_true', dest='prior',
                 help='estimate tangential force using prior.')
    p.add_option('--calibrate', action='store_true', dest='calibrate',
                 help='calibrate the sensor using the ref trajectory.')
    p.add_option('--ram_db', action='store_true', dest='ram_db',
                 help='convert trials to ram_db format.')
    p.add_option('--nm', action='store', dest='name', default='',
                 help='name for the ram_db dict.')
    p.add_option('--simulate_percep', action='store_true', dest='simulate_percep',
                 help='simulate perception.')

    opt, args = p.parse_args()

    if opt.calibrate:
        pkls_nm = glob.glob(opt.dir_nm+'/*pull*.pkl')
        ref_mean, ref_std, ref_config, typ = load_ref_traj(pkls_nm[0])
        cody_pkls = glob.glob(opt.dir_nm+'/*trajector*.pkl')
        cody_mn, cody_std, cody_config, _ = compute_mean_std(cody_pkls, math.radians(1.))
        calibrate_least_squares(ref_mean, cody_mn)
        sys.exit()


    if opt.simulate_percep:
        percep_std = 0.1
        if opt.name == '':
            name = opt.dir_nm.split('/')[0]
        else:
            name = opt.name
        cody_pkls = glob.glob(opt.dir_nm+'/*trajector*.pkl')
        if cody_pkls != []:
            simulate_perception(cody_pkls, percep_std, name+'_noisy_cody')
            known_radius(cody_pkls, name+'_known_rad_cody')
        pr2_pkls = glob.glob(opt.dir_nm+'/pr2*.pkl')
        if pr2_pkls != []:
            simulate_perception(pr2_pkls, percep_std, name+'_noisy_pr2')
            known_radius(pr2_pkls, name+'_known_rad_pr2')

        sys.exit()


    if opt.ram_db:
        if opt.name == '':
            name = opt.dir_nm.split('/')[0]
        else:
            name = opt.name
        cody_pkls = glob.glob(opt.dir_nm+'/*trajector*.pkl')
        convert_to_ram_db(cody_pkls, name+'_cody')
        pr2_pkls = glob.glob(opt.dir_nm+'/pr2*.pkl')
        convert_to_ram_db(pr2_pkls, name+'_pr2')
        sys.exit()

    if opt.dir_nm != '':
        pkls_nm = glob.glob(opt.dir_nm+'/*pull*.pkl')

        pp.figure()
        ref_mean, ref_std, ref_config, typ = load_ref_traj(pkls_nm[0])
        mar.plot_reference_trajectory(ref_config, ref_mean, ref_std, typ, 'Hello')

        pr2_pkls = glob.glob(opt.dir_nm+'/pr2*.pkl')
        if pr2_pkls != []:
            pr2_mn, pr2_std, pr2_config, _ = compute_mean_std(pr2_pkls, math.radians(1.))

            c1 = 'b'
            pr2_config = np.degrees(pr2_config)
            pp.plot(pr2_config, pr2_mn, color=c1, label='PR2')
            pp.fill_between(pr2_config, np.array(pr2_mn)+np.array(pr2_std),
                    np.array(pr2_mn)-np.array(pr2_std), color=c1, alpha=0.5)

        cody_pkls = glob.glob(opt.dir_nm+'/*trajector*.pkl')
        if cody_pkls != []:
            cody_mn, cody_std, cody_config, _ = compute_mean_std(cody_pkls, math.radians(1.))
            m,c = get_cody_calibration()
            cody_mn = (cody_mn - c) / m
            
            cody_mn = cody_mn[1:]
            cody_std = cody_std[1:]
            cody_config = cody_config[:-1]
            c1 = 'r'
            cody_config = np.degrees(cody_config)
            pp.plot(cody_config, cody_mn, color=c1, label='Cody')
            pp.fill_between(cody_config, np.array(cody_mn)+np.array(cody_std),
                    np.array(cody_mn)-np.array(cody_std), color=c1, alpha=0.5)
        pp.legend()
        pp.show()

    if opt.fname:
        plot_pkl(opt.fname)
        pp.legend()
        pp.show()


