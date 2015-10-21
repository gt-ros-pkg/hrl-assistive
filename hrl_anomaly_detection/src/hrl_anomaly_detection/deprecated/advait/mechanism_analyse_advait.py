#
# Assumes that all the logs are in one folder with the file names
# giving the different mechanisms, trial number, open or close etc.
#

import commands
import os
import os.path as pt
import glob
import math, numpy as np
import scipy.signal as ss
import scipy.cluster as clus

import roslib; roslib.load_manifest('modeling_forces')
import modeling_forces.mf_common as mfc
import modeling_forces.door_model as mfd

import hrl_lib.matplotlib_util as mpu
import hrl_lib.util as ut
import scipy.stats as st
import pylab as pb

import matplotlib.pyplot as pp

#import mdp


class pca_plot_gui():
    def __init__(self, legend_list, mech_vec_list, proj_mat, dir_list, mn):
        self.legend_list = legend_list
        self.mech_vec_list = mech_vec_list
        self.proj_mat = proj_mat
        self.dir_list = dir_list
        self.mn = mn

    def pick_cb(self, event):
        if 'shift' != event.key:
            return
        selected = np.matrix([event.xdata, event.ydata]).T
#        print 'selected', selected.A1
        min_list = []
        for i, v in enumerate(self.mech_vec_list):
            p = self.proj_mat[:, 0:2].T * (v-self.mn)
            min_list.append(np.min(ut.norm(p-selected)))

        mech_idx = np.argmin(min_list)
        print 'Selected mechanism was:', self.legend_list[mech_idx]

        plot_tangential_force(self.dir_list[mech_idx], all_trials = True, open = True)
#        mpu.figure()
#        n_dim = 7
#        reconstruction = self.proj_mat[:, 0:n_dim] * (self.proj_mat[:, 0:n_dim].T * (self.mech_vec_list[mech_idx] - self.mn)) + self.mn
#        di = extract_pkls(self.dir_list[mech_idx])
#        nm = get_mech_name(self.dir_list[mech_idx])
#        for i in range(reconstruction.shape[1]):
#            mpu.plot_yx(reconstruction[:,i].A1, color=mpu.random_color(),
#                        plot_title=nm+': %d components'%n_dim)
#        mpu.legend()
#        mpu.figure()
#        plot_velocity(self.dir_list[mech_idx])
        mpu.show()




##
# list all the mechanisms in dir_list and allow user to type the
# numbers of the desired mechanisms.
# @return list of paths to the selected mechanisms.
def input_mechanism_list(dir_list):
    mech_list = []
    for i, m in enumerate(dir_list):
        t = m.split('/')
        mech = t[-1]
        if mech == '':
            mech = t[-2]
        mech_list.append(mech)
        print '%d. %s'%(i, mech)

    print ''
    print 'Enter mechanism numbers that you want to plot'
    s = raw_input()
    num_list = map(int, s.split(' '))
    chosen_list = []
    for n in num_list:
        chosen_list.append(dir_list[n])
    return chosen_list

##
# remove pkls in which the forces are unreasonable large or small
def clean_data_forces(dir):
    l = commands.getoutput('find %s/ -name "*mechanism_trajectories*.pkl"'%dir).splitlines()
    for pkl in l:
        cmd1 = 'rm -f %s'%pkl
        cmd2 = 'svn rm %s'%pkl
        d = ut.load_pickle(pkl)
        radial_mech = d['force_rad_list']
        tangential_mech = d['force_tan_list']

        if len(radial_mech) == 0 or len(tangential_mech) == 0:
            os.system(cmd1)
            os.system(cmd2)
            continue

        max_force = max(np.max(np.abs(radial_mech)),
                        np.max(np.abs(tangential_mech)))
        if max_force > 120.:
            os.system(cmd1)
            os.system(cmd2)

        n_points = len(radial_mech)
        if n_points < 50:
            os.system(cmd1)
            os.system(cmd2)

        if d.has_key('radius'):
            r = d['radius']
            if r != -1:
                ang = np.degrees(d['mechanism_x'])
                if np.max(ang) < 20.:
                    os.system(cmd1)
                    os.system(cmd2)


        if d.has_key('time_list'):
            t_l = d['time_list']
            time_diff_l = np.array(t_l[1:]) - np.array(t_l[0:-1])
            if len(time_diff_l) == 0:
                os.system(cmd1)
                os.system(cmd2)
                continue

            max_time_diff = np.max(time_diff_l)
            if max_time_diff > 0.3:
                print 'max time difference between consec readings:', max_time_diff
                os.system(cmd1)
                os.system(cmd2)


glob_ctr = 0
glob_colors = ['r', 'g', 'b', 'c']
def plot_tangential_filtered(dir_name, fig1 = None, legend = '__nolegend__'):
    if fig1 == None:
        mpu.set_figure_size(4.,4.)
        fig1 = mpu.figure()

    vv = make_vector_mechanism(dir_name, make_unit = False)
    scatter_size = 0
#    color = mpu.random_color()
    global glob_colors, glob_ctr
    color = glob_colors[glob_ctr]
    glob_ctr += 1
    xlabel = 'angle (degrees)'
    ylabel = '$f_{tan}(N)$'
    for v in vv.T:
        mpu.plot_yx(v.A1, axis=None,
                    xlabel=xlabel, ylabel = ylabel, color = color,
                    scatter_size = scatter_size, linewidth = 1,
                    label=legend, alpha = 0.5)
        legend = '__nolegend__'


##
# @param dir_name - directory containing the pkls.
# @param all_trials - plot force for all the trials.
# @param open - Boolean (open or close trial)
# @param filter_speed - mech vel above this will be ignored. for
#                       ROTARY joints only, radians/sec
def plot_tangential_force(dir_name, all_trials, open = True,
                          filter_speed=math.radians(100), fig1 = None):
    if fig1 == None:
        mpu.set_figure_size(7.,5.)
        fig1 = mpu.figure()

    if open:
        trial = 'open'
    else:
        trial = 'close'

    d = extract_pkls(dir_name, open)
    mech_name = get_mech_name(dir_name)

    mechx_l_l = d['mechx_l_l']
    max_config_l = []
    for mech_x_l in mechx_l_l:
        max_config_l.append(np.max(mech_x_l[1:]))
    max_angle = np.min(max_config_l)

    traj_vel_list = []
    for i,ftan_l in enumerate(d['ftan_l_l']):
        mech_x = d['mechx_l_l'][i]
        if d['typ'] == 'rotary':
#            max_angle = math.radians(40)
            type = 'rotary'
        else:
#            max_angle = 0.3
            type = 'prismatic'

        traj_vel = compute_average_velocity(mech_x, d['time_l_l'][i], max_angle, type)
        print 'traj_vel:', math.degrees(traj_vel)
        if traj_vel == -1:
            print 'boo'
            continue
        traj_vel_list.append(traj_vel)

    #vel_color_list = ['r', 'g', 'b', 'y', 'k']
    #vel_color_list = ['#000000', '#A0A0A0', '#D0D0D0', '#E0E0E0', '#F0F0F0']
    #vel_color_list = ['#000000', '#00A0A0', '#00D0D0', '#00E0E0', '#00F0F0']
    #vel_color_list = [ '#%02X%02X%02X'%(r,g,b) for (r,g,b) in [(95, 132, 53), (81, 193, 79), (28, 240, 100), (196, 251, 100)]]
    vel_color_list = [ '#%02X%02X%02X'%(r,g,b) for (r,g,b) in [(95, 132, 53), (28, 240, 100), (196, 251, 100)]]
    traj_vel_sorted = np.sort(traj_vel_list).tolist()

    sorted_color_list = []
    sorted_scatter_list = []
    i = 0
    v_prev = traj_vel_sorted[0]
    legend_list = []
    if d['typ'] == 'rotary':
        l = '%.1f'%math.degrees(v_prev)
        thresh = math.radians(5)
    else:
        l = '%.02f'%(v_prev)
        thresh = 0.05

    t_v = v_prev
    v_threshold = v_prev + thresh * 2
    for j,v in enumerate(traj_vel_sorted):
        if (v - v_prev) > thresh:
            if d['typ'] == 'rotary':
                l = l + ' to %.1f deg/sec'%math.degrees(t_v)
            else:
                l = l + ' to %.1f m/s'%t_v
            legend_list.append(l)
            if d['typ'] == 'rotary':
                l = '%.1f'%math.degrees(v)
            else:
                l = '%.02f'%(v)

            if i >= 2:
                #v_threshold = min(v_threshold, v)
                if d['typ'] == 'rotary':
                    print 'v_threshold:', math.degrees(v_threshold)
                else:
                    print 'v_threshold:', v_threshold

            i += 1
            if i == len(vel_color_list):
                i -= 1

            v_prev = v
        else:
            legend_list.append('__nolegend__')

        t_v = v
        sorted_color_list.append(vel_color_list[i])

#    v_threshold = math.radians(15)
    if d['typ'] == 'rotary':
        l = l + ' to %.1f deg/sec'%math.degrees(t_v)
    else:
        l = l + ' to %.1f m/s'%t_v

    legend_list.append(l)
    legend_list = legend_list[1:]
    giant_list = []

    traj_vel_min = 100000.
    for i,ftan_l in enumerate(d['ftan_l_l']):
        mech_x = d['mechx_l_l'][i]
        trial_num = str(d['trial_num_l'][i])
        color = None
        scatter_size = None

        traj_vel = compute_average_velocity(mech_x, d['time_l_l'][i],
                                            max_angle, d['typ'])
        traj_vel_min = min(traj_vel_min, traj_vel)
        if traj_vel == -1:
            continue
        if traj_vel >= v_threshold:
            continue

        color = sorted_color_list[traj_vel_sorted.index(traj_vel)]
        legend = legend_list[traj_vel_sorted.index(traj_vel)]
        if d['typ'] == 'rotary':
            #traj_vel = compute_trajectory_velocity(mech_x,d['time_l_l'][i],1)
            #if traj_vel >= filter_speed:
            #    continue
            mech_x_degrees = np.degrees(mech_x)
            xlabel = 'Angle (degrees)'
            #ylabel = '$f_{tan}(N)$'
            #ylabel = 'Tangential Force (N)'
            ylabel = 'Opening Force (N)'
        else:
            mech_x_degrees = mech_x
            xlabel = 'Distance (meters)'
            #ylabel = '$f_{tan}(N)$'
            ylabel = 'Tangential Force (N)'
#            n_skip = 65
#            print '>>>>>>>>>>>>> WARNING BEGIN <<<<<<<<<<<<<<<<<'
#            print 'not plotting the last ', n_skip, 'data points for drawers'
#            print '>>>>>>>>>>>>> WARNING END <<<<<<<<<<<<<<<<<'
            n_skip = 1
            mech_x_degrees = mech_x_degrees[:-n_skip]
            ftan_l = ftan_l[:-n_skip]

        mpu.figure(fig1.number)
        #color, scatter_size = None, None
        scatter_size = None
        if color == None:
            color = mpu.random_color()
        if scatter_size == None:
            scatter_size = 1

        giant_list.append((traj_vel, ftan_l, mech_x_degrees, color, legend, xlabel, ylabel, trial_num))

    giant_list_sorted = reversed(sorted(giant_list))
    for traj_vel, ftan_l, mech_x_degrees, color, legend, xlabel, ylabel, trial_num in giant_list_sorted:
        #if traj_vel > traj_vel_min + math.radians(10):
        #    continue
        scatter_size = 0
        fl = make_vector(mech_x_degrees, ftan_l, 60., 1.)
        #mpu.plot_yx(fl, axis=None,
        pp.plot(mech_x_degrees, ftan_l, color=color)
        pp.xlabel(xlabel)
        pp.ylabel(ylabel)
#        mpu.plot_yx(ftan_l, mech_x_degrees, axis=None,
#                    plot_title= '\huge{%s: %s}'%(mech_name, trial), xlabel=xlabel,
#                    ylabel = ylabel, color = color,
#                    #scatter_size = 5, linewidth = 1, label=trial_num)
#                    scatter_size = scatter_size, linewidth = 1, label=legend)
    print '>>>>>>>>>> number of trials <<<<<<<<<<<', len(giant_list)

#    mpu.figure(fig1.number)
#    mpu.legend(display_mode='less_space')
#    spring_force = ut.load_pickle('springloaded_force_filtered.pkl')
#    spring_angle = ut.load_pickle('springloaded_angle.pkl')
#    mpu.plot_yx(spring_force, np.degrees(spring_angle)-1.5, scatter_size=0,
#                linewidth=1, color='r')
#    mpu.pl.xlim(0, 40)

    pp.xlim(0, mech_x_degrees[-1])
    pp.ylim(-2, np.max(ftan_l)+2)
    fig1.subplots_adjust(bottom=.2, top=.99, right=.98, left=0.17)


def radial_tangential_ratio(dir_name):
    d = extract_pkls(dir_name, open=True)
    nm = get_mech_name(dir_name)
    frad_ll = d['frad_l_l']
    mechx_ll = d['mechx_l_l']
    mpu.figure()
    for i,ftan_l in enumerate(d['ftan_l_l']):
        frad_l = frad_ll[i]
        rad_arr = np.array(np.abs(frad_l))
        tan_arr = np.array(np.abs(ftan_l))
        idxs = np.where(np.logical_and(rad_arr > 0.1, tan_arr > 0.1))
        ratio = np.divide(rad_arr[idxs], tan_arr[idxs])
        mpu.plot_yx(ratio, np.degrees(np.array(mechx_ll[i])[idxs]),
                    color = mpu.random_color(), plot_title=nm,
                    ylabel='radial/tangential', xlabel='Angle (degrees)')


##
# get mechanism name from the directory name.
def get_mech_name(d):
    t = d.split('/')
    mech_name = t[-1]
    if mech_name == '':
        mech_name = t[-2]
    return mech_name

##
# get all the information from all the pkls in one directory.
# ASSUMES - all pkls are of the same mechanism (same radius, type ...)
# @param open - extract info for opening trials
def extract_pkls(d, open=True, quiet = False, ignore_moment_list=False):
    if open:
        trial = 'open'
    else:
        trial = 'close'
    l = glob.glob(d+'/*'+trial+'*mechanism_trajectories*.pkl')
    l.sort()
    ftan_l_l, frad_l_l, mechx_l_l = [], [], []
    time_l_l, trial_num_l = [], []
    moment_l_l = []
    typ, rad = None, None
    for p in l:
        d = ut.load_pickle(p)

        if d.has_key('radius'):
            rad = d['radius']
        else:
            print 'WARNING number ONE'
            print 'WARNING: this might cause some trouble for drawers'
            rad = 1.

        if rad != -1:
            if d.has_key('moment_list'):
                moment_l_l.append(d['moment_list'])
            else:
                moment_l_l.append([0 for i in range(len(d['force_rad_list']))])

        trial_num = p.split('_')[-5]
        trial_num_l.append(int(trial_num))
        frad_l_l.append(d['force_rad_list'])
        ftan_l_l.append(d['force_tan_list'])

        if d.has_key('mech_type'):
            typ = d['mech_type']
        else:
            print 'WARNING number TWO'
            print 'WARNING: this might cause some trouble.'
            typ = 'rotary'
            #return None

        mechx_l_l.append(d['mechanism_x'])

        if d.has_key('time_list'):
            t_l = d['time_list']
        else:
            t_l = [0.03*i for i in range(len(ftan_l_l[-1]))]
        time_l_l.append((np.array(t_l)-t_l[0]).tolist())

    r = {}
    r['ftan_l_l'] = ftan_l_l
    r['frad_l_l'] = frad_l_l
    r['mechx_l_l'] = mechx_l_l
    r['typ'] = typ
    r['rad'] = rad
    r['time_l_l'] = time_l_l
    r['trial_num_l'] = trial_num_l
    r['moment_l_l'] = moment_l_l

    return r

##
# take max force magnitude within a bin size
# @param bin_size - depends on the units of poses_list
# @param fn - function to apply to the binned force values (e.g. max, min)
# @param ignore_empty - if True then empty bins are ignored. Else the
#                       value of an empty bin is set to None.
# @param max_pose - maximum value of pose to use if None then derived
#                   from poses_list
# @param empty_value - what to fill in an empty bin (None, np.nan etc.)
def bin(poses_list, ftan_list, bin_size, fn, ignore_empty,
        max_pose=None, empty_value = None):
    if max_pose == None:
        max_dist = max(poses_list)
    else:
        max_dist = max_pose

    poses_array = np.array(poses_list)
    binned_poses_array = np.arange(0., max_dist, bin_size)

    binned_force_list = []
    binned_poses_list = []
    ftan_array = np.array(ftan_list)

    for i in range(binned_poses_array.shape[0]-1):
        idxs = np.where(np.logical_and(poses_array>=binned_poses_array[i],
                                       poses_array<binned_poses_array[i+1]))
        if idxs[0].shape[0] != 0:
            binned_poses_list.append(binned_poses_array[i])
            if i == 0:
                binned_force_list.append(np.nanmax(ftan_array[idxs]))
            else:
                binned_force_list.append(fn(ftan_array[idxs]))
        elif ignore_empty == False:
            binned_poses_list.append(binned_poses_array[i])
            binned_force_list.append(empty_value)

    return binned_poses_list, binned_force_list

##
# makes a scatter plot with the radius along the x axis and the max
# force along the y-axis. different color for each mechanism
# @param dir_name - directory containing the pkls.
def max_force_radius_scatter(dir_name_list, open=True):
    mpu.figure()
    if open:
        trial = 'Opening'
    else:
        trial = 'Closing'

    for d in dir_name_list:
        nm = get_mech_name(d)
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        print 'MECHANISM:', nm
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        ftan_l_l, frad_l_l, mechx_l_l, typ, rad = extract_pkls(d, open)
        fl, rl = [], []
        for n in range(len(ftan_l_l)):
            fmax = np.max(np.abs(ftan_l_l[n]))
            print 'fmax:', fmax
            fl.append(fmax)
            rl.append(rad)
        print 'Aloha'
        print 'len(fl)', len(fl)
        print 'len(rl)', len(rl)
        #mpu.plot_yx(fl, rl)
        plot_title = 'Scatter plot for %s trials'%trial
        mpu.plot_yx(fl, rl, color=mpu.random_color(), label=nm,
                    axis=None, linewidth=0, xlabel='Radius (m)',
                    ylabel='Maximum Force (N)', plot_title=plot_title)
    mpu.legend()

# use all the values of all the bins for all the trials.
# @param plot_type - 'tangential', 'magnitude', 'radial'
def errorbar_one_mechanism(dir_name, open=True, new_figure=True,
                           filter_speed = math.radians(100),
                           plot_type='tangential', color=None,
                           label = None):
    if new_figure:
        mpu.figure()
    nm = get_mech_name(dir_name)

    d = extract_pkls(dir_name, open)
    ftan_l_l = d['ftan_l_l']
    frad_l_l = d['frad_l_l']
    mechx_l_l = d['mechx_l_l']
    time_l_l = d['time_l_l']
    typ = d['typ']
    rad = d['rad']
    
    fn = list

    binned_mechx_l = []
    binned_ftan_ll = []
    use_trials_list = []

    if plot_type == 'radial':
        force_l_l = frad_l_l
    if plot_type == 'tangential':
        force_l_l = ftan_l_l
    if plot_type == 'magnitude':
        force_l_l = []
        for ta, ra in zip(ftan_l_l, frad_l_l):
            force_l_l.append(ut.norm(np.matrix([ta, ra])).A1.tolist())

    n_trials = len(force_l_l)
    for i in range(n_trials):
        if typ == 'rotary':
            traj_vel = compute_trajectory_velocity(mechx_l_l[i],
                                                   time_l_l[i], 1)
            if traj_vel >= filter_speed:
                continue
            t, f = bin(mechx_l_l[i], force_l_l[i], math.radians(1.),
                       fn, ignore_empty=False, max_pose=math.radians(60))
        if typ == 'prismatic':
            t, f = bin(mechx_l_l[i], force_l_l[i], 0.01,
                       fn, ignore_empty=False, max_pose=0.5)
        if len(t) > len(binned_mechx_l):
            binned_mechx_l = t
        binned_ftan_ll.append(f)
        use_trials_list.append(i)

    n_trials = len(binned_ftan_ll)
    n_bins = len(binned_mechx_l)
    force_list_combined = [[] for i in binned_mechx_l]
    for i in range(n_trials):
        force_l = binned_ftan_ll[i]
        for j,p in enumerate(binned_mechx_l):
            if force_l[j] != None:
                if open:
                    if j < 5:
                        force_list_combined[j].append(max(force_l[j]))
                        continue
                else:
                    if (n_trials-j) < 5:
                        force_list_combined[j].append(min(force_l[j]))
                        continue
                force_list_combined[j] += force_l[j]

    plot_mechx_l = []
    mean_l, std_l = [], []
    for i,p in enumerate(binned_mechx_l):
        f_l = force_list_combined[i]
        if len(f_l) == 0:
            continue
        plot_mechx_l.append(p)
        mean_l.append(np.mean(f_l))
        std_l.append(np.std(f_l))

    if open:
        trial = 'Open'
    else:
        trial = 'Close'

    n_sigma = 1
    if typ == 'rotary':
        x_l = np.degrees(plot_mechx_l)
        xlabel='\huge{Angle (degrees)}'
    else:
        x_l = plot_mechx_l
        xlabel='Distance (m)'

    std_arr = np.array(std_l) * n_sigma
    if color == None:
        color = mpu.random_color()
    if label == None:
        label= nm+' '+plot_type

    mpu.plot_errorbar_yx(mean_l, std_arr, x_l, linewidth=1, color=color,
                         plot_title='\huge{Mean \& %d$\sigma$}'%(n_sigma),
                         xlabel=xlabel, label=label,
                         ylabel='\huge{Force (N)}')
    mpu.legend()

# take the max of each bin.
def errorbar_one_mechanism_max(dir_name, open=True,
                               filter_speed=math.radians(100.)):
#    mpu.figure()
    nm = get_mech_name(dir_name)
    d = extract_pkls(dir_name, open)
    ftan_l_l = d['ftan_l_l']
    frad_l_l = d['frad_l_l']
    mechx_l_l = d['mechx_l_l']
    time_l_l = d['time_l_l']
    typ = d['typ']
    rad = d['rad']

    fn = max
    if open == False:
        fn = min

    binned_mechx_l = []
    binned_ftan_ll = []
    use_trials_list = []

    n_trials = len(ftan_l_l)
    for i in range(n_trials):
        if typ == 'rotary':
            traj_vel = compute_trajectory_velocity(mechx_l_l[i],
                                                   time_l_l[i], 1)
            if traj_vel >= filter_speed:
                continue
            t, f = bin(mechx_l_l[i], ftan_l_l[i], math.radians(1.),
                       fn, ignore_empty=False, max_pose=math.radians(60))
        if typ == 'prismatic':
            t, f = bin(mechx_l_l[i], ftan_l_l[i], 0.01,
                       fn, ignore_empty=False, max_pose=0.5)

        if len(t) > len(binned_mechx_l):
            binned_mechx_l = t
        binned_ftan_ll.append(f)
        use_trials_list.append(i)

    binned_ftan_arr = np.array(binned_ftan_ll)
    plot_mechx_l = []
    mean_l, std_l = [], []
    for i,p in enumerate(binned_mechx_l):
        f_l = []
        for j in range(len(use_trials_list)):
            if binned_ftan_arr[j,i] != None:
                f_l.append(binned_ftan_arr[j,i])
        if len(f_l) == 0:
            continue
        plot_mechx_l.append(p)
        mean_l.append(np.mean(f_l))
        std_l.append(np.std(f_l))

    xlabel = 'Angle (degrees)'

    if open:
        trial = 'Open'
    else:
        trial = 'Close'

    n_sigma = 1
    std_arr = np.array(std_l) * n_sigma
    mpu.plot_errorbar_yx(mean_l, std_arr, np.degrees(plot_mechx_l),
                         linewidth=1, plot_title=nm+': '+trial,
                         xlabel='Angle (degrees)',
                         label='Mean \& %d$\sigma$'%(n_sigma),
                         ylabel='Tangential Force (N)',
                         color='y')
    mpu.legend()

def plot_opening_distances_drawers(dir_name_list):
    mpu.figure()
    for d in dir_name_list:
        nm = get_mech_name(d)
        ftan_l_l, frad_l_l, mechx_l_l, typ, rad = extract_pkls(d)
        if rad != -1:
            # ignoring the cabinet doors.
            continue
        dist_opened_list = []
        for x_l in mechx_l_l:
            dist_opened_list.append(x_l[-1] - x_l[0])

        plot_title = 'Opening distance for drawers'
        mpu.plot_yx(dist_opened_list, color=mpu.random_color(), label=nm,
                    axis=None, linewidth=0, xlabel='Nothing',
                    ylabel='Distance opened', plot_title=plot_title)
    mpu.legend()

def handle_height_histogram(dir_name_list, plot_title=''):
    mean_height_list = []
    for d in dir_name_list:
        nm = get_mech_name(d)
        pkl = glob.glob(d+'/mechanism_calc_dict.pkl')
        if pkl == []:
            print 'Mechanism "%s" does not have a mechanism_calc_dict'%nm
            continue
        pkl = pkl[0]
        mech_calc_dict = ut.load_pickle(pkl)
        hb = mech_calc_dict['handle_bottom']
        ht = mech_calc_dict['handle_top']
        mean_height_list.append((hb+ht)/2.)

    #max_height = np.max(mean_height_list)
    max_height = 2.0
    bin_width = 0.1
    bins = np.arange(0.-bin_width/2., max_height+2*bin_width, bin_width)
    hist, bin_edges = np.histogram(np.array(mean_height_list), bins)
    mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                       width=bin_width*0.8, plot_title=plot_title,
                       xlabel='Height (meters)', ylabel='\# of mechanisms')

def plot_handle_height(dir_name_list, plot_title):
    mpu.figure()
    for d in dir_name_list:
        nm = get_mech_name(d)
        pkl = glob.glob(d+'/mechanism_calc_dict.pkl')
        if pkl == []:
            print 'Mechanism "%s" does not have a mechanism_calc_dict'%nm
            continue
        pkl = pkl[0]
        mech_calc_dict = ut.load_pickle(pkl)
        hb = mech_calc_dict['handle_bottom']
        ht = mech_calc_dict['handle_top']

        di = extract_pkls(d, open=True)
        ftan_l_l = di['ftan_l_l']
        frad_l_l = di['frad_l_l']
        mechx_l_l = di['mechx_l_l']
        time_l_l = di['time_l_l']
        typ = di['typ']
        rad = di['rad']
#        ftan_l_l, frad_l_l, mechx_l_l, typ, rad = extract_pkls(d,
#                                                        open=True)
        fl, hl = [], []
        for n in range(len(ftan_l_l)):
            fmax = np.max(np.abs(ftan_l_l[n][0:-50]))
            fl.append(fmax)
            fl.append(fmax)
            hl.append(ht)
            hl.append(hb)

        mpu.plot_yx(hl, fl, color=mpu.random_color(), label=nm,
                    axis=None, linewidth=0, xlabel='Max opening force',
                    ylabel='Handle Height (m)', plot_title=plot_title)
    mpu.legend()

def distance_of_handle_from_edges():
    pass

def plot_handle_height_no_office():
    opt = commands.getoutput('cd aggregated_pkls_Feb11; ls --ignore=*HSI* --ignore=*HRL* --ignore=a.py')
    d_list = opt.splitlines()
    dir_list = []
    for d in d_list:
        dir_list.append('aggregated_pkls_Feb11/'+d)

    plot_title = 'Only homes. Excluding Offices'
    plot_handle_height(dir_list[0:], plot_title)

def plot_handle_height_no_fridge_no_freezer():
    opt = commands.getoutput('cd aggregated_pkls_Feb11; ls --ignore=*refrigerator* --ignore=*freezer* --ignore=a.py')
    d_list = opt.splitlines()
    dir_list = []
    for d in d_list:
        dir_list.append('aggregated_pkls_Feb11/'+d)

    plot_title = 'Excluding Refrigerators and Freezers'
    plot_handle_height(dir_list[0:], plot_title)

##
# returns the median of the velocity.
def compute_velocity(mech_x, time_list, smooth_window):
    x = np.array(mech_x)
    t = np.array(time_list)
    kin_info = {'disp_mech_coord_arr': np.array(mech_x),
                'mech_time_arr': np.array(time_list)}
    vel_arr = mfc.velocity(kin_info, smooth_window)
    return vel_arr

##
# mech_x must be in RADIANS.
def compute_trajectory_velocity(mech_x, time_list, smooth_window):
    vel_arr = compute_velocity(mech_x, time_list, smooth_window)
    filt_vel_arr = vel_arr[np.where(vel_arr>math.radians(2.))]
    median_vel = np.median(filt_vel_arr)
    return median_vel

##
# compute the average velocity = total angle / total time.
def compute_average_velocity(mech_x, time_list, max_angle, type):
    reject_num = 20
    if len(mech_x) < reject_num:
        return -1

    mech_x = mech_x[reject_num:]
    time_list = time_list[reject_num:]
    if mech_x[-1] < max_angle:
        return -1

    if type == 'rotary':
        start_angle = math.radians(1)
    elif type == 'prismatic':
        start_angle = 0.01

    mech_x = np.array(mech_x)
    start_idx = np.where(mech_x > start_angle)[0][0]
    nw = np.where(mech_x > max_angle)
    if len(nw[0]) == 0:
        end_idx = -1
    else:
        end_idx = nw[0][0]

    start_x = mech_x[start_idx]
    end_x = mech_x[end_idx]

    start_time = time_list[start_idx]
    end_time = time_list[end_idx]

    avg_vel = (end_x - start_x) / (end_time - start_time)
    return avg_vel


def plot_velocity(dir_name):
    d = extract_pkls(dir_name, True)
    vel_fig = mpu.figure()
    acc_fig = mpu.figure()
    for i,time_list in enumerate(d['time_l_l']):
        mechx_l = d['mechx_l_l'][i]
        mechx_l, vel, acc, time_list = mfc.kinematic_params(mechx_l, time_list, 10)
        vel_arr = np.array(vel)
        acc_arr = np.array(acc)

        trial_num = d['trial_num_l'][i]

        xarr = np.array(mechx_l)
        idxs = np.where(np.logical_and(xarr < math.radians(20.),
                                       xarr > math.radians(1.)))

        color=mpu.random_color()
        mpu.figure(vel_fig.number)
        mpu.plot_yx(np.degrees(vel_arr[idxs]),
                    np.degrees(xarr[idxs]), color=color,
                    label='%d velocity'%trial_num, scatter_size=0)
        mpu.legend()

        mpu.figure(acc_fig.number)
        mpu.plot_yx(np.degrees(acc_arr[idxs]), np.degrees(xarr[idxs]), color=color,
                    label='%d acc'%trial_num, scatter_size=0)
        mpu.legend()

##
# l list of trials with which to correlate c1
# trial is a list of forces (each element is the max force or some
# other representative value for a given angle)
# lab_list - list of labels
def correlate_trials(c1, l, lab_list):
    mpu.figure()
    x = 0
    corr_list = []
    x_l = []
    for i,c2 in enumerate(l):
        res = ss.correlate(np.array(c1), np.array(c2), 'valid')[0]
        r1 = ss.correlate(np.array(c1), np.array(c1), 'valid')[0]
        r2 = ss.correlate(np.array(c2), np.array(c2), 'valid')[0]
        res = res/math.sqrt(r1*r2) # cross correlation coefficient  http://www.staff.ncl.ac.uk/oliver.hinton/eee305/Chapter6.pdf
        if i == 0 or lab_list[i] == lab_list[i-1]:
            corr_list.append(res)
            x_l.append(x)
        else:
            mpu.plot_yx(corr_list, x_l, color=mpu.random_color(),
                        label=lab_list[i-1], xlabel='Nothing',
                        ylabel='Cross-Correlation Coefficient')
            corr_list = []
            x_l = []
        x += 1
    mpu.plot_yx(corr_list, x_l, color=mpu.random_color(),
                label=lab_list[i-1])
    mpu.legend()

##
# plot errorbars showing 1 sigma for tangential component of the force
# and the total magnitude of the force. (Trying to verify that what we
# are capturing using our setup is consistent across people)
def compare_tangential_total_magnitude(dir):
    mpu.figure()
    errorbar_one_mechanism(dir, open = True,
                           filter_speed = math.radians(30),
                           plot_type = 'magnitude',
                           new_figure = False, color='y',
                           label = '\huge{$\hat F_{normal}$}')

    errorbar_one_mechanism(dir, open = True,
                           filter_speed = math.radians(30),
                           plot_type = 'tangential', color='b',
                           new_figure = False,
                           label = '\huge{$||\hat F_{normal} + \hat F_{plane}||$}')

def max_force_vs_velocity(dir):
    di = extract_pkls(dir, open)
    ftan_l_l = di['ftan_l_l']
    frad_l_l = di['frad_l_l']
    mechx_l_l = di['mechx_l_l']
    time_l_l = di['time_l_l']
    typ = di['typ']
    rad = di['rad']

    nm = get_mech_name(dir)
#    mpu.figure()
    color = mpu.random_color()
    mfl = []
    tvl = []
    for i in range(len(ftan_l_l)):

        xarr = np.array(mechx_l_l[i])
        idxs = np.where(np.logical_and(xarr < math.radians(20.),
                                       xarr > math.radians(1.)))

        max_force = np.max(np.array(ftan_l_l[i])[idxs])
        mechx_short = np.array(mechx_l_l[i])[idxs]
        time_short = np.array(time_l_l[i])[idxs]

        vel_arr = compute_velocity(mechx_l_l[i], time_l_l[i], 5)
        #vel_arr = compute_velocity(mechx_short[i], time_l_l[i], 5)
        vel_short = vel_arr[idxs]
        traj_vel = np.max(vel_short)
        #acc_arr = compute_velocity(vel_arr, time_l_l[i], 1)
        #traj_vel = np.max(acc_arr)
        #traj_vel = compute_trajectory_velocity(mechx_short, time_short, 1)

        mfl.append(max_force)
        tvl.append(traj_vel)

    mpu.plot_yx(mfl, tvl, color = color,
                xlabel = 'Trajectory vel', label = nm,
                ylabel = 'Max tangential force', linewidth=0)
    mpu.legend()

def mechanism_radius_histogram(dir_list, color='b'):
    rad_list = []
    for d in dir_list:
        nm = get_mech_name(d)
        pkl = glob.glob(d+'/mechanism_info.pkl')
        if pkl == []:
            print 'Mechanism "%s" does not have a mechanism_info_dict'%nm
            continue
        pkl = pkl[0]
        md = ut.load_pickle(pkl)
        if md['radius'] != -1:
            rad_list.append(md['radius'])
    max_radius = np.max(rad_list)
    print 'Rad list:', rad_list
    bin_width = 0.05
    bins = np.arange(0.-bin_width/2., max_radius+2*bin_width, bin_width)
    hist, bin_edges = np.histogram(np.array(rad_list), bins)
    print 'Bin Edges:', bin_edges
    print 'Hist:', hist
    h = mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                       width=0.8*bin_width, xlabel='Radius(meters)',
                       ylabel='\# of mechanisms',
                       plot_title='Histogram of radii of rotary mechanisms',
                       color=color)
    return h



def make_vector(mechx, ftan_l, lim, bin_size):
    t, f = bin(mechx, ftan_l, bin_size, max,
               ignore_empty=False, max_pose=lim,
               empty_value = np.nan)
    f = np.array(f)
    t = np.array(t)


    clean_idx = np.where(np.logical_not(np.isnan(f)))
    miss_idx = np.where(np.isnan(f))
    if len(miss_idx[0]) > 0:
        fclean = f[clean_idx]
        mechx_clean = t[clean_idx]
        mechx_miss = t[miss_idx]
        f_inter = mfc.interpolate_1d(mechx_clean, fclean, mechx_miss)
        f[np.where(np.isnan(f))] = f_inter
        #import pdb
        #pdb.set_trace()
    return np.matrix(f).T

# this makes the vector unit norm before returning it. Currently, this
# function is only used for PCA.
def make_vector_mechanism(dir, use_moment = False, make_unit=True):
    print '>>>>>>>>>>>>>>>>>>>>>>'
    print 'dir:', dir
    di = extract_pkls(dir)
    ftan_l_l = di['ftan_l_l']
    frad_l_l = di['frad_l_l']
    mechx_l_l = di['mechx_l_l']
    time_l_l = di['time_l_l']
    moment_l_l = di['moment_l_l']
    typ = di['typ']
    rad = di['rad']

    n_trials = len(ftan_l_l)
    vec_list = []
    tup_list = []
    for i in range(n_trials):
        if typ == 'rotary':
            if use_moment:
                torque_l = moment_l_l[i]
            else:
                torque_l = ftan_l_l[i]

            if len(mechx_l_l[i]) < 30:
                continue

            v = make_vector(mechx_l_l[i], torque_l, lim = math.radians(60.),
                            bin_size = math.radians(1))
            max_angle = math.radians(30)

        if typ == 'prismatic':
            v = make_vector(mechx_l_l[i], ftan_l_l[i], lim = 0.25,
                            bin_size = 0.01)


        traj_vel = compute_average_velocity(mechx_l_l[i], time_l_l[i], max_angle, typ)
        if traj_vel == -1:
            continue
        #v = v / np.linalg.norm(v)
        vec_list.append(v)
        tup_list.append((traj_vel,v))

    if len(vec_list) <= 1:
        return None

    tup_list.sort()
    [vel_list, vec_list] = zip(*tup_list)

    v_prev = vel_list[0]
    t_v = v_prev
    thresh = math.radians(10)
    i = 0
    ret_list = []
    for j,v in enumerate(vel_list):
        # only for different people same mechanism.
        #if v > math.radians(20):
        #    break
        if (v - v_prev) > thresh:
            i += 1
            i += 1
            if i >= 2:
                break
            v_prev = v
        ret_list.append(vec_list[j])

    print '________________________________'
    print 'Number of trials:', len(ret_list)
    print 'Max vel:', math.degrees(vel_list[j-1])
    print 'Min vel:', math.degrees(vel_list[0])

    # selecting only the slowest three trials.
#    tup_list.sort()
#    if len(tup_list) > 3:
#        [acc_list, v_list] = zip(*tup_list)
#        return np.column_stack(v_list[0:3])

    return np.column_stack(ret_list)


##
# one of the figures for the paper.
# showing that different classes have different clusters.
def different_classes_rotary(dir_list):
    mech_vec_list = []
    legend_list = []
    for d in dir_list:
        di = extract_pkls(d)
        if di == None:
            continue
        if di.has_key('typ'):
            typ = di['typ']
            if typ != 'rotary':
            #if typ != 'prismatic':
                continue
        else:
            continue

        v = make_vector_mechanism(d)
        if v == None:
            continue
        mech_vec_list.append(v)
        legend_list.append(get_mech_name(d))

    all_vecs = np.column_stack(mech_vec_list)
    print '>>>>>>>>> all_vecs.shape <<<<<<<<<<<<<', all_vecs.shape
    U, s, _ = np.linalg.svd(np.cov(all_vecs))
    mn = np.mean(all_vecs, 1).A1

    mpu.set_figure_size(3.,3.)
    mpu.figure()
    proj_mat = U[:, 0:2]
    legend_made_list = [False, False, False]
    for i, v in enumerate(mech_vec_list):
        p = proj_mat.T * (v - np.matrix(mn).T)
        if np.any(p[0,:].A1<0):
            print 'First principal component < 0 for some trial of:', legend_list[i]
        color = mpu.random_color()
        if 'ree' in legend_list[i]:
            #color = 'g'
            color = '#66FF33'
            if legend_made_list[0] == False:
                label = 'Freezers'
                legend_made_list[0] = True
            else:
                label = '__nolegend__'
            print 'SHAPE:', p.shape
            print 'p:', p

        elif 'ge' in legend_list[i]:
            color = '#FF6633'
            #color = 'y'
            if legend_made_list[1] == False:
                label = 'Refrigerators'
                legend_made_list[1] = True
            else:
                label = '__nolegend__'
        else:
            #color = 'b'
            color = '#3366FF'
            if legend_made_list[2] == False:
                #label = '\\flushleft Cabinets, Spring \\\\*[-2pt] Loaded Doors'
                label = 'Cabinets'
                legend_made_list[2] = True
            else:
                label = '__nolegend__'
        pp.plot(p[0,:].A1, p[1,:].A1, color = color, linewidth=0,
                ms=6, marker='o', mew=0, mec=color, label = label)
    pp.xlabel('First Principal Component')
    pp.ylabel('Second Principal Component')
    pp.axis('equal')
    pp.axhline(y=0., color = 'k', ls='--')
    pp.axvline(x=0., color = 'k', ls='--')
    #mpu.legend(loc='upper center', display_mode = 'less_space', draw_frame = True)
    mpu.legend(loc='center left', display_mode = 'less_space',
               draw_frame = True)

    mpu.figure()
    mn = np.mean(all_vecs, 1).A1
    mn = mn/np.linalg.norm(mn)
    pp.plot(mn, color = '#FF3300', marker='o', mew=0,
                label = 'mean (normalized)', ms=3)

    c_list = ['#00CCFF', '#643DFF']
    for i in range(2):
        pp.plot(U[:,i].flatten(), color = c_list[i], marker='o',
                mew=0,
                    label = 'Eigenvector %d'%(i+1), ms=3)

    pp.axhline(y=0., color = 'k', ls='--')
    mpu.legend(display_mode = 'less_space', draw_frame=False)

    pp.show()

##
# one of the figures for the paper.
# showing that different classes have different clusters.
def different_classes_rotary_RAM(dir_list):
    mech_vec_list = []
    legend_list = []
    for d in dir_list:
        di = extract_pkls(d)
        if di == None:
            continue
        if di.has_key('typ'):
            typ = di['typ']
            if typ != 'rotary':
            #if typ != 'prismatic':
                continue
        else:
            continue

        v = make_vector_mechanism(d)
        if v == None:
            continue
        mech_vec_list.append(v)
        legend_list.append(get_mech_name(d))

    all_vecs = np.column_stack(mech_vec_list)
    print '>>>>>>>>> all_vecs.shape <<<<<<<<<<<<<', all_vecs.shape
    U, s, _ = np.linalg.svd(np.cov(all_vecs))
    mn = np.mean(all_vecs, 1).A1

    mpu.set_figure_size(10.,6.)
    pp.figure()
    proj_mat = U[:, 0:2]
    legend_made_list = [False, False, False, False, False]
    for i, v in enumerate(mech_vec_list):
        p = proj_mat.T * (v - np.matrix(mn).T)
        if np.any(p[0,:].A1<0):
            print 'First principal component < 0 for some trial of:', legend_list[i]
        color = mpu.random_color()
        if 'HSI' in legend_list[i] and 'net' in legend_list[i]:
            color = 'y'
            marker = 'v'
            ms = 4
            if legend_made_list[0] == False:
                label = 'Office Cabinet'
                legend_made_list[0] = True
            else:
                label = '__nolegend__'
        elif 'kitchen' in legend_list[i]:
            color = 'k'
            marker = 'x'
            ms = 4
            if legend_made_list[1] == False:
                label = 'Kitchen Cabinet'
                legend_made_list[1] = True
            else:
                label = '__nolegend__'
        elif 'ree' in legend_list[i]:
            marker = '+'
            ms = 4
            color = '#66FF33'
            if legend_made_list[2] == False:
                label = 'Freezers'
                legend_made_list[2] = True
            else:
                label = '__nolegend__'
        elif 'ge' in legend_list[i]:
            marker = '^'
            ms = 4
            color = '#FF6633'
            if legend_made_list[3] == False:
                label = 'Refrigerators'
                legend_made_list[3] = True
            else:
                label = '__nolegend__'
        elif 'lass' in legend_list[i]:
            color = 'c'
            marker = 'o'
            ms = 4
            if legend_made_list[4] == False:
                label = 'Springloaded door'
                legend_made_list[4] = True
            else:
                label = '__nolegend__'
        else:
            continue

        mec = color
        if marker == '^' or marker == 'v':
            color = 'w'
        pp.plot(p[0,:].A1, p[1,:].A1, color = color, linewidth=0,
                ms=ms, marker=marker, mew=1, mec=mec, label = label)
    pp.xlabel('1st Principal Component')
    pp.ylabel('2nd Principal Component')
    pp.axis('equal')
    pp.axhline(y=0., color = 'k', ls='--')
    pp.axvline(x=0., color = 'k', ls='--')
    #mpu.legend(loc='upper center', display_mode = 'less_space', draw_frame = True)
    mpu.legend(loc='bottom left', display_mode = 'normal',
               draw_frame = True)
    f = pp.gcf()
    f.subplots_adjust(bottom=.18, top=.94, right=.98, left=0.18)

    mpu.set_figure_size(8.,5.)
    pp.figure()
    mn = np.mean(all_vecs, 1).A1
    mn = mn/np.linalg.norm(mn)
    pp.plot(mn, color = '#FF3300', marker='+', mew=1,
                label = 'mean (normalized)', ms=5, mec='#FF3300')

    c_list = ['#00CCFF', '#643DFF']
    for i in range(2):
        pp.plot(U[:,i].flatten(), color = c_list[i], marker='o',
                mew=0, label = 'Eigenvector %d'%(i+1), ms=3,
                mec=c_list[i])

    pp.axhline(y=0., color = 'k', ls='--')
    mpu.legend(display_mode = 'normal', draw_frame=False)
    f = pp.gcf()
    f.subplots_adjust(bottom=.12, top=.94, right=.96, left=0.12)

    pp.show()


##
# makes a scatter plot with the radius along the x axis and the max
# force along the y-axis. different color for each mechanism
# @param dir_name - directory containing the pkls.
def max_force_hist(dir_name_list, open=True, type=''):
    if open:
        trial = 'Opening'
    else:
        trial = 'Closing'

    fls, freezer_list, fridge_list, springloaded_list = [],[],[],[]
    broiler_list = []
    num_mech = 0
    lens = []

    max_angle = math.radians(15.)
    max_dist = 0.1

    for d in dir_name_list:
        nm = get_mech_name(d)
        ep = extract_pkls(d, open, ignore_moment_list=True)
        if ep == None:
            continue
        ftan_l_l = ep['ftan_l_l']
        frad_l_l = ep['frad_l_l']
        mechx_l_l = ep['mechx_l_l']
        typ = ep['typ']
        rad = ep['rad']

        fl, rl = [], []
        for n in range(len(ftan_l_l)):
            ftan_l = ftan_l_l[n]

            mechx_a = np.array(mechx_l_l[n]) 
            if type == 'prismatic':
                indices = np.where(mechx_a < max_dist)[0]
            else:
                indices = np.where(mechx_a < max_angle)[0]

            if len(indices) > 0:
                ftan_l = np.array(ftan_l)[indices].tolist()

            fmax = np.max(np.abs(ftan_l))
            fl.append(fmax)
            rl.append(rad)

        #fmax_max = np.max(fl)
        fmax_max = np.min(fl)
        if type == 'rotary':
            if 'ree' in nm:
                freezer_list.append(fmax_max)
            elif 'naveen_microwave' in nm:
                # putting microwave in freezers
                freezer_list.append(fmax_max)
                if fmax_max < 5.:
                    print 'nm:', nm
            elif 'ge' in nm:
                fridge_list.append(fmax_max)
            elif fmax > 60.:
                springloaded_list.append(fmax_max)
        else:
            if 'ven' in nm:
                broiler_list.append(fmax_max)
            if fmax_max > 10.:
                print 'nm:', nm, 'fmax:', fmax_max

        fls.append(fmax_max)
        num_mech += 1
        lens.append(len(fl))
    
    if len(fls) > 0:
        max_force = np.max(fls)
        bin_width = 2.5
        bins = np.arange(0.-bin_width/2., max_force+2*bin_width, bin_width)
        if type == 'rotary':
            mpu.set_figure_size(1.5, 1.5)
            mpu.figure()
            hist, bin_edges = np.histogram(fls, bins)
            h = mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                               width=0.8*bin_width, xlabel='Force (Newtons)',
                               ylabel='\# of mechanisms',
                               color='b', label='Cabinets')
            max_freq = np.max(hist)

            hist, bin_edges = np.histogram(freezer_list + fridge_list, bins)
            h = mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                               width=0.8*bin_width, xlabel='Force (Newtons)',
                               ylabel='\# of mechanisms',
                               color='y', label='Appliances')

            hist, bin_edges = np.histogram(springloaded_list, bins)
            h = mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                               width=0.8*bin_width, xlabel='Force (Newtons)',
                               ylabel='\# of mechanisms',
                               color='g', label='Spring Loaded Doors')
            mpu.pl.xticks(np.arange(0.,max_force+2*bin_width, 10.))
            mpu.legend(display_mode='less_space', handlelength=1.)

            pb.xlim(-bin_width, max_force+bin_width)
            pb.ylim(0, max_freq+0.5)
        else:
            mpu.set_figure_size(1.5, 1.5)
            mpu.figure()
            hist, bin_edges = np.histogram(fls, bins)
            h = mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                               width=0.8*bin_width, xlabel='Force (Newtons)',
                               ylabel='\# of mechanisms',
                               color='b', label='Drawers')
            max_freq = np.max(hist)

            hist, bin_edges = np.histogram(broiler_list, bins)
            h = mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                               width=0.8*bin_width, xlabel='Force (Newtons)',
                               ylabel='\# of mechanisms',
                               color='y', label='Broilers')


        mpu.figure()
        bin_width = 2.5
        bins = np.arange(0.-bin_width/2., max_force+2*bin_width, bin_width)
        hist, bin_edges = np.histogram(fls, bins)
        h = mpu.plot_histogram(bin_edges[:-1]+bin_width/2., hist,
                           width=0.8*bin_width, xlabel='Force (Newtons)',
                           ylabel='\# of mechanisms',
                           color='b', label='All')
        mpu.legend()
        pb.xlim(-bin_width, max_force+bin_width)
        pb.ylim(0, np.max(hist)+1)
        mpu.pl.xticks(np.arange(0.,max_force+2*bin_width, 5.))
        mpu.pl.yticks(np.arange(0.,np.max(hist)+0.5, 1.))
    else:
        print "OH NO! FLS <= 0"


def dimen_reduction_mechanisms(dir_list, dimen = 2):
    mech_vec_list = []
    legend_list = []
    dir_list = filter_dir_list(dir_list)
    dir_list_new = []
    for d in dir_list:
        v = make_vector_mechanism(d)
        if v == None:
            continue
        print 'v.shape:', v.shape
        mech_vec_list.append(v)
        legend_list.append(get_mech_name(d))
        dir_list_new.append(d)

    all_vecs = np.column_stack(mech_vec_list)
    #U, s, _ = np.linalg.svd(np.cov(all_vecs))
    normalized_all_vecs = all_vecs
    print '>>>>>>>>>>>> all_vecs.shape <<<<<<<<<<<<<<<', all_vecs.shape
    #normalized_all_vecs = (all_vecs - all_vecs.mean(1))

    #Rule this out as it places equal value on the entire trajectory but we want
    #to focus modeling efforts on the beginning of the trajectory
    #normalized_all_vecs = normalized_all_vecs / np.std(normalized_all_vecs, 1)

    #Removes the effect of force scaling... not sure if we want this
    #normalized_all_vecs = normalized_all_vecs / ut.norm(normalized_all_vecs)
    U, s, _ = np.linalg.svd(np.cov((normalized_all_vecs)))

    perc_account = np.cumsum(s) / np.sum(s)
    mpu.plot_yx([0]+list(perc_account))

    mpu.set_figure_size(5.,5.)
    mpu.figure()
    mn = np.mean(all_vecs, 1).A1
    mpu.plot_yx(mn/np.linalg.norm(mn), color = '#FF3300',
                label = 'mean (normalized)', scatter_size=5)

    c_list = ['#%02x%02x%02x'%(r,g,b) for (r,g,b) in [(152, 32, 176), (23,94,16)]]
    c_list = ['#00CCFF', '#643DFF']
    for i in range(2):
        mpu.plot_yx(U[:,i].flatten(), color = c_list[i],
                    label = 'Eigenvector %d'%(i+1), scatter_size=5)

    mpu.pl.axhline(y=0., color = 'k')
    mpu.legend(display_mode='less_space')

    if dimen == 2:
        fig = mpu.figure()
        proj_mat = U[:, 0:2]
        for i, v in enumerate(mech_vec_list[:]):
            p = proj_mat.T * (v - np.matrix(mn).T)
            color = mpu.random_color()
            label = legend_list[i]

            mpu.plot_yx(p[1,:].A1, p[0,:].A1, color = color,
                        linewidth = 0, label = label,
                        xlabel='\huge{First Principal Component}',
                        ylabel='\huge{Second Principal Component}',
                        axis = 'equal', picker=0.5)

        mpu.pl.axhline(y=0., color = 'k', ls='--')
        mpu.pl.axvline(x=0., color = 'k', ls='--')
        mpu.legend()

        ppg = pca_plot_gui(legend_list, mech_vec_list, U,
                           dir_list_new, np.matrix(mn).T)
        fig.canvas.mpl_connect('button_press_event', ppg.pick_cb)
        mpu.show()

def filter_dir_list(dir_list, typ = 'rotary', name = None):
    filt_list = []
    for d in dir_list:
        nm = get_mech_name(d)
        if name != None:
            if name not in nm:
                continue

#        trial = 'open'
#        l = glob.glob(d+'/*'+trial+'*mechanism_trajectories*.pkl')
#        di = ut.load_pickle(l[0])
#        m_typ = di['typ']
#        if m_typ != typ:
#            continue
#        filt_list.append(d)

        di = extract_pkls(d, quiet=True)
        if di == None:
            continue
        if di.has_key('typ'):
            m_typ = di['typ']
            if m_typ != typ:
                continue
            filt_list.append(d)
        else:
            continue
    return filt_list



if __name__ == '__main__':
    import optparse
    p = optparse.OptionParser()
    p.add_option('-d', '--dir', action='store', default='',
                 type='string', dest='dir', help='directory with logged data')
    p.add_option('--check_data', action='store_true', dest='check_data',
                 help='count the number of trajectories for each mechanism')
    p.add_option('--rearrange', action='store_true', dest='rearrange',
                 help='rearrange aggregated pkls into separate folders for each mechanism')
    p.add_option('--clean', action='store_true', dest='clean',
                 help='remove pkls with corrupted data')

    p.add_option('--max_force_hist', action='store_true',
                 dest='max_force_hist', help='histogram of max forces')
    p.add_option('--max_force_radius_scatter', action='store_true',
                 dest='max_force_radius_scatter', help='scatter plot of max force vs radius')
    p.add_option('--opening_distances_drawers', action='store_true',
                 dest='opening_distances_drawers', help='opening distances for drawers')
    p.add_option('--plot_handle_height', action='store_true',
                 dest='plot_handle_height', help='handle height above the ground')
    p.add_option('--plot_radius', action='store_true',
                 dest='plot_radius', help='histogram of radii of the mechanisms')
    p.add_option('--correlate', action='store_true',
                 dest='correlate', help='correlation across different trials')
    p.add_option('--consistent_across_people', action='store_true',
                 dest='consistent_across_people',
                 help='plot mean and std for tangential and total magnitude of the force')
    p.add_option('--dimen_reduc', action='store_true',
                 dest='dimen_reduc',  help='try dimen reduction')
    p.add_option('--independence', action='store_true', 
                 dest='independence', help='test for conditional independence')
    p.add_option('--mech_models', action='store_true',
                 dest='mech_models',  help='fit mechanical models to data')
    p.add_option('--different_classes_rotary', action='store_true',
                 dest='different_classes_rotary',
                 help='scatter plot showing the differnt mechanism classes')

    opt, args = p.parse_args()

    dir_list = commands.getoutput('ls -d %s/*/'%(opt.dir)).splitlines()

#    drawers = 0
#    cabinets = 0
#    for d in dir_list:
#        nm = get_mech_name(d)
#        di = extract_pkls(d)
#        if di == None:
#            print 'di was None for:', nm
#            continue
#        if di['rad'] == -1:
#            drawers += 1
#        else:
#            cabinets += 1
#
#    print 'drawers:', drawers
#    print 'cabinets:', cabinets
#    import sys; sys.exit()

    if opt.clean:
        clean_data_forces(opt.dir)
    elif opt.rearrange:
        # listing all the different mechanisms
        pkl_list = commands.getoutput('ls %s/*.pkl'%(opt.dir)).splitlines()
        mech_name_list = []
        for p in pkl_list:
            nm = '_'.join(p.split('/')[-1].split('_')[:-5])
            print 'p:', p
            print 'nm:', nm
            mech_name_list.append(nm)
        mech_name_list = list(set(mech_name_list))
        print 'List of unique mechanisms:', mech_name_list

        for mech_name in mech_name_list:
            nm = '%s/%s'%(opt.dir, mech_name)
            os.system('mkdir %s'%nm)
            os.system('mv %s/*%s*.pkl %s'%(opt.dir, mech_name, nm))
    elif opt.check_data:
        mech_list = []
        for i, m in enumerate(dir_list):
            t = m.split('/')
            mech = t[-1]
            if mech == '':
                mech = t[-2]
            mech_list.append(mech)
            print '%d. %s'%(i, mech)

        for i,d in enumerate(dir_list):
            mech_nm = mech_list[i]
            print '------------------------------------'
            print 'Mechanism name:', mech_nm
            for trial in ['open', 'close']:
                l = glob.glob(d+'/*'+trial+'*mechanism_trajectories*.pkl')
                l.sort()
                print 'Number of %s trials: %d'%(trial, len(l))
    elif opt.max_force_radius_scatter:
        max_force_radius_scatter(dir_list[0:], open=True)
        max_force_radius_scatter(dir_list[0:], open=False)
        mpu.show()
    elif opt.max_force_hist:
        print 'Found %d mechanisms' % len(dir_list[0:])
        max_force_hist(filter_dir_list(dir_list[0:], typ='rotary'), open=True, type='rotary')
        max_force_hist(filter_dir_list(dir_list[0:], typ='prismatic'), open=True, type='prismatic')
        mpu.show()
    elif opt.opening_distances_drawers:
        plot_opening_distances_drawers(dir_list[0:])
        mpu.show()
    elif opt.plot_radius:
        l1 = filter_dir_list(dir_list, name='ree')
        l2 = filter_dir_list(dir_list, name='ge')
        print 'LEN:', len(filter_dir_list(dir_list, name=None))
        bar1 = mechanism_radius_histogram(filter_dir_list(dir_list, name=None))
        bar2 = mechanism_radius_histogram(l1+l2, color='y')
        labels = ['Other', 'Freezers and Refrigerators']
        mpu.pl.legend([bar1[0],bar2[0]], labels, loc='best')
        mpu.show()
    elif opt.plot_handle_height:
        #plot_title = 'Opening force at different heights'
        #plot_handle_height(dir_list[:], plot_title)

        #plot_handle_height_no_fridge_no_freezer()
#        plot_handle_height_no_office()
        #handle_height_histogram(dir_list, plot_title='Homes excluding \n refrigerators and freezers')
        #handle_height_histogram(filter_dir_list(dir_list, name='ge'),
        #                        plot_title = 'Refrigerators')
        handle_height_histogram(filter_dir_list(dir_list, name='ree'),
                                plot_title = 'Freezers')
        mpu.show()
    elif opt.correlate:
        cl = []
        lab_list = []
        ch_list =  input_mechanism_list(dir_list)
        for i, d in enumerate(ch_list):
            nm = get_mech_name(d)
            di = extract_pkls(d, open)
            ftan_l_l = di['ftan_l_l']
            frad_l_l = di['frad_l_l']
            mechx_l_l = di['mechx_l_l']
            time_l_l = di['time_l_l']
            typ = di['typ']
            rad = di['rad']
            #ftan_l_l, frad_l_l, mechx_l_l, typ, rad = extract_pkls(d, open)
            for j in range(len(ftan_l_l)):
                if typ == 'rotary':
                    traj_vel = compute_trajectory_velocity(mechx_l_l[j],time_l_l[j],1)
                    if traj_vel > math.radians(30):
                        continue
                    t, f = bin(mechx_l_l[j], ftan_l_l[j], math.radians(1.),
                               max, ignore_empty=True, max_pose=math.radians(60))
                if typ == 'prismatic':
                    t, f = bin(mechx_l_l[j], ftan_l_l[j], 0.01,
                               max, ignore_empty=True, max_pose=0.3)
                cl.append(f)
                lab_list.append(nm)
        correlate_trials(cl[0], cl[:], lab_list)
        mpu.show()
    elif opt.consistent_across_people:
        ch_list =  input_mechanism_list(dir_list)
        for dir in ch_list:
            compare_tangential_total_magnitude(dir)
        mpu.show()
    elif opt.different_classes_rotary:
        #different_classes_rotary(dir_list)
        different_classes_rotary_RAM(dir_list)
    elif opt.independence:
        test_independence_mechanism(dir_list)

    elif opt.dimen_reduc:
        #ch_list =  input_mechanism_list(dir_list)
        #dimen_reduction_mechanisms(ch_list)
        #dimen_reduction_mechanisms(filter_dir_list(dir_list, name='HSI_Suite_210_brown_cabinet_right'))
        dimen_reduction_mechanisms(dir_list)
    else:
#        filt_list = filter_dir_list(dir_list)
#        for dir in filt_list:
#            max_force_vs_velocity(dir)
#        mpu.show()

        d_list = input_mechanism_list(dir_list)
#        d_list = filter_dir_list(dir_list, typ='prismatic')
#        mpu.figure()
        traj_vel_l = []

        mpu.set_figure_size(10.,7.)
        fig1 = mpu.figure()
        for n,dir in enumerate(d_list):
            #make_vector_mechanism(dir)


#            vel_l = []
#            nm = get_mech_name(dir)
#            print '>>>>>>>>>> nm:', nm
#            d = extract_pkls(dir, ignore_moment_list=True)
#            for i, mech_x in enumerate(d['mechx_l_l']):
#                time_list = d['time_l_l'][i]
#                #v = compute_average_velocity(mech_x, time_list, max_angle=math.radians(30), type='rotary')
#                v = compute_average_velocity(mech_x, time_list, max_angle=0.2, type='prismatic')
#                print 'v:', v
#                if v == -1:
#                    continue
#                vel_l.append(v)
#            sorted_vel_l = sorted(vel_l)
#            if len(sorted_vel_l) > 6:
#                vel_l = sorted_vel_l[0:6]
#            else:
#                vel_l = sorted_vel_l
#            traj_vel_l += vel_l
#
#        #print 'mean angular velocity:', math.degrees(np.mean(traj_vel_l))
#        #print 'std angular velocity:', math.degrees(np.std(traj_vel_l))
#
#        print 'mean angular velocity:', np.mean(traj_vel_l)
#        print 'std angular velocity:', np.std(traj_vel_l)

            #plot_tangential_force(dir, all_trials = True, open = True,
            #                      filter_speed = math.radians(30))

            plot_tangential_force(dir, all_trials = True, open = True,
                                  fig1 = None)
            
            #plot_tangential_filtered(dir, fig1 = fig1, legend = str(n+1))

#        mpu.legend(display_mode='less_space', draw_frame=False)
        mpu.show()




