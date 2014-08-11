
import scipy.optimize as so

import math, numpy as np
import pylab as pl
import sys, optparse, time
import copy

## from mayavi import mlab

import roslib; roslib.load_manifest('hrl_anomaly_detection')

import hrl_lib.util as ut
## import roslib; roslib.load_manifest('modeling_forces')
## #import cody_arms.arms as ca

## roslib.load_manifest('hrl_cody_arms')
## import hrl_cody_arms.cody_arm_kinematics as cak

import hrl_lib.matplotlib_util as mpu
## import hrl_lib.util as ut, hrl_lib.transforms as tr
## import hrl_tilting_hokuyo.display_3d_mayavi as d3m

## import segway_motion_calc as smc

class JointTrajectory():
    ''' class to store joint trajectories.
        data only - use for pickling.
    '''
    def __init__(self):
        self.time_list = [] # time in seconds
        self.q_list = [] #each element is a list of 7 joint angles.
        self.qdot_list = [] #each element is a list of 7 joint angles.
        self.qdotdot_list = [] #each element is a list of 7 joint angles.

## class to store trajectory of a coord frame executing planar motion (x,y,a)
#data only - use for pickling
class PlanarTrajectory():
    def __init__(self):
        self.time_list = [] # time in seconds
        self.x_list = []
        self.y_list = []
        self.a_list = []

class CartesianTajectory():
    ''' class to store trajectory of cartesian points.
        data only - use for pickling
    '''
    def __init__(self):
        self.time_list = [] # time in seconds
        self.p_list = [] #each element is a list of 3 coordinates
        self.v_list = [] #each element is a list of 3 coordinates (velocity)

class ForceTrajectory():
    ''' class to store time evolution of the force at the end effector.
        data only - use for pickling
    '''
    def __init__(self):
        self.time_list = [] # time in seconds
        self.f_list = [] #each element is a list of 3 coordinates

## ##
## # @param traj - JointTrajectory
## # @return CartesianTajectory after performing FK on traj to compute
## # cartesian position, velocity
## def joint_to_cartesian(traj, arm):
##     #firenze = ca.M3HrlRobot(end_effector_length = 0.17318)
##     if arm == 'right_arm':
##         arm = 'r'
##     else:
##         arm = 'l'

##     firenze = cak.CodyArmKinematics(arm)
##     firenze.set_tooltip(np.matrix([0.,0.,-0.12]).T)

##     pts = []
##     cart_vel = []
##     for i in range(len(traj.q_list)):
##         q = traj.q_list[i]
##         p, _ = firenze.FK(q)
##         pts.append(p.A1.tolist())

##         if traj.qdot_list != [] and traj.qdot_list[0] != None:
##             qdot = traj.qdot_list[i]
##             jac = firenze.Jacobian(q)
##             vel = jac * np.matrix(qdot).T
##             cart_vel.append(vel.A1[0:3].tolist())

##     ct = CartesianTajectory()
##     ct.time_list = copy.copy(traj.time_list)
##     ct.p_list = copy.copy(pts)
##     ct.v_list = copy.copy(cart_vel)
##     #return np.matrix(pts).T
##     return ct

## def plot_forces_quiver(pos_traj,force_traj,color='k'):
##     import arm_trajectories as at
##     #if traj.__class__ == at.JointTrajectory:
##     if isinstance(pos_traj,at.JointTrajectory):
##         pos_traj = joint_to_cartesian(pos_traj)

##     pts = np.matrix(pos_traj.p_list).T
##     label_list = ['X coord (m)', 'Y coord (m)', 'Z coord (m)']
##     x = pts[0,:].A1.tolist()
##     y = pts[1,:].A1.tolist()

##     forces = np.matrix(force_traj.f_list).T
##     u = (-1*forces[0,:]).A1.tolist()
##     v = (-1*forces[1,:]).A1.tolist()
##     pl.quiver(x,y,u,v,width=0.002,color=color,scale=100.0)
## #    pl.quiver(x,y,u,v,width=0.002,color=color)
##     pl.axis('equal')

## ##
## # @param xaxis - x axis for the graph (0,1 or 2)
## # @param zaxis - for a 3d plot. not implemented.
## def plot_cartesian(traj, xaxis=None, yaxis=None, zaxis=None, color='b',label='_nolegend_',
##                    linewidth=2, scatter_size=10, plot_velocity=False):

##     import arm_trajectories as at
##     #if traj.__class__ == at.JointTrajectory:
##     if isinstance(traj,at.JointTrajectory):
##         traj = joint_to_cartesian(traj)

##     pts = np.matrix(traj.p_list).T
##     label_list = ['X coord (m)', 'Y coord (m)', 'Z coord (m)']
##     x = pts[xaxis,:].A1.tolist()
##     y = pts[yaxis,:].A1.tolist()

##     if plot_velocity:
##         vels = np.matrix(traj.v_list).T
##         xvel = vels[xaxis,:].A1.tolist()
##         yvel = vels[yaxis,:].A1.tolist()

##     if zaxis == None:
##         mpu.plot_yx(y, x, color, linewidth, '-', scatter_size, label,
##                     axis = 'equal', xlabel = label_list[xaxis],
##                     ylabel = label_list[yaxis],)
##         if plot_velocity:
##             mpu.plot_quiver_yxv(y, x, np.matrix([xvel,yvel]),
##                                 width = 0.001, scale = 1.)
##         mpu.legend()
##     else:
##         from numpy import array
##         from enthought.mayavi.api import Engine
##         engine = Engine()
##         engine.start()
##         if len(engine.scenes) == 0:
##             engine.new_scene()

##         z = pts[zaxis,:].A1.tolist()
##         time_list = [t-traj.time_list[0] for t in traj.time_list]
##         mlab.plot3d(x,y,z,time_list,tube_radius=None,line_width=4)
##         mlab.axes()
##         mlab.xlabel(label_list[xaxis])
##         mlab.ylabel(label_list[yaxis])
##         mlab.zlabel(label_list[zaxis])
##         mlab.colorbar(title='Time')

##         # ------------------------------------------- 
##         axes = engine.scenes[0].children[0].children[0].children[1]
##         axes.axes.position = array([ 0.,  0.])
##         axes.axes.label_format = '%-#6.2g'
##         axes.title_text_property.font_size=4

## return two lists containing the radial and tangential components of the forces.
# @param f_list - list of forces. (each force is a list of 2 or 3 floats)
# @param p_list - list of positions. (each position is a list of 2 or 3 floats)
# @param cx - x coord of the center of the circle.
# @param cy - y coord of the center of the circle.
# @return list of magnitude of radial component, list of magnitude
# tangential component, list of the force along the remaining
# direction
def compute_radial_tangential_forces(f_list,p_list,cx,cy):
    f_rad_l,f_tan_l, f_res_l = [], [], []
    for f,p in zip(f_list,p_list):
        rad_vec = np.array([p[0]-cx,p[1]-cy])
        rad_vec = rad_vec/np.linalg.norm(rad_vec)
        tan_vec = (np.matrix([[0,-1],[1,0]]) * np.matrix(rad_vec).T).A1
        f_vec = np.array([f[0],f[1]])

        f_tan_mag = np.dot(f_vec, tan_vec)
        f_rad_mag = np.dot(f_vec, rad_vec)

#        f_res_mag = np.linalg.norm(f_vec- rad_vec*f_rad_mag - tan_vec*f_tan_mag)
        f_rad_mag = abs(f_rad_mag)
        f_tan_mag = abs(f_tan_mag)

        f_rad_l.append(f_rad_mag)
        f_tan_l.append(f_tan_mag)
        f_res_l.append(abs(f[2]))

    return f_rad_l, f_tan_l, f_res_l
        


## def fit_circle_priors(rad_guess, x_guess, y_guess, pts, sigma_r,
##                       sigma_xy, sigma_pts, verbose=True):
##     global x_prior, y_prior
##     x_prior = x_guess
##     y_prior = y_guess
##     def error_function(params):
##         center = np.matrix((params[0],params[1])).T
##         rad = params[2]
##         err_pts = ut.norm(pts-center).A1 - rad
##         lik = np.dot(err_pts, err_pts) / (sigma_pts * sigma_pts)
##         pri = ((rad - rad_guess)**2) / (sigma_r * sigma_r)
##         #pri += ((x_prior - center[0,0])**2) / (sigma_xy * sigma_xy)
##         #pri += ((y_prior - center[1,0])**2) / (sigma_xy * sigma_xy)
##         return (lik + pri)

##     params_1 = [x_prior, y_prior, rad_guess]
##     r = so.fmin_bfgs(error_function, params_1, full_output=1,
##                      disp = verbose, gtol=1e-5)
##     opt_params_1,fopt_1 = r[0],r[1]

##     y_prior = y_guess + 2*rad_guess
##     params_2 = [x_prior, y_prior, rad_guess]
##     r = so.fmin_bfgs(error_function, params_2, full_output=1,
##                      disp = verbose, gtol=1e-5)
##     opt_params_2,fopt_2 = r[0],r[1]

##     if fopt_2<fopt_1:
##         return opt_params_2[2],opt_params_2[0],opt_params_2[1]
##     else:
##         return opt_params_1[2],opt_params_1[0],opt_params_1[1]



## find the x and y coord of the center of the circle and the radius that
# best matches the data.
# @param rad_guess - guess for the radius of the circle
# @param x_guess - guess for x coord of center
# @param y_guess - guess for y coord of center.
# @param pts - 2xN np matrix of points.
# @param method - optimization method. ('fmin' or 'fmin_bfgs')
# @param verbose - passed onto the scipy optimize functions. whether to print out the convergence info.
# @return r,x,y  (radius, x and y coord of the center of the circle)
def fit_circle(rad_guess, x_guess, y_guess, pts, method, verbose=True,
               rad_fix = False):
    def error_function(params):
        center = np.matrix((params[0],params[1])).T
        if rad_fix:
            rad = rad_guess
        else:
            rad = params[2]

        err = ut.norm(pts-center).A1 - rad
        res = np.dot(err,err)
        #if not rad_fix and rad < 0.3:
        #    res = res*(0.3-rad)*100
        return res

    params_1 = [x_guess,y_guess]
    if not rad_fix:
        params_1.append(rad_guess)
    if method == 'fmin':
        r = so.fmin(error_function,params_1,xtol=0.0002,ftol=0.000001,full_output=1,disp=verbose)
        opt_params_1,fopt_1 = r[0],r[1]
    elif method == 'fmin_bfgs':
        r = so.fmin_bfgs(error_function, params_1, full_output=1,
                         disp = verbose, gtol=1e-5)
        opt_params_1,fopt_1 = r[0],r[1]
    else:
        raise RuntimeError('unknown method: '+method)

    params_2 = [x_guess,y_guess+2*rad_guess]
    if not rad_fix:
        params_2.append(rad_guess)
    if method == 'fmin':
        r = so.fmin(error_function,params_2,xtol=0.0002,ftol=0.000001,full_output=1,disp=verbose)
        opt_params_2,fopt_2 = r[0],r[1]
    elif method == 'fmin_bfgs':
        r = so.fmin_bfgs(error_function, params_2, full_output=1,
                         disp = verbose, gtol=1e-5)
        opt_params_2,fopt_2 = r[0],r[1]
    else:
        raise RuntimeError('unknown method: '+method)

    if fopt_2<fopt_1:
        if rad_fix:
            return rad_guess,opt_params_2[0],opt_params_2[1]
        else:
            return opt_params_2[2],opt_params_2[0],opt_params_2[1]
    else:
        if rad_fix:
            return rad_guess,opt_params_1[0],opt_params_1[1]
        else:
            return opt_params_1[2],opt_params_1[0],opt_params_1[1]


## ## changes the cartesian trajectory to put everything in the same frame.
## # NOTE - velocity transformation does not work if the segway is also
## # moving. This is because I am not logging the velocity of the segway.
## # @param pts - CartesianTajectory
## # @param st - object of type PlanarTrajectory (segway trajectory)
## # @return CartesianTajectory
## def account_segway_motion(cart_traj, force_traj, st):
##     ct = CartesianTajectory()
##     ft = ForceTrajectory()
##     for i in range(len(cart_traj.p_list)):
##         x,y,a = st.x_list[i], st.y_list[i], st.a_list[i]
##         p_tl = np.matrix(cart_traj.p_list[i]).T
##         p_ts = smc.tsTtl(p_tl, x, y, a)
##         p = p_ts
##         ct.p_list.append(p.A1.tolist())

##         # transform forces to the world frame.
##         f_tl = np.matrix(force_traj.f_list[i]).T
##         f_ts = smc.tsRtl(f_tl, a)
##         f = f_ts
##         ft.f_list.append(f.A1.tolist())

##         # this is incorrect. I also need to use the velocity of the
##         # segway. Unclear whether this is useful right now, so not
##         # implementing it. (Advait. Jan 6, 2010.)
##         if cart_traj.v_list != []:
##             v_tl = np.matrix(cart_traj.v_list[i]).T
##             v_ts = smc.tsRtl(v_tl, a)
##             ct.v_list.append(v_ts.A1.tolist())

##     ct.time_list = copy.copy(cart_traj.time_list)
##     ft.time_list = copy.copy(force_traj.time_list)
##     return ct, ft

## # @param cart_traj - CartesianTajectory
## # @param z_l - list of zenither heights
## # @return CartesianTajectory
## def account_zenithering(cart_traj, z_l):
##     ct = CartesianTajectory()
##     h_start = z_l[0]
##     for i in range(len(cart_traj.p_list)):
##         h = z_l[i]
##         p = cart_traj.p_list[i]
##         p[2] += h - h_start
##         ct.p_list.append(p)

##         # this is incorrect. I also need to use the velocity of the
##         # zenither. Unclear whether this is useful right now, so not
##         # implementing it. (Advait. Jan 6, 2010.)
##         if cart_traj.v_list != []:
##             ct.v_list.append(cart_traj.v_list[i])

##     ct.time_list = copy.copy(cart_traj.time_list)
##     return ct

## ##
## # remove the initial part of the trjectory in which the hook is not moving.
## # @param ct - cartesian trajectory of the end effector in the world frame.
## # @return 2xN np matrix, reject_idx
## def filter_cartesian_trajectory(ct):
##     pts_list = ct.p_list
##     ee_start_pos = pts_list[0]
##     l = [pts_list[0]]

##     for i, p in enumerate(pts_list[1:]):
##         l.append(p)
##         pts_2d = (np.matrix(l).T)[0:2,:]
##         st_pt = pts_2d[:,0]
##         end_pt = pts_2d[:,-1]
##         dist_moved = np.linalg.norm(st_pt-end_pt)
##         #if dist_moved < 0.1:
##         if dist_moved < 0.03:
##             reject_idx = i

##     pts_2d = pts_2d[:,reject_idx:]
##     return pts_2d, reject_idx

## ##
## # remove the  last part of the trjectory in which the hook might have slipped off
## # @param ct - cartesian trajectory of the end effector in the world frame.
## # @param ft - force trajectory
## # @return cartesian trajectory with the zero force end part removed, force trajectory
## def filter_trajectory_force(ct, ft):
##     vel_list = copy.copy(ct.v_list)
##     pts_list = copy.copy(ct.p_list)
##     time_list = copy.copy(ct.time_list)
##     ft_list = copy.copy(ft.f_list)
##     f_mag_list = ut.norm(np.matrix(ft.f_list).T).A1.tolist()

##     if len(pts_list) != len(f_mag_list):
##         print 'arm_trajectories.filter_trajectory_force: force and end effector lists are not of the same length.'
##         print 'Exiting ...'
##         sys.exit()

##     n_pts = len(pts_list)
##     i = n_pts - 1
##     hook_slip_off_threshold = 1.5 # from compliant_trajectories.py
##     while i > 0:
##         if f_mag_list[i] < hook_slip_off_threshold:
##             pts_list.pop()
##             time_list.pop()
##             ft_list.pop()
##             if vel_list != []:
##                 vel_list.pop()
##         else:
##             break
##         i -= 1

##     ct2 = CartesianTajectory()
##     ct2.time_list = time_list
##     ct2.p_list = pts_list
##     ct2.v_list = vel_list

##     ft2 = ForceTrajectory()
##     ft2.time_list = copy.copy(time_list)
##     ft2.f_list = ft_list
##     return ct2, ft2


## if __name__ == '__main__':
##     p = optparse.OptionParser()
##     p.add_option('-f', action='store', type='string', dest='fname',
##                  help='pkl file to use.', default='')
##     p.add_option('--xy', action='store_true', dest='xy',
##                  help='plot the x and y coordinates of the end effector.')
##     p.add_option('--yz', action='store_true', dest='yz',
##                  help='plot the y and z coordinates of the end effector.')
##     p.add_option('--xz', action='store_true', dest='xz',
##                  help='plot the x and z coordinates of the end effector.')
##     p.add_option('--plot_ellipses', action='store_true', dest='plot_ellipses',
##                  help='plot the stiffness ellipse in the x-y plane')
##     p.add_option('--pfc', action='store_true', dest='pfc',
##                  help='plot the radial and tangential components of the force.')
##     p.add_option('--pff', action='store_true', dest='pff',
##                  help='plot the force field corresponding to a stiffness ellipse.')
##     p.add_option('--pev', action='store_true', dest='pev',
##                  help='plot the stiffness ellipses for different combinations of the rel stiffnesses.')
##     p.add_option('--plot_forces', action='store_true', dest='plot_forces',
##                  help='plot the force in the x-y plane')
##     p.add_option('--plot_forces_error', action='store_true', dest='plot_forces_error',
##                  help='plot the error between the computed and measured (ATI) forces in the x-y plane')
##     p.add_option('--xyz', action='store_true', dest='xyz',
##                  help='plot in 3d the coordinates of the end effector.')
##     p.add_option('-r', action='store', type='float', dest='rad',
##                  help='radius of the joint.', default=None)
##     p.add_option('--noshow', action='store_true', dest='noshow',
##                  help='do not display the figure (use while saving figures to disk)')
##     p.add_option('--exptplot', action='store_true', dest='exptplot',
##                  help='put all the graphs of an experiment as subplots.')
##     p.add_option('--sturm', action='store_true', dest='sturm',
##                  help='make log files to send to sturm')
##     p.add_option('--icra_presentation_plot', action='store_true',
##                  dest='icra_presentation_plot',
##                  help='plot explaining CEP update.')

##     opt, args = p.parse_args()
##     fname = opt.fname
##     xy_flag = opt.xy
##     yz_flag = opt.yz
##     xz_flag = opt.xz
##     plot_forces_flag = opt.plot_forces
##     plot_ellipses_flag = opt.plot_ellipses
##     plot_forces_error_flag = opt.plot_forces_error
##     plot_force_components_flag = opt.pfc
##     plot_force_field_flag = opt.pff
##     xyz_flag = opt.xyz
##     rad = opt.rad
##     show_fig = not(opt.noshow)
##     plot_ellipses_vary_flag = opt.pev
##     expt_plot = opt.exptplot
##     sturm_output = opt.sturm


##     if plot_ellipses_vary_flag:
##         show_fig=False
##         i = 0
##         ratio_list1 = [0.1,0.3,0.5,0.7,0.9] # coarse search
##         ratio_list2 = [0.1,0.3,0.5,0.7,0.9] # coarse search
##         ratio_list3 = [0.1,0.3,0.5,0.7,0.9] # coarse search
## #        ratio_list1 = [0.7,0.8,0.9,1.0]
## #        ratio_list2 = [0.7,0.8,0.9,1.0]
## #        ratio_list3 = [0.3,0.4,0.5,0.6,0.7]
## #        ratio_list1 = [1.0,2.,3.0]
## #        ratio_list2 = [1.,2.,3.]
## #        ratio_list3 = [0.3,0.4,0.5,0.6,0.7]

##         inv_mean_list,std_list = [],[]
##         x_l,y_l,z_l = [],[],[]
##         s0 = 0.2
##         #s0 = 0.4
##         for s1 in ratio_list1:
##             for s2 in ratio_list2:
##                 for s3 in ratio_list3:
##                     i += 1
##                     s_list = [s0,s1,s2,s3,0.8]
##                     #s_list = [s1,s2,s3,s0,0.8]
##                     print '################## s_list:', s_list
##                     m,s = plot_stiff_ellipse_map(s_list,i)
##                     inv_mean_list.append(1./m)
##                     std_list.append(s)
##                     x_l.append(s1)
##                     y_l.append(s2)
##                     z_l.append(s3)

##         ut.save_pickle({'x_l':x_l,'y_l':y_l,'z_l':z_l,'inv_mean_list':inv_mean_list,'std_list':std_list},
##                        'stiff_dict_'+ut.formatted_time()+'.pkl')
##         d3m.plot_points(np.matrix([x_l,y_l,z_l]),scalar_list=inv_mean_list,mode='sphere')
##         mlab.axes()
##         d3m.show()

##         sys.exit()

##     if fname=='':
##         print 'please specify a pkl file (-f option)'
##         print 'Exiting...'
##         sys.exit()

##     d = ut.load_pickle(fname)
##     actual_cartesian_tl = joint_to_cartesian(d['actual'], d['arm'])
##     actual_cartesian = actual_cartesian_tl
##     #actual_cartesian, _ = account_segway_motion(actual_cartesian_tl,d['force'], d['segway'])
##     if d.has_key('zenither_list'):
##         actual_cartesian = account_zenithering(actual_cartesian,
##                                                d['zenither_list'])

##     eq_cartesian_tl = joint_to_cartesian(d['eq_pt'], d['arm'])
##     eq_cartesian = eq_cartesian_tl
##     #eq_cartesian, _ = account_segway_motion(eq_cartesian_tl, d['force'], d['segway'])
##     if d.has_key('zenither_list'):
##         eq_cartesian = account_zenithering(eq_cartesian, d['zenither_list'])

##     cartesian_force_clean, _ = filter_trajectory_force(actual_cartesian,
##                                                        d['force'])
##     pts_2d, reject_idx = filter_cartesian_trajectory(cartesian_force_clean)

##     if rad != None:
##         #rad = 0.39 # lab cabinet recessed.
##         #rad = 0.42 # kitchen cabinet
##         #rad = 0.80 # lab glass door
##         pts_list = actual_cartesian.p_list
##         eq_pts_list = eq_cartesian.p_list
##         ee_start_pos = pts_list[0]
##         x_guess = ee_start_pos[0]
##         y_guess = ee_start_pos[1] - rad
##         print 'before call to fit_rotary_joint'
##         force_list = d['force'].f_list

##         if sturm_output:
##             str_parts = fname.split('.')
##             sturm_file_name = str_parts[0]+'_sturm.log'
##             print 'Sturm file name:', sturm_file_name
##             sturm_file = open(sturm_file_name,'w')
##             sturm_pts = cartesian_force_clean.p_list
##             print 'len(sturm_pts):', len(sturm_pts)
##             print 'len(pts_list):', len(pts_list)

##             for i,p in enumerate(sturm_pts[1:]):
##                 sturm_file.write(" ".join(map(str,p)))
##                 sturm_file.write('\n')

##             sturm_file.write('\n')
##             sturm_file.close()

##         rad_guess = rad
##         rad, cx, cy = fit_circle(rad_guess,x_guess,y_guess,pts_2d,
##                                  method='fmin_bfgs',verbose=False)
##         print 'rad, cx, cy:', rad, cx, cy
##         c_ts = np.matrix([cx, cy, 0.]).T
##         start_angle = tr.angle_within_mod180(math.atan2(pts_2d[1,0]-cy,
##                                pts_2d[0,0]-cx) - math.pi/2)
##         end_angle = tr.angle_within_mod180(math.atan2(pts_2d[1,-1]-cy,
##                                pts_2d[0,-1]-cx) - math.pi/2)
##         mpu.plot_circle(cx, cy, rad, start_angle, end_angle,
##                         label='Actual\_opt', color='r')


##     if opt.icra_presentation_plot:
##         mpu.set_figure_size(30,20)
##         rad = 1.0
##         x_guess = pts_2d[0,0]
##         y_guess = pts_2d[1,0] - rad

##         rad_guess = rad
##         rad, cx, cy = fit_circle(rad_guess,x_guess,y_guess,pts_2d,
##                                  method='fmin_bfgs',verbose=False)
##         print 'Estimated rad, cx, cy:', rad, cx, cy

##         start_angle = tr.angle_within_mod180(math.atan2(pts_2d[1,0]-cy,
##                                pts_2d[0,0]-cx) - math.pi/2)
##         end_angle = tr.angle_within_mod180(math.atan2(pts_2d[1,-1]-cy,
##                                pts_2d[0,-1]-cx) - math.pi/2)

##         subsample_ratio = 1
##         pts_2d_s = pts_2d[:,::subsample_ratio]

##         cep_force_clean, force_new = filter_trajectory_force(eq_cartesian,
##                                                              d['force'])
##         cep_2d = np.matrix(cep_force_clean.p_list).T[0:2,reject_idx:]

##         # first draw the entire CEP and end effector trajectories
##         mpu.figure()
##         mpu.plot_yx(pts_2d_s[1,:].A1, pts_2d_s[0,:].A1, color='b',
##                     label = '\huge{End Effector Trajectory}', axis = 'equal', alpha = 1.0,
##                     scatter_size=7, linewidth=0, marker='x',
##                     marker_edge_width = 1.5)

##         cep_2d_s = cep_2d[:,::subsample_ratio]
##         mpu.plot_yx(cep_2d_s[1,:].A1, cep_2d_s[0,:].A1, color='g',
##                     label = '\huge{Equilibrium Point Trajectory}', axis = 'equal', alpha = 1.0,
##                     scatter_size=10, linewidth=0, marker='+',
##                     marker_edge_width = 1.5)

##         mpu.plot_circle(cx, cy, rad, start_angle, end_angle,
##                         label='\huge{Estimated Kinematics}', color='r',
##                         alpha=0.7)
##         mpu.plot_radii(cx, cy, rad, start_angle, end_angle,
##                        interval=end_angle-start_angle, color='r',
##                        alpha=0.7)
##         mpu.legend()
##         fig_name = 'epc'
##         fig_number = 1
##         mpu.savefig(fig_name+'%d'%fig_number)
##         fig_number += 1

##         # now zoom in to a small region to show the force
##         # decomposition.
##         zoom_location = 10
##         pts_2d_zoom = pts_2d[:,:zoom_location]
##         cep_2d_zoom = cep_2d[:,:zoom_location]

## #        image_name = 'anim'
## #        for i in range(zoom_location):
## #            mpu.figure()
## #            mpu.plot_yx(pts_2d_zoom[1,:].A1, pts_2d_zoom[0,:].A1, color='w',
## #                        axis = 'equal', alpha = 1.0)
## #            mpu.plot_yx(cep_2d_zoom[1,:].A1, cep_2d_zoom[0,:].A1, color='w',
## #                        axis = 'equal', alpha = 1.0)
## #            mpu.plot_yx(pts_2d_zoom[1,:i+1].A1, pts_2d_zoom[0,:i+1].A1, color='b',
## #                        label = '\huge{End Effector Trajectory}', axis = 'equal', alpha = 1.0,
## #                        scatter_size=7, linewidth=0, marker='x',
## #                        marker_edge_width = 1.5)
## #            mpu.plot_yx(cep_2d_zoom[1,:i+1].A1, cep_2d_zoom[0,:i+1].A1, color='g',
## #                        label = '\huge{Equilibrium Point Trajectory}', axis = 'equal', alpha = 1.0,
## #                        scatter_size=10, linewidth=0, marker='+',
## #                        marker_edge_width = 1.5)
## #            mpu.pl.xlim(0.28, 0.47)
## #            mpu.legend()
## #            mpu.savefig(image_name+'%d.png'%(i+1))

##         mpu.figure()
##         mpu.plot_yx(pts_2d_zoom[1,:].A1, pts_2d_zoom[0,:].A1, color='b',
##                     label = '\huge{End Effector Trajectory}', axis = 'equal', alpha = 1.0,
##                     scatter_size=7, linewidth=0, marker='x',
##                     marker_edge_width = 1.5)
##         mpu.plot_yx(cep_2d_zoom[1,:].A1, cep_2d_zoom[0,:].A1, color='g',
##                     label = '\huge{Equilibrium Point Trajectory}', axis = 'equal', alpha = 1.0,
##                     scatter_size=10, linewidth=0, marker='+',
##                     marker_edge_width = 1.5)
##         mpu.pl.xlim(0.28, 0.47)
##         mpu.legend()
##         #mpu.savefig('two.png')
##         mpu.savefig(fig_name+'%d'%fig_number)
##         fig_number += 1

##         rad, cx, cy = fit_circle(1.0,x_guess,y_guess,pts_2d_zoom,
##                                  method='fmin_bfgs',verbose=False)
##         print 'Estimated rad, cx, cy:', rad, cx, cy
##         start_angle = tr.angle_within_mod180(math.atan2(pts_2d[1,0]-cy,
##                                pts_2d[0,0]-cx) - math.pi/2)
##         end_angle = tr.angle_within_mod180(math.atan2(pts_2d_zoom[1,-1]-cy,
##                                pts_2d_zoom[0,-1]-cx) - math.pi/2)
##         mpu.plot_circle(cx, cy, rad, start_angle, end_angle,
##                         label='\huge{Estimated Kinematics}', color='r',
##                         alpha=0.7)
##         mpu.pl.xlim(0.28, 0.47)
##         mpu.legend()
##         #mpu.savefig('three.png')
##         mpu.savefig(fig_name+'%d'%fig_number)
##         fig_number += 1

##         current_pos = pts_2d_zoom[:,-1]
##         radial_vec = current_pos - np.matrix([cx,cy]).T
##         radial_vec = radial_vec / np.linalg.norm(radial_vec)
##         tangential_vec = np.matrix([[0,-1],[1,0]]) * radial_vec
##         mpu.plot_quiver_yxv([pts_2d_zoom[1,-1]],
##                             [pts_2d_zoom[0,-1]],
##                             radial_vec, scale=10., width = 0.002)
##         rad_text_loc = pts_2d_zoom[:,-1] + np.matrix([0.001,0.01]).T
##         mpu.pl.text(rad_text_loc[0,0], rad_text_loc[1,0], '$\hat v_{rad}$', fontsize = 30)
##         mpu.plot_quiver_yxv([pts_2d_zoom[1,-1]],
##                             [pts_2d_zoom[0,-1]],
##                             tangential_vec, scale=10., width = 0.002)

##         tan_text_loc = pts_2d_zoom[:,-1] + np.matrix([-0.012, -0.011]).T
##         mpu.pl.text(tan_text_loc[0,0], tan_text_loc[1,0], s = '$\hat v_{tan}$', fontsize = 30)
##         mpu.pl.xlim(0.28, 0.47)
##         mpu.legend()
##         #mpu.savefig('four.png')
##         mpu.savefig(fig_name+'%d'%fig_number)
##         fig_number += 1

##         wrist_color = '#A0A000'
##         wrist_force = -np.matrix(force_new.f_list[zoom_location]).T
##         frad = (wrist_force[0:2,:].T * radial_vec)[0,0] * radial_vec
##         mpu.plot_quiver_yxv([pts_2d_zoom[1,-1]],
##                             [pts_2d_zoom[0,-1]],
##                             wrist_force, scale=50., width = 0.002,
##                             color=wrist_color)
##                             #color='y')
##         wf_text = rad_text_loc + np.matrix([-0.06,0.015]).T
##         mpu.pl.text(wf_text[0,0], wf_text[1,0], color=wrist_color,
##                     fontsize = 25, s = 'Total Force')

##         mpu.plot_quiver_yxv([pts_2d_zoom[1,-1]],
##                             [pts_2d_zoom[0,-1]],
##                             frad, scale=50., width = 0.002,
##                             color=wrist_color)
##         frad_text = rad_text_loc + np.matrix([0.,0.015]).T
##         mpu.pl.text(frad_text[0,0], frad_text[1,0], color=wrist_color, s = '$\hat F_{rad}$', fontsize = 30)

##         mpu.pl.xlim(0.28, 0.47)
##         mpu.legend()
##         #mpu.savefig('five.png')
##         mpu.savefig(fig_name+'%d'%fig_number)
##         fig_number += 1

##         frad = (wrist_force[0:2,:].T * radial_vec)[0,0]
##         hook_force_motion = -(frad - 5) * radial_vec * 0.001
##         tangential_motion = 0.01 * tangential_vec
##         total_cep_motion = hook_force_motion + tangential_motion

##         mpu.plot_quiver_yxv([cep_2d_zoom[1,-1]],
##                             [cep_2d_zoom[0,-1]],
##                             hook_force_motion, scale=0.1, width = 0.002)
##         hw_text = cep_2d_zoom[:,-1] + np.matrix([-0.002,-0.012]).T
##         mpu.pl.text(hw_text[0,0], hw_text[1,0], color='k', fontsize=20,
##                     s = '$h[t]$')
##         mpu.pl.xlim(0.28, 0.47)
##         mpu.legend()
##         #mpu.savefig('six.png')
##         mpu.savefig(fig_name+'%d'%fig_number)
##         fig_number += 1

##         mpu.plot_quiver_yxv([cep_2d_zoom[1,-1]],
##                             [cep_2d_zoom[0,-1]],
##                             tangential_motion, scale=0.1, width = 0.002)
##         mw_text = cep_2d_zoom[:,-1] + np.matrix([-0.018,0.001]).T
##         mpu.pl.text(mw_text[0,0], mw_text[1,0], color='k', fontsize=20,
##                     s = '$m[t]$')
##         mpu.pl.xlim(0.28, 0.47)
##         mpu.legend()
##         #mpu.savefig('seven.png')
##         mpu.savefig(fig_name+'%d'%fig_number)
##         fig_number += 1

##         mpu.plot_quiver_yxv([cep_2d_zoom[1,-1]],
##                             [cep_2d_zoom[0,-1]],
##                             total_cep_motion, scale=0.1, width = 0.002)
##         cep_text = cep_2d_zoom[:,-1] + np.matrix([-0.058,-0.023]).T
##         mpu.pl.text(cep_text[0,0], cep_text[1,0], color='k', fontsize=20,
##                     s = '$x_{eq}[t]$ = &x_{eq}[t-1] + m[t] + h[t]$')
##         mpu.pl.xlim(0.28, 0.47)
##         mpu.legend()
##         #mpu.savefig('eight.png')
##         mpu.savefig(fig_name+'%d'%fig_number)
##         fig_number += 1

##         new_cep = cep_2d_zoom[:,-1] + total_cep_motion
##         mpu.plot_yx(new_cep[1,:].A1, new_cep[0,:].A1, color='g',
##                     axis = 'equal', alpha = 1.0,
##                     scatter_size=10, linewidth=0, marker='+',
##                     marker_edge_width = 1.5)
##         mpu.pl.xlim(0.28, 0.47)
##         mpu.legend()
##         #mpu.savefig('nine.png')
##         mpu.savefig(fig_name+'%d'%fig_number)
##         fig_number += 1

##         mpu.show()

##     if xy_flag:
##         st_pt = pts_2d[:,0]
##         end_pt = pts_2d[:,-1]

##         if expt_plot:
##             pl.subplot(233)

##         plot_cartesian(actual_cartesian, xaxis=0, yaxis=1, color='b',
##                        label='FK', plot_velocity=False)
##         plot_cartesian(eq_cartesian, xaxis=0,yaxis=1,color='g',label='Eq Point')

##     elif yz_flag:
##         plot_cartesian(actual_cartesian,xaxis=1,yaxis=2,color='b',label='FK')
##         plot_cartesian(eq_cartesian, xaxis=1,yaxis=2,color='g',label='Eq Point')
##     elif xz_flag:
##         plot_cartesian(actual_cartesian,xaxis=0,yaxis=2,color='b',label='FK')
##         plot_cartesian(eq_cartesian, xaxis=0,yaxis=2,color='g',label='Eq Point')


##     if plot_forces_flag or plot_forces_error_flag or plot_ellipses_flag or plot_force_components_flag or plot_force_field_flag:
##         #        arm_stiffness_list = d['stiffness'].stiffness_list
##         #        scale = d['stiffness'].stiffness_scale
##         #        asl = [min(scale*s,1.0) for s in arm_stiffness_list]
##         #        ftraj_jinv,ftraj_stiff,ftraj_torque,k_cart_list=compute_forces(d['actual'],d['eq_pt'],
##         #                                                                       d['torque'],asl)
##         if plot_forces_flag:
##             plot_forces_quiver(actual_cartesian,d['force'],color='k')
##             #plot_forces_quiver(actual_cartesian,ftraj_jinv,color='y')
##             #plot_forces_quiver(actual_cartesian,ftraj_stiff,color='y')

##         if plot_ellipses_flag:
##             #plot_stiff_ellipses(k_cart_list,actual_cartesian)
##             if expt_plot:
##                 subplotnum=234
##             else:
##                 pl.figure()
##                 subplotnum=111
##             plot_stiff_ellipses(k_cart_list,eq_cartesian,subplotnum=subplotnum)

##         if plot_forces_error_flag:
##             plot_error_forces(d['force'].f_list,ftraj_jinv.f_list)
##             #plot_error_forces(d['force'].f_list,ftraj_stiff.f_list)

##         if plot_force_components_flag:
##             p_list = actual_cartesian.p_list
##             #cx = 45.
##             #cy = -0.3
##             frad_list,ftan_list,_ = compute_radial_tangential_forces(d['force'].f_list,p_list,cx,cy)
##             #frad_list,ftan_list,_ = compute_radial_tangential_forces(d['force_raw'].f_list,p_list,cx,cy)
##             if expt_plot:
##                 pl.subplot(235)
##             else:
##                 pl.figure()

##             time_list = d['force'].time_list
##             time_list = [t-time_list[0] for t in time_list]
##             x_coord_list = np.matrix(p_list)[:,0].A1.tolist()
##             mpu.plot_yx(frad_list,x_coord_list,scatter_size=50,color=time_list,cb_label='time',axis=None)
##             pl.xlabel('x coord of end effector (m)')
##             pl.ylabel('magnitude of radial force (N)')
##             pl.title(d['info'])
##             if expt_plot:
##                 pl.subplot(236)
##             else:
##                 pl.figure()
##             mpu.plot_yx(ftan_list,x_coord_list,scatter_size=50,color=time_list,cb_label='time',axis=None)
##             pl.xlabel('x coord of end effector (m)')
##             pl.ylabel('magnitude of tangential force (N)')
##             pl.title(d['info'])

##         if plot_force_field_flag:
##             plot_stiffness_field(k_cart_list[0],plottitle='start')
##             plot_stiffness_field(k_cart_list[-1],plottitle='end')


##     str_parts = fname.split('.')
##     if d.has_key('strategy'):
##         addon = ''
##         if opt.xy:
##             addon = '_xy'
##         if opt.xz:
##             addon = '_xz'
##         fig_name = str_parts[0]+'_'+d['strategy']+addon+'.png'
##     else:
##         fig_name = str_parts[0]+'_res.png'

##     if expt_plot:
##         f = pl.gcf()
##         curr_size = f.get_size_inches()
##         f.set_size_inches(curr_size[0]*2,curr_size[1]*2)
##         f.savefig(fig_name)

##     if show_fig:
##         pl.show()
##     else:
##         print '################################'
##         print 'show_fig is FALSE'
##         if not(expt_plot):
##             pl.savefig(fig_name)

##     if xyz_flag:
##         plot_cartesian(traj, xaxis=0,yaxis=1,zaxis=2)
##         mlab.show()








