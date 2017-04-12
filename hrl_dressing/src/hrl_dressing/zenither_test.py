#!/usr/bin/python

import roslib
import rospkg
roslib.load_manifest('zenither')
# from zenither import Zenither
import zenither.zenither as zenither
# from zenither import Zenither
import sys,time,os,optparse
from collections import deque
import rospy

import math, numpy as np
import hrl_lib.util as ut
from hrl_lib.util import save_pickle, load_pickle

def move_position_fast(z,pos,vel,acc):
    acceleration = z.limit_acceleration(acc)
    velocity = z.limit_velocity(vel)
    z.set_pos_absolute(pos)
    z.set_velocity(velocity)
    z.set_acceleration(acceleration)
    z.go()


def calibrate(z):
    z.nadir()
    z.calibrated = True # this is a hack!
    pos = z.get_position_meters()
    while True:
        time.sleep(0.5)
        new_pos = z.get_position_meters()
        if abs(new_pos-pos)<0.005:
            time.sleep(0.5)
            break
        pos = new_pos

    print 'Hit the end stop.'
    print 'Setting the origin.'
    z.estop()
    z.engage_brake()
    z.set_origin()
    print '__________________________'
    print 'Calibration Over'
    pos = z.get_position_meters()
    print 'Current position is: ', pos
    # st = 0.9
    # z.move_position(st)
    # rospy.sleep(5.0)
    # print 'Current position is: ', pos
    # print 'Movement Over'
    #go_pos = 0.1
    #print 'Going to position: ', go_pos
    #print '__________________________'
    #z.move_position(go_pos)



def time_to_reach(dist,vel,acc):

    # v = u + at (u=0)
    # s = ut + (at^2)/2
    t0 = vel/acc
    d0 = acc*t0*t0/2.
    d_remain = dist - 2*d0
    t_maxvel = d_remain/vel
    return t0+t_maxvel+t0


# returns two deques. position and time
def log_until_stopped(z,pos,tolerance=0.005):
    t0 = time.time()
    dt = deque()
    dx = deque()
    new_pos = z.get_position_meters()
    compare_pos = new_pos
    not_moved_count = 0

    while True:
        new_pos = z.get_position_meters()
        dx.append(new_pos)
        dt.append(time.time()-t0)
        if abs(new_pos-pos) < tolerance:
            break

    return dx,dt


# Trying to confirm that the units of acceln and velocity are indeed
# SI units. Sinusoidal trajectory is the next step
def test_linear(z):
    st = 0.13
    z.move_position(st)
    dist = -0.13
    vel = 0.4
    acc = 0.5
    pos = st+dist
    t0 = time.time()
    move_position_fast(z,pos,vel,acc)
    deck_x,deck_t = log_until_stopped(z,pos)
    t1 = time.time()
    print 'measured time:', t1-t0
    print 'calculated time:', time_to_reach(dist,vel,acc)
    z.engage_brake()

    #mpu.plot_yx(deck_x,deck_t,scatter_size=0,linewidth=1,axis=None,
    #            ylabel='position (meters)',xlabel='time (s)')
    #mpu.show()

def move_to_place(z, x):
    pos = x
    vel = 0.1
    acc = 0.1
    #pos = pos
    t0 = time.time()
    move_position_fast(z,pos,vel,acc)

# Trying to confirm that the units of acceln and velocity are indeed
# SI units. Sinusoidal trajectory is the next step
def sleeve(z):
    import rospy
    #rospy.init_node('zenither_node')
    #st = 0
    #z.move_position(st)
    print 'Moved to 0'
    pos = 0.9
    vel = 0.1
    acc = 0.1
    #pos = pos
    t0 = time.time()
    move_position_fast(z,pos,vel,acc)
    print 'Moved to 0.9'
    rospy.sleep(8.5)
    z.estop()
    print 'estop'
    # dist = 0.1
    #vel = 0.4
    #acc = 0.4
    #pos = 0.1
    #t0 = time.time()
    #move_position_fast(z,pos,vel,acc)
    #print 'Moved to 0.1'
    #rospy.sleep(1.5)
    z.estop()
    print 'estop'
    print z.get_position_meters()

    # for i in xrange(4):
    #     st = 0.8
    #     z.move_position(st)
    #     dist = -0.75
    #     vel = 0.1
    #     acc = 0.1
    #     pos = st+dist
    #     t0 = time.time()
    #     move_position_fast(z,pos,vel,acc)
    #     deck_x,deck_t = log_until_stopped(z,pos)
    #     t1 = time.time()
    #     print 'measured time:', t1-t0
    #     print 'calculated time:', time_to_reach(dist,vel,acc)
    print 'Movement done!'

    #mpu.plot_yx(deck_x,deck_t,scatter_size=0,linewidth=1,axis=None,
    #            ylabel='position (meters)',xlabel='time (s)')
    #mpu.show()

def test_sinusoid(z,A0,A,freq):
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('efri')
    # res_dict = np.array([1,1,1])
    # save_pickle(res_dict,'/home/ari/sinusoid.pkl')
    # save_pickle(res_dict, ''.join([pkg_path, 'sinusoid_data.pkl']))

    w = 2*math.pi*freq

    period = 1./freq
    t_end = 10*period
    if t_end>10:
        t_end = 5*period

    z.move_position(A0+A,0.1,0.05,blocking=True)
    time.sleep(2.0)
    #z.move_position(A0)

    z.use_velocity_mode()

    deck_t = deque()
    deck_x = deque()

    t0 = rospy.Time.now()
    # t0 = time.time()
    while True:
        tnow = rospy.Time.now()
        # tnow = time.time()
        t_diff = rospy.Time.now().to_sec()-t0.to_sec()
        v = -w*A*math.sin(w*t_diff)
        a = -w*w*A*np.cos(w*t_diff)

        z.set_velocity(v)
        z.set_acceleration(a)
        z.go()

        time.sleep(0.025)
        new_pos = z.get_position_meters()
        deck_x.append(new_pos)
        t_diff = rospy.Time.now().to_sec()-t0.to_sec()
        deck_t.append(t_diff)

        if t_diff >= t_end:
            break

    z.estop()
    time.sleep(0.1)
    z.use_position_mode()
    time.sleep(0.1)
    z.move_position(A0,0.1,0.05,blocking=True)

    res_dict = {}
    res_dict['x'] = deck_x
    res_dict['t'] = deck_t
    res_dict['A0'] = A0
    res_dict['A'] = A
    res_dict['freq'] = freq
    res_dict['t_end'] = t_end

    ut.save_pickle(res_dict, '/home/ari/sinusoid.pkl')

if __name__ == '__main__':

    #import matplotlib_util.util as mpu
    rospy.init_node('zenither_test')
    p = optparse.OptionParser()
    p.add_option('--test_linear', action='store_true', dest='test_lin',
                 help='constant acceln and max vel for zenither.')
    p.add_option('--calib', action='store_true', dest='calib',
                 help='calibrate the zenither')
    p.add_option('--test_sine', action='store_true', dest='test_sine',
                 help='acceln and vel according to a sinusoid.')
    p.add_option('--sleeve', action='store_true', dest='sleeve',
                 help='Move actuator to pull sleeve on arm.')
    p.add_option('--move', action='store_true', dest='move',
                 help='Move actuator to a fixed location.')
    p.add_option('--sine_expt', action='store_true', dest='sine_expt',
                 help='run the sinusoid experiment.')
    p.add_option('--cmp_sine_pkl', action='store', dest='cmp_sine_pkl',
                 type='string', default='',
                 help='pkl saved by test_sine.')

    opt, args = p.parse_args()

    # z = zenither.Zenither(robot='test_rig')
    z = zenither.Zenither(robot='henrybot')

    if opt.calib:
        calibrate(z)

    if not z.calibrated:
        calibrate(z)

    if opt.move:
        # move_to_place(z, 0.43)
        z.disengage_brake()
        rospy.sleep(1)
        # rospy.sleep(5)
        move_to_place(z, 0.00)
        rospy.sleep(1)
        z.engage_brake()

    if opt.test_lin:
        test_linear(z)

    if opt.test_sine:
        A0 = 0.5
        A = 0.05 #0.02
        freq = 0.1 #0.50
        test_sinusoid(z,A0,A,freq)

    if opt.sleeve:
        sleeve(z)

    if opt.sine_expt:
        A0 = 0.5
        A = 0.01
        freq = 0.25
        z.move_position(A0,0.1,0.05)

        import m3.toolbox as m3t
        print 'Connect the robot to the slider'
        print 'Hit Enter to start the sinusoid'
        k=m3t.get_keystroke()
        if k == '\r':
            test_sinusoid(z,A0,A,freq)
        else:
            print 'You did not hit enter.'

    if opt.cmp_sine_pkl != '':
        import trajectories as traj
        d = ut.load_pickle(opt.cmp_sine_pkl)

        deck_x = d['x']
        deck_t = d['t']
        A0 = d['A0']
        A = d['A']
        freq = d['freq']
        t_end = d['t_end']

        n_cycles = int(round(t_end*freq))
        x,xdot,xddot,time_arr = traj.generate_sinusoidal_trajectory(A0,A,
                                            freq,n_cycles,200)

        mpu.plot_yx(deck_x,deck_t,scatter_size=0,linewidth=1,axis=None,
                    ylabel='position (meters)',xlabel='time (s)',
                    label='measured')
        mpu.plot_yx(x,time_arr,scatter_size=0,label='synthetic',linewidth=1,
                    color='g',axis=None)
        mpu.legend()
        mpu.show()



