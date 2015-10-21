
import math
import numpy as np
import copy
import sys, time, os
import thread

import hrl_lib.util as ut


class MekaArmSettings():
    def __init__(self, stiffness_scale=1.0,stiffness_list=[0.7,0.7,0.8,0.8,0.3],
                 control_mode='theta_gc'):
        ''' stiffness_list: list of 5 stiffness values for joints 0-4.
            stiffness_scale: common scaling factor for all the joints
            control_mode: 'theta_gc' or 'torque_gc'
        '''
        self.stiffness_scale = stiffness_scale
        # for safety of wrist roll. Advait Jun 18, 2010.
        # changed to 0.2 from 0.3 (Advait, Sept 19, 2010)
        stiffness_list[4] = min(stiffness_list[4], 0.2)
        self.stiffness_list = stiffness_list
        self.control_mode = control_mode

class M3HrlRobot():
    def __init__(self, connect=True, right_arm_settings=None,
                 left_arm_settings=None, end_effector_length=0.12818):
        ''' connect -  connect to the robot or not.
            can do FK and IK without connecting.
            right_arm_settings, left_arm_settings: objects of class
                        MekaArmSettings
        '''
        if connect:
            self.arm_settings = {}  # dict is set in set_arm_settings
            self.initialize_joints(right_arm_settings, left_arm_settings)
            #self.initialize_gripper()
            self.left_arm_ft = {'force': np.matrix(np.zeros((3,1),dtype='float32')),
                                'torque': np.matrix(np.zeros((3,1),dtype='float32'))}
            self.right_arm_ft = {'force': np.matrix(np.zeros((3,1),dtype='float32')),
                                 'torque': np.matrix(np.zeros((3,1),dtype='float32'))}
            self.fts_bias = {'left_arm': self.left_arm_ft, 'right_arm': self.right_arm_ft}

        # create joint limit dicts
        self.joint_lim_dict = {}
        self.joint_lim_dict['right_arm'] = {'max':[math.radians(a) for a in [ 100.00, 60.,  77.5, 144., 122.,  65.,  65.]],
                                            'min':[math.radians(a) for a in [ -47.61,  0., -77.5,   0., -80., -65., -65.]]}
        self.joint_lim_dict['left_arm'] = {'max':[math.radians(a) for a in [ 100.00,   20.,  77.5, 144.,   80.,  65.,  65.]],
                                           'min':[math.radians(a) for a in [ -47.61, -122., -77.5,   0., -125., -65., -65.]]}

        self.setup_kdl_mekabot(end_effector_length)
        q_guess_pkl_l = os.environ["HRLBASEPATH"] + "/src/libraries/mekabot/q_guess_left_dict.pkl"
        q_guess_pkl_r = os.environ["HRLBASEPATH"] + "/src/libraries/mekabot/q_guess_right_dict.pkl"
        self.q_guess_dict_left = ut.load_pickle(q_guess_pkl_l)
        self.q_guess_dict_right = ut.load_pickle(q_guess_pkl_r)

        self.jep = None # see set_joint_angles

        # kalman filtering force vector. (self.step and bias_wrist_ft)
        self.Q_force, self.R_force = {}, {}
        self.xhat_force, self.P_force = {}, {}

        self.Q_force['right_arm'] = [1e-3, 1e-3, 1e-3]
        self.R_force['right_arm'] = [0.2**2, 0.2**2, 0.2**2]
        self.xhat_force['right_arm'] = [0., 0., 0.]
        self.P_force['right_arm'] = [1.0, 1.0, 1.0]

        self.Q_force['left_arm'] = [1e-3, 1e-3, 1e-3]
        self.R_force['left_arm'] = [0.2**2, 0.2**2, 0.2**2]
        self.xhat_force['left_arm'] = [0., 0., 0.]
        self.P_force['left_arm'] = [1.0, 1.0, 1.0]

