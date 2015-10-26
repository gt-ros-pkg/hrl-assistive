#!/usr/local/bin/python

import sys, os
import numpy as np, math
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import rospy
import inspect
import warnings
import random

# Util
import hrl_lib.util as ut

# Matplot
import matplotlib.pyplot as plt
import matplotlib.pyplot as pp
import hrl_lib.matplotlib_util as mpu

# Sklearn
from sklearn import cross_validation

TOL=0.0001

class traj_data():
    def __init__(self):

        pth       = os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook/'
        blocked_thresh_dict = ut.load_pickle(pth+'blocked_thresh_dict.pkl') # ['mean_charlie', 'mean_known_mech']

        # Get data
        self.semantic = blocked_thresh_dict['mean_charlie'] # each category has (n_std, mn, std)  <= force profiles
        self.second_time = blocked_thresh_dict['mean_known_mech'] # (Ms(mn_mn, var_mn, mn_std, var_std), n_std)=(tuple(4),float)        

        self.force_table = None  # discrete force table        
        self.force_max   = 0.0
        self.force_resol = 0.5

        self.trans_size = None
        self.trans_mat = None        
        self.trans_mat = None
        self.trans_prob_mat = None
        self.start_prob_vec = None

        self.means = None
        self.vars  = None
        
        pass

    
    def find_nearest(self, array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx], idx

    
    def reset_trans_mat(self):

        self.trans_size = int(np.ceil(self.force_max / self.force_resol)) + 1
        self.trans_mat = np.zeros((self.trans_size, self.trans_size))
        self.trans_prob_mat = np.zeros((self.trans_size, self.trans_size))
        self.start_prob_vec = np.zeros((self.trans_size,1)) + 1.0/float(self.trans_size)
        
    def update_trans_mat_all(self, test_dict):
        for key in test_dict.keys():
            self.update_trans_mat(test_dict[key])
        
        self.set_trans_mat()
        
        
    def update_trans_mat(self, profile):

        for i in xrange(len(profile)):
            # skip first element
            if i==0: continue

            _, x_idx = self.find_nearest(self.force_table, profile[i-1])
            _, y_idx = self.find_nearest(self.force_table, profile[i])

            self.trans_mat[x_idx,y_idx] += 1.0
            ## print x_idx, y_idx, profile[i-1], profile[i]
                        
    def set_trans_mat(self):

        for j in xrange(self.trans_size):
            total = np.sum(self.trans_mat[:,j])
            if total == 0: 
                self.trans_prob_mat[:,j] = 1.0 / float(self.trans_size)
            else:
                self.trans_prob_mat[:,j] = self.trans_mat[:,j] / total

    
    def discrete_profile(self, plot=False):
        
        # Get max force
        for key in self.semantic.keys():
            if self.force_max < max(self.semantic[key][1]):
                self.force_max = max(self.semantic[key][1])

        self.force_max = np.ceil(self.force_max)

        # Discrete Force list
        self.force_table = np.arange(0.0, self.force_max+0.000001, self.force_resol)
        
        # Discretize it
        test_dict = {}
        for key in self.semantic.keys():

            # Discrete Force profile
            force_profile = np.zeros(self.semantic[key][1].shape)
            for i,f in enumerate(self.semantic[key][1]):
                force_profile[i], _ = self.find_nearest(self.force_table,f)

            test_dict[key] = force_profile

        # Reset transition probability matrix wrt force_max
        self.reset_trans_mat()

            
        # plot
        if plot:
            mpu.set_figure_size(10, 7.0)
            f = pp.figure()
            f.set_facecolor('w')

            for key in self.semantic.keys():
                pp.plot(self.semantic[key][1], 'r-')
                pp.plot(test_dict[key],'b-')
                
            
            pp.ylim(-3., 16.)
            pp.xlim(0., 34.)
            f.subplots_adjust(bottom=.15, top=.99, right=.99)
            ## mpu.legend(loc='lower left')
            ## pp.savefig('collision_detection_hsi_kitchen_pr2.pdf')
            pp.show()

        return test_dict

        
    ## def feature_to_mu_sigma(fvec, nState): 

    ##     index = 0
    ##     m,n = np.shape(fvec)
    ##     #print m,n
    ##     mu = np.matrix(np.zeros((nState,1)))
    ##     sigma = np.matrix(np.zeros((nState,1)))
    ##     DIVS = m/float(nState)

    ##     while (index < nState):
    ##         m_init = index*DIVS
    ##         temp_fvec = fvec[(m_init):(m_init+DIVS),0:]
    ##         #if index == 1:
    ##             #print temp_fvec
    ##         mu[index] = scp.mean(temp_fvec)
    ##         sigma[index] = scp.std(temp_fvec)
    ##         index = index+1

    ##     return mu,sigma



        
    
    def test(self):

        print self.semantic.keys()
        print self.semantic['kitchen_cabinet_pr2'][1]
        print max(self.semantic['kitchen_cabinet_pr2'][1])

        # With respect to all semantic data, convert those into discrete action-observation
        
        return


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()
    
    td        = traj_data()
    test_dict = td.discrete_profile(False)

    td.update_trans_mat_all(test_dict)


    ## print td.trans_mat
    ## print td.trans_prob_mat      
    
    ## kf = cross_validation.KFold(len(test_dict.keys()), n_folds = 2, shuffle=True)
    ## for train_index, test_index in kf:
    ##     print train_index, test_index

    
    ## td.sampling_histories()

    ## data_path = os.getcwd()    
    ## td.test()

    

    
