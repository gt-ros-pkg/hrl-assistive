#!/usr/bin/env python

# Hidden Markov Model Implementation for Dressing
import matplotlib
import numpy as np
import matplotlib.pyplot as pp
import scipy as scp
import roslib; roslib.load_manifest('sandbox_tapo_darpa_m3')
import rospy
import hrl_lib.util as ut
import hrl_lib.matplotlib_util as mpu
import pickle
import optparse
import unittest
import ghmm
import ghmmwrapper
import random

import sys
sys.path.insert(0, '/home/tapo/svn/robot1_data/usr/tapo/data_code/dressing/')
from data_organizer import Fmat_original

class HMM_Model:
    def __init__(self, Fmat_train, Fmat_test, categories):
        self.F = ghmm.Float() # emission domain of HMM model
        self.Fmat_train = Fmat_train
        self.Fmat_test = Fmat_test
        self.train_trials_per_category = np.size(Fmat_train,0)/np.size(categories)
        self.test_trials_per_category  = np.size(Fmat_test,0)/np.size(categories)
                                
    # Getting mean-std / mean-covariance
    def mean_cov(self, Fmat, start_Trials, end_Trials, number_states):
        j = 0
        mu_force = np.zeros((number_states,1))
        sigma = np.zeros((number_states,1))      
        while (j < number_states):
            mu_force[j] = np.random.random()
            sigma[j] = np.random.random()
            j = j+1
        #print mu_force, sigma
        #raw_input("Press Enter to continue...")
        return mu_force, sigma

    def calculate_A_B_pi(self, number_states, flag):
        # A - Transition Matrix
        if number_states == 3:
            A  = [[0.2, 0.5, 0.3],
                  [0.0, 0.5, 0.5],
                  [0.0, 0.0, 1.0]]
        elif number_states == 5:
            A  = [[0.2, 0.35, 0.2, 0.15, 0.1],
                  [0.0, 0.2, 0.45, 0.25, 0.1],
                  [0.0, 0.0, 0.2, 0.55, 0.25],
                  [0.0, 0.0, 0.0, 0.2, 0.8],
                  [0.0, 0.0, 0.0, 0.0, 1.0]]
        elif number_states == 10:
            A  = [[0.1, 0.25, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
                  [0.0, 0.1, 0.25, 0.25, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05],
                  [0.0, 0.0, 0.1, 0.25, 0.25, 0.2, 0.05, 0.05, 0.05, 0.05],
                  [0.0, 0.0, 0.0, 0.1, 0.3, 0.30, 0.20, 0.1, 0.05, 0.05],
                  [0.0, 0.0, 0.0, 0.0, 0.1, 0.30, 0.30, 0.20, 0.05, 0.05],
                  [0.0, 0.0, 0.0, 0.0, 0.00, 0.1, 0.35, 0.30, 0.20, 0.05],
                  [0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.2, 0.30, 0.30, 0.20],
                  [0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00, 0.2, 0.50, 0.30],
                  [0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00, 0.00, 0.4, 0.60],
                  [0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        
        # B - Emission Matrix, parameters of emission distributions in pairs of (mu, sigma)
    
        B = [0.0]*number_states
        
        if flag == 'Missed':
            mu_force, sigma = self.mean_cov(self.Fmat_train, 0, self.train_trials_per_category, number_states)
        elif flag == 'Good':
            mu_force, sigma = self.mean_cov(self.Fmat_train, self.train_trials_per_category, self.train_trials_per_category*2, number_states)
        elif flag == 'High':
            mu_force, sigma = self.mean_cov(self.Fmat_train, self.train_trials_per_category*2, self.train_trials_per_category*3, number_states)
        elif flag == 'Caught':
            mu_force, sigma = self.mean_cov(self.Fmat_train, self.train_trials_per_category*3, self.train_trials_per_category*4, number_states)
                  
        for num_states in range(number_states):
            B[num_states] = [mu_force[num_states][0],sigma[num_states][0]]
                
        # pi - initial probabilities per state
        
        if number_states == 3:
            pi = [1./3.] * 3
        elif number_states == 5:
            pi = [0.2] * 5
        elif number_states == 10:
            pi = [0.1] * 10
        
        #print B
        #raw_input("Press Enter to continue...")   
        return A, B, pi

    def create_model(self, flag, number_states):
          
        A, B, pi = self.calculate_A_B_pi(number_states, flag)

        # generate models from parameters
        model = ghmm.HMMFromMatrices(self.F,ghmm.GaussianDistribution(self.F), A, B, pi)
        #model = ghmm.HMMFromMatrices(F,ghmm.MultivariateGaussianDistribution(F), A, B, pi)
        return model

    def train(self, model, flag):
        index = np.size(self.Fmat_train,0)/np.size(categories)
        if flag == 'Missed':
            final_ts = ghmm.SequenceSet(self.F,self.Fmat_train[0:index])
        elif flag == 'Good':
            final_ts = ghmm.SequenceSet(self.F,self.Fmat_train[index:2*index])
        elif flag == 'High':
            final_ts = ghmm.SequenceSet(self.F,self.Fmat_train[2*index:3*index])
        elif flag == 'Caught':
            final_ts = ghmm.SequenceSet(self.F,self.Fmat_train[3*index:4*index])
                
        model.baumWelch(final_ts)
        return model

    def test(self, model, ts_obj):

        # Find Viterbi Path
        final_ts_obj = ghmm.EmissionSequence(self.F,ts_obj.tolist())
        path_obj = model.viterbi(final_ts_obj)
        
        return path_obj
 
def create_datasets(mat, trials, fold_num):
    if fold_num == 1:    	
    	start_train = 0
    	end_train = 3
    	start_test = end_train
    	end_test = trials
        
    elif fold_num == 2:
        start_train = 3
    	end_train = trials
    	start_test = 0
    	end_test = start_train
        
    train_missed = mat[trials*0+start_train:trials*0+end_train]
    test_missed  = mat[trials*0+start_test:trials*0+end_test]
        
    train_good = mat[trials*1+start_train:trials*1+end_train]
    test_good  = mat[trials*1+start_test:trials*1+end_test]
        
    train_high = mat[trials*2+start_train:trials*2+end_train]
    test_high  = mat[trials*2+start_test:trials*2+end_test]
        
    train_caught = mat[trials*3+start_train:trials*3+end_train]
    test_caught  = mat[trials*3+start_test:trials*3+end_test]
        
    return sum([train_missed, train_good, train_high, train_caught],[]), sum([test_missed, test_good, test_high, test_caught],[])

def run_crossvalidation(Fmat, categories, n_folds):
    confusion_mat = [[0] * np.size(categories) for i in range(np.size(categories))]
    for fold in range(1, n_folds+1):
        training_set, testing_set = create_datasets(Fmat, np.size(Fmat,0)/np.size(categories), fold)
        hMM = HMM_Model(training_set, testing_set, categories)     
        states = 10
        path = np.array([[0] * np.size(categories) for i in range(np.size(testing_set,0))])
        path_max = []
        model = []
        model_trained = []
        for i in range(np.size(categories)):
            model.append(hMM.create_model(categories[i], states))
            model_trained.append(hMM.train(model[i], categories[i]))
        for i in range(np.size(testing_set,0)):
            for j in range(np.size(categories)):
                value = hMM.test(model_trained[j], testing_set[i])[1]
                path[i][j] = value if value != -float('Inf') else -50000
        for i in range(np.size(testing_set,0)):
            path_max.append(max(path[i]))
            #print max(path[i])
            for j in range(np.size(categories)):
                if path_max[i] == path[i][j]:
                    confusion_mat[j][i/(np.size(testing_set,0)/np.size(categories))] = confusion_mat[j][i/(np.size(testing_set,0)/np.size(categories))] + 1
        #print path_max
    cmat = np.matrix(confusion_mat)
    return cmat

def show_confusion(cmat):
    total = float(cmat.sum())
    true = float(np.trace(cmat))
    percent_acc = "{0:.2f}".format((true/total)*100)

    # Plot Confusion Matrix
    Nlabels = np.size(categories)
    fig = pp.figure()
    ax = fig.add_subplot(111)
    figplot = ax.matshow(cmat, interpolation = 'nearest', origin = 'upper', extent=[0, Nlabels, 0, Nlabels])
    ax.set_title('Performance of HMM Models : Accuracy = ' + str(percent_acc))
    pp.xlabel("Targets")
    pp.ylabel("Predictions")
    ax.set_xticks([0.5,1.5,2.5,3.5])
    ax.set_xticklabels(['Missed', 'Good', 'High', 'Caught'])
    ax.set_yticks([3.5,2.5,1.5,0.5])
    ax.set_yticklabels(['Missed', 'Good', 'High', 'Caught'])
    figbar = fig.colorbar(figplot)

    i = 0    
    while (i < 4):
        j = 0
        while (j < 4):
            pp.text(j+0.5,3.5-i,cmat[i,j])
            j = j+1
        i = i+1

if __name__ == '__main__':  
    
    input_Fmat = Fmat_original
    
    categories = ['Missed', 'Good', 'High', 'Caught']
    n_folds = 2
   
    result_mat = run_crossvalidation(input_Fmat, categories, n_folds)
    show_confusion(result_mat)
    pp.show()
        

    
    
   

    
    
