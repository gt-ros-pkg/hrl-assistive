#!/usr/bin/env python

# Hidden Markov Model Implementation for Dressing
import matplotlib
import numpy as np
import matplotlib.pyplot as pp

import ghmmwrapper
import ghmm
import random

import math

import sys
from data_organizer import Fmat_original, exp_list, Subject_armlength
from hmm_a_matrix import getA

class HMM_Model:
    def __init__(self, Fmat_train, Fmat_test, categories, train_per_category, test_per_category):
        self.F = ghmm.Float() # emission domain of HMM model
        self.Fmat_train = Fmat_train
        self.Fmat_test = Fmat_test
        self.categories = categories
        self.train_trials_per_category = train_per_category
        self.test_trials_per_category  = test_per_category
                                
    # Getting mean-std / mean-covariance
    def mean_cov(self, Fmat, start_Trials, end_Trials, number_states):
        j = 0

        mu_force = [0.0] * number_states    # mean value
        sigma = [0.0] * number_states       # covariance matrix
        while j < number_states:
            interval = int(len(Fmat[start_Trials]) / number_states)
            all_data = Fmat[start_Trials][j * interval : (j+1) * interval]
            for i in range(start_Trials+1, end_Trials):
                interval = int(len(Fmat[i]) / number_states)
                if j != number_states - 1:
                    all_data = np.vstack([all_data, Fmat[i][j * interval : (j+1) * interval]])
                else:
                    all_data = np.vstack([all_data, Fmat[i][j * interval : -1]])

            mu_force[j] = np.mean(np.array(all_data), 0).tolist()
            sigma[j] = (np.cov(np.transpose(np.array(all_data))) * 1).flatten().tolist()
            # if the covariance matrix is close to zero
            # if np.linalg.norm(np.array(sigma[j])) <= 0.00001:
            #     #sigma[j] = (np.identity(len(mu_force[j]))*1).flatten().tolist()
            #     sigma[j] = (np.ones(len(mu_force[j])*len(mu_force[j]))*10000).tolist()
            j = j+1
        return mu_force, sigma

    def calculate_A_B_pi(self, number_states, flag):
        # A - Transition Matrix
        A = getA(number_states)
        '''
        A = []
            for i in range(number_states):
                row = np.zeros(number_states)
                for j in range(i, number_states):
                    row[j] = (number_states - j)
                A.append(row.tolist())
        '''


        
        # B - Emission Matrix, parameters of emission distributions in pairs of (mu, sigma)
    
        B = [0.0]*number_states
        accum = [0.0] * len(self.categories)
        accum[0] = self.train_trials_per_category[0]
        for i in range(1, len(self.categories)):
            accum[i] = accum[i-1] + self.train_trials_per_category[i]

        if flag == exp_list[0]:
            mu_force, sigma = self.mean_cov(self.Fmat_train, 0, accum[0], number_states)
        elif flag == exp_list[1]:
            mu_force, sigma = self.mean_cov(self.Fmat_train, accum[0], accum[1], number_states)
        elif flag == exp_list[2]:
            mu_force, sigma = self.mean_cov(self.Fmat_train, accum[1], accum[2], number_states)
        elif flag == exp_list[3]:
            mu_force, sigma = self.mean_cov(self.Fmat_train, accum[2], accum[3], number_states)

        for num_states in range(number_states):
            B[num_states] = [mu_force[num_states],sigma[num_states]]

        # pi - initial probabilities per state
        if number_states == 3:
            pi = [1./3.] * 3
        elif number_states == 5:
            pi = [0.2] * 5
        elif number_states == 10:
            pi = [0.1] * 10
            # pi = [0.0] * 10
            # pi[0] = 1.0
        elif number_states == 15:
            pi = [1./15.] * 15
            # pi = [0.0] * 15
            # pi[0] = 1.0
        elif number_states == 20:
            pi = [1./20.] * 20
            # pi = [0.0] * 20
            # pi[0] = 1.0
        elif number_states == 30:
            pi = [1./30.] * 30
            # pi = [0.0] * 30
            # pi[0] = 1.0
        elif number_states == 40:
            pi = [1./40.] * 40
            #pi = np.square(np.array(range(40, 0, -1))).tolist()
            # pi = [0.0] * 40
            # pi[0] = 1.0

            #pi = [0.5, 0.25, 0.125, 0.0625, 0.0625/6, 0.0625/6, 0.0625/6, 0.0625/6, 0.0625/6, 0.0625/6]

        print 'Matrix B'
        print B
        #raw_input("Press Enter to continue...")
        return A, B, pi

    def create_model(self, flag, number_states):
          
        A, B, pi = self.calculate_A_B_pi(number_states, flag)

        # generate models from parameters
        #model = ghmm.HMMFromMatrices(self.F,ghmm.GaussianDistribution(self.F), A, B, pi)
        model = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), A, B, pi)
        model.normalize()
        return model

    def train(self, model, flag, accum_idx):
        input_data = []
        start = 0
        end = 0
        if flag == exp_list[0]:
            start = 0
            end = accum_idx[0]
        elif flag == exp_list[1]:
            start = accum_idx[0]
            end = accum_idx[1]
        elif flag == exp_list[2]:
            start = accum_idx[1]
            end = accum_idx[2]
        elif flag == exp_list[3]:
            start = accum_idx[2]
            end = accum_idx[3]
        for i in range(start, end):
            input_data.append(self.Fmat_train[i].flatten().tolist())

        '''
        if (flag == exp_list[1]):
            pp.figure(3)
            print len(self.Fmat_train), start, end
            for i in range(start, end):
                #pp.plot(np.transpose(self.Fmat_train[i])[0], np.transpose(self.Fmat_train[i])[1], 'o')
                pp.plot(np.transpose(self.Fmat_train[i])[0])
            pp.show()
            abc
        '''

        final_ts = ghmm.SequenceSet(self.F, input_data)

        #final_ts = model.sample(10, 50)

        # print the input data
        print final_ts

        model.baumWelch(final_ts)
        # print the optimized model
        #print model
        if math.isnan(model.getInitial(0)):
            print 'model has nan'
            abc
        #abc
        return model

    def test(self, model, ts_obj):
        # Find Viterbi Path
        final_ts_obj = ghmm.EmissionSequence(self.F, ts_obj.flatten().tolist())
        path_obj = model.viterbi(final_ts_obj)
        return path_obj

# Converts 0.1mps data to be synced with 0.15mps data
def warp_slow_to_fast(data):
    interpolated_data = []
    for i in xrange(len(data)-1):
        interpolated_data.append(data[i])
        interpolated_data.append(data[i] + 0.5*(data[i+1]-data[i]))
    interpolated_data.append(data[len(data)-1])
    interpolated_data = np.array(interpolated_data)
    warped_data = []
    for i in xrange(len(interpolated_data)):
        if i%3 == 0:
            warped_data.append(interpolated_data[i])
    return np.array(warped_data)

def get_armlength_percent_index(armlength, train_percentage, positions):
    for i in xrange(len(positions)):
        if positions[i]*100 > armlength * train_percentage:
            return i
    return len(positions)

def create_datasets(mat, fold_num, n_folds, generalizability, train_percentage):   # generalizaility takes 'subject', 'velocity' or 'none'
    trains = [[], [],[],[]]
    tests = [[],[],[],[]]

    if generalizability == 'none':   # do n_folds across all data
        exp_num = [0.0] * len(exp_list)     # number of trials for each category
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                for k in range(len(mat[i][j])):
                    exp_num[k] += len(mat[i][j][k])
        training_part = [0.0] * len(exp_list)
        for i in range(len(exp_num)):
            training_part[i] = [int(exp_num[i] / n_folds * int(fold_num-1)), int(exp_num[i] / n_folds * int(fold_num))]
            if fold_num == n_folds:
                training_part[i][1] = exp_num[i]
        for e in range(len(exp_num)):
            accum_num = 0
            for sub in range(len(mat)):
                for vel in range(len(mat[sub])):
                    for profile in mat[sub][vel][e]:
                        if accum_num < training_part[e][0] or accum_num >= training_part[e][1]:
                            if vel == 0:
                                warped_data = warp_slow_to_fast(profile)
                                perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, warped_data[:, 0])
                                trains[e].append(warped_data[0:perc, 1:3])
                            else:
                                perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, profile[:, 0])
                                trains[e].append(profile[0:perc, 1:3])
                        else:
                            tests[e].append([profile, Subject_armlength[sub]])
                        accum_num += 1
    elif generalizability == 'subject':
        subject_part = [int(len(mat)) / n_folds * int(fold_num-1), int(len(mat)) / n_folds * int(fold_num)]
        if fold_num == n_folds:
                subject_part[1] = len(mat)
        for sub in range(len(mat)):
            for vel in range(len(mat[sub])):
                for condition in range(len(mat[sub][vel])):
                    for profile in mat[sub][vel][condition]:
                        if sub < subject_part[0] or sub >= subject_part[1]:
                            if vel == 0:
                                warped_data = warp_slow_to_fast(profile)
                                perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, warped_data[:, 0])
                                trains[condition].append(warped_data[0:perc, 1:3])
                            else:
                                perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, profile[:, 0])
                                trains[condition].append(profile[0:perc, 1:3])
                        else:
                            tests[condition].append([profile, Subject_armlength[sub]])
    elif generalizability == 'velocity':
        n_folds = 2
        velocity_part = [fold_num-1, fold_num]
        for sub in range(len(mat)):
            for vel in range(len(mat[sub])):
                for condition in range(len(mat[sub][vel])):
                    for profile in mat[sub][vel][condition]:
                        if vel < velocity_part[0] or vel >= velocity_part[1]:
                            perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, profile[:, 0])
                            trains[condition].append(profile[0:perc, 1:3])
                        else:
                            tests[condition].append([profile, Subject_armlength[sub]])

    return sum([trains[0], trains[1], trains[2], trains[3]],[]), sum([tests[0], tests[1], tests[2], tests[3]],[]), [len(trains[0]), len(trains[1]), len(trains[2]), len(trains[3])], [len(tests[0]), len(tests[1]), len(tests[2]), len(tests[3])]

def run_crossvalidation(Fmat, categories, n_folds, data_div, max_percent, train_percentage, generalizability_test):
    confusion_mat = [[0] * np.size(categories) for i in range(np.size(categories))]
    sub_test_numbers = []
    sub_test_correct = []
    for i in xrange(data_div):
        sub_test_numbers.append([0.0, 0.0, 0.0, 0.0])
        sub_test_correct.append([0.0, 0.0, 0.0, 0.0])
    for fold in range(1, n_folds+1):
        training_set, testing_set, training_trials, testing_trials = create_datasets(Fmat, fold, n_folds, generalizability_test, train_percentage)
        hMM = HMM_Model(training_set, testing_set, categories, training_trials, testing_trials)
        states = 10
        path = np.array([[0] * np.size(categories) for i in range(np.size(testing_set,0))])
        path_max = []
        model = []
        model_trained = []
        accum = [0.0] * np.size(categories)
        accum[0] = training_trials[0]
        for i in range(1, len(categories)):
            accum[i] = accum[i-1] + training_trials[i]
        for i in range(np.size(categories)):
            model.append(hMM.create_model(categories[i], states))
            model_trained.append(hMM.train(model[i], categories[i], accum))
            print 'trained model' , i
        for p in range(0, data_div):
            for test in range(np.size(testing_set,0)):
                max_value = -50000001
                max_category = -1
                actual_cat = 0
                if test < testing_trials[0]:
                    actual_cat = 0
                elif test < testing_trials[0] + testing_trials[1]:
                    actual_cat = 1
                elif test < testing_trials[0] + testing_trials[1] + testing_trials[2]:
                    actual_cat = 2
                elif test < testing_trials[0] + testing_trials[1] + testing_trials[2] + testing_trials[3]:
                    actual_cat = 3
                for j in range(np.size(categories)):
                    test_data_len = get_armlength_percent_index(testing_set[test][1], (p+1.0)/data_div * max_percent, testing_set[test][0][:, 0])
                    value = hMM.test(model_trained[j], testing_set[test][0][0:test_data_len, 1:3])[1]
                    path[test][j] = value if value != -float('Inf') else -50000000 # eventually will get the path for 100% data
                    if value > max_value:
                        max_value = value
                        max_category = j
                if actual_cat == max_category:
                    sub_test_correct[p][actual_cat] += 1
                sub_test_numbers[p][actual_cat] += 1
        for i in range(np.size(testing_set,0)):
            path_max.append(max(path[i]))
            #print max(path[i])
            for j in range(np.size(categories)):
                if path_max[i] == path[i][j]:
                    actual_cat = 0
                    if i < testing_trials[0]:
                        actual_cat = 0
                    elif i < testing_trials[0] + testing_trials[1]:
                        actual_cat = 1
                    elif i < testing_trials[0] + testing_trials[1] + testing_trials[2]:
                        actual_cat = 2
                    elif i < testing_trials[0] + testing_trials[1] + testing_trials[2] + testing_trials[3]:
                        actual_cat = 3
                    confusion_mat[j][actual_cat] = confusion_mat[j][actual_cat] + 1
                    break
        #print path_max
    cmat = np.matrix(confusion_mat)

    accuracies = []
    for i in range(data_div):
        accuracies.append([sub_test_correct[i][j]/sub_test_numbers[i][j] for j in range(4)] +[sum(sub_test_correct[i])/sum(sub_test_numbers[i])])

    return cmat, accuracies

def show_confusion(cmat):
    total = float(cmat.sum())
    true = float(np.trace(cmat))
    percent_acc = "{0:.2f}".format((true/total)*100)

    # Plot Confusion Matrix
    Nlabels = np.size(categories)
    fig = pp.figure(0)
    ax = fig.add_subplot(111)
    figplot = ax.matshow(cmat, interpolation = 'nearest', origin = 'upper', extent=[0, Nlabels, 0, Nlabels])
    ax.set_title('Performance of HMM Models : Accuracy = ' + str(percent_acc))
    pp.xlabel("Targets")
    pp.ylabel("Predictions")
    ax.set_xticks([0.5,1.5,2.5,3.5])
    ax.set_xticklabels(exp_list)
    ax.set_yticks([3.5,2.5,1.5,0.5])
    ax.set_yticklabels(exp_list)
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
    
    categories = exp_list
    n_folds = 10
    generalizability_test_options = ['none', 'subject', 'velocity']
    generalizability_test = generalizability_test_options[2]
    if generalizability_test == 'velocity':
        n_folds = 2

    percentages = []
    accuracies = []
    percent_div = 30
    max_percent_armlength = 3.0
    for i in range(0, percent_div):
        percentages.append((i+1) * max_percent_armlength / percent_div)
    result_mat, accuracies = run_crossvalidation(input_Fmat, categories, n_folds, percent_div, max_percent_armlength, 3.0, generalizability_test)
    show_confusion(result_mat)

    fig1 = pp.figure(1)
    ax1 = fig1.add_subplot(231)
    ax1.plot(np.array(percentages)*100., np.transpose(np.matrix(accuracies))[0].tolist()[0], 'o-')
    ax1.set_title('Missed Type; Limited Data for Testing, Full for Training')
    ax1.set_xlabel('Percent of arm length of test data (arm ~ 0.3m)')
    ax1.set_ylabel('True Positive Rate (%)')
    ax2 = fig1.add_subplot(232)
    ax2.plot(np.array(percentages)*100., np.transpose(np.matrix(accuracies))[1].tolist()[0], 'o-')
    ax2.set_title('Good Type; Limited Data for Testing, Full for Training')
    ax2.set_xlabel('Percent of arm length of test data (arm ~ 0.3m)')
    ax2.set_ylabel('True Positive Rate (%)')
    ax3 = fig1.add_subplot(233)
    ax3.plot(np.array(percentages)*100., np.transpose(np.matrix(accuracies))[2].tolist()[0], 'o-')
    ax3.set_title('Caught Fist Type; Limited Data for Testing, Full for Training')
    ax3.set_xlabel('Percent of arm length of test data (arm ~ 0.3m)')
    ax3.set_ylabel('True Positive Rate (%)')
    ax4 = fig1.add_subplot(234)
    ax4.plot(np.array(percentages)*100., np.transpose(np.matrix(accuracies))[3].tolist()[0], 'o-')
    ax4.set_title('Caught Other Type; Limited Data for Testing, Full for Training')
    ax4.set_xlabel('Percent of arm length of test data (arm ~ 0.3m)')
    ax4.set_ylabel('True Positive Rate (%)')
    ax5 = fig1.add_subplot(235)
    ax5.plot(np.array(percentages)*100., np.transpose(np.matrix(accuracies))[4].tolist()[0], 'go-')
    ax5.set_title('Total True Positive Rate; Limited Data for Testing, Full for Training')
    ax5.set_xlabel('Percent of arm length of test data (arm ~ 0.3m)')
    ax5.set_ylabel('True Positive Rate (%)')
    pp.show()
    print('maximum accuracy:', max(accuracies))
    print('percentage at:', percentages[accuracies.index(max(accuracies))])
        

    
    
   

    
    