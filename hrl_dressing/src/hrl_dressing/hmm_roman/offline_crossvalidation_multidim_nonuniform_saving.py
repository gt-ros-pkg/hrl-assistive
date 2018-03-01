#!/usr/bin/env python

# Hidden Markov Model Implementation for Dressing
import matplotlib
import numpy as np
import matplotlib.pyplot as pp

import ghmmwrapper
import ghmm
import random

import math
import copy

import roslib
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle, save_pickle

import sys
from data_organizer import Fmat_original, exp_list, Subject_armlength, Maximal_initial_position
from hmm_a_matrix import getA

import matplotlib.pyplot as plt

datafrom = 1
datato = 3

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
            sigma[j] = (np.cov(np.transpose(np.array(all_data))) * 10).flatten().tolist()
            # if the covariance matrix is close to zero
            #if np.linalg.norm(np.array(sigma[j])) <= 0.00001:
                #sigma[j] = (np.identity(len(mu_force[j]))*1).flatten().tolist()
            #    sigma[j] = (np.ones(len(mu_force[j])*len(mu_force[j]))*10000).tolist()
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
        else:
            mu_force, sigma = self.mean_cov(self.Fmat_train, accum[exp_list.index(flag)-1], accum[exp_list.index(flag)], number_states)

        for num_states in range(number_states):
            B[num_states] = [mu_force[num_states],sigma[num_states]]

        # pi - initial probabilities per state
        if number_states == 3:
            pi = [1./3.] * 3
            pi[0] = 1.0
        elif number_states == 5:
            pi = [0.2] * 5
            pi[0] = 1.0
        elif number_states == 10:
            pi = [0.1] * 10
            # pi = [0.0] * 10
            #pi[0] = 1.0
        elif number_states == 15:
            pi = [1./15.] * 15
            #pi = [0.0] * 15
            #pi[0] = 1.0
        elif number_states == 20:
            pi = [1./20.] * 20
            # pi = [0.0] * 20
            #pi[0] = 1.0
        elif number_states == 30:
            pi = [1./30.] * 30
            # pi = [0.0] * 30
            #pi[0] = 1.0
        elif number_states == 40:
            pi = [1./40.] * 40
            #pi = np.square(np.array(range(40, 0, -1))).tolist()
            # pi = [0.0] * 40
            #pi[0] = 1.0

            #pi = [0.5, 0.25, 0.125, 0.0625, 0.0625/6, 0.0625/6, 0.0625/6, 0.0625/6, 0.0625/6, 0.0625/6]

        print 'Matrix B'
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
        else:
            start = accum_idx[exp_list.index(flag)-1]
            end = accum_idx[exp_list.index(flag)]
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
            pp.figure(6)
            print len(self.Fmat_train), start, end
            for i in range(start, end):
                #pp.plot(np.transpose(self.Fmat_train[i])[0], np.transpose(self.Fmat_train[i])[1], 'o')
                pp.plot(np.transpose(self.Fmat_train[i])[1])
            pp.show()
            abc
        '''

        final_ts = ghmm.SequenceSet(self.F, input_data)

        #final_ts = model.sample(10, 50)

        # print the input data

        model.baumWelch(final_ts)
        # print the optimized model
        #print model
        if math.isnan(model.getInitial(0)):
            print 'model has nan'
            #abc
        #abc
        #print model
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

def get_position_trunc_index(position, positions):
    for i in xrange(len(positions)):
        if positions[i] > position:
            return i
    return len(positions)

def get_maxforce_percent_index(maxforce, percentage, forces):
    for i in xrange(len(forces)):
        magnitude = 0
        for j in xrange(len(forces[i])):
            magnitude += (forces[i][j] * forces[i][j])
        if math.sqrt(magnitude) > maxforce * percentage:
            #print 'percentage of data: ', i * 1.0 / len(forces)
            if i == 0:
                i = 1
            return i
    return len(forces)

def create_datasets(mat, fold_num, n_folds, generlizability, train_percentage):   # generalizaility takes 'subject', 'velocity' or 'none'
    trains = []
    tests = []
    for i in range(len(exp_list)):
        trains.append([])
        tests.append([])

    if generlizability == 'none':   # do n_folds across all data
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
                        if True:  # accum_num < training_part[e][0] or accum_num >= training_part[e][1]:
                            if vel == 0:
                                warped_data = warp_slow_to_fast(profile)
                                #perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, warped_data[:, 0])
                                perc = get_maxforce_percent_index(20, train_percentage, warped_data[:, 1:3])
                                trains[e].append(warped_data[:, datafrom:datato])
                            else:
                                #perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, profile[:, 0])
                                perc = get_maxforce_percent_index(20, train_percentage, profile[:, 1:3])
                                trains[e].append(profile[:, datafrom:datato])
                        if True: #accum_num < training_part[e][0] or accum_num >= training_part[e][1]:
                            tests[e].append([profile, Subject_armlength[sub]])
                        accum_num += 1
    elif generlizability == 'subject':
        subject_part = [int(len(mat)) / n_folds * int(fold_num-1), int(len(mat)) / n_folds * int(fold_num)]
        if fold_num == n_folds:
            subject_part[1] = len(mat)
        count = 0
        for sub in range(len(mat)):
            for vel in range(len(mat[sub])):
                for condition in range(len(mat[sub][vel])):
                    for profile in mat[sub][vel][condition]:
                        #if sub < subject_part[0] or sub >= subject_part[1]:
                        if sub < subject_part[0] or sub >= subject_part[1]:
                            if vel == 0:
                                warped_data = warp_slow_to_fast(profile)
                                #perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, warped_data[:, 0])
                                #perc = get_maxforce_percent_index(max_force_mag, train_percentage, warped_data[:, 1:3])
                                perc = get_position_trunc_index(train_percentage, warped_data[:, 0])
                                #print perc * 1.0 / len(warped_data)
                                trains[condition].append(warped_data[0:perc, datafrom:datato])
                            else:
                                #perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, profile[:, 0])
                                #perc = get_maxforce_percent_index(max_force_mag, train_percentage, profile[:, 1:3])
                                perc = get_position_trunc_index(train_percentage, profile[:, 0])
                                #print perc * 1.0 / len(profile)
                                trains[condition].append(profile[0:perc, datafrom:datato])
                        else:
                            tests[condition].append([profile, Subject_armlength[sub]])
    elif generlizability == 'velocity':
        velocity_part = [fold_num-1, fold_num]
        for sub in range(len(mat)):
            for vel in range(len(mat[sub])):
                for condition in range(len(mat[sub][vel])):
                    for profile in mat[sub][vel][condition]:
                        if vel < velocity_part[0] or vel >= velocity_part[1]:
                            #perc = get_armlength_percent_index(Subject_armlength[sub], train_percentage, profile[:, 0])
                            #perc = get_maxforce_percent_index(max_force_mag, train_percentage, profile[:, 1:3])
                            perc = get_position_trunc_index(train_percentage, profile[:, 0])
                            trains[condition].append(profile[0:perc, 1:3])
                        else:
                            tests[condition].append([profile, Subject_armlength[sub]])

    return sum([trains[i] for i in xrange(len(trains))],[]), sum([tests[i] for i in xrange(len(tests))],[]), [len(trains[i]) for i in xrange(len(trains))], [len(tests[i]) for i in xrange(len(tests))]

def run_crossvalidation(Fmat, categories, n_folds, test_percentages, train_percentage):
    test_percentages = [1.0]
    train_percentage = 1.0
    confusion_mat = [[0] * np.size(categories) for i in range(np.size(categories))]
    sub_test_numbers = []
    sub_test_labeled = []
    sub_test_correct = []
    for i in xrange(len(test_percentages)):
        zero = []
        for j in xrange(len(exp_list)):
            zero.append(0.0)
        sub_test_numbers.append(copy.copy(zero))
        sub_test_correct.append(copy.copy(zero))
        sub_test_labeled.append(copy.copy(zero))
    # for fold in range(1, n_folds+1):
    training_set, testing_set, training_trials, testing_trials = create_datasets(Fmat, 1, n_folds, 'none', train_percentage)
    # print 'testing_set'
    # print testing_set
    # print 'training set:'
    # print training_set
    # print 'testing set:'
    # print testing_set
    # print 'training trials:'
    # print training_trials
    # print 'testing trials:'
    # print testing_trials
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
        model_trained_save = hMM.train(model[i], categories[i], accum)
        print 'trained model' , i
    # save_pickle(hMM, model'/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/')
        ghmm.HMMwriteList('/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/hmm_rig_subjects_'+categories[i][:len(categories[i])-1]+'.xml', [model_trained_save], ghmm.GHMM_FILETYPE_XML)
        model_trained[i] = ghmm.HMMOpen('/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/hmm_rig_subjects_'+categories[i][:len(categories[i])-1]+'.xml')
    # hMM.write('/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/hmm_rig_subjects.xml')

    data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing'
    saved_data = load_pickle(data_path + '/subject0/formatted_three/0.1mps/good/force_profile_1.pkl')
    scalar = 1
    test_data = np.dstack([saved_data[:, 1], (saved_data[:,2])*1, (saved_data[:,4])*1])[0]
    test_data = np.dstack([saved_data[:,1], (saved_data[:,4]) * scalar, (saved_data[:, 2]) * scalar])[0]
    temp = test_data[:, 1:3]
    test_data_len = get_position_trunc_index([1.0], test_data[:, 0])
    for j in range(np.size(categories)):
        value = hMM.test(model_trained[j], temp[0:test_data_len])[1]
        print 'category:', categories[j]
        print 'value: ', value
        # final_ts_obj = ghmm.EmissionSequence(self.F, np.array(temp).flatten().tolist())
        # pathobj = self.myHmms[modelid].viterbi(final_ts_obj)

    for p in range(0, len(test_percentages)):
        for test in range(np.size(testing_set,0)):
            max_value = -50000001
            max_category = -1
            actual_cat = 0
            for j in xrange(len(testing_trials)):
                if test < sum(testing_trials[0:j+1]):
                    actual_cat = j
                    break
            for j in range(np.size(categories)):
                #test_data_len = get_armlength_percent_index(testing_set[test][1], test_percentages[p], testing_set[test][0][:, 0])
                #test_data_len = get_maxforce_percent_index(max_force_mag, test_percentages[p], testing_set[test][0][:, datafrom:datato])
                test_data_len = get_position_trunc_index(test_percentages[p], testing_set[test][0][:, 0])
                #print test_data_len
                if test_data_len == 0:
                    test_data_len = 1
                # test_data_len = 1
                # print 'testing_set[test]', testing_set[test]
                value = hMM.test(model_trained[j], testing_set[test][0][0:test_data_len, datafrom:datato])[1]
                plotX = testing_set[test][0][0:test_data_len, datafrom:datato][:,0]
                plotY = testing_set[test][0][0:test_data_len, datafrom:datato][:,1]
                # plt.plot(plotX)
                # plt.plot(plotY)
                #
                # plt.show()
                path[test][j] = value if value != -float('Inf') else -50000000 # eventually will get the path for 100% data
                if value > max_value:
                    max_value = value
                    max_category = j
            # print 'max_category:', max_category
            if max_category == -1:
                max_category = random.randint(0, len(categories)-1)
            if actual_cat == max_category:
                sub_test_correct[p][actual_cat] += 1
            sub_test_numbers[p][actual_cat] += 1
            sub_test_labeled[p][max_category] += 1
            if p == len(test_percentages)-1:
                confusion_mat[max_category][actual_cat] = confusion_mat[max_category][actual_cat] + 1
        #print path_max
    cmat = np.matrix(confusion_mat)


    accuracies = []
    for i in range(len(test_percentages)):
        accuracies.append([sub_test_correct[i][j]/sub_test_numbers[i][j] for j in range(len(categories))] +[sum(sub_test_correct[i])/sum(sub_test_numbers[i])])
    recalls = []
    for i in range(len(test_percentages)):
        for j in range(len(categories)):
            if sub_test_labeled[i][j] == 0:
                sub_test_labeled[i][j] = 0.00001
        recalls.append([sub_test_correct[i][j]/sub_test_labeled[i][j] for j in range(len(categories))])
    return cmat, accuracies, recalls

def show_confusion(cmat):
    total = float(cmat.sum())
    true = float(np.trace(cmat))
    percent_acc = "{0:.2f}".format((true/total)*100)

    # Plot Confusion Matrix
    Nlabels = np.size(categories)
    fig = pp.figure(0)
    ax = fig.add_subplot(111)
    figplot = ax.matshow(cmat, interpolation = 'nearest', origin = 'upper', extent=[0, Nlabels, 0, Nlabels], cmap=pp.cm.gray_r)
    ax.set_title('Performance of HMM Models : Accuracy = ' + str(percent_acc))
    pp.xlabel("Targets")
    pp.ylabel("Predictions")
    if len(categories) == 3:
        ax.set_xticks([0.5,1.5,2.5])
    else:
        ax.set_xticks([0.5,1.5,2.5,3.5])
    ax.set_xticklabels(exp_list)
    if len(categories) == 3:
        ax.set_yticks([2.5,1.5,0.5])
    else:
        ax.set_yticks([3.5,2.5,1.5,0.5])
    ax.set_yticks([2.5,1.5,0.5])
    ax.set_yticklabels(exp_list)
    figbar = fig.colorbar(figplot)

    max_val = np.max(cmat)
    i = 0
    while (i < len(categories)):
        j = 0
        while (j < len(categories)):
            if cmat[i, j] < 0.3 * max_val:
                pp.text(j+0.5,-0.5 + len(categories)-i,cmat[i,j], size='xx-large', color='black')
            else:
                pp.text(j+0.5,-0.5 + len(categories)-i,cmat[i,j], size='xx-large', color='white')
            j = j+1
        i = i+1

if __name__ == '__main__':
    testname = 'subject_gen_5states_position_2categories'
    input_Fmat = Fmat_original
    
    categories = exp_list
    n_folds = 12

    percentages = []
    input_percent = []
    accuracies = []

    max_force_mag = 14

    percent_div = 100
    max_percent_armlength = 1.0
    max_distance = 1.0
    for i in range(0, percent_div):
        percentages.append((i+1) * max_percent_armlength / percent_div * max_force_mag)
        input_percent.append((i+1) * max_percent_armlength / percent_div)


    # position based thresholding
    percentages = np.arange(Maximal_initial_position, 0.85, 0.01).tolist()
    input_percent = np.arange(Maximal_initial_position, 0.85, 0.01).tolist()

    #percentages.append(0.505228792688)
    #input_percent.append(0.505228792688)

    '''
    additional_points = np.arange(1, max_force_mag, 1).tolist() + [3.68191569242]
    print additional_points

    for pt in additional_points:
        if not pt in percentages:
            percentages.append(pt)
            input_percent.append(pt*1.0/max_force_mag)
    '''
    percentages.sort()
    input_percent.sort()


    datato = 3
    datafrom = 1
    # result_mat, accuracies, recalls = run_crossvalidation(input_Fmat, categories, n_folds, input_percent, 1.0)
    result_mat, accuracies, recalls =run_crossvalidation(input_Fmat, categories, n_folds, input_percent, 1.0)
    # distance based result
    # result_mat, accuracies, recalls = run_crossvalidation(input_Fmat, categories, n_folds, input_percent, max_distance)

    '''
    # additional data
    additional_accuracies = []
    additional_recalls = []
    result_mat2, additional_accuracies, additional_recalls = run_crossvalidation(input_Fmat, categories, n_folds, [additional_points[i]*1.0/max_force_mag for i in range(len(additional_points))], 1.0)

    # z axis only
    datato = 2
    datafrom = 1
    mat, xonly_accuracies, xonly_recalls = run_crossvalidation(input_Fmat, categories, n_folds, input_percent, 1.0)

    # x axis only
    datafrom = 2
    datato = 3
    mat, zonly_accuracies, zonly_recalls = run_crossvalidation(input_Fmat, categories, n_folds, input_percent, 1.0)
    '''
    print result_mat
    show_confusion(result_mat)

    #show_confusion(lpm_mat)

    pp.figure(1)
    '''
    pp.subplot(231)
    pp.plot(percentages, np.transpose(np.matrix(accuracies))[0].tolist()[0], 'o-')
    pp.plot(percentages, np.transpose(np.matrix(recalls))[0].tolist()[0], 'ro-')
    pp.title('missed')
    pp.subplot(232)
    pp.plot(percentages, np.transpose(np.matrix(accuracies))[1].tolist()[0], 'o-')
    pp.plot(percentages, np.transpose(np.matrix(recalls))[1].tolist()[0], 'ro-')
    pp.title('good')
    pp.subplot(233)
    pp.plot(percentages, np.transpose(np.matrix(accuracies))[2].tolist()[0], 'o-')
    pp.plot(percentages, np.transpose(np.matrix(recalls))[2].tolist()[0], 'ro-')
    if len(categories) == 4:
        pp.title('caught fist')
        pp.subplot(234)
        pp.plot(percentages, np.transpose(np.matrix(accuracies))[3].tolist()[0], 'o-')
        pp.plot(percentages, np.transpose(np.matrix(recalls))[3].tolist()[0], 'ro-')
        pp.title('caught other')
        pp.subplot(235)
        pp.plot(percentages, np.transpose(np.matrix(accuracies))[4].tolist()[0], 'go-')
    elif len(categories) == 3:
        pp.title('caught')
        pp.subplot(234)
        pp.plot(percentages, np.transpose(np.matrix(accuracies))[3].tolist()[0], 'go-')
    pp.title('total')
    '''
    # pp.title('Accuracy w.r.t force threshold 3 states')
    # pp.plot(percentages, np.transpose(np.matrix(accuracies))[-1].tolist()[0], 'g-')

    '''
    pp.plot(percentages, np.transpose(np.matrix(xonly_accuracies))[3].tolist()[0], 'b-')
    pp.plot(percentages, np.transpose(np.matrix(zonly_accuracies))[3].tolist()[0], 'y-')
    pp.plot(additional_points[0:-1], np.transpose(np.matrix(additional_accuracies))[3].tolist()[0][0:-1], 'go')
    pp.plot(additional_points[-1], np.transpose(np.matrix(additional_accuracies))[3].tolist()[0][-1], 'ro')
    '''

    # pp.ylabel("Precisions")
    # pp.xlabel("Force Threshold (N)")
    # pp.show()
    # print('maximum accuracy:', max(accuracies, key=lambda x:x[-1]))
    # print('percentage at:', percentages[accuracies.index(max(accuracies))])
    #
    #
    # # output the results to file for future rendering
    # outfile = open('results_'+testname+'.txt', 'w')
    # for i in range(len(result_mat)):
    #     line = ''
    #     for j in range(len(result_mat[i])):
    #         line += str(result_mat[i].tolist()[j]) + ' '
    #     line += '\n'
    #     outfile.write(line)
    #
    # outfile.write(str(len(percentages)) + '\n')
    # for i in range(len(percentages)):
    #     outfile.write(str(percentages[i]) + '\n')
    #
    # for i in range(len(np.transpose(np.matrix(accuracies))[-1].tolist()[0])):
    #     outfile.write(str(np.transpose(np.matrix(accuracies))[-1].tolist()[0][i]) + '\n')
    #
    # outfile.write('noxz\n0\n')
    # '''
    # outfile.write('hasxz\n')  # whetehr output xz axis
    # for i in range(len(np.transpose(np.matrix(xonly_accuracies))[3].tolist()[0])):
    #     outfile.write(str(np.transpose(np.matrix(xonly_accuracies))[3].tolist()[0][i]) + '\n')
    # for i in range(len(np.transpose(np.matrix(zonly_accuracies))[3].tolist()[0])):
    #     outfile.write(str(np.transpose(np.matrix(zonly_accuracies))[3].tolist()[0][i]) + '\n')
    #
    # outfile.write(str(len(additional_accuracies)) + '\n')
    # for i in range(len(additional_points)):
    #     outfile.write(str(additional_points[i]) + '\n')
    # for i in range(len(np.transpose(np.matrix(additional_accuracies))[3].tolist()[0])):
    #     outfile.write(str(np.transpose(np.matrix(additional_accuracies))[3].tolist()[0][i]) + '\n')
    # '''
    # outfile.close()
    
    
   

    
    
