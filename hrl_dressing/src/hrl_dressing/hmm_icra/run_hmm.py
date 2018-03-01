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

# from run_dtw import run_dtw

from scipy.misc import logsumexp

import sys
from data_organizer_gown import Fmat_original, exp_list, Maximal_initial_position, test_subjects, get_position_trunc_index, get_maxforce_percent_index, subject_list
from hmm_a_matrix import getA

# from draw_distribution import draw_distribution

datafrom = 1
datato = 3

leave_one_out = False

class HMM_Model:
    def __init__(self, Fmat, categories):
        # print 'Fmat length: ', len(Fmat)
        # print 'Fmat[0] length: ', len(Fmat[0])
        # print 'Fmat[0][0] length: ', len(Fmat[0][0])
        # print 'Fmat[0][1] length: ', len(Fmat[0][1])
        # print 'Fmat[1] length: ', len(Fmat[1])
        # print 'Fmat[1][0] length: ', len(Fmat[1][0])
        # print 'Fmat[1][1] length: ', len(Fmat[1][1])
        # print 'Fmat[2] length: ', len(Fmat[2])
        # print 'Fmat[2][0] length: ', len(Fmat[2][0])
        # print 'Fmat[2][1] length: ', len(Fmat[2][1])
        self.F = ghmm.Float() # emission domain of HMM model
        self.Fmat = Fmat
        self.categories = categories

    # Getting mean-std / mean-covariance
    def mean_cov(self, flag, number_states, dataset_id):
        data_to_use = []
        if not leave_one_out:
            data_to_use = self.Fmat[self.categories.index(flag)][0][dataset_id]
        else:
            for i in range(len(self.Fmat[self.categories.index(flag)][0])):
                if i != dataset_id:
                    data_to_use = data_to_use + self.Fmat[self.categories.index(flag)][0][i]
        j = 0
        mu_force = [0.0] * number_states    # mean value
        sigma = [0.0] * number_states       # covariance matrix
        while j < number_states:
            interval = int(len(data_to_use[0]) / number_states)
            all_data = data_to_use[0][j * interval : (j+1) * interval][:, datafrom:datato]
            for i in range(1, len(data_to_use)):
                interval = int(len(data_to_use[i]) / number_states)
                if j != number_states - 1:
                    all_data = np.vstack([all_data, data_to_use[i][j * interval : (j+1) * interval][:, datafrom:datato]])
                else:
                    all_data = np.vstack([all_data, data_to_use[i][j * interval : -1][:, datafrom:datato]])

            mu_force[j] = np.mean(np.array(all_data), 0).tolist()
            sigma[j] = (np.cov(np.transpose(np.array(all_data))) + 1. * np.identity(len(all_data[0]))).flatten().tolist()
            # if the covariance matrix is close to zero
            #if np.linalg.norm(np.array(sigma[j])) <= 0.00001:
                #sigma[j] = (np.identity(len(mu_force[j]))*1).flatten().tolist()
            #    sigma[j] = (np.ones(len(mu_force[j])*len(mu_force[j]))*10000).tolist()
            j = j+1
        return mu_force, sigma

    def calculate_A_B_pi(self, number_states, flag, dataset_id):
        # A - Transition Matrix
        A = getA(number_states)


        
        # B - Emission Matrix, parameters of emission distributions in pairs of (mu, sigma)
    
        B=[]

        mu_force, sigma = self.mean_cov(flag, number_states, dataset_id)

        for num_states in range(number_states):
            B.append([mu_force[num_states],sigma[num_states]])

        # pi - initial probabilities per state
        if number_states == 3:
            pi = [1./3.] * 3
            pi[0] = 1.0
        elif number_states == 5:
            pi = [0.2] * 5
            #pi[0] = 1.0
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

        #raw_input("Press Enter to continue...")
        return A, B, pi

    def create_model(self, flag, number_states, dataset_id):
          
        A, B, pi = self.calculate_A_B_pi(number_states, flag, dataset_id)

        # generate models from parameters
        model = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), A, B, pi)
        #model.normalize()

        return model

    def train(self, model, flag, dataset_id):
        input_data = []
        start = 0
        if not leave_one_out:
            end = len(self.Fmat[self.categories.index(flag)][0][dataset_id])
            for i in range(start, end):
                input_data.append(self.Fmat[self.categories.index(flag)][0][dataset_id][i][:, datafrom:datato].flatten().tolist())
        else:
            for dataid in range(len(self.Fmat[self.categories.index(flag)][0])):
                if dataid != dataset_id:
                    end = len(self.Fmat[self.categories.index(flag)][0][dataid])
                    for i in range(start, end):
                        input_data.append(self.Fmat[self.categories.index(flag)][0][dataid][i][:, datafrom:datato].flatten().tolist())
        random.shuffle(input_data)
        final_ts = ghmm.SequenceSet(self.F, input_data)



        model.baumWelch(final_ts)
        # print the optimized model

        if math.isnan(model.getInitial(0)):
            print 'model has nan'
            abc
        else:
            pp.figure(5)
            tc=['r', 'g', 'b']
            tc2=['k', 'c', 'y', 'm', 'w', '#eeefff', '#eeaaff', '#55efff', '#feefff', '#ffeeaa', '#eeecca', '#bbefff']
            # for i in range(model.N):
            #     draw_distribution(np.array(model.getEmission(i, 0)[0]), np.array(model.getEmission(i, 0)[1]).reshape(2,2), colors=(tc[self.categories.index(flag)], tc2[dataset_id]))

        return model

    def test(self, model, ts_obj):
        # Find Viterbi Path
        final_ts_obj = ghmm.EmissionSequence(self.F, np.array(ts_obj).flatten().tolist())
        path_obj = model.viterbi(final_ts_obj)
        return path_obj



#return 1 if list1 > list2
def compare_sum_exponentials(list1, list2):
    max_exponent = max(max(list1), max(list2))
    dif = 0
    for num in list1:
        dif += np.exp(num-max_exponent)
    for num in list2:
        dif -= np.exp(num-max_exponent)
    if abs(dif) < 0.00001:
        print 'similarrrrrrerewrewrerewrew'
    #print list1, list2, dif
    return dif > 0

def run_hmm(Fmat, categories, states, test_percentages):


    # test number and correct number per percentage
    test_numbers_perc = []
    test_correct_perc = []

    acc = []

    for p in range(len(test_percentages)):
        test_numbers_perc.append(0)
        test_correct_perc.append(0)

    #confusion_mat = [[0] * np.size(categories) for i in range(np.size(categories))]
    confusion_mat = [[0] * 3 for i in range(3)]
    print Fmat[0][0][0]
    hMM = HMM_Model(Fmat, categories)
    model = []
    model_trained = []
    for i in range(np.size(categories)):
        model_category = []
        model_trained_category = []
        for j in range(len(Fmat[i][0])):
            model_category.append(hMM.create_model(categories[i], states, j))
            model_trained_category.append(hMM.train(model_category[j], categories[i], j))
            print 'trained model' , i, j
        model.append(model_category)
        model_trained.append(model_trained_category)

    #fig = pp.figure(1)
    print 'start testing'
    print test_percentages
    for test in range(len(categories)):
        print 'test:', test
        actual_cat = test
        # print 'Fmat[test][0]', Fmat[test][0][0]
        # print 'Fmat[test][1]', Fmat[test][1]
        for test_subject_id in range(len(Fmat[test][1])):
            for testdata_id in range(len(Fmat[test][1][test_subject_id])):
                print 'test_subject_id', test_subject_id
                print 'testdata_id', testdata_id
                testdata = Fmat[test][1][test_subject_id][testdata_id]
                for p in range(0, len(test_percentages)):
                    for j in range(len(Fmat[test][0])):
                        if test_subjects[j] == subject_list[test_subject_id] and not leave_one_out:
                            print 'got here this'
                            continue
                        if test_subjects[j] != subject_list[test_subject_id] and leave_one_out:
                            print 'I got here instead'
                            if subject_list[test_subject_id] not in test_subjects:
                                print subject_list[test_subject_id], "not in test subjects"
                            #else:
                            #    continue
                            continue
                        max_value = -1000000
                        max_category = -1
                        # print 'testdata', testdata
                        perc = get_position_trunc_index(test_percentages[p], testdata[:, 0])
                        #if perc >= len(testdata[:, 0])-1:
                        #    perc -= 1
                        #perc = get_maxforce_percent_index(test_percentages[p], testdata[:, datafrom:datato])
                        values = []
                        for modelid in range(len(categories)):
                            pathobj = hMM.test(model_trained[modelid][j], testdata[:, datafrom:datato])
                            values.append(pathobj[1])
                            if pathobj[1] > max_value:
                                max_value = pathobj[1]
                                max_category = modelid

                        print 'max_category', max_category
                        print 'actual_cat', actual_cat

                        ''' take category with max score '''
                        if max_category == -1:
                            max_category = random.randint(0, len(categories)-1)
                        if actual_cat == max_category:
                            test_correct_perc[p] += 1
                        # print 'If I got here things should be ok'
                        test_numbers_perc[p] += 1
                        if p == len(test_percentages)-1:
                            confusion_mat[max_category][actual_cat] = confusion_mat[max_category][actual_cat] + 1
                            if actual_cat != max_category:
                                print actual_cat, testdata_id, values[actual_cat], values[max_category]
                                #pp.figure(10+actual_cat)
                                #pp.plot(testdata[0:perc, 1], 'r', alpha=0.2)
                            #if actual_cat == max_category:
                                #pp.figure(10+actual_cat)
                                #pp.plot(testdata[0:perc, 1], 'g', alpha=0.2)


        print 'one class done'
        #print path_max
    cmat = np.matrix(confusion_mat)
    print 'test_numbers_perc', test_numbers_perc
    for i in range(len(test_percentages)):
        acc.append(test_correct_perc[i]*1.0 / test_numbers_perc[i])
    print 'test done'
    return cmat, acc

def show_confusion(cmat):
    total = float(cmat.sum())
    true = float(np.trace(cmat))
    percent_acc = "{0:.2f}".format((true/total)*100)

    # Plot Confusion Matrix
    Nlabels = np.size(exp_list)
    print('here?')
    fig = pp.figure(0)
    print('here?')
    ax = fig.add_subplot(111)
    figplot = ax.matshow(cmat, interpolation = 'nearest', origin = 'upper', extent=[0, Nlabels, 0, Nlabels], cmap=pp.cm.gray_r)
    ax.set_title('Performance of HMM Models : Accuracy = ' + str(percent_acc))
    pp.xlabel("Targets")
    pp.ylabel("Predictions")
    print('here?')
    if len(exp_list) == 3:
        ax.set_xticks([0.5,1.5,2.5])
    else:
        ax.set_xticks([0.5,1.5])
    print('here?')
    ax.set_xticklabels(exp_list)
    if len(exp_list) == 3:
        ax.set_yticks([2.5,1.5,0.5])
    else:
        ax.set_yticks([1.5,0.5])
    ax.set_yticklabels(exp_list)
    figbar = fig.colorbar(figplot)
    print('here?')
    max_val = np.max(cmat)
    i = 0
    while (i < len(exp_list)):
        j = 0
        while (j < len(exp_list)):
            if cmat[i, j] < 0.3 * max_val:
                pp.text(j+0.5,-0.5 + len(exp_list)-i,cmat[i,j], size='xx-large', color='black')
            else:
                pp.text(j+0.5,-0.5 + len(exp_list)-i,cmat[i,j], size='xx-large', color='white')
            j = j+1
        i = i+1

if __name__ == '__main__':
    testname = 'final_sim_all'
    states = 10

    input_Fmat = Fmat_original
    
    categories = exp_list

     # position based thresholding
    percentages = np.arange(Maximal_initial_position, 0.85, 10.01).tolist()
    input_percent = np.arange(Maximal_initial_position, 0.85, 10.01).tolist()

    #percentages = np.arange(0.0, 14, 0.5).tolist()
    #input_percent = np.arange(0.0, 14, 0.5).tolist()

    percentages.sort()
    input_percent.sort()

    accuracies = []

    result_mat, accuracies = run_hmm(input_Fmat, categories, states, input_percent)
    #result_mat, accuracies = run_dtw(input_Fmat, categories, states, input_percent)

    print 'run hmm done'
    show_confusion(result_mat)
    pp.figure(1)

    pp.title('Accuracy w.r.t force position 10 states')
    pp.plot(percentages, accuracies, 'g-')

    pp.ylabel("Precisions")
    pp.xlabel("Position Threshold (m)")
    pp.ylim([0.2, 1])
    pp.show()
    print('maximum accuracy:', max(accuracies))
    print('percentage at:', percentages[accuracies.index(max(accuracies))])


    # output the results to file for future rendering
    outfile = open('results_'+testname+'.txt', 'w')
    for i in range(len(result_mat)):
        line = ''
        for j in range(len(result_mat[i])):
            line += str(result_mat[i].tolist()[j]) + ' '
        line += '\n'
        outfile.write(line)

    outfile.write(str(len(percentages)) + '\n')
    for i in range(len(percentages)):
        outfile.write(str(percentages[i]) + '\n')

    for i in range(len(accuracies)):
        outfile.write(str(accuracies[i]) + '\n')

    outfile.write('noxz\n0\n')
    outfile.close()

    pp.show()
   

    
    
