
import numpy as np
import matplotlib.pyplot as plt

class plotGenerator:
    def __init__(self, forcesList, distancesList, anglesList, pdfList, timesList, forcesTrueList, distancesTrueList, anglesTrueList,
            pdfTrueList, testForcesList, testDistancesList, testAnglesList, testPdfList, testTimesList,
            testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdfTrueList):

        self.forces, self.distances, self.angles, self.pdfs, self.times, self.forcesTrue, self.distancesTrue, self.anglesTrue, \
            self.pdfsTrue, self.testForces, self.testDistances, self.testAngles, self.testPdfs, self.testTimes, \
            self.testForcesTrue, self.testDistancesTrue, self.testAnglesTrue, self.testPdfsTrue = forcesList, distancesList, anglesList, pdfList, timesList, forcesTrueList, distancesTrueList, anglesTrueList, \
            pdfTrueList, testForcesList, testDistancesList, testAnglesList, testPdfList, testTimesList, \
            testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdfTrueList

    def plotOneTrueSet(self, k=0):
        fig = plt.figure()
        ax1 = plt.subplot(412)
        ax1.plot(self.times[k], self.forcesTrue[k], label='Force')
        ax1.set_ylabel('Magnitude (N)', fontsize=16)
        # ax1.set_xticks(np.arange(0, np.max(self.times[0]), 2.5))
        ax1.set_yticks(np.arange(8, 10, 0.5))
        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(411)
        ax2.plot(self.times[k], self.distancesTrue[k], label='Kinematics')
        ax2.set_ylabel('Distance (m)', fontsize=16)
        ax2.set_yticks(np.arange(0, 0.6, 0.1))
        ax2.legend()
        ax2.grid()

        ax3 = plt.subplot(414)
        ax3.plot(self.times[k], self.anglesTrue[k], label='Kinematics')
        ax3.set_ylabel('Angle (rad)', fontsize=16)
        ax3.set_xlabel('Time (sec)', fontsize=16)
        ax3.set_yticks(np.arange(0, 0.9, 0.2))
        ax3.legend()
        ax3.grid()

        ax4 = plt.subplot(413)
        ax4.plot(self.times[k], np.array(self.pdfsTrue[k]) * 100, label='Vision')
        ax4.set_ylabel('Magnitude (m)', fontsize=16)
        # ax4.set_yticks(np.arange(34, 37.5, 1))
        ax4.set_yticks(np.arange(4.6, 5.4, 0.2))
        ax4.legend()
        ax4.grid()

        plt.show()

    def distributionOfSequences(self):
        fig = plt.figure()
        ax1 = plt.subplot(412)
        ax1.set_ylabel('Force\nMagnitude (N)', fontsize=16)
        ax1.set_xticks(np.arange(0, 25, 5))
        ax1.set_yticks(np.arange(8, 10, 0.5))
        # ax1.set_yticks(np.arange(np.min(self.forcesTrue), np.max(self.forcesTrue), 1.0))
        ax1.grid()
        ax2 = plt.subplot(411)
        ax2.set_ylabel('Kinematic\nDistance (m)', fontsize=16)
        ax2.set_xticks(np.arange(0, 25, 5))
        ax2.set_yticks(np.arange(0, 1.0, 0.2))
        ax2.set_ylim([0, 0.9])
        # ax2.set_yticks(np.arange(np.min(self.distancesTrue), np.max(self.distancesTrue), 0.2))
        ax2.grid()
        ax3 = plt.subplot(414)
        ax3.set_ylabel('Kinematic\nAngle (rad)', fontsize=16)
        ax3.set_xlabel('Time (sec)', fontsize=16)
        ax3.set_xticks(np.arange(0, 25, 5))
        ax3.set_yticks(np.arange(0, 1.5, 0.3))
        ax3.set_ylim([0, 1.2])
        # ax3.set_yticks(np.arange(np.min(self.anglesTrue), np.max(self.anglesTrue), 0.2))
        ax3.grid()
        ax4 = plt.subplot(413)
        ax4.set_ylabel('Visual\nMagnitude (m)', fontsize=16)
        ax4.set_xticks(np.arange(0, 25, 5))
        # ax4.set_yticks(np.arange(2, 4, 0.5))
        # ax4.set_yticks(np.arange(np.min(np.array(self.pdfsTrue) * 100), np.max(np.array(self.pdfsTrue) * 100), 0.1))
        ax4.grid()

        print 'Force min/max:', np.min(self.forcesTrue), np.max(self.forcesTrue)
        print 'Distance min/max:', np.min(self.distancesTrue), np.max(self.distancesTrue)
        print 'Angle min/max:', np.min(self.anglesTrue), np.max(self.anglesTrue)
        print 'PDF min/max:', np.min(np.array(self.pdfsTrue) * 100), np.max(np.array(self.pdfsTrue) * 100)

        for force, distance, angle, pdf, time in zip(self.forcesTrue, self.distancesTrue, self.anglesTrue, np.array(self.pdfsTrue) * 100, self.times):
            ax1.plot(time, force)
            ax2.plot(time, distance)
            ax3.plot(time, angle)
            ax4.plot(time, pdf)

        plt.show()




def fig_eval(test_title, cross_data_path, nDataSet, onoff_type, check_methods, check_dims, \
             prefix, nState=20, \
             opr='robot', attr='id', bPlot=False, \
             cov_mult=[1.0, 1.0, 1.0, 1.0], renew=False, test=False, disp=None, rm_run=False, sim=False):

    # anomaly check method list
    use_ml_pkl = False
    false_dataSet = None
    count = 0

    for method in check_methods:
        for i in xrange(nDataSet):

            pkl_file = os.path.join(cross_data_path, "dataSet_"+str(i))
            dd = ut.load_pickle(pkl_file)

            train_aXData1 = dd['ft_force_mag_train_l']
            train_aXData2 = dd['audio_rms_train_l']
            train_chunks  = dd['train_chunks']
            test_aXData1 = dd['ft_force_mag_test_l']
            test_aXData2 = dd['audio_rms_test_l']
            test_chunks  = dd['test_chunks']

            # min max scaling for training data
            aXData1_scaled, min_c1, max_c1 = dm.scaling(train_aXData1, scale=10.0)
            aXData2_scaled, min_c2, max_c2 = dm.scaling(train_aXData2, scale=10.0)
            labels = [True]*len(train_aXData1)
            train_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, train_chunks, labels)

            # test data!!
            aXData1_scaled, _, _ = dm.scaling(test_aXData1, min_c1, max_c1, scale=10.0)
            aXData2_scaled, _, _ = dm.scaling(test_aXData2, min_c2, max_c2, scale=10.0)
            labels = [False]*len(test_aXData1)
            test_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, test_chunks, labels)

            if sim == True:
                false_aXData1 = dd['ft_force_mag_sim_false_l']
                false_aXData2 = dd['audio_rms_sim_false_l']
                false_chunks  = dd['sim_false_chunks']
                false_anomaly_start = dd['anomaly_start_idx']

                # generate simulated data!!
                aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
                aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)
                labels = [False]*len(false_aXData1)
                false_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, false_chunks, labels)
                false_dataSet.sa['anomaly_idx'] = false_anomaly_start
            else:
                false_aXData1 = dd['ft_force_mag_false_l']
                false_aXData2 = dd['audio_rms_false_l']
                false_chunks  = dd['false_chunks']
                false_anomaly_start = dd['anomaly_start_idx']

                ## print np.shape(false_aXData1), np.shape(false_aXData2)
                ## print false_chunks
                ## for k in xrange(len(false_aXData2)):
                ##     print np.shape(false_aXData1[k]), np.shape(false_aXData2[k])

                aXData1_scaled, _, _ = dm.scaling(false_aXData1, min_c1, max_c1, scale=10.0)
                aXData2_scaled, _, _ = dm.scaling(false_aXData2, min_c2, max_c2, scale=10.0)
                labels = [False]*len(false_aXData1)
                false_dataSet = dm.create_mvpa_dataset(aXData1_scaled, aXData2_scaled, false_chunks, labels)
                false_dataSet.sa['anomaly_idx'] = false_anomaly_start

            for check_dim in check_dims:

                # save file name
                res_file = prefix+'_dataset_'+str(i)+'_'+method+'_roc_'+opr+'_dim_'+str(check_dim)+'.pkl'
                res_file = os.path.join(cross_test_path, res_file)

                mutex_file_part = 'running_dataset_'+str(i)+'_dim_'+str(check_dim)+'_'+method
                mutex_file_full = mutex_file_part+'_'+strMachine+'.txt'
                mutex_file      = os.path.join(cross_test_path, mutex_file_full)

                if os.path.isfile(res_file):
                    count += 1
                    continue
                elif hcu.is_file(cross_test_path, mutex_file_part):
                    continue
                ## elif os.path.isfile(mutex_file): continue
                os.system('touch '+mutex_file)

                if check_dim is not 2:
                    x_train1 = train_dataSet.samples[:,check_dim,:]
                    lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, nEmissionDim=1, \
                                             check_method=method)
                    if check_dim==0: lhm.fit(x_train1, cov_mult=[cov_mult[0]]*4, use_pkl=use_ml_pkl)
                    elif check_dim==1: lhm.fit(x_train1, cov_mult=[cov_mult[3]]*4, use_pkl=use_ml_pkl)
                else:
                    x_train1 = train_dataSet.samples[:,0,:]
                    x_train2 = train_dataSet.samples[:,1,:]
                    ## plot_all(x_train1, x_train2)

                    lhm = learning_hmm_multi(nState=nState, trans_type=trans_type, check_method=method)
                    lhm.fit(x_train1, x_train2, cov_mult=cov_mult, use_pkl=use_ml_pkl)

                # find a minimum sensitivity gain
                if check_dim == 2:
                    x_test1 = test_dataSet.samples[:,0]
                    x_test2 = test_dataSet.samples[:,1]
                else:
                    x_test1 = test_dataSet.samples[:,check_dim]

                min_ths = 0
                min_ind = 0
                if method == 'progress':
                    min_ths = np.zeros(lhm.nGaussian)+10000
                    min_ind = np.zeros(lhm.nGaussian)
                elif method == 'globalChange':
                    min_ths = np.zeros(2)+10000

                n = len(x_test1)
                for i in range(n):
                    m = len(x_test1[i])

                    # anomaly_check only returns anomaly cases only
                    for j in range(2,m):

                        if check_dim == 2:
                            ths, ind = lhm.get_sensitivity_gain(x_test1[i][:j], x_test2[i][:j])
                        else:
                            ths, ind = lhm.get_sensitivity_gain(x_test1[i][:j])

                        if ths == []: continue

                        if method == 'progress':
                            if min_ths[ind] > ths:
                                min_ths[ind] = ths
                                print "Minimum threshold: ", min_ths[ind], ind
                        elif method == 'globalChange':
                            if min_ths[0] > ths[0]:
                                min_ths[0] = ths[0]
                            if min_ths[1] > ths[1]:
                                min_ths[1] = ths[1]
                            print "Minimum threshold: ", min_ths[0], min_ths[1]
                        else:
                            if min_ths > ths:
                                min_ths = ths
                                print "Minimum threshold: ", min_ths

                tp, fn, fp, tn, delay_l = anomaly_check_offline(lhm, [], false_dataSet, min_ths, check_dim=check_dim)

                d = dict()
                d['fn']    = fn
                d['tn']    = tn
                d['tp']    = tp
                d['fp']    = fp
                d['ths']   = ths
                d['delay_l'] = delay_l
                d['false_detection_l'] = false_detection_l

                try:
                    ut.save_pickle(d,res_file)
                except:
                    print "There is the targeted pkl file"

    if count == len(check_methods)*nDataSet*len(check_dims):
        print "#############################################################################"
        print "All file exist ", count
        print "#############################################################################"
    else:
        return


    if bPlot:

        fig = pp.figure()
        for method in check_methods:

            fn_l = np.zeros(nDataSet)
            tp_l = np.zeros(nDataSet)
            tn_l = np.zeros(nDataSet)
            fp_l = np.zeros(nDataSet)
            delay_l = []
            fd_l = []
            fpr_l = np.zeros(nDataSet)

            for i in xrange(nDataSet):
                # save file name
                res_file = prefix+'_dataset_'+str(i)+'_'+method+'_roc_'+opr+'_dim_'+str(check_dim)+'.pkl'
                res_file = os.path.join(cross_test_path, res_file)

                d = ut.load_pickle(res_file)
                fn_l[i] = d['fn']; tp_l[i] = d['tp']
                tn_l[i] = d['tn']; fp_l[i] = d['fp']
                delay_l.append([d['delay_l']])
                fd_l.append([d['false_detection_l']])
                ## print d['false_detection_l']

            for i in xrange(nDataSet):
                if fp_l[i]+tn_l[i] != 0:
                    fpr_l[i] = fp_l[i]/(fp_l[i]+tn_l[i])*100.0


            tot_fpr = np.sum(fp_l)/(np.sum(fp_l)+np.sum(tn_l))*100.0
            print method, tot_fpr


        pp.bar(range(nDataSet+1), np.hstack([fpr_l,np.array([tot_fpr])]))
        pp.ylim([0.0, 100])
