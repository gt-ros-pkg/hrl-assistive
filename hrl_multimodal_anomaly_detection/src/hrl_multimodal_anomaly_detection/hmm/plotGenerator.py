import os
import numpy as np
import matplotlib.pyplot as plt

class plotGenerator:
    def __init__(self, forcesList, distancesList, anglesList, audioList, timesList,\
                 forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList,\
                 testForcesList, testDistancesList, testAnglesList, testAudioList, testTimesList,
                 testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList):

        self.forces, self.distances, self.angles, self.audio, self.times, \
        self.forcesTrue, self.distancesTrue, self.anglesTrue, self.audioTrue, \
        self.testForces, self.testDistances, self.testAngles, self.testAudio, self.testTimes,\
        self.testForcesTrue, self.testDistancesTrue, self.testAnglesTrue, self.testAudioTrue =\
        [np.array(x) for x in [forcesList, distancesList, anglesList, audioList, timesList,\
                               forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList,\
                               testForcesList, testDistancesList, testAnglesList, testAudioList, testTimesList,\
                               testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList]]

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
        ax4.plot(self.times[k], self.audioTrue[k], label='Audio')
        ax4.set_ylabel('Magnitude (dec)', fontsize=16)
        ax4.set_yticks(np.arange(70, 150, 20))
        # ax4.set_yticks(np.arange(4.6, 5.4, 0.2))
        ax4.legend()
        ax4.grid()

        plt.show()

    def distributionOfSequences(self, useTest=False, numSuccess=0):
        fig = plt.figure()
        ax1 = plt.subplot(412)
        ax1.set_ylabel('Force\nMagnitude (N)', fontsize=16)
        ax1.set_xticks(np.arange(0, 25, 5))
        # ax1.set_yticks(np.arange(8, 10, 0.5))
        # ax1.set_yticks(np.arange(np.min(self.forcesTrue), np.max(self.forcesTrue), 1.0))
        # ax1.grid()
        ax2 = plt.subplot(411)
        ax2.set_ylabel('Kinematic\nDistance (m)', fontsize=16)
        ax2.set_xticks(np.arange(0, 25, 5))
        # ax2.set_yticks(np.arange(0, 1.0, 0.2))
        # ax2.set_ylim([0, 1.0])
        # ax2.set_yticks(np.arange(np.min(self.distancesTrue), np.max(self.distancesTrue), 0.2))
        # ax2.grid()
        ax3 = plt.subplot(414)
        ax3.set_ylabel('Kinematic\nAngle (rad)', fontsize=16)
        ax3.set_xlabel('Time (sec)', fontsize=16)
        ax3.set_xticks(np.arange(0, 25, 5))
        # ax3.set_yticks(np.arange(0, 1.5, 0.3))
        # ax3.set_ylim([0, 1.5])
        # ax3.set_yticks(np.arange(np.min(self.anglesTrue), np.max(self.anglesTrue), 0.2))
        # ax3.grid()
        ax4 = plt.subplot(413)
        ax4.set_ylabel('Audio\nMagnitude (dec)', fontsize=16)
        ax4.set_xticks(np.arange(0, 25, 5))
        # ax4.set_yticks(np.arange(2, 4, 0.5))
        # ax4.set_yticks(np.arange(np.min(np.array(self.pdfsTrue) * 100), np.max(np.array(self.pdfsTrue) * 100), 0.1))
        # ax4.grid()

        if not useTest:
            forces, distances, angles, audios, times = self.forcesTrue, self.distancesTrue, self.anglesTrue, self.audioTrue, self.times
        else:
            forces, distances, angles, audios, times = self.testForcesTrue[numSuccess:], self.testDistancesTrue[numSuccess:], self.testAnglesTrue[numSuccess:], self.testAudioTrue[numSuccess:], self.testTimes[numSuccess:]

        print 'Force min/max:', np.min(forces), np.max(forces)
        print 'Distance min/max:', np.min(distances), np.max(distances)
        print 'Angle min/max:', np.min(angles), np.max(angles)
        print 'Audio min/max:', np.min(audios), np.max(audios)

        # Plot successful test data as same color
        if useTest:
            for force, distance, angle, audio, time in zip(self.testForcesTrue[:numSuccess], self.testDistancesTrue[:numSuccess], self.testAnglesTrue[:numSuccess], self.testAudioTrue[:numSuccess], self.testTimes[:numSuccess]):
                ax1.plot(time, force, c='k')
                ax2.plot(time, distance, c='k')
                ax3.plot(time, angle, c='k')
                ax4.plot(time, audio, c='k')

        for force, distance, angle, audio, time in zip(forces, distances, angles, audios, times):
            ax1.plot(time, force)
            ax2.plot(time, distance)
            ax3.plot(time, angle)
            ax4.plot(time, audio)

        plt.show()


    def distributionOfSequencesScaled(self, useTest=False, numSuccess=0, save_pdf=False):
        fig = plt.figure()
        ax1 = plt.subplot(412)
        ax1.set_ylabel('Force\nMagnitude (N)', fontsize=16)
        ax1.set_xticks(np.arange(0, 25, 5))
        # ax1.set_yticks(np.arange(8, 10, 0.5))
        # ax1.set_yticks(np.arange(np.min(self.forcesTrue), np.max(self.forcesTrue), 1.0))
        # ax1.grid()
        ax2 = plt.subplot(411)
        ax2.set_ylabel('Kinematic\nDistance (m)', fontsize=16)
        ax2.set_xticks(np.arange(0, 25, 5))
        # ax2.set_yticks(np.arange(0, 1.0, 0.2))
        # ax2.set_ylim([0, 1.0])
        # ax2.set_yticks(np.arange(np.min(self.distancesTrue), np.max(self.distancesTrue), 0.2))
        # ax2.grid()
        ax3 = plt.subplot(414)
        ax3.set_ylabel('Kinematic\nAngle (rad)', fontsize=16)
        ax3.set_xlabel('Time (sec)', fontsize=16)
        ax3.set_xticks(np.arange(0, 25, 5))
        # ax3.set_yticks(np.arange(0, 1.5, 0.3))
        # ax3.set_ylim([0, 1.5])
        # ax3.set_yticks(np.arange(np.min(self.anglesTrue), np.max(self.anglesTrue), 0.2))
        # ax3.grid()
        ax4 = plt.subplot(413)
        ax4.set_ylabel('Audio\nMagnitude (dec)', fontsize=16)
        ax4.set_xticks(np.arange(0, 25, 5))
        # ax4.set_yticks(np.arange(2, 4, 0.5))
        # ax4.set_yticks(np.arange(np.min(np.array(self.pdfsTrue) * 100), np.max(np.array(self.pdfsTrue) * 100), 0.1))
        # ax4.grid()

        ## print 'Force min/max:', np.min(forces), np.max(forces)
        ## print 'Distance min/max:', np.min(distances), np.max(distances)
        ## print 'Angle min/max:', np.min(angles), np.max(angles)
        ## print 'Audio min/max:', np.min(audios), np.max(audios)

        # training data
        count = 0
        for force, distance, angle, audio, time in zip(self.forces, self.distances, self.angles, self.audio, self.times):
            ## if count == 15 : 
            ##     count = count + 1
            ##     continue
            
            ## ax1.plot(time, force, 'b')
            ## ax2.plot(time, distance, 'b', label=str(count))
            ## ax3.plot(time, angle, 'b')
            ## ax4.plot(time, audio, 'b')
            ax1.plot(time, force)
            ax2.plot(time, distance)
            ax3.plot(time, angle)
            ax4.plot(time, audio, label=str(count))
            count = count + 1
        
        # Plot successful test data as same color
        if useTest and numSuccess>0:
            for force, distance, angle, audio, time in zip(self.testForces[:numSuccess], 
                                                           self.testDistances[:numSuccess], 
                                                           self.testAngles[:numSuccess], 
                                                           self.testAudio[:numSuccess], 
                                                           self.testTimes[:numSuccess]):
                ax1.plot(time, force, c='g')
                ax2.plot(time, distance, c='g')
                ax3.plot(time, angle, c='g')
                ax4.plot(time, audio, c='g')
        ax4.legend(loc=4,prop={'size':16})

        # Plot abnormal test data as same color
        if useTest and False:
            for force, distance, angle, audio, time in zip(self.testForces[numSuccess:], 
                                                           self.testDistances[numSuccess:], 
                                                           self.testAngles[numSuccess:], 
                                                           self.testAudio[numSuccess:], 
                                                           self.testTimes[numSuccess:]):
                ax1.plot(time, force, 'r')
                ax2.plot(time, distance, 'r')
                ax3.plot(time, angle, 'r')
                ax4.plot(time, audio, 'r')

        if save_pdf == True:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')
        else:
            plt.show()        
        
    def quickPlotModalities(self):
        # Quickly plot modality data without any axes labels
        # for modality in [self.forcesTrue[5:6] + self.testForcesTrue[6:7], self.distancesTrue[5:6] + self.testDistancesTrue[6:7], self.anglesTrue[5:6] + self.testAnglesTrue[6:7], self.pdfsTrue[5:6] + self.testPdfsTrue[6:7]]:
        for modality in [self.forcesTrue, self.distancesTrue, self.anglesTrue, self.audioTrue]:
            for index, (modal, times) in enumerate(zip(modality, self.times)): # self.times[5:6] + self.testTimes[6:7]
                plt.plot(times, modal, label='%d' % index)
            plt.legend()
            plt.show()
