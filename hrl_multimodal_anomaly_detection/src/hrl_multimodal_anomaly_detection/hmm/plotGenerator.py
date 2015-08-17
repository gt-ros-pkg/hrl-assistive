
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
        ax4.plot(self.times[k], np.array(self.pdfsTrue[k]), label='Audio')
        ax4.set_ylabel('Magnitude (dec)', fontsize=16)
        ax4.set_yticks(np.arange(70, 150, 20))
        # ax4.set_yticks(np.arange(4.6, 5.4, 0.2))
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
        # ax1.grid()
        ax2 = plt.subplot(411)
        ax2.set_ylabel('Kinematic\nDistance (m)', fontsize=16)
        ax2.set_xticks(np.arange(0, 25, 5))
        ax2.set_yticks(np.arange(0, 1.0, 0.2))
        ax2.set_ylim([0, 1.0])
        # ax2.set_yticks(np.arange(np.min(self.distancesTrue), np.max(self.distancesTrue), 0.2))
        # ax2.grid()
        ax3 = plt.subplot(414)
        ax3.set_ylabel('Kinematic\nAngle (rad)', fontsize=16)
        ax3.set_xlabel('Time (sec)', fontsize=16)
        ax3.set_xticks(np.arange(0, 25, 5))
        ax3.set_yticks(np.arange(0, 1.5, 0.3))
        ax3.set_ylim([0, 1.5])
        # ax3.set_yticks(np.arange(np.min(self.anglesTrue), np.max(self.anglesTrue), 0.2))
        # ax3.grid()
        ax4 = plt.subplot(413)
        ax4.set_ylabel('Audio\nMagnitude (dec)', fontsize=16)
        ax4.set_xticks(np.arange(0, 25, 5))
        # ax4.set_yticks(np.arange(2, 4, 0.5))
        # ax4.set_yticks(np.arange(np.min(np.array(self.pdfsTrue) * 100), np.max(np.array(self.pdfsTrue) * 100), 0.1))
        # ax4.grid()

        print 'Force min/max:', np.min(self.forcesTrue), np.max(self.forcesTrue)
        print 'Distance min/max:', np.min(self.distancesTrue), np.max(self.distancesTrue)
        print 'Angle min/max:', np.min(self.anglesTrue), np.max(self.anglesTrue)
        print 'Audio min/max:', np.min(np.array(self.pdfsTrue)), np.max(np.array(self.pdfsTrue))

        for force, distance, angle, pdf, time in zip(self.forcesTrue, self.distancesTrue, self.anglesTrue, np.array(self.pdfsTrue), self.times):
            ax1.plot(time, force)
            ax2.plot(time, distance)
            ax3.plot(time, angle)
            ax4.plot(time, pdf)

        plt.show()

    def quickPlotModalities(self):
        # Quickly plot modality data without any axes labels
        # for modality in [self.forcesTrue[5:6] + self.testForcesTrue[6:7], self.distancesTrue[5:6] + self.testDistancesTrue[6:7], self.anglesTrue[5:6] + self.testAnglesTrue[6:7], self.pdfsTrue[5:6] + self.testPdfsTrue[6:7]]:
        for modality in [self.forcesTrue, self.distancesTrue, self.anglesTrue, self.pdfsTrue]:
            for index, (modal, times) in enumerate(zip(modality, self.times)): # self.times[5:6] + self.testTimes[6:7]
                plt.plot(times, modal, label='%d' % index)
            plt.legend()
            plt.show()
