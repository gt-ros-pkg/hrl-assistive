
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
        ax1 = plt.subplot(411)
        ax1.plot(self.times[k], self.forcesTrue[k], label='Force')
        ax1.set_ylabel('Magnitude (N)')
        # ax1.set_xticks(np.arange(0, np.max(self.times[0]), 2.5))
        ax1.set_yticks(np.arange(8, 11, 1.0))
        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(self.times[k], self.distancesTrue[k], label='Kinematics')
        ax2.set_ylabel('Distance (m)')
        ax2.set_yticks(np.arange(0, 0.6, 0.2))
        ax2.legend()
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(self.times[k], self.anglesTrue[k], label='Kinematics')
        ax3.set_ylabel('Angle (rad)')
        ax3.set_yticks(np.arange(0, 0.9, 0.2))
        ax3.legend()
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(self.times[k], np.array(self.pdfsTrue[k]) * 100, label='Vision')
        ax4.set_ylabel('Magnitude (m)')
        ax4.set_xlabel('Time (sec)')
        # ax4.set_yticks(np.arange(34, 37.5, 1))
        ax4.set_yticks(np.arange(2.4, 2.8, 0.1))
        ax4.legend(loc=4)
        ax4.grid()

        plt.show()

    def distributionOfSequences(self):
        fig = plt.figure()
        ax1 = plt.subplot(411)
        ax1.set_ylabel('Force Magnitude (N)')
        ax1.set_xticks(np.arange(0, 25, 5))
        ax1.set_yticks(np.arange(8, 12, 1.0))
        # ax1.set_yticks(np.arange(np.min(self.forcesTrue), np.max(self.forcesTrue), 1.0))
        ax1.grid()
        ax2 = plt.subplot(412)
        ax2.set_ylabel('Kinematic Distance (m)')
        ax2.set_xticks(np.arange(0, 25, 5))
        ax2.set_yticks(np.arange(0, 1.0, 0.2))
        ax2.set_ylim([0, 0.9])
        # ax2.set_yticks(np.arange(np.min(self.distancesTrue), np.max(self.distancesTrue), 0.2))
        ax2.grid()
        ax3 = plt.subplot(413)
        ax3.set_ylabel('Kinematic Angle (rad)')
        ax3.set_xticks(np.arange(0, 25, 5))
        ax3.set_yticks(np.arange(0, 1.4, 0.4))
        ax3.set_ylim([0, 1.2])
        # ax3.set_yticks(np.arange(np.min(self.anglesTrue), np.max(self.anglesTrue), 0.2))
        ax3.grid()
        ax4 = plt.subplot(414)
        ax4.set_ylabel('Visual Magnitude (m)')
        ax4.set_xlabel('Time (sec)')
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

