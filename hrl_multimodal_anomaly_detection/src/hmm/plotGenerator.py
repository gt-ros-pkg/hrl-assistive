
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

