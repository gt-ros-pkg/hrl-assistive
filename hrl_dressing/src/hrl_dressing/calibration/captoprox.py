import glob, time
import cPickle as pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

def firstDeriv(x, t):
    # First derivative of measurements with respect to time
    dx = np.zeros(x.shape, np.float)
    dx[0:-1] = np.diff(x, axis=0)/np.diff(t, axis=0)
    dx[-1] = (x[-1] - x[-2])/(t[-1] - t[-2])
    return dx

class CapToProx:
    def __init__(self, files='singlesensor_armmount_newholder_0.*.pkl', plot=True, saveMat=False):
        self.capacitance = np.empty(0)
        self.position = np.empty((0, 3))

        filenames = glob.glob(files)
        for filename in filenames:
            if '5' in filename or '4' in filename:
                continue
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.capacitance = np.concatenate([self.capacitance, data['capacitance']], axis=0)
            self.position = np.concatenate([self.position, data['posDiff']], axis=0)

            print np.shape(self.position), np.shape(self.capacitance)

        print 'Min height:', np.min(self.position[:, 2])
        print 'Min cap:', np.min(self.capacitance), 'Max cap:', np.max(self.capacitance)
        if np.min(self.position[:, 2]) < 0:
            self.position[:, 2] -= np.min(self.position[:, 2])

        if plot:
            self.plot()

        d = {'capacitance': self.capacitance, 'position': self.position[:, 2]}

        if saveMat:
            scipy.io.savemat('singlesensor_armmount_newholder_combined', d)
            # scipy.io.savemat('calibration/participant_armmount2_0_combined', d)
            with open('singlesensor_armmount_newholder_combined.pkl', 'wb') as f:
            # with open('calibration/participant_armmount2_0_combined.pkl', 'wb') as f:
                pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    def plot(self):
        # [+ away from robot, + left (robot's perspective) towards center of robot, + up towards the sky]
        plt.subplot(1, 2, 1)
        plt.scatter(self.position[:, 0], self.position[:, 2], c=self.capacitance, cmap='inferno')
        plt.subplot(1, 2, 2)
        plt.scatter(self.capacitance, self.position[:, 2], c=self.capacitance, cmap='inferno')
        plt.xlabel('Lateral from hand (X)')
        plt.ylabel('Height from hand (Z)')
        plt.colorbar()
        plt.show()


# Plot data
captoprox = CapToProx(files='singlesensor_armmount_newholder_0.*.pkl', plot=True, saveMat=True)

