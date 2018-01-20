import numpy as np
import random
from scipy import stats

import hrl_lib.circular_buffer as cb

INIT_WINS = 3 #number of windows to use for calculating window

#TODO: function stable() may need changes to avoid some disadvantages / flaws
#      currently, shape of state is assumed to be [a_1, ..., a_n]

#random idea/possible extension for gesture control:
# http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4360147
# we may be able to use steady state to extract features (like std/slope/r_value)
# need more literature survey if you want to go deeper

class SteadyStateDetector:
    #win_size: moving window size for slope calculation
    #state_shape: shape of the state (e.g. 3x1 matrix of sensor reading)
    #duration: how many "windows" are used in stable calculation 
    #        (e.g. 10 moving windows of win_size low slope = stable)
    #mode:linreg      - finds the "slope" of the window using lin regression
    #     std monitor - finds the standard deviation of the windows (low std = stable)
    #overlap: how much each windows overlap. If negative, overlap = winsize + overlap
    #         (e.g. win_size of 10 and overlap of -1 will give overlap of 9. so 
    #          adjacent windows differ by 1 element)
    def __init__(self, win_size, state_shape, duration, mode="linreg", overlap=0):
        self.mode = mode #"linreg"
        self.win_size = win_size
        self.cb = cb.CircularBuffer(win_size * INIT_WINS, state_shape)
        self.time_cb = cb.CircularBuffer(win_size * INIT_WINS, (1,))
        if overlap < 0:
            if np.abs(overlap) >= win_size:
                print "overlap is bigger than win_size"
            self.overlap = win_size + overlap
        else:
            if np.abs(overlap) >= win_size:
                print "overlap is bigger than win_size"
            self.overlap = overlap
        self.cnt = 0
        if self.mode == "linreg":
            self.slope = cb.CircularBuffer(duration, state_shape)
        elif self.mode == "std monitor":
            assert overlap == -1 
            self.slope = cb.CircularBuffer(duration, state_shape) # Hold slope/std for each window?
            # What are these??
            self.avg   = []
            self.std   = []
            self.var   = []
            self.stable_stds = []
            self.stable_means = []

    #appends current state. It automatically calculates slope for moving windows.
    def append(self, state, time):
        #print "old state", state
        if self.mode == 'std monitor':
            if len(self.stable_stds) != 0:
                temp = []
                for i, std in enumerate(self.stable_stds):
                    #temp.append(state[i]/std)
                    temp.append((state[i] - self.stable_means[i])/std)
                #print "temp ", temp
                state = temp
        #print "new state ", state
        if len(self.cb) > 0:
            old_state = self.cb[0].copy().copy()
            #old_time = self.time_cb[0].copy().copy()
        if len(self.time_cb) > 0:
            old_time = self.time_cb[0].copy().copy()
        self.cb.append(state)
        self.time_cb.append(time)
        #check if enough is appended for slope calc
        if len(self.cb) == self.cb.size:
            #check if windows overlap is correct before calc slope
            if self.cnt <= self.overlap:
                self.cnt = self.cb.size - 1
                slope = self.calc_slope(old_state, old_time)
                print slope
                self.slope.append(slope)
            else:
                self.cnt -= 1

    #calculates slope.
    def calc_slope(self, old_state, old_time):
    #def calc_slope(self, old_state):
        """Returns array of lin reg slopes or std devs depending on mode."""
        if len(self.cb) + 1 < self.cb.size:
            print "Not enough data to extract slope"
        else:
            #Linear regression - calculates slope for each element/sensor reading/channel
            if self.mode == "linreg":
                slope_arr = []
                for i in xrange(len(self.cb[0])):
                    arr  = np.asarray(self.cb)[:, i]
                    t_arr = np.asarray(self.time_cb)[:, 0]
                    temp = stats.linregress(t_arr, arr)
                    #temp also gives p_value and r_value which may be useful.
                    #eg. r_value indicates likelihood of relation b/t time and state
                    #    (low r_value may mean its noise while high r_value may mean relation) 
                    slope, intercept, r_value, p_value, std_err = temp
                    slope_arr.append(slope)
                return slope_arr

            #Standard deviation - calculates standard deviation for each element within window
            elif self.mode == "std monitor":
                variances = []
                stds = [] # Will hold std dev for each signal
                # ???
                # first calculation of variances?
                if len(self.std) == 0:
                    for i in xrange(len(self.cb[0])):
                        arr = np.asarray(self.cb)[:, i] # array consisting of only 1 signal
                        var = np.var(arr)
                        print "first var ", var
                        self.avg.append(np.mean(arr))
                        stds.append(np.sqrt(var))
                        variances.append(var)
                    if len(self.stable_stds) == 0:
                        if 0.0 in self.stable_stds:
                            return stds
                        self.stable_stds = stds#.copy().copy()
                        self.stable_means = self.avg
                        self.cb = cb.CircularBuffer(self.win_size, self.cb[0].shape)
                        self.time_cb = cb.CircularBuffer(self.win_size, (1,))
                        return stds
                # Update variance/std dev
                else:
                    for i in xrange(len(self.cb[0])):
                        arr = np.asarray(self.cb)[:, i] # array consisting of only 1 signal
                        var = np.var(arr)
                        self.avg.append(np.mean(arr))
                        stds.append(np.sqrt(var))
                        variances.append(var)
                    """
                    for i, var in enumerate(self.var):
                        avg1 = self.avg[i].copy().copy()
                        self.avg[i] = self.avg[i] + ((self.cb[-1][i] - old_state[i])/self.cb.size)
                        avg2 = self.avg[i]
                        curr_var = var + ((self.cb[-1][i] ** 2 - old_state[i] ** 2) / float(self.cb.size))
                        curr_var = curr_var - (avg2 ** 2 - avg1 ** 2)
                        variances.append(curr_var)
                        print curr_var
                        stds.append(np.sqrt(curr_var))
                    """
                self.std = stds
                self.var = variances
                return self.std

    #checks for stability.
    def stable(self, thresh):
        """Return boolean of whether state is steady."""
        if self.mode == "linreg":
            #unstable if for any windows, L2 distance of the slope is bigger than threshold
            #stable otherwise. (L2 dist = sqrt(a_0^2 + ... + a_n^2) 
            #threshold is a float
            if len(self.slope) == self.slope.size:
                for slope in self.slope:
                    norm = np.sqrt(np.dot(slope, slope))
                    if norm > thresh:
                        return False
                return True
            else:
                return False
            #flaws you might need to consider (significance of deviation may vary from each channel.
            #(e.g. 1st channel's difference of .1 may indicate unstable while nth channel's difference
            # of 100 may be completely stable due to level of noise. But its current calculation
            # makes it hard to detect the difference of .1 in channel 1 )

        elif self.mode == "std monitor":
            #unstable if std deviation is bigger than the threshold. stable otherwise.
            #threshold is list of std_dev (e.g. [1.0, 3.0, ...]) as each sensor reading
            #                              may vary in how noisy it is)
            if len(self.slope) == self.slope.size:
                for slope in self.slope:
                    #bool_arr = slope > thresh
                    bool_arr = slope < thresh
                    #print bool_arr
                    for boolean in bool_arr:
                        if not boolean:
                            return False
                return True
            #flaws: you have to manually pick std deviation for each channel.
            # Without a method of choosing a 'good' std threshold, it may be criticized
            # (simply put, it would benefit from academically sound method of choosing std that works well)
            # (idea: run a simple SVM/linreg/etc to determine a boundary b/t stable/unstable
            #  on somewhat decently sized samples. May take too long thou)
            # if we do hand picked threshold. We can state the above method future work/consideration


    def size(self):
        return len(self.cb)

