#!/usr/bin/env python

import os
import time
import numpy as np
import cPickle as pickle
from scipy import interpolate
import matplotlib.pyplot as plt
import icra2015Batch as onlineHMM
import matplotlib.animation as animation

directory = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/onlineDataRecordings/'
# fileName = directory + 't2/t2_f_success.pkl'
# fileName = directory + 's10/s10_f_success.pkl'
# fileName = directory + 's11/ash_b_success1.pkl'
# fileName = directory + 's11/ash_b_failure_bowl.pkl'
fileName = directory + 's11/ash_b_failure_collision.pkl'
isNewFormat = True

parts = fileName.split('/')[-1].split('_')
subject = parts[0]
task = parts[1]

with open(fileName, 'rb') as f:
    data = pickle.load(f)
    forces = data['forces']
    distances = data['distances']
    angles = data['angles']
    audios = data['audios']
    forcesRaw = data['forcesRaw']
    distancesRaw = data['distancesRaw']
    anglesRaw = data['anglesRaw']
    audiosRaw = data['audioRaw']
    times = data['times']
    anomalyOccured = data['anomalyOccured']
    if isNewFormat:
        minThresholds = data['minThreshold']
        likelihoods = data['likelihoods']

print np.shape(times)
        
if isNewFormat:
    ll_likelihood = [x[0] for x in likelihoods]
    ll_state_idx = [x[1] for x in likelihoods]
    ll_likelihood_mu = [x[2] for x in likelihoods]
    ll_likelihood_std = [x[3] for x in likelihoods]
else:
    # Predefined settings
    downSampleSize = 100 #200
    scale = 1.0 #10
    nState = 10 #20
    cov_mult = 5.0
    cutting_ratio = [0.0, 0.7] #[0.0, 0.7]
    isScooping = task == 's' or task == 'b'
    if isScooping: ml_thres_pkl='ml_scooping_thres.pkl'
    else: ml_thres_pkl='ml_feeding_thres.pkl'

    saveDataPath = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/hmm/batchDataFiles/%s_%d_%d_%d_%d.pkl'
    # Setup HMM to perform online anomaly detection
    hmm, minVals, maxVals, minThresholds \
    = onlineHMM.iteration(downSampleSize=downSampleSize,
                          scale=scale, nState=nState,
                          cov_mult=cov_mult, verbose=False,
                          isScooping=isScooping, use_pkl=False,
                          train_cutting_ratio=cutting_ratio,
                          findThresholds=True, ml_pkl=ml_thres_pkl,
                          savedDataFile=saveDataPath % (('scooping' if isScooping else 'feeding'),
                                        downSampleSize, scale, nState, int(cov_mult)))

    ll_likelihood, ll_state_idx, ll_likelihood_mu, ll_likelihood_std = hmm.allLikelihoods(forces, distances, angles, audios)

def plotDataAndLikelihood():
    fig = plt.figure()
    ax1 = plt.subplot(411)
    # ax1.plot(times, trainData[0][:3], c='k')
    ax1.plot(times, forces, c='b')
    ax1.set_ylabel('Force')

    ax2 = plt.subplot(412)
    # ax2.plot(times, trainData[1][:3], c='k')
    ax2.plot(times, distances, c='b')
    ax2.set_ylabel('Distance')

    ax3 = plt.subplot(413)
    # ax3.plot(times, trainData[2][:3], c='k')
    ax3.plot(times, angles, c='b')
    ax3.set_ylabel('Angle')

    ax4 = plt.subplot(414)
    # ax4.plot(times, trainData[3][:3], c='k')
    ax4.plot(times, audios, c='b')
    ax4.set_ylabel('Audio')

    # ax5 = plt.subplot(515)
    # ax5.plot(times, trainData[0][:3], c='k')
    # ax5.plot(times, ll_likelihood, c='b')
    # ax5.set_ylabel('Log-likelihood')

    plt.show()

# plotDataAndLikelihood()


animateThreshold = True


print 'Times length:', len(times), 'Likelihood length:', len(ll_likelihood)
if len(ll_likelihood) > len(times):
    i = len(times)
    ll_likelihood, ll_state_idx, ll_likelihood_mu, ll_likelihood_std = ll_likelihood[-i:], ll_state_idx[-i:], ll_likelihood_mu[-i:], ll_likelihood_std[-i:]
    print 'New times length:', len(times), 'New likelihood length:', len(ll_likelihood)
elif len(ll_likelihood) < len(times):
    times = times[:len(ll_likelihood)]
    print 'New times length:', len(times), 'New likelihood length:', len(ll_likelihood)

# Determine thresholds
thresholdValues = []
for index in xrange(len(ll_likelihood)):
    minIndex = ll_state_idx[index]
    thresholdValues.append(ll_likelihood_mu[index] + minThresholds[minIndex]*ll_likelihood_std[index])


# Extrapolate data to see anomaly
# timeDiff = times[-1] - times[-2]
# times.append(times[-1] + timeDiff)
# times.append(times[-1] + timeDiff)
# lineDiff = ll_likelihood[-1] - ll_likelihood[-2]
# print ll_likelihood[-1], ll_likelihood[-2], lineDiff
# ll_likelihood.append(ll_likelihood[-1] + lineDiff)
# ll_likelihood.append(ll_likelihood[-1] + lineDiff)
# expectedDiff = ll_likelihood_mu[-1] - ll_likelihood_mu[-2]
# print ll_likelihood_mu[-1], ll_likelihood_mu[-2], expectedDiff
# ll_likelihood_mu.append(ll_likelihood_mu[-1] + expectedDiff)
# ll_likelihood_mu.append(ll_likelihood_mu[-1] + expectedDiff)
# thresholdDiff = thresholdValues[-1] - thresholdValues[-2]
# thresholdValues.append(thresholdValues[-1] + thresholdDiff)
# thresholdValues.append(thresholdValues[-1] + thresholdDiff)

def interpData(oldTimes, newTimes, data):
    dataInterp = interpolate.splrep(oldTimes, data, s=0)
    return interpolate.splev(newTimes, dataInterp, der=0)

# Interpolate data to 24 FPS
fps = 24
frames = int(times[-1] * fps)
newTimes = np.linspace(0, times[-1], frames)
ll_likelihood = interpData(times, newTimes, ll_likelihood)
ll_likelihood_mu = interpData(times, newTimes, ll_likelihood_mu)
thresholdValues = interpData(times, newTimes, thresholdValues)
times = newTimes

# Animation
fig, ax = plt.subplots()
# ax.set_title('Log-likelihood')
ax.set_xlabel('Time [s]', fontsize=16)
ax.set_ylabel('Log-likelihood', fontsize=16)

line, = ax.plot(times, ll_likelihood, 'b', linewidth=2.0, label='Log-likelihood')
expected, = ax.plot(times, ll_likelihood_mu, 'm', linewidth=2.0, label='Expected log-likelihood')
threshold, = ax.plot(times, thresholdValues, '--', color=0.75, linewidth=2.0, label='Threshold')
legend = ax.legend(loc=2)

# Increase legend line width
for label in legend.get_lines():
    label.set_linewidth(2.0)

# ax3.plot(x*(1./10.), ll_likelihood, 'b', label='Actual from \n test data')
# ax3.plot(x*(1./10.), ll_likelihood_mu, 'r', label='Expected from \n trained model')
# ax3.plot(x*(1./10.), ll_likelihood_mu + ll_thres_mult*ll_likelihood_std, 'r--', label='Threshold')

def animate(i):
    # Update the plots
    line.set_xdata(times[:i])
    line.set_ydata(ll_likelihood[:i])
    # print 'Length:', len(ll_likelihood[:i]), len(ll_likelihood)
    if animateThreshold:
        expected.set_xdata(times[:i])
        expected.set_ydata(ll_likelihood_mu[:i])
        threshold.set_xdata(times[:i])
        threshold.set_ydata(thresholdValues[:i])
    return line,

# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(times, mask=True))
    if animateThreshold:
        expected.set_ydata(np.ma.array(times, mask=True))
        threshold.set_ydata(np.ma.array(times, mask=True))
    return line,

interval = 1000 / len(ll_likelihood) * times[-1]
# fps = int(len(ll_likelihood) / times[-1])
# fps = 24
print 'Max time:', times[-1], 'Interval:', interval, 'FPS:', fps
ani = animation.FuncAnimation(fig, animate, np.arange(1, len(ll_likelihood) + 1), init_func=init, interval=25, blit=True)
location = time.strftime(os.path.join(os.path.dirname(__file__), 'likelihood_%m-%d-%Y_%H-%M-%S.mp4'))
ani.save(location, fps=fps)
# plt.show()

print 'Animation saved to:', location

