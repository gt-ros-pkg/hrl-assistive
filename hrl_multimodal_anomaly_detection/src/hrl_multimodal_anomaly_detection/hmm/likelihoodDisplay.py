#!/usr/bin/env python

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import icra2015Batch as onlineHMM
import matplotlib.animation as animation

fileName = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/onlineDataRecordings/t2_f_success.pkl'

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

print 'Times length:', len(times), 'Likelihood length:', len(ll_likelihood)

fig = plt.figure()
plt.plot(times, ll_likelihood)
plt.show()

# Animation
fig, ax = plt.subplots()

line, = ax.plot(times, ll_likelihood)

def animate(i):
    # Update the plot
    line.set_ydata(ll_likelihood[:i])
    return line,

# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(times, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(ll_likelihood)), init_func=init, interval=25, blit=True)
plt.show()

