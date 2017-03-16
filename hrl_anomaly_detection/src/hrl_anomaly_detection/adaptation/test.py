import numpy as np
import sys, os
import ghmm
import ghmmwrapper

sys.path.insert(0, os.path.expanduser('~')+'/catkin_ws/src/hrl-assistive/hrl_anomaly_detection/src/hrl_anomaly_detection/hmm')
import learning_util as util



t = np.linspace(0.0, np.pi*1.0, 100)
n = 20
noise_mag = 0.1
scaler = np.array( [1.0]*len(t) )
## scaler[0:5] = 0.1
## scaler[5] = 0.2
## scaler[6] = 0.4
## scaler[7] = 0.6
## scaler[8] = 0.8

## print np.shape(scaler), np.shape(t)
## print np.random.normal(-noise_mag, noise_mag, np.shape(t))*scaler

mag=1.0
X = []
for i in xrange(n):

    x1 = np.array([mag*np.cos(0)]*20) + np.random.normal(-noise_mag, noise_mag, 20)
    x2 = np.array([mag*np.sin(0)]*20) + np.random.normal(-noise_mag, noise_mag, 20)
    
    x1 = np.concatenate([x1, mag*np.cos(t) + np.random.normal(-noise_mag, noise_mag, np.shape(t))*scaler ])
    x2 = np.concatenate([x2, mag*np.sin(t) + np.random.normal(-noise_mag, noise_mag, np.shape(t))*scaler ])

    X.append( np.vstack([ x1.reshape(1,len(x1)), x2.reshape(1,len(x2)) ]) )
X = np.swapaxes(X, 0,1)



nEmissionDim = 2
nState       = 20
F = ghmm.Float()
cov_mult = [1.0]*(nEmissionDim**2)
cov_type = 'diag'


# Transition probability matrix (Initial transition probability, TODO?)
A = util.init_trans_mat(nState, mode='linear').tolist()
## A = np.zeros((nState, nState))
## for i in xrange(nState):
##     for j in xrange(i,i+3):
##         if j > nState-1: continue
##         if j==i:   A[i,j] = 0.4
##         if j==i+1: A[i,j] = 0.4
##         if j==i+2: A[i,j] = 0.2           
##     A[i]/=np.sum(A[i])


mus, cov = util.vectors_to_mean_cov(X, nState, nEmissionDim, cov_type=cov_type)

# cov: state x dim x dim
for i in xrange(nEmissionDim):
    for j in xrange(nEmissionDim):
        cov[:, i, j] *= cov_mult[nEmissionDim*i + j]
        
# Emission probability matrix
B = [0] * nState
for i in range(nState):
    B[i] = [[mu[i] for mu in mus]]
    B[i].append(cov[i].flatten())

# pi - initial probabilities per state 
pi = [0.0] * nState
pi[0] = 1.0

ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
print 'Creating Training Data'            
X_train = util.convert_sequence(X) # Training input
X_train = X_train.tolist()
print "training data size: ", np.shape(X_train)

## ml.cmodel.getState(0).setOutProb(1, 0, 0.8)
## print ml.cmodel.getState(0).getOutProb(1)
## print ml.cmodel.getState(0).getOutNum(1)
#if cov_type=='diag': ml.setDiagonalCovariance(0)



final_seq = ghmm.SequenceSet(F, X_train)
print 'Run Baum Welch method with (samples, length)', np.shape(X_train)
ret = ml.baumWelch(final_seq, 10000)

######################### save/load params######################################

## [out_a_num, vec_num, mat_num, u_denom] = ml.getBaumWelchParams()
## ml.setBaumWelchParams(out_a_num, vec_num, mat_num, u_denom)
## [out_a_num2, vec_num2, mat_num2, u_denom2] = ml.getBaumWelchParams()

## if np.sum(np.array(out_a_num) - np.array(out_a_num2)) + np.sum(np.array(vec_num) - np.array(vec_num2)) + np.sum(np.array(mat_num) - np.array(mat_num2)) + np.sum(np.array(u_denom) - np.array(u_denom2)) != 0.0:
##     print "get/set baumwelch param error"
##     sys.exit()

######################### likelihood ###########################################
## # new target traj
## X2 = []
## for i in xrange(n):
##     x1 = np.cos(t+np.pi/2.) + np.random.normal(-0.2, 0.2, np.shape(t) )
##     x2 = np.sin(t+np.pi/2.) + np.random.normal(-0.2, 0.2, np.shape(t) )
##     X2.append( np.vstack([ x1.reshape(1,len(t)), x2.reshape(1,len(t)) ]) )
## X2 = np.swapaxes(X2, 0,1)

## X_test = util.convert_sequence(X2) # Training input
## X_test = X_test.tolist()

ll_likelihoods = []
for i in xrange(1,n):

    l_likelihood = []
    for j in xrange(4, len(X_train[i])/nEmissionDim):    
        final_seq = ghmm.EmissionSequence(F, np.array(X_train)[i,:j*nEmissionDim].tolist() )        
        logp = ml.loglikelihood(final_seq)
        l_likelihood.append( logp )

    ll_likelihoods.append(l_likelihood)
ll_likelihoods = np.array(ll_likelihoods)



m = 100
X2 = []
obs_seq = ml.sample(m, len(X_train[0])/nEmissionDim, seed=3586662)
for j in xrange(m):
    X2.append(np.array(obs_seq[j]).reshape(len(X_train[0])/nEmissionDim,2).T)
X2 = np.swapaxes(X2, 0, 1)

X_test = util.convert_sequence(X2) # Training input
print np.shape(X2), np.shape(X_train)

lls = []
for i in xrange(1,m):
    l_likelihood = []
    for j in xrange(4, len(X_test[i])/nEmissionDim):    
        final_seq = ghmm.EmissionSequence(F, np.array(X_test)[i,:j*nEmissionDim].tolist() )        
        logp = ml.loglikelihood(final_seq)
        l_likelihood.append( logp )

    lls.append(l_likelihood)
lls = np.array(lls)


print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
import matplotlib.pyplot as plt

fig = plt.figure()
fig.add_subplot(411)
plt.plot(X[0].T, 'b-')
## plt.plot(X2[0].T, 'g-')
plt.plot(X2[0].T, 'r-', alpha=0.5)
fig.add_subplot(412)
plt.plot(X[1].T, 'b-')
## plt.plot(X2[1].T, 'g-')
plt.plot(X2[1].T, 'r-', alpha=0.5)

fig.add_subplot(413)
plt.plot(ll_likelihoods.T, 'b-')
plt.plot(lls.T, 'r-')

fig.add_subplot(414)
plt.plot(mus[0], 'b-')
plt.plot(mus[0]+cov[:,0,0], 'r--')
plt.plot(mus[0]-cov[:,0,0], 'r--')


plt.show()
