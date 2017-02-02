import numpy as np
import sys, os
import ghmm
import ghmmwrapper
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser('~')+'/catkin_ws/src/hrl-assistive/hrl_anomaly_detection/src/hrl_anomaly_detection/hmm')
import learning_util as util



t = np.linspace(0.0, np.pi*2.0, 100)
n = 10 

X = []
for i in xrange(n):
    if i<1:
        x1 = np.cos(t) + np.random.normal(-0.3, 0.3, np.shape(t) )
        x2 = np.sin(t) + np.random.normal(-0.3, 0.3, np.shape(t) )
    else:
        x1 = np.cos(t) + np.random.normal(-0.2, 0.2, np.shape(t) )
        x2 = np.sin(t) + np.random.normal(-0.2, 0.2, np.shape(t) )        
    X.append( np.vstack([ x1.reshape(1,len(t)), x2.reshape(1,len(t)) ]) )
X = np.swapaxes(X, 0,1)


nEmissionDim = 2
nState       = 20
F = ghmm.Float()
cov_mult = [0.05]*(nEmissionDim**2)
cov_type = 'full'


# Transition probability matrix (Initial transition probability, TODO?)
A = util.init_trans_mat(nState).tolist()
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
if cov_type=='diag': ml.setDiagonalCovariance(0)

final_seq = ghmm.SequenceSet(F, X_train)
print 'Run Baum Welch method with (samples, length)', np.shape(X_train)
ret = ml.baumWelch(final_seq, 10000)
[A, B, pi] = ml.asMatrices()

######################### save/load params######################################

[out_a_num, vec_num, mat_num, u_denom] = ml.getBaumWelchParams()
ml.setBaumWelchParams(out_a_num, vec_num, mat_num, u_denom)
[out_a_num2, vec_num2, mat_num2, u_denom2] = ml.getBaumWelchParams()

if np.sum(np.array(out_a_num) - np.array(out_a_num2)) + np.sum(np.array(vec_num) - np.array(vec_num2)) + np.sum(np.array(mat_num) - np.array(mat_num2)) + np.sum(np.array(u_denom) - np.array(u_denom2)) != 0.0:
    print "get/set baumwelch param error"
    sys.exit()



######################### Prediction ###########################################

for k in xrange(10):
    # new target traj
    X2 = []
    x1 = np.cos(t/2.0+np.pi/8.)  + np.random.normal(-0.2, 0.2, np.shape(t))
    x2 = np.sin(t/2.0+np.pi/8.)  + np.random.normal(-0.2, 0.2, np.shape(t))
    X2.append( np.vstack([ x1.reshape(1,len(t)), x2.reshape(1,len(t)) ]) )
    X2 = np.swapaxes(X2, 0,1)

    X_test = util.convert_sequence(X2) # Training input
    X_test = X_test.tolist()
    print "target data: ", np.shape(X_test)

    # new emission for partial sequence
    B1 = []
    for i in xrange(nState):    
        B1.append( [ B[i][0][0], B[i][1][0] ] )

    ml1 = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), \
                               A, B1, pi)

    startIdx = 4
    x2_pred = []
    for i in xrange(len(t)):

        if i > startIdx:
            final_ts_obj = ghmm.EmissionSequence(F, x1[:i].tolist())
            try:
                (alpha, scale) = ml1.forward(final_ts_obj)
            except:
                print "No alpha is available !!"
                sys.exit()

            t_o = 0.0
            for j in xrange(nState):
                t_o += alpha[-1][j]*(B[j][0][1] + B[j][1][1]/B[j][1][0]*(x1[i]-B[j][0][0]))

            x2_pred.append(t_o)
        else:
            x2_pred.append(x2[i])



    fig = plt.figure()
    fig.add_subplot(211)
    plt.plot(X[0].T, 'b-')
    plt.plot(X2[0].T, 'g-')
    fig.add_subplot(212)
    plt.plot(X[1].T, 'b-')
    plt.plot(X2[1].T, 'g-')
    plt.plot(x2_pred, 'r-')
    plt.show()
