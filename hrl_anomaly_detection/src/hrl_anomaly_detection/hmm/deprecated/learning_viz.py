#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

import learning_util as util
import warnings







def path_disp(self, X):
    warnings.simplefilter("always", DeprecationWarning)
    
    X = [np.array(x) for x in X]
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n, m = np.shape(X[0])
    if self.verbose: print n, m
    x = np.arange(0., float(m))*(1./43.)
    path_mat  = np.zeros((self.nState, m))
    zbest_mat = np.zeros((self.nState, m))

    path_l = []
    for i in xrange(n):
        x_test = [x[i:i+1,:] for x in X]

        if self.nEmissionDim == 1:
            X_test = x_test[0]
        else:
            X_test = util.convert_sequence(x_test, emission=False)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
        path,_    = self.ml.viterbi(final_ts_obj)
        post = self.ml.posterior(final_ts_obj)

        use_last = False
        for j in xrange(m):
            ## sum_post = np.sum(post[j*2+1])
            ## if sum_post <= 0.1 or sum_post > 1.1 or sum_post == float('Inf') or use_last == True:
            ##     use_last = True
            ## else:
            add_post = np.array(post[j])/float(n)
            path_mat[:, j] += add_post

        path_l.append(path)
        for j in xrange(m):
            zbest_mat[path[j], j] += 1.0

    path_mat /= np.sum(path_mat, axis=0)

    # maxim = np.max(path_mat)
    # path_mat = maxim - path_mat

    zbest_mat /= np.sum(zbest_mat, axis=0)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fig = plt.figure()
    plt.rc('text', usetex=True)

    ax1 = plt.subplot(111)
    im  = ax1.imshow(path_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper',
                     extent=[0, float(m)*(1.0/10.), 20, 1], aspect=0.85)

    ## divider = make_axes_locatable(ax1)
    ## cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
    ax1.set_xlabel("Time (sec)", fontsize=18)
    ax1.set_ylabel("Hidden State Index", fontsize=18)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])

    ## for p in path_l:
    ##     ax1.plot(x, p, '*')

    ## ax2 = plt.subplot(212)
    ## im2 = ax2.imshow(zbest_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper',
    ##                  extent=[0,float(m)*(1.0/43.),20,1], aspect=0.1)
    ## plt.colorbar(im2, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
    ## ax2.set_xlabel("Time [sec]", fontsize=18)
    ## ax2.set_ylabel("Hidden State", fontsize=18)


    ## ax3 = plt.subplot(313)
    # fig.savefig('test.pdf')
    # fig.savefig('test.png')
    plt.grid()
    plt.show()


def likelihood_disp(self, X, X_true, Z, Z_true, axisTitles, ths_mult, figureSaveName=None):
    warnings.simplefilter("always", DeprecationWarning)
    
    n, m = np.shape(X[0])
    n2, m2 = np.shape(Z[0])
    if self.verbose: print "Input sequence X1: ", n, m
    if self.verbose: print 'Anomaly: ', self.anomaly_check(X, ths_mult)

    X_test = util.convert_sequence(X, emission=False)
    Z_test = util.convert_sequence(Z, emission=False)

    x = np.arange(0., float(m))
    z = np.arange(0., float(m2))
    ll_likelihood = np.zeros(m)
    ll_state_idx  = np.zeros(m)
    ll_likelihood_mu  = np.zeros(m)
    ll_likelihood_std = np.zeros(m)
    ll_thres_mult = np.zeros(m)
    for i in xrange(1, m):
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:i*self.nEmissionDim].tolist())
        logp = self.ml.loglikelihood(final_ts_obj)
        post = np.array(self.ml.posterior(final_ts_obj))

        # Find the best posterior distribution
        min_index, min_dist = self.findBestPosteriorDistribution(post[i-1])

        ll_likelihood[i] = logp
        ll_state_idx[i]  = min_index
        ll_likelihood_mu[i]  = self.ll_mu[min_index]
        ll_likelihood_std[i] = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]
        ll_thres_mult[i] = ths_mult

    # state blocks
    block_flag = []
    block_x    = []
    block_state= []
    text_n     = []
    text_x     = []
    for i, p in enumerate(ll_state_idx):
        if i is 0:
            block_flag.append(0)
            block_state.append(0)
            text_x.append(0.0)
        elif ll_state_idx[i] != ll_state_idx[i-1]:
            if block_flag[-1] is 0: block_flag.append(1)
            else: block_flag.append(0)
            block_state.append( int(p) )
            text_x[-1] = (text_x[-1]+float(i-1))/2.0 - 0.5 #
            text_x.append(float(i))
        else:
            block_flag.append(block_flag[-1])
        block_x.append(float(i))
    text_x[-1] = (text_x[-1]+float(m-1))/2.0 - 0.5 #

    block_flag_interp = []
    block_x_interp    = []
    for i in xrange(len(block_flag)):
        block_flag_interp.append(block_flag[i])
        block_flag_interp.append(block_flag[i])
        block_x_interp.append( float(block_x[i]) )
        block_x_interp.append(block_x[i]+0.5)


    # y1 = (X1_true[0]/scale1[2])*(scale1[1]-scale1[0])+scale1[0]
    # y2 = (X2_true[0]/scale2[2])*(scale2[1]-scale2[0])+scale2[0]
    # y3 = (X3_true[0]/scale3[2])*(scale3[1]-scale3[0])+scale3[0]
    # y4 = (X4_true[0]/scale4[2])*(scale4[1]-scale4[0])+scale4[0]
    Y = [x_true[0] for x_true in X_true]

    ZY = [np.mean(z_true, axis=0) for z_true in Z_true]

    ## matplotlib.rcParams['figure.figsize'] = 8,7
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fig = plt.figure()
    plt.rc('text', usetex=True)

    for index, (y, zy, title) in enumerate(zip(Y, ZY, axisTitles)):
        ax = plt.subplot('%i1%i' % (len(X) + 1, index + 1))
        ax.plot(x*(1./10.), y)
        ax.plot(z*(1./10.), zy, 'r')
        y_min = np.amin(y)
        y_max = np.amax(y)
        collection = collections.BrokenBarHCollection.span_where(np.array(block_x_interp)*(1./10.),
                                                                 ymin=0, ymax=y_max+0.5,
                                                                 # ymin=y_min - y_min/15.0, ymax=y_max + y_min/15.0,
                                                                 where=np.array(block_flag_interp)>0,
                                                                 facecolor='green',
                                                                 edgecolor='none', alpha=0.3)
        ax.add_collection(collection)
        ax.set_ylabel(title, fontsize=16)
        ax.set_xlim([0, x[-1]*(1./10.)])
        ax.set_ylim([y_min - 0.25, y_max + 0.5])
        # ax.set_ylim([y_min - y_min/15.0, y_max + y_min/15.0])

        # Text for progress
        if index == 0:
            for i in xrange(len(block_state)):
                if i%2 is 0:
                    if i<10:
                        ax.text((text_x[i])*(1./10.), y_max+0.15, str(block_state[i]+1))
                    else:
                        ax.text((text_x[i]-1.0)*(1./10.), y_max+0.15, str(block_state[i]+1))
                else:
                    if i<10:
                        ax.text((text_x[i])*(1./10.), y_max+0.06, str(block_state[i]+1))
                    else:
                        ax.text((text_x[i]-1.0)*(1./10.), y_max+0.06, str(block_state[i]+1))

    ax = plt.subplot('%i1%i' % (len(X) + 1, len(X) + 1))
    ax.plot(x*(1./10.), ll_likelihood, 'b', label='Actual from \n test data')
    ax.plot(x*(1./10.), ll_likelihood_mu, 'r', label='Expected from \n trained model')
    ax.plot(x*(1./10.), ll_likelihood_mu + ll_thres_mult*ll_likelihood_std, 'r--', label='Threshold')
    # ax.set_ylabel(r'$log P({\mathbf{X}} | {\mathbf{\theta}})$',fontsize=18)
    ax.set_ylabel('Log-likelihood', fontsize=16)
    ax.set_xlim([0, x[-1]*(1./10.)])

    # ax.legend(loc='upper left', fancybox=True, shadow=True, ncol=3, prop={'size':14})
    lgd = ax.legend(loc='upper center', fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(0.5, -0.5), prop={'size':14})
    ax.set_xlabel('Time (sec)', fontsize=16)

    plt.subplots_adjust(bottom=0.15)

    if figureSaveName is None:
        plt.show()
    else:
        # fig.savefig('test.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
        fig.savefig(figureSaveName, bbox_extra_artists=(lgd,), bbox_inches='tight')

