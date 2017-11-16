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

import os, sys
import numpy as np
import hrl_lib.util as ut



def sampleWithWindow(X, window=5):
    '''
    X : sample x length x features
    return: (sample x length-window+1) x features
    '''
    if window < 1:
        print "Wrong window size"
        sys.exit()

    X_new = []
    for i in xrange(len(X)): # per sample
        for j in xrange(len(X[i])-window+1): # per time
            X_new.append( X[i][j:j+window].tolist() ) # per sample
    
    return X_new


def create_dataset(X, window_size=5, step=5):
    '''
    dataset: timesteps x dim
    '''    
    x = []
    y = []
    for j in range(len(X)-step-window_size):
        x.append(X[j:(j+window_size), :].tolist())
        y.append(X[j+step:(j+step+window_size), :].tolist())
    return np.array(x), np.array(y)



def graph_variations(x_true, x_pred_mean, x_pred_std=None, scaler_dict=None, save_pdf=False,
                     **kwargs):
    '''
    x_true: timesteps x dim
    '''

    # visualization
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import gridspec
    from matplotlib import rc
    import itertools
    colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
    shapes = itertools.cycle(['x','v', 'o', '+'])

    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.rcParams['ps.fonttype'] = 42
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #rc('text', usetex=True)
    ## matplotlib.rcParams['text.usetex'] = True

    # unscale
    if scaler_dict is None: param_dict = None
    else: param_dict = scaler_dict.get('param_dict', None)

    def unscale(x, std=False):
        if type(x) is list: x = np.array(x)
        x = x*2.*scaler_dict['scale'] - scaler_dict['scale']
        x = scaler_dict['scaler'].inverse_transform(x)
        if std is False:
            x = x*(np.array(param_dict['feature_max'])-np.array(param_dict['feature_min']))+\
              np.array(param_dict['feature_min'])
        else:
            x = x*(param_dict['feature_max']-param_dict['feature_min'])
        return x

    print np.shape(x_true), np.shape(x_pred_mean)

    
    if param_dict is not None and False:
        x_true      = unscale(x_true)
        x_pred_mean = unscale(x_pred_mean)
        x_pred_std  = unscale(x_pred_std, std=True)
    #--------------------------------------------------------------------

    
    nDim = len(x_true[0])
    if nDim > 6: nDim = 6
    
    fig = plt.figure(figsize=(6, 6))
    for k in xrange(nDim):
        ax = fig.add_subplot(nDim,1,k+1)
        #plt.rc('text', usetex=True) 
        ax.plot(np.array(x_true)[:,k], '-b', label='Inputs')
        ax.plot(np.array(x_pred_mean)[:,k], '-r', )#label=r'$\mu$')
        if x_pred_std is not None and len(x_pred_std)>0:
            ax.fill_between(range(len(x_pred_mean)),
                            np.array(x_pred_mean)[:,k]+np.array(x_pred_std)[:,k],
                            np.array(x_pred_mean)[:,k]-np.array(x_pred_std)[:,k],
                            facecolor='red', alpha=0.5, linewidth=0)
            ## plt.plot(np.array(x_pred_mean)[:,k]+np.array(x_pred_std)[:,k], '--r', )#label=r'$\mu\pm\sigma$')
            ## plt.plot(np.array(x_pred_mean)[:,k]-np.array(x_pred_std)[:,k], '--r')
        #plt.ylim([-0.1,1.1])

        if k==0:
            ax.set_ylabel('Sound'+'\n'+'Energy', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center')
            ## ax.set_ylabel('Audio [RMS]')
        elif k==1: 
            ax.set_ylabel('1st Joint'+'\n'+'Torque(Nm)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center')
            ## ax.set_ylabel('1st Joint \n effort')            
        elif k==2: 
            ax.set_ylabel('Accumulated'+'\n'+'Force'+'\n'+'on Spoon(N)',
                          rotation='horizontal', verticalalignment='center',
                          horizontalalignment='center')
            ## ax.set_ylabel('Accumulated \n force [N]')
        elif k==3: 
            ax.set_ylabel('Spoon-Mouth'+'\n'+'Distance(m)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center')
            ## ax.set_ylabel('Distance [m]')
            
        ax.yaxis.set_label_coords(-0.2,0.5)            
        #ax.set_ylabel(param_dict['feature_names'][k])

        ax.locator_params(axis='y', nbins=3)
        if k < nDim-1: ax.tick_params(axis='x', bottom='off', labelbottom='off')

    if param_dict is not None:
        x_tick = [param_dict['timeList'][0],
                  (param_dict['timeList'][-1]-param_dict['timeList'][0])/2.0,
                  param_dict['timeList'][-1]]
        ax.set_xticks(np.linspace(0, len(x_pred_mean), len(x_tick)))        
        ax.set_xticklabels(x_tick)
        ax.set_xlabel('Time [s]', fontsize=18)
        fig.subplots_adjust(left=0.25) 

    if save_pdf or True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        fig.savefig('test.eps')
        os.system('cp test.p* ~/Dropbox/HRL/')        
        os.system('cp test.e* ~/Dropbox/HRL/')        
    #else:
    plt.show()

    #ut.get_keystroke('Hit a key to proceed next')  


def graph_data_score(data, score, scaler_dict=None, save_pdf=False, **kwargs):
    '''
    x_true: timesteps x dim
    '''

    # visualization
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import gridspec
    from matplotlib import rc
    import matplotlib.patches as patches
    import itertools
    colors = itertools.cycle(['g', 'm', 'c', 'k', 'y', 'r', 'b', ])
    shapes = itertools.cycle(['x','v', 'o', '+'])

    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.rcParams['ps.fonttype'] = 42
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #rc('text', usetex=True)
    matplotlib.rcParams['text.usetex'] = True

    x_true, x_pred_mean, x_pred_std = data
    s, s_pred_mean, s_pred_bnd = score
    nVizData = kwargs.get('max_viz_data', 4)    

    # unscale
    if scaler_dict is None: param_dict = None
    else: param_dict = scaler_dict.get('param_dict', None)

    def unscale(x, std=False):
        if type(x) is list: x = np.array(x)
        x = x*2.*scaler_dict['scale'] - scaler_dict['scale']
        x = scaler_dict['scaler'].inverse_transform(x)
        if std is False:
            x = x*(np.array(param_dict['feature_max'])-np.array(param_dict['feature_min']))+\
              np.array(param_dict['feature_min'])
        else:
            x = x*(np.array(param_dict['feature_max'])-np.array(param_dict['feature_min']))
        return x

    print np.shape(x_true), np.shape(x_pred_mean)

    
    if param_dict is not None:
        x_true      = unscale(x_true[:])
        x_pred_mean = unscale(x_pred_mean[:])
        x_pred_std  = unscale(x_pred_std[:], std=True)
    #--------------------------------------------------------------------

    # Save Data
    prefix   = kwargs.get('prefix', '')
    sd = {}
    sd['nDim']   = len(x_true[0])
    sd['s']      = s
    sd['s_pred_mean'] = s_pred_mean
    sd['s_pred_bnd']  = s_pred_bnd
    sd['x_true'] = x_true
    sd['x_pred_mean'] = x_pred_mean
    sd['x_pred_std']  = x_pred_std
    sd['param_dict']  = param_dict
    save_pkl = os.path.join('./'+prefix+'_data_score.pkl')
    ut.save_pickle(sd, save_pkl)

    
    #--------------------------------------------------------------------
    nDim = len(x_true[0])
    if nDim > nVizData: nDim = nVizData
    
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1,1,1,1,0.4,2]) 

    # Data visualization -------------------------------------------------
    for k in xrange(nDim):
        ax = fig.add_subplot(gs[k])
        #plt.rc('text', usetex=True) 
        ax.plot(np.array(x_true)[:,k], '-b', label='Inputs')
        ax.plot(np.array(x_pred_mean)[:,k], '-r', )#label=r'$\mu$')
        if x_pred_std is not None and len(x_pred_std)>0:
            if k==1: mc = 0.1
            else: mc = 1.
            ax.fill_between(range(len(x_pred_mean)),
                            np.array(x_pred_mean)[:,k]+mc*np.array(x_pred_std)[:,k],
                            np.array(x_pred_mean)[:,k]-mc*np.array(x_pred_std)[:,k],
                            facecolor='red', alpha=0.3, linewidth=0)
        if k==0:
            ax.set_ylabel('Sound'+'\n'+'Energy', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=12)
            ## ax.set_ylabel('Audio [RMS]')
        elif k==1: 
            ax.set_ylabel('1st Joint'+'\n'+'Torque(Nm)', rotation='horizontal',
                          verticalalignment='center',
                         horizontalalignment='center', fontsize=12)
            ## ax.set_ylabel('1st Joint \n effort')            
        elif k==2: 
            ax.set_ylabel('Accumulated'+'\n'+'Force'+'\n'+'on Spoon(N)',
                          rotation='horizontal', verticalalignment='center',
                          horizontalalignment='center', fontsize=12)
            ## ax.set_ylabel('Accumulated \n force [N]')
        elif k==3: 
            ax.set_ylabel('Spoon-Mouth'+'\n'+'Distance(m)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=12)
            ## ax.set_ylabel('Distance [m]')
            
        ax.yaxis.set_label_coords(-0.22,0.5)            
        #ax.set_ylabel(param_dict['feature_names'][k])

        ax.locator_params(axis='y', nbins=3)
        #if k < nDim-1:
        ax.tick_params(axis='x', bottom='off', labelbottom='off')
        ax.set_xlim([0, len(x_pred_mean)])


    # Score visualization ------------------------------------------------------
    ax2 = fig.add_subplot(gs[k+1+1])
    plt.plot(s, '-b')
    plt.plot(s_pred_mean, '-r')
    plt.plot(s_pred_bnd, ':r') 
    ax2.set_ylabel('Anomaly \n Score',
                  rotation='horizontal', verticalalignment='center',
                  horizontalalignment='center', fontsize=12)
    ax2.yaxis.set_label_coords(-0.22,0.5)            
    ax2.locator_params(axis='y', nbins=3)
    ax2.set_ylim([-1, 5])
    
    ax2.set_xlim([0, len(s_pred_bnd)])

    anomaly = [True, ]
    for i in range(len(s)):
        if s[i]-s_pred_bnd[i]>0:
            ax2.add_patch( patches.Rectangle((i,-1),1,6, facecolor='peru', edgecolor='none' ))

    # --------------------------------------------------------------------------

            
    ax = fig.add_subplot(gs[0])
    axbox = ax.get_position()
    import matplotlib.lines as mlines
    blue_line = mlines.Line2D([], [], color='blue', alpha=0.5, markersize=30, label='Observations')
    red_line  = mlines.Line2D([], [], color='red', markersize=15,
                              label=r'Predicted distributions $\mu_{\bf x} \pm \sigma_{\bf x}$')

    handles = [blue_line,red_line]
    labels = [h.get_label() for h in handles]
    lg1 = plt.legend(handles=handles, labels=labels, loc=(axbox.x0-0.4, axbox.y0+0.5), #loc='upper center',
               ncol=2, shadow=False, fancybox=False, edgecolor='k', prop={'size': 12})

    blue_line = mlines.Line2D([], [], color='blue', alpha=0.5, markersize=30, label='Current')
    red_line  = mlines.Line2D([], [], color='red', markersize=15, label=r'Expected')
    red_dash_line  = mlines.Line2D([], [], color='red', ls=':', markersize=15, label=r'Threshold')
    handles = [blue_line,red_line, red_dash_line]
    labels = [h.get_label() for h in handles]
    axbox = ax2.get_position()
    lg2 = plt.legend(handles=handles, labels=labels, ncol=3, loc=(axbox.x0-0.28, axbox.y0-4.4),
                     shadow=False, fancybox=False, edgecolor='k', prop={'size': 12})
    ax.add_artist(lg1)
    ax.add_artist(lg2)


    if param_dict is not None:
        x_tick = [0,
                  (param_dict['timeList'][-1]-0)/2.0,
                  param_dict['timeList'][-1]]
        ax2.set_xticks(np.linspace(0, len(x_pred_mean), len(x_tick)))        
        ax2.set_xticklabels(x_tick)
        ax2.set_xlabel('Time [s]', fontsize=16)
        fig.subplots_adjust(left=0.28) 


    if save_pdf:
        prefix   = kwargs.get('prefix', '')
        fig.savefig(prefix+'data_score.pdf')
        fig.savefig(prefix+'data_score.png')
        fig.savefig(prefix+'data_score.eps')
        #os.system('mv *data_score.* ~/Dropbox/HRL/')        
        #ut.get_keystroke('Hit a key to proceed next')  
    else:
        plt.show()




def graph_latent_space(normalTestData, abnormalTestData, enc_z, timesteps=1, batch_size=None,
                       method='lstm_vae', save_pdf=False):

    print "latent variable visualization"
    if method == 'lstm_vae_offline':
        z_mean_n = enc_z_mean.predict(normalTestData)
        z_mean_a = enc_z_mean.predict(abnormalTestData)
        viz_latent_space(z_mean_n, z_mean_a)
    else:
        tgt_idx = 10
        
        #if batch_size is not None:
        z_mean_n = []
        z_mean_n_s = []
        z_mean_n_e = []
        for i in xrange(len(normalTestData)):

            x = normalTestData[i:i+1]
            for j in xrange(batch_size-1):
                x = np.vstack([x,normalTestData[i:i+1]])            

            z_mean=[]
            for j in xrange(len(x[0])-timesteps+1):
                if (method.find('lstm_vae_custom') >=0 or method.find('lstm_dvae_custom') >=0\
                    or method.find('phase')>=0) and method.find('pred')<0:
                    x_in = np.concatenate((x[:,j:j+timesteps],
                                           np.zeros((len(x), timesteps,1))), axis=-1)
                elif method.find('lstm_dvae_pred') >=0:
                    x_in = np.concatenate((x[:,j:j+timesteps],
                                           np.zeros((len(x), timesteps,1)),
                                           x[:,j:j+timesteps]), axis=-1)
                else:
                    x_in = x[:,j:j+timesteps]
                z = enc_z.predict(x_in, batch_size=batch_size)

                z_mean.append( z[0] )
                if j==0: z_mean_n_s.append(z[0])
                if j==len(x[0])-timesteps: z_mean_n_e.append(z[0])
                    
            z_mean_n.append(z_mean)

        z_mean_a = []
        z_mean_a_s = []
        z_mean_a_e = []
        for i in xrange(len(abnormalTestData)):

            x = abnormalTestData[i:i+1]
            for j in xrange(batch_size-1):
                x = np.vstack([x,abnormalTestData[i:i+1]])            

            z_mean=[]
            for j in xrange(len(x[0])-timesteps+1):
                if (method.find('lstm_vae_custom')>=0 or method.find('lstm_dvae_custom')>=0\
                    or method.find('phase')>=0) and method.find('pred')<0:
                    x_in = np.concatenate((x[:,j:j+timesteps],
                                           np.zeros((len(x), timesteps,1))), axis=-1)
                elif method.find('lstm_dvae_pred')>=0:
                    x_in = np.concatenate((x[:,j:j+timesteps],
                                           np.zeros((len(x), timesteps,1)),
                                           x[:,j:j+timesteps]), axis=-1)
                else:
                    x_in = x[:,j:j+timesteps]                    
                z = enc_z.predict(x_in, batch_size=batch_size)
                z_mean.append( z[0] )
                if j==0: z_mean_a_s.append(z[0])
                if j==len(x[0])-timesteps: z_mean_a_e.append(z[0])
            z_mean_a.append(z_mean)

        

        viz_latent_space(z_mean_n, z_mean_a, z_n_se=(z_mean_n_s, z_mean_n_e), save_pdf=save_pdf)
    


def viz_latent_space(z_n, z_a=None, z_n_se=None, save_pdf=False, **kwargs):
    '''
    z_n: latent variable from normal data
    z_n: latent variable from abnormal data
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if type(z_n) is list: z_n = np.array(z_n)
    if type(z_a) is list: z_a = np.array(z_a)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(6, 6))
    if np.shape(z_n)[-1]>2:
        print np.shape(z_n), np.shape(z_a)
        z_dim = np.shape(z_n)[-1]
        n_z_n   = len(z_n)
        n_z_n_l = len(z_n[0])
        n_z_a   = len(z_a)
        n_z_a_l = len(z_a[0])
        
        from sklearn.decomposition import KernelPCA
        ml = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=False, \
                       gamma=0.01)
        z_n = ml.fit_transform(z_n.reshape(-1, z_dim))
        z_a = ml.transform(z_a.reshape(-1, z_dim))                
        ## ax = fig.add_subplot(111, projection='3d')
        print np.shape(z_n), np.shape(z_a)
        z_n = z_n.reshape(n_z_n, n_z_n_l, 2)
        z_a = z_a.reshape(n_z_a, n_z_a_l, 2)
        
    s = 121    
    
    if z_a is not None:        
        for z in z_a:
            if np.shape(z)[-1] == 2:
                plt.scatter(z[:,0], z[:,1], color='r', s=0.5*s, marker='^', alpha=.4, label='Anomalous')
            else:
                ax.scatter(z[:,0], z[:,1], z[:,2], color='r', s=0.5*s, marker='^', alpha=.4, label='Anomalous')

    for z in z_n:
        if np.shape(z)[-1] == 2:
            plt.scatter(z[:,0], z[:,1], color='b', s=0.5*s, alpha=.4, label='Non-anomalous')
        else:
            ax.scatter(z[:,0], z[:,1], z[:,2], color='b', s=0.5*s, alpha=.4, label='Non-anomalous') 

    if z_n_se is not None:
        z_n_s, z_n_e = z_n_se
        if np.shape(z_n_s)[-1]>2:
            z_n_s = ml.transform(z_n_s)
            z_n_e = ml.transform(z_n_e)
        
        if np.shape(z_n_s)[-1] == 2:
            plt.scatter(np.array(z_n_s)[:,0], np.array(z_n_s)[:,1], color='g', s=1.*s, marker='x')
            plt.scatter(np.array(z_n_e)[:,0], np.array(z_n_e)[:,1], color='y', s=1.*s, marker='x')
        else:
            ax.scatter(np.array(z_n_s)[:,0], np.array(z_n_s)[:,1], np.array(z_n_s)[:,2],
                        color='g', s=1.*s, marker='x')
            ax.scatter(np.array(z_n_e)[:,0], np.array(z_n_e)[:,1], np.array(z_n_e)[:,2],
                        color='k', s=1.*s, marker='x')

    ## if kwargs.get('z_mean_tgt', None) is not None:
    ##     z_mean_tgt = kwargs['z_mean_tgt']
    ##     if np.shape(z_mean_tgt)[-1]>2:            
    ##         z_mean_tgt = ml.transform(z_mean_tgt)
    ##     print np.shape(z_mean_tgt)
    ##     plt.plot(z_mean_tgt[:,0], z_mean_tgt[:,1], '-k', marker='o', ms=5, lw=2)
        

    ## ax = plt.gca()
    ## ax.axes.get_xaxis().set_visible(False)
    ## ax.axes.get_yaxis().set_visible(False)
    #plt.legend(handles=[ax1, ax2], loc=3, ncol=2)

    ## import matplotlib.cm as cm
    ## colors = iter(cm.rainbow(np.linspace(0, 1, np.shape(z_n)[-2])))
    ## for i in range(len(z_n[0])): 
    ##     plt.scatter(z_n[:,i,0], z_n[:,i,1], color=next(colors), s=0.5*s, alpha=.4, label='Non-anomalous')
        
    ## sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=1))
    ## sm._A = []
    ## cbar = plt.colorbar(sm)
    ## cbar.set_ticks([0,1])
    ## cbar.set_ticklabels(['Start','End'])

    if save_pdf:
        fig.savefig('latent_space.pdf')
        fig.savefig('latent_space.png')
        fig.savefig('latent_space.eps')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()



def graph_score_distribution(scores_n, scores_a, param_dict, save_pdf=False):

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    import matplotlib.lines as mlines
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ## from mpl_toolkits.mplot3d import Axes3D
    ## ax = fig.add_subplot(111, projection='3d')
    ## for i, s in enumerate(scores_tr_n):
    ##     ax.scatter(zs_tr_n[i,:,0], zs_tr_n[i,:,1], scores_tr_n[i,:,0], c='g', marker='o')
    ## for i, s in enumerate(scores_te_n):
    ##     ax.scatter(zs_te_n[i,:,0], zs_te_n[i,:,1], scores_te_n[i,:,0], c='b', marker='x')
    ## for i, s in enumerate(scores_te_a):
    ##     ax.scatter(zs_te_a[i,:,0], zs_te_a[i,:,1], scores_te_a[i,:,0], c='r', marker='^')
    #for i, s in enumerate(scores_tr_n):
    #    plt.plot(s, '-g')

    #for i, s in enumerate(scores_n):
    #    if i==0: plt.plot(s, '--b', label='Non-anomalous data')
    #    else: plt.plot(s, '--b')
    ## for i, s in enumerate(scores_a):
    ##     if i==0: h1 = plt.plot(s, '-r', alpha=0.5, label='Anomalous data')
    ##     else: plt.plot(s, '-r', alpha=0.5)

    # ---------------------------------------------------------------
    s_ab_mu  = np.mean(scores_a, axis=0)[:,0]
    s_ab_std = np.std(scores_a, axis=0)[:,0]
    print np.shape(scores_a), np.shape(s_ab_mu), np.shape(s_ab_std)
    
    plt.plot(s_ab_mu, '-r', linewidth=2)
    h2 = ax.fill_between(range(len(scores_a[0])),
                         s_ab_mu+s_ab_std,
                         s_ab_mu-s_ab_std, alpha=0.5, facecolor='red', linewidth=0, label='Anomalous data')


    # ---------------------------------------------------------------
    s_nor_mu  = np.mean(scores_n, axis=0)[:,0]
    s_nor_std = np.std(scores_n, axis=0)[:,0]

    plt.plot(s_nor_mu, '-b', linewidth=2)
    h2 = ax.fill_between(range(len(scores_n[0])),
                         s_nor_mu+s_nor_std,
                         s_nor_mu-s_nor_std, alpha=0.5, facecolor='blue', linewidth=0, label='Non-anomalous data')
    # ---------------------------------------------------------------

    if param_dict is not None:
        x_tick = [param_dict['timeList'][0],
                  (param_dict['timeList'][-1]-param_dict['timeList'][0])/2.0,
                  param_dict['timeList'][-1]]
        ax.set_xticks(np.linspace(0, len(s_nor_mu), len(x_tick)))        
        ax.set_xticklabels(x_tick)
        ax.set_xlabel('Time [s]', fontsize=18)
        ## fig.subplots_adjust(left=0.25) 

    ax.set_ylim([-10,60])
    ax.set_ylabel('Anomaly Score', fontsize=18)


    blue_line = mlines.Line2D([], [], color='blue', alpha=0.5, markersize=30, label='Non-anomalous Data')
    green_line = mlines.Line2D([], [], color='red', markersize=15, label='Anomalous Data')

    handles = [blue_line,green_line]
    labels = [h.get_label() for h in handles]
    fig.legend(handles=handles, labels=labels, loc='upper center', ncol=2, shadow=True, fancybox=True)
    
    if save_pdf :
        fig.savefig('anomaly_score.pdf')
        fig.savefig('anomaly_score.png')
        fig.savefig('anomaly_score.eps')
        os.system('mv anomaly_score.* ~/Dropbox/HRL/')        
    else:
        plt.show()


    


def get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                 init_param_dict=None, init_raw_param_dict=None, id_num=0, raw_feature=False,
                 depth=False, ros_bag_image=False, kfold_split=True):
    from hrl_anomaly_detection import data_manager as dm
    
    ## Parameters # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    if raw_feature:
        AE_dict = param_dict['AE']
    
    #------------------------------------------
    if os.path.isdir(save_data_path) is False:
        os.system('mkdir -p '+save_data_path)

    if init_param_dict is None:
        crossVal_pkl = os.path.join(save_data_path, 'cv_'+task_name+'.pkl')
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        init_param_dict = d['param_dict']
        if raw_feature:
            init_raw_param_dict = d['raw_param_dict']

    #------------------------------------------
    crossVal_pkl = os.path.join(save_data_path, 'cv_td_'+task_name+'_'+str(id_num)+'.pkl')
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        td = ut.load_pickle(crossVal_pkl)
    else:
        if raw_feature is False:
            # Extract data from designated location
            td = dm.getDataLOPO(subjects, task_name, raw_data_path, save_data_path,\
                                downSampleSize=data_dict['downSampleSize'],\
                                init_param_dict=init_param_dict,\
                                handFeatures=data_dict['isolationFeatures'], \
                                cut_data=data_dict['cut_data'],\
                                data_renew=data_renew, max_time=data_dict['max_time'],
                                pkl_prefix='tgt_', depth=depth, id_num=id_num,\
                                ros_bag_image=ros_bag_image)


            if ros_bag_image is False:
                td['successData'], td['failureData'], td['success_files'], td['failure_files'], td['kFoldList'] \
                  = dm.LOPO_data_index(td['successDataList'], td['failureDataList'],\
                                       td['successFileList'], td['failureFileList'])
            else:
                (td['successData'], td['success_image_list'], td['success_d_image_list']),(td['failureData'], td['failure_image_list'], td['failure_d_image_list']), td['success_files'], td['failure_files'], td['kFoldList'] \
                  = dm.LOPO_data_index(td['successDataList'], td['failureDataList'],\
                                       td['successFileList'], td['failureFileList'],\
                                       success_image_list = td['success_image_list'], \
                                       failure_image_list = td['failure_image_list'],\
                                       success_d_image_list = td['success_d_image_list'], \
                                       failure_d_image_list = td['failure_d_image_list'])
                                       
        else:
            # Extract data from designated location
            td = dm.getRawDataLOPO(subjects, task_name, raw_data_path, save_data_path,\
                                   downSampleSize=data_dict['downSampleSize'],\
                                   init_param_dict=init_param_dict,\
                                   init_raw_param_dict=init_raw_param_dict,\
                                   handFeatures=data_dict['isolationFeatures'], \
                                   rawFeatures=AE_dict['rawFeatures'],\
                                   cut_data=data_dict['cut_data'],\
                                   data_renew=data_renew, max_time=data_dict['max_time'],
                                   pkl_prefix='tgt_', depth=depth, id_num=id_num,
                                   ros_bag_image=ros_bag_image)

            if kfold_split:
                # Get flatten array
                if ros_bag_image is False:
                    td['successData'], td['failureData'], td['success_files'], td['failure_files'],\
                    td['kFoldList'] \
                      = dm.LOPO_data_index(td['successRawDataList'], td['failureRawDataList'],\
                                           td['successFileList'], td['failureFileList'])
                else:                
                    (td['successData'], td['success_image_list'], td['success_d_image_list']),\
                      (td['failureData'], td['failure_image_list'], td['failure_d_image_list']), \
                      td['success_files'], td['failure_files'], td['kFoldList'] \
                      = dm.LOPO_data_index(td['successRawDataList'], td['failureRawDataList'],\
                                           td['successFileList'], td['failureFileList'],\
                                           success_image_list = td['success_image_list'], \
                                           failure_image_list = td['failure_image_list'],\
                                           success_d_image_list = td['success_d_image_list'], \
                                           failure_d_image_list = td['failure_d_image_list'])
            else:
                if ros_bag_image is False:
                    print "Not implemented"
                    sys.exit()
                else:
                    print "Not implemented"
                    sys.exit()
                    ## (td['successData'], td['success_image_list'], td['sucess_d_image_list']),\
                    ##   (td['failureData'], td['failure_image_list'], td['failure_d_image_list'])\
                    ##   td['success_files'], td['failure_files']
                    ##   = dm.flatten_LOPO_data(td['successRawDataList'], td['failureRawDataList'],\
                    ##                          td['successFileList'], td['failureFileList'],\
                    ##                          success_image_list = td['success_image_list'], \
                    ##                          failure_image_list = td['failure_image_list'],\
                    ##                          success_d_image_list = td['success_d_image_list'], \
                    ##                          failure_d_image_list = td['failure_d_image_list'])


        def get_label_from_filename(file_names):

            labels = []
            for f in file_names:
                labels.append( int(f.split('/')[-1].split('_')[0]) )

            return labels
        
        td['failure_labels'] = get_label_from_filename(td['failure_files'])


        ut.save_pickle(td, crossVal_pkl)
    
    if raw_feature is False:
        #------------------------------------------
        # select feature for detection
        feature_list = []
        for feature in param_dict['data_param']['handFeatures']:
            idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
            feature_list.append(idx)

        td['successData']    = np.array(td['successData'])[feature_list]
        td['failureData']    = np.array(td['failureData'])[feature_list]

    if ros_bag_image :
        print "complement data"
        print np.shape(td['successData']), np.shape(td['success_files']),\
        np.shape(td['success_image_list']), np.shape(td['success_d_image_list'])
        print np.shape(td['failureData']), np.shape(td['failure_files']),\
        np.shape(td['failure_image_list']), np.shape(td['failure_d_image_list'])

    return td


def get_scaled_data(normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, aligned=True,
                    scale=1.8):
    '''
    Remove outlier and scale into 0-1 range
    '''

    if aligned is False:
        # dim x sample x length => sample x length x dim
        normalTrainData   = np.swapaxes(normalTrainData, 0,1 )
        normalTrainData   = np.swapaxes(normalTrainData, 1,2 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 0,1 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 1,2 )

        # dim x sample x length => sample x length x dim
        normalTestData   = np.swapaxes(normalTestData, 0,1 )
        normalTestData   = np.swapaxes(normalTestData, 1,2 )
        abnormalTestData = np.swapaxes(abnormalTestData, 0,1 )
        abnormalTestData = np.swapaxes(abnormalTestData, 1,2 )
        

    # normalization => (sample x dim) ----------------------------------
    from sklearn import preprocessing
    ## scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = preprocessing.StandardScaler() 


    normalTrainData_scaled   = scaler.fit_transform(normalTrainData.reshape(-1,len(normalTrainData[0][0])))
    abnormalTrainData_scaled = scaler.transform(abnormalTrainData.reshape(-1,len(abnormalTrainData[0][0])))
    normalTestData_scaled    = scaler.transform(normalTestData.reshape(-1,len(normalTestData[0][0])))
    abnormalTestData_scaled  = scaler.transform(abnormalTestData.reshape(-1,len(abnormalTestData[0][0])))

    # rescale 95%of values into 0-1
    def rescaler(x, mean, var):
        
        max_val = scale #1.9#mean+3.0*np.sqrt(var)
        min_val = -scale #mean-3.0*np.sqrt(var)
        return (x-min_val)/( max_val-min_val )
    
    normalTrainData_scaled   = rescaler(normalTrainData_scaled, scaler.mean_, scaler.var_)
    abnormalTrainData_scaled = rescaler(abnormalTrainData_scaled, scaler.mean_, scaler.var_)
    normalTestData_scaled    = rescaler(normalTestData_scaled, scaler.mean_, scaler.var_)
    abnormalTestData_scaled  = rescaler(abnormalTestData_scaled, scaler.mean_, scaler.var_)

    # reshape
    normalTrainData   = normalTrainData_scaled.reshape(np.shape(normalTrainData))
    abnormalTrainData = abnormalTrainData_scaled.reshape(np.shape(abnormalTrainData))
    normalTestData   = normalTestData_scaled.reshape(np.shape(normalTestData))
    abnormalTestData  = abnormalTestData_scaled.reshape(np.shape(abnormalTestData))

    return normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, scaler


def get_scaled_data2(x, scaler, aligned=True, scale=1.8):
    if aligned is False:
        # dim x sample x length => sample x length x dim
        x = np.swapaxes(x, 0, 1 )
        x = np.swapaxes(x, 1, 2 )

    x_scaled = scaler.transform(x.reshape(-1,len(x[0][0])))

    # rescale 95%of values into 0-1
    def rescaler(X, mean, var):
        
        max_val = scale #1.9#mean+3.0*np.sqrt(var)
        min_val = -scale #mean-3.0*np.sqrt(var)
        return (X-min_val)/( max_val-min_val )

    x_scaled = rescaler(x_scaled, scaler.mean_, scaler.var_)
    x        = x_scaled.reshape(np.shape(x))

    return x

        

def get_ext_feeding_data(task_name, save_data_path, param_dict, d, raw_feature=False, ros_bag_image=False):
    if raw_feature is False: d['raw_param_dict'] = None
    
    subjects = ['Andrew', 'Britteney', 'Joshua', 'Jun', 'Kihan', 'Lichard', 'Shingshing', 'Sid', 'Tao']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/CORL2017/'
    td1 = get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                       init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
                       depth=True, id_num=1, raw_feature=raw_feature, ros_bag_image=ros_bag_image)

    subjects = ['ari', 'park', 'jina', 'linda', 'sai', 'hyun']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/ICRA2017/'
    td2 = get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                       init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
                       id_num=2, raw_feature=raw_feature, ros_bag_image=ros_bag_image)

    subjects = []
    for i in xrange(1,23):
        subjects.append('day'+str(i))
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/ICRA2018/'
    td3 = get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                       init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
                       id_num=3, raw_feature=raw_feature, ros_bag_image=ros_bag_image)

    return td1, td2, td3
    

def get_roc(tp_l, tn_l, fp_l, fn_l):

    tp_ll = []
    fp_ll = []
    tn_ll = []
    fn_ll = []  
    for i in xrange(len(tp_l)):
        tp_ll.append( tp_l[i])
        fp_ll.append( fp_l[i])
        tn_ll.append( tn_l[i])
        fn_ll.append( fn_l[i])



    tpr_l = np.array(tp_ll).astype(float)/(np.array(tp_ll).astype(float)+
                                           np.array(fn_ll).astype(float))*100.0
    fpr_l = np.array(fp_ll).astype(float)/(np.array(fp_ll).astype(float)+
                                           np.array(tn_ll).astype(float))*100.0

    from sklearn import metrics 
    return metrics.auc(fpr_l, tpr_l, True)

