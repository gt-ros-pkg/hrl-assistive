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

import numpy as np
import scipy
import sys

def get_reconstruction_err_prob(x, x_mean, x_std, alpha=1.0):
    '''
    Return minimum value for alpha x \sum P(x_i(t) ; mu, std) over time 
    '''
    if len(np.shape(x))>2: x = x[0]
    
    p_l     = []
    for k in xrange(len(x_mean[0])): # per dim
        p = []
        for l in xrange(len(x_mean)): # per length
            p.append(scipy.stats.norm(x_mean[l,k], x_std[l,k]).pdf(x[l,k])) # length

        p = [val if not np.isinf(val).any() and not np.isnan(val).any() and val > 0
             else 1e-50 for val in p]
        p_l.append(p) # dim x length

    # find min 
    ## return np.mean(alpha.dot( np.log(np.array(p_l)) )) 
    return [-np.amin(alpha.dot( np.log(np.array(p_l)) ))]


def get_reconstruction_err(x, x_mean, alpha=1.0):
    ''' Return minimum value for alpha \sum alpha * |x(i)-x_mean(i)|^2 over time '''
    if len(np.shape(x))>2:
        x = x[0]
        x_mean = x_mean[0]

    p_l = np.sum(alpha * ((x_mean-x)**2), axis=-1)
    # find min 
    return [np.amin(p_l)]


def get_reconstruction_err_lld(x, x_mean, alpha=1.0):
    '''
    Return reconstruction error likelihood
    Return minimum value for alpha \sum alpha * |x(i)-x_mean(i)|^2 over time 
    '''

    if len(np.shape(x))>2:
        x = x[0]

    p_l = np.sum(alpha * ((x_mean-x)**2), axis=-1)
        
    ## p_l     = []
    ## for k in xrange(len(x_mean[0])): # per dim
    ##     p = []
    ##     for l in xrange(len(x_mean)): # per length
            
    ##         p.append( alpha * (x_mean[l,k] - x[l,k])**2 ) # length

    ##     p_l.append(p) # dim x length
    ## p_l = np.sum(p_l, axis=0)

    # find min 
    return [np.amin(p_l)]


def get_reconstruction_err_vec(x, x_mean, alpha=1.0):
    '''
    Return error vector
    '''

    if len(np.shape(x))>2:
        x = x[0]
        
    return np.linalg.norm(x-x_mean, axis=-1)



def get_lower_bound(x, x_mean, x_std, z_std, enc_z_mean, enc_z_logvar, nDim, method=None, p=None,
                    alpha=None):
    '''
    No fixed batch
    x: batch x length x dim
    '''
    if len(np.shape(x))>2:
        if (method.find('lstm_vae_custom')>=0 or method.find('lstm_dvae_custom')>=0 or \
            method.find('phase')>=0) and method.find('pred')<0 and method.find('input')<0:
            x_in = np.concatenate((x, p), axis=-1)
        elif method.find('pred')>=0 or method.find('input')>=0:
            x_in = np.concatenate((x, p, x), axis=-1) 
        else:
            x_in = x
        
        batch_size = len(x)
        z_mean    = enc_z_mean.predict(x_in, batch_size=batch_size)[0]
        z_log_var = enc_z_logvar.predict(x_in, batch_size=batch_size)[0]
        x = x[0]
    else:
        z_mean    = enc_z_mean.predict(x)[0]
        z_log_var = enc_z_logvar.predict(x)[0]        

    if alpha is None: alpha = np.array([1.0]*nDim)

    # x: length x dim ?
    log_p_x_z = -0.5 * ( np.sum( (alpha*(x-x_mean)/x_std)**2, axis=-1) \
                         + float(nDim) * np.log(2.0*np.pi) + np.sum(np.log(x_std**2), axis=-1) )
    if len(np.shape(log_p_x_z))>1:
        xent_loss = np.mean(-log_p_x_z, axis=-1)
    else:
        xent_loss = -log_p_x_z

    ## if method == 'lstm_vae_custom' or method == 'lstm_vae_custom2':
    ##     p=0
    ##     if len(np.shape(z_log_var))>1:            
    ##         kl_loss = - 0.5 * np.sum(1 + z_log_var -np.log(z_std*z_std) - (z_mean-p)**2
    ##                                 - np.exp(z_log_var)/(z_std*z_std), axis=-1)
    ##     else:
    ##         kl_loss = - 0.5 * np.sum(1 + z_log_var -np.log(z_std*z_std) - (z_mean-p)**2
    ##                                 - np.exp(z_log_var)/(z_std*z_std))
    ##     kl_loss = 0
    ## else:
    ##     if len(np.shape(z_log_var))>1:
    ##         kl_loss = - 0.5 * np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var), axis=-1)
    ##     else:
    ##         kl_loss = - 0.5 * np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var))
    kl_loss = 0

    if len(xent_loss + kl_loss)>1:
        return [np.mean(xent_loss + kl_loss)], z_mean, z_log_var
    else:
        return xent_loss + kl_loss, z_mean, z_log_var


def get_err_vec(x, x_mean, x_std, nDim, method=None, p=None, alpha=None):
    '''
    No fixed batch
    x: batch x length x dim
    '''
    if len(np.shape(x))>2:
        if (method.find('lstm_vae_custom')>=0 or method.find('lstm_dvae_custom')>=0 or \
            method.find('phase')>=0) and method.find('pred')<0 and method.find('input')<0:
            x_in = np.concatenate((x, p), axis=-1)
        elif method.find('pred')>=0 or method.find('input')>=0:
            x_in = np.concatenate((x, p, x), axis=-1) 
        else:
            x_in = x
        
        batch_size = len(x)
        x = x[0]

    if alpha is None: alpha = np.array([1.0]*nDim)

    # x: length x dim ?
    log_p_x_z = -0.5 * ( (alpha*(x-x_mean)/x_std)**2 \
                         + float(nDim) * np.log(2.0*np.pi) + np.log(x_std**2) )
    ## log_p_x_z = (x-x_mean)**2
        
    return log_p_x_z
