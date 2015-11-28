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

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np



def plot_time_distribution(ax, data_seq, time_seq, x_label=None, y_label=None, title=None, y_lim=None):

    ax.plot(data_seq)

    ## print np.shape(data_seq), np.shape(time_seq)   
    ## ax.plot(time_seq, data_seq)
    
    ## x_tick = np.arange(0, x_range[-1], 2.0)
    ## ax.set_xticks(np.linspace(0, len(image[0]), len(x_tick)))        
    ## ax.set_xticklabels(x_tick)

    if y_lim is not None: ax.set_ylim(y_lim)
    
    if title is not None: ax.set_title(title)
    if x_label is not None: ax.set_xlabel(x_label)
    if y_label is not None: ax.set_ylabel(y_label)

def plot_space_time_distribution(ax, image, x_range, y_range, x_label=None, y_label=None, title=None):
    ax.imshow(image, aspect='auto', origin='lower', interpolation='none')
    y_tick = np.arange(y_range[0], y_range[-1]+0.01, 30)
    ax.set_yticks(np.linspace(0, len(image), len(y_tick)))
    ax.set_yticklabels(y_tick)
    x_tick = np.arange(0, x_range[-1], 5.0)
    ax.set_xticks(np.linspace(0, len(image[0]), len(x_tick)))        
    ax.set_xticklabels(x_tick)
    
    if title is not None: ax.set_title(title)
    if x_label is not None: ax.set_xlabel(x_label)
    if y_label is not None: ax.set_ylabel(y_label)
