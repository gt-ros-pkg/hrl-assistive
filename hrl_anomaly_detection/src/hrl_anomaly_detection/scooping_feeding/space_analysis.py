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

# image
from astropy.convolution.kernels import CustomKernel
from astropy.convolution import convolve, convolve_fft

def time_correlation(subject_names, task_name, raw_data_path, processed_data_path, rf_center, local_range, \
                     nSet=1, downSampleSize=200, success_viz=True, failure_viz=False, \
                     save_pdf=False, \
                     feature_list=['crossmodal_targetRelativeDist'], data_renew=False):

    if success_viz or failure_viz: bPlot = True
    else: bPlot = False

    data_pkl = os.path.join(processed_data_path, task_name+'_test.pkl')
    if os.path.isfile(data_pkl) and data_renew == False:
        data_dict = ut.load_pickle(data_pkl)

        fileNameList     = data_dict['fileNameList']
        # Audio
        audioTimesList   = data_dict['audioTimesList']
        audioPowerList   = data_dict['audioPowerList']
        min_audio_power  = data_dict['min_audio_power']

        # Fabric force
        fabricTimesList  = data_dict['fabricTimesList']
        fabricValueList  = data_dict['fabricValueList']

        # Vision change
        visionTimesList         = data_dict['visionChangeTimesList']
        visionChangeMagList     = data_dict['visionChangeMagList']
        
    else:
        success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

        #-------------------------------- Success -----------------------------------
        success_data_pkl = os.path.join(processed_data_path, subject+'_'+task+'_success')
        raw_data_dict, _ = loadData(success_list, isTrainingData=False,
                                    downSampleSize=downSampleSize,\
                                    local_range=local_range, rf_center=rf_center,\
                                    renew=data_renew, save_pkl=success_data_pkl)

        fileNameList = raw_data_dict['fileNameList']
        # Audio
        audioTimesList   = raw_data_dict['audioTimesList']
        audioPowerList   = raw_data_dict['audioPowerList']

        # Fabric force
        fabricTimesList  = raw_data_dict['fabricTimesList']
        fabricValueList  = raw_data_dict['fabricValueList']

        visionTimesList         = raw_data_dict['visionChangeTimesList']
        visionChangeMagList     = raw_data_dict['visionChangeMagList']

        ## min_audio_power = np.mean( [np.mean(x) for x in audioPowerList] )
        min_audio_power = np.min( [np.max(x) for x in audioPowerList] )

        #-------------------------------- Failure -----------------------------------
        failure_data_pkl = os.path.join(processed_data_path, subject+'_'+task+'_failure')
        raw_data_dict, _ = loadData(failure_list, isTrainingData=False,
                                    downSampleSize=downSampleSize,\
                                    local_range=local_range, rf_center=rf_center,\
                                    renew=data_renew, save_pkl=failure_data_pkl)

        data_dict = {}

        fileNameList += raw_data_dict['fileNameList']
        data_dict['fileNameList'] = fileNameList
        # Audio
        audioTimesList += raw_data_dict['audioTimesList']
        audioPowerList += raw_data_dict['audioPowerList']
        data_dict['audioTimesList'] = audioTimesList
        data_dict['audioPowerList'] = audioPowerList

        # Fabric force
        fabricTimesList += raw_data_dict['fabricTimesList']
        fabricValueList += raw_data_dict['fabricValueList']
        data_dict['fabricTimesList']  = fabricTimesList
        data_dict['fabricValueList']  = fabricValueList

        visionTimesList += raw_data_dict['visionChangeTimesList']
        visionChangeMagList += raw_data_dict['visionChangeMagList']
        data_dict['visionChangeTimesList']   = visionTimesList    
        data_dict['visionChangeMagList']     = visionChangeMagList

        data_dict['min_audio_power'] = min_audio_power
        ut.save_pickle(data_dict, data_pkl)


    max_audio_power    = 5000 #np.median( [np.max(x) for x in audioPowerList] )
    max_fabric_value   = 3.0
    max_vision_change  = 80 #? 
    
    max_audio_delay    = 1.0
    max_fabric_delay   = 0.1
    max_vision_delay   = 1.0

    nSample      = len(audioTimesList)    
    label_list = ['no_failure', 'falling', 'slip', 'touch']
    cause_list   = []

    x_cors = []
    x_diffs= []
    y_true = []

    for i in xrange(nSample):

        # time
        max_time1 = np.max(audioTimesList[i])
        max_time2 = np.max(fabricTimesList[i])
        max_time3 = np.max(visionTimesList[i])
        max_time  = min([max_time1, max_time2, max_time3]) # min of max time
        new_times     = np.linspace(0.0, max_time, downSampleSize)
        time_interval = new_times[1]-new_times[0]

        if 'success' in fileNameList[i]: cause = 'no_failure'
        else:
            cause = os.path.split(fileNameList[i])[-1].split('.pkl')[0].split('failure_')[-1]

        # -------------------------------------------------------------
        # ------------------- Auditory --------------------------------
        audioTime    = audioTimesList[i]
        audioPower   = audioPowerList[i]
    
        discrete_time_array = hdl.discretization_array(audioTime, [0.0, max_time], len(new_times))
        audio_raw = np.zeros(np.shape(discrete_time_array))

        last_time_idx = -1
        for j, time_idx in enumerate(discrete_time_array):
            if time_idx < 0: time_idx = 0
            if time_idx >= len(new_times): time_idx=len(new_times)-1
                
            if audioPower[j] > max_audio_power:
                audio_raw[time_idx] = 1.0
            elif audioPower[j] > min_audio_power:
                s = ((audioPower[j]-min_audio_power)/(max_audio_power-min_audio_power)) #**2
                if audio_raw[time_idx] < s: audio_raw[time_idx] = s

            last_time_idx = time_idx

        # -------------------------------------------------------------
        # Convoluted data
        time_1D_kernel = get_time_kernel(max_audio_delay, time_interval)
        time_1D_kernel = CustomKernel(time_1D_kernel)

        # For color scale
        audio_min = np.amin(audio_raw)
        audio_max = np.amax(audio_raw)
        audio_smooth = convolve(audio_raw, time_1D_kernel, boundary='extend')
        ## if audio_min != audio_max:
        ##     audio_smooth = (audio_smooth - audio_min)/(audio_max-audio_min)
        ## else:
        ##     audio_smooth = audio_smooth - audio_min

        # -------------------------------------------------------------
        # ------------------- Fabric Force ----------------------------
        fabricTime   = fabricTimesList[i]
        fabricValue  = fabricValueList[i]

        discrete_time_array = hdl.discretization_array(fabricTime, [0.0, max_time], len(new_times))
        fabric_raw = np.zeros(np.shape(discrete_time_array))

        last_time_idx = -1
        for j, time_idx in enumerate(discrete_time_array):
            if time_idx < 0: time_idx = 0
            if time_idx >= len(new_times): time_idx=len(new_times)-1

            f = [fabricValue[0][j], fabricValue[1][j], fabricValue[2][j]]
            mag = 0.0
            for k in xrange(len(f[0])):
                s = np.linalg.norm(np.array([f[0][k],f[1][k],f[2][k]]))

                if s > max_fabric_value: mag = 1.0
                elif s > 0.0 or s/max_fabric_value>mag: mag = (s-0.0)/(max_fabric_value-0.0)

            #
            if last_time_idx == time_idx:
                if fabric_raw[time_idx] < mag: fabric_raw[time_idx] = mag
            else:
                fabric_raw[time_idx] = mag

            last_time_idx = time_idx

        # -------------------------------------------------------------
        # Convoluted data
        time_1D_kernel = get_time_kernel(max_fabric_delay, time_interval)
        time_1D_kernel = CustomKernel(time_1D_kernel)
                
        # For color scale
        fabric_min = np.amin(fabric_raw)
        fabric_max = np.amax(fabric_raw)
        fabric_smooth = convolve(fabric_raw, time_1D_kernel, boundary='extend')
        ## if fabric_min != fabric_max:
        ##     fabric_smooth = (fabric_smooth - fabric_min)/(fabric_max-fabric_min)
        ## else:
        ##     fabric_smooth = fabric_smooth - fabric_min

        # -------------------------------------------------------------
        # ------------------- Vision Change ---------------------------
        visionTime      = visionTimesList[i]
        visionChangeMag = visionChangeMagList[i]

        discrete_time_array = hdl.discretization_array(visionTime, [0.0, max_time], len(new_times))
        vision_raw          = np.zeros(np.shape(discrete_time_array))

        last_time_idx = -1
        for j, time_idx in enumerate(discrete_time_array):
            if time_idx < 0: time_idx = 0
            if time_idx >= len(new_times): time_idx=len(new_times)-1

            mag = visionChangeMag[j]
            if mag > max_vision_change: mag = 1.0
            elif mag > 0.0: mag = (mag-0.0)/(max_vision_change-0.0)

            #
            if last_time_idx == time_idx:
                if vision_raw[time_idx] < mag: vision_raw[time_idx] = mag
            else:
                vision_raw[time_idx] = mag

            last_time_idx = time_idx

        # -------------------------------------------------------------
        # Convoluted data
        time_1D_kernel = get_time_kernel(max_vision_delay, time_interval)
        time_1D_kernel = CustomKernel(time_1D_kernel)
                
        # For color scale
        vision_min = np.amin(vision_raw)
        vision_max = np.amax(vision_raw)
        vision_smooth = convolve(vision_raw, time_1D_kernel, boundary='extend')
        ## if vision_min != vision_max:
        ##     vision_smooth = (vision_smooth - vision_min)/(vision_max-vision_min)
        ## else:
        ##     vision_smooth = vision_smooth - vision_min

        # -------------------------------------------------------------
        #-----------------Multi modality ------------------------------
        
        pad = int(np.floor(4.0/time_interval))

        cor_seq1, time_diff1 = cross_1D_correlation(fabric_raw, vision_raw, pad)
        cor_seq2, time_diff2 = cross_1D_correlation(fabric_raw, audio_raw, pad)
        cor_seq3, time_diff3 = cross_1D_correlation(vision_raw, audio_raw, pad)

        # Normalization
        ## if np.amax(cor_seq1) > 1e-6: cor_seq1 /= np.amax(cor_seq1)
        ## if np.amax(cor_seq2) > 1e-6: cor_seq2 /= np.amax(cor_seq2)
        ## if np.amax(cor_seq3) > 1e-6: cor_seq3 /= np.amax(cor_seq3)

        # -------------------------------------------------------------
        # Visualization
        # -------------------------------------------------------------
        if bPlot:
            y_lim=[0,1.0]
            
            fig = plt.figure(figsize=(12,8))

            ax = fig.add_subplot(3,3,1)
            plot_time_distribution(ax, audio_raw, new_times, x_label='Time [sec]', title='Raw Audio Data',\
                                   y_lim=y_lim)
            ax = fig.add_subplot(3,3,2)        
            plot_time_distribution(ax, audio_smooth, new_times, x_label='Time [sec]', title='Smooth Audio Data',\
                                   y_lim=y_lim)
            
            ax = fig.add_subplot(3,3,4)
            plot_time_distribution(ax, fabric_raw, new_times, x_label='Time [sec]', title='Raw Fabric Skin Data',\
                                   y_lim=y_lim)
            ax = fig.add_subplot(3,3,5)
            plot_time_distribution(ax, fabric_smooth, new_times, x_label='Time [sec]', \
                                   title='Smooth Fabric Skin Data',\
                                   y_lim=y_lim)
            
            ax = fig.add_subplot(3,3,7)
            plot_time_distribution(ax, vision_raw, new_times, x_label='Time [sec]', title='Raw Vision Data',\
                                   y_lim=y_lim)
            ax = fig.add_subplot(3,3,8)
            plot_time_distribution(ax, vision_smooth, new_times, x_label='Time [sec]', \
                                   title='Smooth Vision Data',\
                                   y_lim=y_lim)

            ax = fig.add_subplot(3,3,3)
            plot_time_distribution(ax, cor_seq1, None, x_label='Time [sec]', title='Fabric-vision Correlation',\
                                   y_lim=[0,1])
            ax = fig.add_subplot(3,3,6)
            plot_time_distribution(ax, cor_seq2, None, x_label='Time [sec]', title='Fabric-audio Correlation',\
                                   y_lim=[0,1])
            ax = fig.add_subplot(3,3,9)
            plot_time_distribution(ax, cor_seq3, None, x_label='Time [sec]', title='Vision-audio Correlation',\
                                   y_lim=[0,1])


            plt.suptitle('Anomaly: '+cause, fontsize=20)                        
            plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=0.5)

        if bPlot:
            if save_pdf:
                fig.savefig('test.pdf')
                fig.savefig('test.png')
                os.system('cp test.p* ~/Dropbox/HRL/')        
            else:
                plt.show()

        # -------------------------------------------------------------
        # Classification
        # -------------------------------------------------------------

        x_cors.append([cor_seq1, cor_seq2, cor_seq3])
        x_diffs.append([time_diff1, time_diff2, time_diff3])
        cause_list.append(cause)

        for idx, anomaly in enumerate(label_list):
            if cause == anomaly: y_true.append(idx)

    # data preprocessing
    aXData = []
    chunks = []
    labels = []
    for i, x_cor in enumerate(x_cors):
        X = []
        Y = []
        ## print "image range: ", i , np.amax(image), np.amin(image)
        ## if np.amax(image) < 1e-6: continue

        for ii in xrange(len(x_cor[0])):
            ## if x_cor[0][ii] < 0.01 and x_cor[1][ii] < 0.01 and x_cor[2][ii] < 0.01:
            ##     continue
            X.append([ x_cor[0][ii],x_cor[1][ii],x_cor[2][ii] ])
            Y.append(y_true[i])

        if X==[]: continue
            
        aXData.append(X)
        chunks.append(cause_list[i])
        labels.append(y_true[i])
        
    data_set = create_mvpa_dataset(aXData, chunks, labels)

    # save data
    d = {}
    d['label_list'] = label_list
    d['mvpa_dataset'] = data_set
    ## d['c'] = c
    ut.save_pickle(d, 'st_svm.pkl')


                

def space_time_analysis(subject_names, task_name, raw_data_path, processed_data_path, \
                        nSet=1, downSampleSize=200, success_viz=True, failure_viz=False, \
                        save_pdf=False, data_renew=False):

    if success_viz or failure_viz: bPlot = True
    else: bPlot = False
    
    data_pkl = os.path.join(processed_data_path, task_name+'_test.pkl')
    if os.path.isfile(data_pkl) and data_renew == False:
        data_dict = ut.load_pickle(data_pkl)

        fileNameList     = data_dict['fileNameList']
        # Audio
        audioTimesList   = data_dict['audioTimesList']
        audioAzimuthList = data_dict['audioAzimuthList']
        audioPowerList   = data_dict['audioPowerList']

        # Fabric force
        fabricTimesList  = data_dict['fabricTimesList']
        fabricCenterList = data_dict['fabricCenterList']
        fabricNormalList = data_dict['fabricNormalList']
        fabricValueList  = data_dict['fabricValueList']

        # Vision change
        visionTimesList         = data_dict['visionChangeTimesList']
        visionChangeCentersList = data_dict['visionChangeCentersList']
        visionChangeMagList     = data_dict['visionChangeMagList']
        
        min_audio_power  = data_dict['min_audio_power']
    else:
        success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

        #-------------------------------- Success -----------------------------------
        success_data_pkl     = os.path.join(processed_data_path, subject+'_'+task+'_success')
        raw_data_dict, _ = loadData(success_list, isTrainingData=False,
                                    downSampleSize=downSampleSize,\
                                    global_data=True,\
                                    renew=data_renew, save_pkl=success_data_pkl)

        # Audio
        audioTimesList   = raw_data_dict['audioTimesList']
        audioAzimuthList = raw_data_dict['audioAzimuthList']
        audioPowerList   = raw_data_dict['audioPowerList']

        # Fabric force
        fabricTimesList  = raw_data_dict['fabricTimesList']
        fabricCenterList = raw_data_dict['fabricCenterList']
        fabricNormalList = raw_data_dict['fabricNormalList']
        fabricValueList  = raw_data_dict['fabricValueList']

        # Vision change
        visionTimesList         = raw_data_dict['visionChangeTimesList']
        visionChangeCentersList = raw_data_dict['visionChangeCentersList']
        visionChangeMagList     = raw_data_dict['visionChangeMagList']


        ## min_audio_power = np.mean( [np.mean(x) for x in audioPowerList] )
        min_audio_power = np.min( [np.max(x) for x in audioPowerList] )

        #-------------------------------- Failure -----------------------------------
        failure_data_pkl     = os.path.join(processed_data_path, subject+'_'+task+'_failure')
        raw_data_dict, _ = loadData(failure_list, isTrainingData=False,
                                    downSampleSize=downSampleSize,\
                                    global_data=True,\
                                    renew=data_renew, save_pkl=failure_data_pkl)

        data_dict = {}

        fileNameList += raw_data_dict['fileNameList']
        data_dict['fileNameList'] = fileNameList
        # Audio
        audioTimesList   += raw_data_dict['audioTimesList']
        audioAzimuthList += raw_data_dict['audioAzimuthList']
        audioPowerList   += raw_data_dict['audioPowerList']
        data_dict['audioTimesList']   = audioTimesList
        data_dict['audioAzimuthList'] = audioAzimuthList
        data_dict['audioPowerList']   = audioPowerList

        # Fabric force
        fabricTimesList += raw_data_dict['fabricTimesList']
        fabricCenterList+= raw_data_dict['fabricCenterList']
        fabricNormalList+= raw_data_dict['fabricNormalList']
        fabricValueList += raw_data_dict['fabricValueList']
        data_dict['fabricTimesList']  = fabricTimesList
        data_dict['fabricCenterList'] = fabricCenterList
        data_dict['fabricNormalList'] = fabricNormalList
        data_dict['fabricValueList']  = fabricValueList

        visionTimesList += raw_data_dict['visionChangeTimesList']
        visionChangeCentersList += raw_data_dict['visionChangeCentersList']
        visionChangeMagList += raw_data_dict['visionChangeMagList']
        data_dict['visionChangeTimesList']   = visionTimesList    
        data_dict['visionChangeCentersList'] = visionChangeCentersList
        data_dict['visionChangeMagList']     = visionChangeMagList

        data_dict['min_audio_power'] = min_audio_power
        ut.save_pickle(data_dict, data_pkl)


    nSample = len(audioTimesList)
    azimuth_interval = 2.0
    audioSpace = np.arange(-90, 90+0.01, azimuth_interval)

    max_audio_power    = 5000 #np.median( [np.max(x) for x in audioPowerList] )
    max_fabric_value   = 3.0
    max_vision_change  = 80 #?

    limit = [[0.25, 1.0], [-0.2, 1.0], [-0.5,0.5]]
    
    max_audio_azimuth = 15.0
    max_audio_delay   = 1.0
    max_fabric_offset = 10.0
    max_fabric_delay  = 1.0
    max_vision_offset = 0.05
    max_vision_delay  = 1.0
    label_list = ['no_failure', 'falling', 'slip', 'touch']

    X_images = []
    y_true   = []
    cause_list = []
    
    for i in xrange(nSample):

        # time
        max_time1 = np.max(audioTimesList[i])
        max_time2 = np.max(fabricTimesList[i])
        max_time3 = np.max(visionTimesList[i])
        max_time  = min([max_time1, max_time2, max_time3]) # min of max time
        new_times     = np.linspace(0.0, max_time, downSampleSize)
        time_interval = new_times[1]-new_times[0]

        if 'success' in fileNameList[i]: cause = 'no_failure'
        else:
            cause = os.path.split(fileNameList[i])[-1].split('.pkl')[0].split('failure_')[-1]
        
        # -------------------------------------------------------------
        # ------------------- Auditory --------------------------------
        audioTime    = audioTimesList[i]
        audioAzimuth = audioAzimuthList[i]
        audioPower   = audioPowerList[i]

        discrete_azimuth_array = hdl.discretization_array(audioAzimuth, [-90,90], len(audioSpace))
        discrete_time_array    = hdl.discretization_array(audioTime, [0.0, max_time], len(new_times))

        image = np.zeros((len(new_times),len(audioSpace)))
        last_time_idx = -1
        for j, time_idx in enumerate(discrete_time_array):
            if time_idx < 0: time_idx = 0
            if time_idx >= len(new_times): time_idx=len(new_times)-1
                
            s = np.zeros(len(audioSpace))
            if audioPower[j] > max_audio_power:
                s[discrete_azimuth_array[j]] = 1.0
            elif audioPower[j] > min_audio_power:
                try:
                    s[discrete_azimuth_array[j]] = ((audioPower[j]-min_audio_power)/
                                                    (max_audio_power-min_audio_power)) #**2
                except:
                    print "size error???"
                    print np.shape(audioPower), np.shape(discrete_time_array), j
                    print fileNameList[i]
                    ## sys.exit()

            #
            if last_time_idx == time_idx:
                for k in xrange(len(s)):
                    if image[time_idx,k] < s[k]: image[time_idx,k] = s[k]
            else:
                # image: (Ang, N)
                if len(np.shape(s))==1: s = np.array([s])
                image[time_idx,:] = s
            last_time_idx = time_idx

        audio_raw_image = image.T

        # -------------------------------------------------------------
        # Convoluted data
        ## g = Gaussian1DKernel(max_audio_kernel_y) # 8*std
        ## a = g.array
        gaussian_2D_kernel = get_space_time_kernel(max_audio_delay, max_audio_azimuth, \
                                                   time_interval, azimuth_interval)
        gaussian_2D_kernel = CustomKernel(gaussian_2D_kernel)

        # For color scale
        image_min = np.amin(audio_raw_image.flatten())
        image_max = np.amax(audio_raw_image.flatten())        
        audio_smooth_image = convolve(audio_raw_image, gaussian_2D_kernel, boundary='extend')
        if image_max != image_min:
            audio_smooth_image = (audio_smooth_image-image_min)/(image_max-image_min)#*image_max
        else:
            audio_smooth_image = (audio_smooth_image-image_min)
                
        # -------------------------------------------------------------
        ## Clustering
        ## audio_clustered_image, audio_label_list = \
        ##   space_time_clustering(audio_smooth_image, max_audio_delay, max_audio_azimuth, \
        ##                         azimuth_interval, time_interval, 4)

        # -------------------------------------------------------------
        # ------------------- Fabric Force ----------------------------
        fabricTime   = fabricTimesList[i]
        fabricCenter = fabricCenterList[i]
        fabricValue  = fabricValueList[i]

        ## discrete_azimuth_array = hdl.discretization_array(audioAzimuth, [-90,90], len(audioSpace))
        discrete_time_array    = hdl.discretization_array(fabricTime, [0.0, max_time], len(new_times))

        image = np.zeros((len(new_times),len(audioSpace)))
        last_time_idx = -1
        for j, time_idx in enumerate(discrete_time_array):
            if time_idx < 0: time_idx = 0
            if time_idx >= len(new_times): time_idx=len(new_times)-1
            s = np.zeros(len(audioSpace))

            # Estimate space
            xyz  = [fabricCenter[0][j], fabricCenter[1][j], fabricCenter[2][j]]
            fxyz = [fabricValue[0][j], fabricValue[1][j], fabricValue[2][j]] 
            for k in xrange(len(xyz[0])):
                if xyz[0][k]==0 and xyz[1][k]==0 and xyz[2][k]==0: continue
                y   = xyz[1][k]/np.linalg.norm( np.array([ xyz[0][k],xyz[1][k],xyz[2][k] ]) )
                ang = np.arcsin(y)*180.0/np.pi 
                mag = np.linalg.norm(np.array([fxyz[0][k],fxyz[1][k],fxyz[2][k]]))

                ang_idx = hdl.discretize_single(ang, [-90,90], len(audioSpace))
                if mag > max_fabric_value:
                    s[ang_idx] = 1.0
                elif mag > 0.0:
                    s[ang_idx] = ((mag-0.0)/(max_fabric_value-0.0))

            #
            if last_time_idx == time_idx:
                ## print "fabrkc: ", np.shape(image), np.shape(s), last_time_idx, time_idx
                for k in xrange(len(s)):
                    if image[time_idx,k] < s[k]: image[time_idx,k] = s[k]
            else:
                # image: (Ang, N)
                if len(np.shape(s))==1: s = np.array([s])
                ## print np.shape(image), time_idx, np.shape(image[time_idx,:]), np.shape(s)
                image[time_idx,:] = s

            # clustering label
            ## for k in xrange(len(z)):
            ##     if z[k,0] > 0.01: X.append([j,k]) #temp # N x Ang

            # For color scale
            ## image[0,0]=1.0
            last_time_idx = time_idx
            
        fabric_raw_image = image.T

        # -------------------------------------------------------------
        # Convoluted data
        gaussian_2D_kernel = get_space_time_kernel(max_fabric_delay, max_fabric_azimuth, \
                                                   time_interval, azimuth_interval)        
        gaussian_2D_kernel = CustomKernel(gaussian_2D_kernel)

        # For color scale
        image_min = np.amin(fabric_raw_image.flatten())
        image_max = np.amax(fabric_raw_image.flatten())        
        fabric_smooth_image = convolve(fabric_raw_image, gaussian_2D_kernel, boundary='extend')
        if image_max != image_min:
            fabric_smooth_image = (fabric_smooth_image-image_min)/(image_max-image_min)#*image_max
        else:
            fabric_smooth_image = (fabric_smooth_image-image_min)
        #image[0,0] = 1.0

        # -------------------------------------------------------------
        # Clustering
        ## fabric_clustered_image, fabric_label_list = space_time_clustering(fabric_smooth_image, \
        ##                                                                   max_fabric_delay, \
        ##                                                                   max_fabric_azimuth,\
        ##                                                                   azimuth_interval, time_interval, 4)
                                     
        # -------------------------------------------------------------
        #-----------------Multi modality ------------------------------
        x_pad = int(np.floor(60.0/azimuth_interval))
        y_pad = int(np.floor(4.0/time_interval))
        cor_image, azimuth_diff, time_diff = cross_correlation(fabric_smooth_image, audio_smooth_image, \
                                                               x_pad, y_pad)
        # Normalization
        if np.amax(cor_image) > 1e-6:
            cor_image /= np.amax(cor_image)
            
        # -------------------------------------------------------------
        # Visualization
        # -------------------------------------------------------------
        if bPlot:
            fig = plt.figure(figsize=(12,8))
            
            ax = fig.add_subplot(2,3,1)
            plot_space_time_distribution(ax, audio_raw_image, new_times, audioSpace, \
                                         x_label='Time [sec]', y_label='Azimuth [deg]', \
                                         title='Raw Audio Data')
            ax = fig.add_subplot(2,3,4)        
            plot_space_time_distribution(ax, audio_smooth_image, new_times, audioSpace, \
                                         x_label='Time [sec]', y_label='Azimuth [deg]', \
                                         title='Smooth Audio Data')
            ## ax = fig.add_subplot(3,3,7)
            ## plot_space_time_distribution(ax, audio_clustered_image, new_times, audioSpace, \
            ##                              x_label='Time [msec]', y_label='Azimuth [deg]', title='Auditory Map')
            ax = fig.add_subplot(2,3,2)
            plot_space_time_distribution(ax, fabric_raw_image, new_times, audioSpace, \
                                         x_label='Time [sec]', title='Raw Fabric Skin Data')
            ax = fig.add_subplot(2,3,5)
            plot_space_time_distribution(ax, fabric_smooth_image, new_times, audioSpace, \
                                         x_label='Time [sec]', title='Smooth Fabric Skin Data')
            ## ax = fig.add_subplot(3,3,8)
            ## plot_space_time_distribution(ax, fabric_clustered_image, new_times, audioSpace, \
            ##                              x_label='Time [msec]', title='Fabric Skin Map')

            ax = fig.add_subplot(2,3,6)
            ax.imshow(cor_image, aspect='auto', origin='lower', interpolation='none')
            ## plot_space_time_distribution(ax, cor_image, [0, y_pad*azimuth_interval], [0, x_pad*time_interval])

            y_tick = np.arange(-len(cor_image)/2*azimuth_interval, len(cor_image)/2*azimuth_interval, 15.0)
            ax.set_yticks(np.linspace(0, len(cor_image), len(y_tick)))        
            ax.set_yticklabels(y_tick)
            x_tick = np.arange(0, len(cor_image[0])*time_interval, 0.5)
            ax.set_xticks(np.linspace(0, len(cor_image[0]), len(x_tick)))        
            ax.set_xticklabels(x_tick)
            ax.set_title("Cross correlation")
            ax.set_xlabel("Time delay [sec]")
            ax.set_ylabel("Angular difference [degree]")

            ## ax = fig.add_subplot(3,3,6)
            ## image = audio_smooth_image * fabric_smooth_image
            ## plot_space_time_distribution(ax, image, new_times, audioSpace, \
            ##                              x_label='Time [msec]', title='Multimodal Map')
            
            plt.suptitle('Anomaly: '+cause, fontsize=20)                        
            plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=0.5)

        
        # -------------------------------------------------------------
        # Classification
        # -------------------------------------------------------------

        cause = cause.split('_')[0]
        X_images.append(cor_image)
        X_diff.append([azimuth_diff, time_diff])
        cause_list.append(cause)
        print "diff", azimuth_diff, time_diff, cause, os.path.split(fileNameList[i])[-1].split('.pkl')[0]

        for idx, anomaly in enumerate(label_list):
            if cause == anomaly: y_true.append(idx)

        if bPlot:
            if save_pdf:
                fig.savefig('test.pdf')
                fig.savefig('test.png')
                os.system('cp test.p* ~/Dropbox/HRL/')
                ut.get_keystroke('Hit a key to proceed next')
            else:
                plt.show()
                ## sys.exit()

    # data preprocessing
    ## c = []
    aXData = []
    chunks = []
    labels = []
    for i, image in enumerate(X_images):
        X = []
        Y = []
        print "image range: ", i , np.amax(image), np.amin(image)
        if np.amax(image) < 1e-6: continue
        for ix in xrange(len(image)):
            for iy in xrange(len(image[0])):
                if image[ix,iy] > 0.3:
                    X.append([float(ix), float(iy), float(image[ix,iy])])
                    Y.append(y_true[i])
                ## c.append(['ro' if y_true[i] == 0 else 'bx']*len(y_true[i]))
        aXData.append(X)
        chunks.append(cause_list[i])
        labels.append(y_true[i])
        

    data_set = create_mvpa_dataset(aXData, chunks, labels)

    # save data
    d = {}
    d['label_list'] = label_list
    d['mvpa_dataset'] = data_set
    ## d['c'] = c
    ut.save_pickle(d, 'st_svm.pkl')
    

def correlation_confusion_matrix(save_pdf=False, verbose=False):
    
    d          = ut.load_pickle('st_svm.pkl')
    dataSet    = d['mvpa_dataset']
    label_list = d['label_list']

    # leave-one-out data set
    splits = []
    for i in xrange(len(dataSet)):

        test_ids = Dataset.get_samples_by_attr(dataSet, 'id', i)
        test_dataSet = dataSet[test_ids]
        train_ids     = [val for val in dataSet.sa.id if val not in test_dataSet.sa.id]
        train_ids     = Dataset.get_samples_by_attr(dataSet, 'id', train_ids)
        train_dataSet = dataSet[train_ids]

        splits.append([train_dataSet, test_dataSet])

        ## print "test"
        ## space_time_anomaly_check_offline(0, train_dataSet, test_dataSet, label_list=label_list)
        ## sys.exit()

    if verbose: print "Start to parallel job"
    r = Parallel(n_jobs=-1)(delayed(space_time_anomaly_check_offline)(i, train_dataSet, test_dataSet, \
                                                                      label_list)\
                            for i, (train_dataSet, test_dataSet) in enumerate(splits))
    y_pred_ll, y_true_ll = zip(*r)

    print y_pred_ll
    print y_true_ll

    import operator
    y_pred = reduce(operator.add, y_pred_ll)
    y_true = reduce(operator.add, y_true_ll)

    ## count = 0
    ## for i in xrange(len(y_pred)):
    ##     if y_pred[i] == y_true[i]: count += 1.0

    ## print "Classification Rate: ", count/float(len(y_pred))*100.0

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest')
    plt.colorbar()
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list, rotation=45)
    plt.yticks(tick_marks, label_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
        ## ut.get_keystroke('Hit a key to proceed next')
    else:
        plt.show()



def space_time_anomaly_check_offline(idx, train_dataSet, test_dataSet, label_list=None):

    print "Parallel Run ", idx
    
    train_aXData = train_dataSet.samples
    train_chunks = train_dataSet.sa.chunks
    train_labels = train_dataSet.sa.targets

    test_aXData = test_dataSet.samples
    test_chunks = test_dataSet.sa.chunks
    test_labels = test_dataSet.sa.targets

    # data conversion
    ## print np.shape(train_aXData[0,0]), np.shape(train_labels[0])
    ## print np.shape(train_aXData), np.shape(train_labels)
    ## print np.shape(test_aXData), np.shape(test_labels)

    aXData = None
    aYData = []
    for j in xrange(len(train_aXData)):
        if aXData is None: aXData = train_aXData[j,0]
        else: aXData = np.vstack([aXData, train_aXData[j,0]])
        aYData += [ train_labels[j] for x in xrange(len(train_aXData[j,0])) ]

    ## ml = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=1.0)
    ml = svm.SVC()
    ml.decision_function_shape = "ovr"
    ml.fit(aXData,aYData)

    y_true = []
    y_pred = []
    for i in xrange(len(test_labels)):
        y_true.append(test_labels[0])
        res = ml.predict(test_aXData[0,0])

        score = np.zeros((len(label_list)))
        for s in res:
            for idx in range(len(label_list)):
                if s == idx: score[idx] += 1.0

        y_pred.append( np.argmax(score) )

        print "score: ", score, " true: ", y_true[-1], " pred: ", y_pred[-1]
        
        ## if np.sum(res) >= 0.0: y_pred.append(1)
        ## else: y_pred.append(-1)

    ## print "%d / %d : true=%f, pred=%f " % (i, len(dataSet), np.sum(y_true[-1]), np.sum(y_pred[-1]))
    return y_pred, y_true


def space_time_class_viz(save_pdf=False):
    
    d = ut.load_pickle('st_svm.pkl')
    X = d['X'] 
    Y = d['Y'] 
    ## c = d['c'] 

    ## print "before: ", np.shape(X), np.shape(np.linalg.norm(X, axis=0))
    X = np.array(X)
    X = (X - np.min(X, axis=0))/(np.max(X, axis=0)-np.min(X, axis=0))
    print "Size of Data", np.shape(X)
    print np.max(X, axis=0)

    ## fig = plt.figure()
    ## ax = fig.add_subplot(111)
    ## ax.scatter(X[:,0], X[:,1], c=c)

    ## if save_pdf:
    ##     fig.savefig('test.pdf')
    ##     fig.savefig('test.png')
    ##     os.system('cp test.p* ~/Dropbox/HRL/')
    ##     ## ut.get_keystroke('Hit a key to proceed next')
    ## else:
    ##     plt.show()
    ## sys.exit()
    

    # SVM
    from sklearn import svm
    bv = {}    
    bv['svm_gamma1'] = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=1.0)
    ## bv['svm_gamma2'] = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=0.2)
    ## bv['svm_gamma3'] = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=0.4)
    ## bv['svm_gamma4'] = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=1.0)

    fig = plt.figure()
    for idx, key in enumerate(bv.keys()):
        print "Running SVM with "+key, " " , idx
        ml = bv[key]

        ## ax = fig.add_subplot(2, 2, idx + 1)#, projection='3d')        
        ml.fit(X,Y)

        xx, yy = np.meshgrid(np.arange(0.0, 1.0, 0.2),
                             np.arange(0.0, 1.0, 0.2))

        for j in range(1,5):
            ax = fig.add_subplot(2, 2, j)
            Z = ml.decision_function(np.c_[xx.ravel(), yy.ravel(), np.ones(np.shape(yy.ravel()))*0.15*j])
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
            plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
            plt.axis('off')
                
        ## y_pred = ml.predict(X)

        ## c = []
        ## for i in xrange(len(y_pred)):
        ##     if y_pred[i] == 0:
        ##         c.append('r')
        ##     else:
        ##         c.append('b')                
        ## ax.scatter(X[:,0], X[:,1], c=c)
        bv[key] = ml

    from sklearn.externals import joblib
    

    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
        ## ut.get_keystroke('Hit a key to proceed next')
    else:
        plt.show()
        
    ## plt.close()

    ## print np.shape(X_diff), np.shape(y_true)
    ## fig2 = plt.figure('delay')
    ## for i in xrange(len(y_true)):
    ##     if y_true[i]==0: plt.plot(X_diff[i][0], X_diff[i][1], 'ro', ms=10 )
    ##     if y_true[i]==1: plt.plot(X_diff[i][0], X_diff[i][1], 'gx', ms=10 )
    ##     if y_true[i]==2: plt.plot(X_diff[i][0], X_diff[i][1], 'b*', ms=10 )
    ##     if y_true[i]==3: plt.plot(X_diff[i][0], X_diff[i][1], 'm+', ms=10 )
    ## plt.xlim([ np.amin(np.array(X_diff)[:,0])-1, np.amax(np.array(X_diff)[:,0])+1])
    ## plt.show()
            

        ## sys.exit()

    
    
def offline_classification(subject_names, task_name, raw_data_path, processed_data_path, \
                           nSet=1, downSampleSize=200, \
                           save_pdf=False, data_renew=False):

    
    data_pkl = os.path.join(processed_data_path, 'test.pkl')
    if os.path.isfile(data_pkl):
        data_dict = ut.load_pickle(data_pkl)

        fileNameList     = data_dict['fileNameList']
        # Audio
        audioTimesList   = data_dict['audioTimesList']
        audioAzimuthList = data_dict['audioAzimuthList']
        audioPowerList   = data_dict['audioPowerList']

        # Fabric force
        fabricTimesList  = data_dict['fabricTimesList']
        fabricCenterList = data_dict['fabricCenterList']
        fabricNormalList = data_dict['fabricNormalList']
        fabricValueList  = data_dict['fabricValueList']
        min_audio_power  = data_dict['min_audio_power']
    else:
        success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

        #-------------------------------- Success -----------------------------------
        success_data_pkl     = os.path.join(processed_data_path, subject+'_'+task+'_success')
        raw_data_dict, _ = loadData(success_list, isTrainingData=False,
                                    downSampleSize=downSampleSize,\
                                    global_data=True,\
                                    renew=data_renew, save_pkl=success_data_pkl)

        # Audio
        audioTimesList   = raw_data_dict['audioTimesList']
        audioAzimuthList = raw_data_dict['audioAzimuthList']
        audioPowerList   = raw_data_dict['audioPowerList']

        ## min_audio_power = np.mean( [np.mean(x) for x in audioPowerList] )
        min_audio_power = np.min( [np.max(x) for x in audioPowerList] )

        #-------------------------------- Failure -----------------------------------
        failure_data_pkl     = os.path.join(processed_data_path, subject+'_'+task+'_failure')
        raw_data_dict, _ = loadData(failure_list, isTrainingData=False,
                                    downSampleSize=downSampleSize,\
                                    global_data=True,\
                                    renew=data_renew, save_pkl=failure_data_pkl)

        fileNameList     = raw_data_dict['fileNameList']
        # Audio
        audioTimesList   = raw_data_dict['audioTimesList']
        audioAzimuthList = raw_data_dict['audioAzimuthList']
        audioPowerList   = raw_data_dict['audioPowerList']

        # Fabric force
        fabricTimesList  = raw_data_dict['fabricTimesList']
        fabricCenterList = raw_data_dict['fabricCenterList']
        fabricNormalList = raw_data_dict['fabricNormalList']
        fabricValueList  = raw_data_dict['fabricValueList']

        data_dict = {}
        data_dict['fileNameList'] = fileNameList
        # Audio
        data_dict['audioTimesList'] = audioTimesList
        data_dict['audioAzimuthList'] = audioAzimuthList
        data_dict['audioPowerList'] = audioPowerList

        # Fabric force
        data_dict['fabricTimesList'] = fabricTimesList
        data_dict['fabricCenterList'] = fabricCenterList
        data_dict['fabricNormalList'] = fabricNormalList
        data_dict['fabricValueList'] = fabricValueList

        data_dict['min_audio_power'] = min_audio_power
        ut.save_pickle(data_dict, data_pkl)

    # Parameter set
    ## downSampleSize = 1000

    azimuth_interval = 2.0
    audioSpace = np.arange(-90, 90, azimuth_interval)

    max_audio_azimuth  = 15.0
    max_audio_delay    = 1.0
    max_fabric_azimuth = 10.0
    max_fabric_delay   = 1.0

    max_audio_power    = 5000 #np.median( [np.max(x) for x in audioPowerList] )
    max_fabric_value   = 3.0


    label_list = ['sound', 'force', 'forcesound']
    y_true = []
    y_pred = []
    #
    for i in xrange(len(fileNameList)):

        # time
        max_time1 = np.max(audioTimesList[i])
        max_time2 = np.max(fabricTimesList[i])
        if max_time1 > max_time2: # min of max time
            max_time = max_time2
        else:
            max_time = max_time1            
        new_times     = np.linspace(0.0, max_time, downSampleSize)
        time_interval = new_times[1]-new_times[0]
        
        # ------------------- Data     --------------------------------
        audioTime    = audioTimesList[i]
        audioAzimuth = audioAzimuthList[i]
        audioPower   = audioPowerList[i]

        fabricTime   = fabricTimesList[i]
        fabricCenter = fabricCenterList[i]
        fabricValue  = fabricValueList[i]
        
        # ------------------- Auditory --------------------------------
        discrete_azimuth_array = hdl.discretization_array(audioAzimuth, [-90,90], len(audioSpace))
        discrete_time_array    = hdl.discretization_array(audioTime, [0.0, max_time], len(new_times))

        image = np.zeros((len(new_times),len(audioSpace)))
        last_time_idx = -1
        for j, time_idx in enumerate(discrete_time_array):
            if time_idx < 0: time_idx = 0
            if time_idx >= len(new_times): time_idx=len(new_times)-1
                
            s = np.zeros(len(audioSpace))
            if audioPower[j] > max_audio_power:
                s[discrete_azimuth_array[j]] = 1.0
            elif audioPower[j] > min_audio_power:                
                s[discrete_azimuth_array[j]] = ((audioPower[j]-min_audio_power)/
                                                (max_audio_power-min_audio_power)) #**2

            #
            if last_time_idx == time_idx:
                for k in xrange(len(s)):
                    if image[time_idx,k] < s[k]: image[time_idx,k] = s[k]
            else:
                # image: (Ang, N)
                if len(np.shape(s))==1: s = np.array([s])
                image[time_idx,:] = s
            last_time_idx = time_idx

        image = image.T

        gaussian_2D_kernel = get_space_time_kernel(max_audio_delay, max_audio_azimuth, \
                                                   time_interval, azimuth_interval)
        gaussian_2D_kernel = CustomKernel(gaussian_2D_kernel)
        audio_image = convolve(image, gaussian_2D_kernel, boundary='extend')

        clustered_audio_image, audio_label_list = space_time_clustering(audio_image, max_audio_delay, \
                                                                       max_audio_azimuth, \
                                                                       azimuth_interval, time_interval, 4)

        # ------------------- Fabric Force ----------------------------
        discrete_time_array    = hdl.discretization_array(fabricTime, [0.0, max_time], len(new_times))

        image = np.zeros((len(new_times),len(audioSpace)))
        last_time_idx = -1
        for j, time_idx in enumerate(discrete_time_array):
            if time_idx < 0: time_idx = 0
            if time_idx >= len(new_times): time_idx=len(new_times)-1
            s = np.zeros(len(audioSpace))

            # Estimate space
            xyz  = [fabricCenter[0][j], fabricCenter[1][j], fabricCenter[2][j]]
            fxyz = [fabricValue[0][j], fabricValue[1][j], fabricValue[2][j]] 
            for k in xrange(len(xyz[0])):
                if xyz[0][k]==0 and xyz[1][k]==0 and xyz[2][k]==0: continue
                y   = xyz[1][k]/np.linalg.norm( np.array([ xyz[0][k],xyz[1][k],xyz[2][k] ]) )
                ang = np.arcsin(y)*180.0/np.pi 
                mag = np.linalg.norm(np.array([fxyz[0][k],fxyz[1][k],fxyz[2][k]]))

                ang_idx = hdl.discretize_single(ang, [-90,90], len(audioSpace))
                if mag > max_fabric_value:
                    s[ang_idx] = 1.0
                elif mag > 0.0:
                    s[ang_idx] = ((mag-0.0)/(max_fabric_value-0.0))

            if last_time_idx == time_idx:
                for k in xrange(len(s)):
                    if image[time_idx,k] < s[k]: image[time_idx,k] = s[k]
            else:
                if len(np.shape(s))==1: s = np.array([s])
                image[time_idx,:] = s

            last_time_idx = time_idx
            
        image = image.T

        # Convoluted data
        gaussian_2D_kernel = get_space_time_kernel(max_fabric_delay, max_fabric_azimuth, \
                                                   time_interval, azimuth_interval)        
        gaussian_2D_kernel = CustomKernel(gaussian_2D_kernel)
        fabric_image = convolve(image, gaussian_2D_kernel, boundary='extend')

        clustered_fabric_image, fabric_label_list = space_time_clustering(image, max_fabric_delay , \
                                                                          max_fabric_azimuth, \
                                                                          azimuth_interval, time_interval, 4)

        #-----------------Multi modality ------------------------------
        image = audio_image * fabric_image
        max_delay   = np.max([max_audio_delay, max_fabric_delay])
        max_azimuth = np.max([max_audio_azimuth, max_fabric_azimuth])
        clustered_image, label_list = space_time_clustering(image, max_fabric_delay , max_fabric_azimuth, \
                                                            azimuth_interval, time_interval, 4)


        # -------------------------------------------------------------
        # Classification
        # -------------------------------------------------------------
        audio_score = np.zeros((len(audio_label_list))) if len(audio_label_list) > 0 else [0]       
        fabric_score = np.zeros((len(fabric_label_list))) if len(fabric_label_list) > 0 else [0]       
        multi_score = np.zeros((len(label_list))) if len(label_list) > 0 else [0]      
        for ii in xrange(len(clustered_image)):
            for jj in xrange(len(clustered_image[ii])):
                # audio
                y = clustered_audio_image[ii,jj]
                if y > 0: audio_score[int(y)-1]+=1
                # fabric
                y = clustered_fabric_image[ii,jj]
                if y > 0: fabric_score[int(y)-1]+=1
                # multimodal
                y = clustered_image[ii,jj]
                if y > 0: multi_score[int(y)-1]+=1

        cause = os.path.split(fileNameList[i])[-1].split('.pkl')[0].split('failure_')[-1].split('_')[0]
        image_area = float(len(clustered_image)*len(clustered_image[0]))
        if multi_score is not [] and np.max(multi_score)/image_area > 0.02:
            estimated_cause = 'forcesound'            
            print "Force and sound :: ", cause
        else:
            if np.max(audio_score)/image_area > np.max(fabric_score)/image_area:
                estimated_cause = 'sound'
                print "sound :: ", cause
            else:
                estimated_cause = 'force'
                print "Skin contact force :: ", cause

        for ii, real_anomaly in enumerate(label_list):
            if real_anomaly == cause:
                y_true.append(ii)
                break 
            
        for ii, est_anomaly in enumerate(label_list):
            if est_anomaly == estimated_cause:
                y_pred.append(ii)                
                break
                    

    print y_true
    print y_pred


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest')
    plt.colorbar()
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list, rotation=45)
    plt.yticks(tick_marks, label_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
        ## ut.get_keystroke('Hit a key to proceed next')
    else:
        plt.show()
        

if __name__ == '__main__':

    p.add_option('--timeCorrelation', '--tc', action='store_true', dest='bTimeCorr',
                 default=False, help='Plot time correlation.')
    p.add_option('--spaceTimeAnalysis', '--st', action='store_true', dest='bSpaceTimeAnalysis',
                 default=False, help='Plot space-time correlation.')
    p.add_option('--classification', '--c', action='store_true', dest='bClassification',
                 default=False, help='Evaluate classification performance.')

    elif opt.bTimeCorr:
        '''
        time correlation alaysis
        '''
        task    = 'touching'    
        target_data_set = 0
        rf_center    = 'kinForearmPos'
        feature_list = ['unimodal_audioPower',\
                        ##'unimodal_kinVel',\
                        'unimodal_visionChange',\
                        'unimodal_fabricForce',\
                        ]
        local_range = 0.15
        success_viz = False
        failure_viz = False
                        
        time_correlation([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                         nSet=target_data_set, downSampleSize=downSampleSize, \
                         success_viz=success_viz, failure_viz=failure_viz,\
                         save_pdf=opt.bSavePdf,
                         feature_list=feature_list, data_renew=opt.bDataRenew)


    elif opt.bSpaceTimeAnalysis:
        '''
        space time receptive field
        '''
        task    = 'touching'    
        target_data_set = 0
        success_viz = False
        failure_viz = False
        space_time_analysis([subject], task, raw_data_path, save_data_path,\
                            nSet=target_data_set, downSampleSize=downSampleSize, \
                            success_viz=success_viz, failure_viz=failure_viz,\
                            save_pdf=opt.bSavePdf, data_renew=opt.bDataRenew)
        ## space_time_class_viz(save_pdf=opt.bSavePdf)

    elif opt.bClassification:
        '''
        Get classification evaluation result
        '''        
        task    = 'touching'    
        target_data_set = 0
        correlation_confusion_matrix(save_pdf=False, verbose=True)
        ## offline_classification([subject], task, raw_data_path, save_data_path,\
        ##                        nSet=target_data_set, downSampleSize=downSampleSize, \
        ##                        save_pdf=opt.bSavePdf, data_renew=opt.bDataRenew)
