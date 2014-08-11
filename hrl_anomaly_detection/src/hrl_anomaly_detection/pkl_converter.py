#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import numpy as np

# ROS library
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut

import scipy.io
import klepto

def getAllFiles(dirName):

    lFile = []
    for root, dirs, files in os.walk(dirName):

        if root.find('.svn') >= 0: continue
                
        for sub_dir in dirs:
            if sub_dir.find('.svn') >= 0: continue
            lFile = lFile + getAllFiles(sub_dir)

        for sub_file in files:
            lFile.append(os.path.join(root,sub_file))

    return lFile
   

if __name__ == '__main__':

    dirName='/home/dpark/svn/robot1_data/usr/advait/ram_www'
    ## dirName='../../matlab/data/'

    lFile = getAllFiles(dirName)

    for f in lFile:

        fileName, fileExtension = os.path.splitext(f)
        if fileExtension.find('mat')>=0:
            ## strPath, strFile = os.path.split(f)
            new_fileName = fileName+'_new.pkl'
            if os.path.isfile(new_fileName): continue

            # Load mat files
            mat_data = scipy.io.loadmat(f)
            
            # Get Data
            pkl_data  = {}
            lKeys = mat_data.keys()
            for key in lKeys:
                if key.find('__version__') < 0 and key.find('__header__') < 0 and  key.find('__globals__'):
                    pkl_data[key] = mat_data[key]

                    if len(mat_data[key].flat) == 1:
                                                
                        if isinstance(mat_data[key].flat[0], np.unicode):
                            pkl_data[key] = str(mat_data[key].flat[0])
                        else:
                            pkl_data[key] = mat_data[key].flat[0]

            # Save as pkl files
            ut.save_pickle(pkl_data,new_fileName)

        elif fileName.find('klepto')>=0 and fileExtension.find('.pyc') < 0:

            new_fileName = fileName[:-7]+'_new.pkl'
            ## if os.path.isfile(new_fileName): continue
            
            strPath, strFile = os.path.split(f)
            
            import imp 

            try:
                m = imp.load_source("klepto_data", f)
                import klepto_data
            except:
                print "failed to load ", f
                sys.exit()

            ## print klepto_data.memo.keys()
            
            ut.save_pickle(klepto_data.memo, new_fileName)
