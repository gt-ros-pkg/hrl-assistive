
import numpy as np


def getParams(task, param_dict):

    nPoints = param_dict['ROC']['nPoints']

    if task == "pushing_toolcase":
        param_dict['ROC']['svm_param_range'] = np.logspace(-2, 0.15, nPoints) 
    if task == "scooping":
        param_dict['ROC']['hmmsvm_no_dL_param_range'] = np.logspace(-2.7, -1.3, nPoints) 
        param_dict['ROC']['hmmsvm_dL_param_range'] = np.logspace(-2.15, -0.87, nPoints) 
        param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-3, -1.3, nPoints)
        param_dict['ROC']['svm_param_range'] = np.logspace(-2.52, -.5, nPoints) 
    if task == "feeding":
        param_dict['ROC']['hmmsvm_dL_param_range'] = np.logspace(-3.7, 0.7, nPoints) 
        param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-3, 1.2, nPoints)
        param_dict['ROC']['svm_param_range'] = np.logspace(1.0, -2.4, nPoints) 

    return param_dict
    
