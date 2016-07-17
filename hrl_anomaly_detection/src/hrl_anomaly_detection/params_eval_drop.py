



def getParm(task, param_dict):


    if opt.task == "pushing_toolcase":
        param_dict['ROC']['svm_param_range'] = np.logspace(-2, 0.15, nPoints) 
    if opt.task == "scooping":
        param_dict['ROC']['hmmsvm_no_dL_param_range'] = np.logspace(-3.5, 0.0, nPoints) 
        param_dict['ROC']['hmmsvm_dL_param_range'] = np.logspace(-2.5, 0.0, nPoints) 
        param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-3.2, -1.0, nPoints)
        param_dict['ROC']['svm_param_range'] = np.logspace(-2.8, 0.2, nPoints) 
    if opt.task == "feeding":
        param_dict['ROC']['hmmsvm_dL_param_range'] = np.logspace(-3.7, 0.7, nPoints) 
        param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-3, 1.2, nPoints)
        param_dict['ROC']['svm_param_range'] = np.logspace(1.0, -2.4, nPoints) 

    return param_dict
    
