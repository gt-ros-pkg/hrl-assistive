def cross_validate(train, test,  model, params):
    '''
    train : (X, y)

    '''
    
    #from grab_data import grab_data
    import os
    #data,target = grab_data('/home/sgeadmin/iris', 4, 4, 150)
    #data, target = grab_data(path_file, 54, 0, 581012)
    #data, target = grab_data(path_file, 11, 11, 4898)
    train_data = train[0]
	train_target = train[1]
	test_data =test[0]
    test_target = test[1]
    model.set_params(**params)
	#return psutil.Process(os.getpid()).memory_info().rss / 1e6  
    model.fit(train_data, train_target)
    score = model.score(test_data, test_target)
    return score, params

def cross_validate_local(train, test, path_file, model, params):
	from grab_data import grab_data
    data, target = grab_data(path_file, 11, 11, 4898)
    trainSet = (data[train], target[train])
    testSet = (data[test], target[test])
    return cross_validate(trainSet, testSet, model, params)

"""def grab_data(path_file, n_attrib, target_loc, n_inst):
	#change accordingly
        import numpy as np
        obj = open(path_file)
        target = np.ndarray(n_inst)
        data = np.ndarray((n_inst, n_attrib))
        for i, line in enumerate(obj):
                splitted = line.split(',')
                splitted[-1] = splitted[-1].split()[0]
                #datprint splitted
                for j, ind in enumerate(splitted):
                        if j is not target_loc:
                                if j > target_loc:
                                        data[i][j-1] = splitted[j]
                                else:
                                        data[i][j] = splitted[j]
                        else:
                                target[i] = splitted[target_loc]
        return (data, target)"""
