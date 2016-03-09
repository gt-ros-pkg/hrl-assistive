
def cross_validate(train_data, test_data,  model, params):
    '''
    train_data : [x,y]
    '''

    train_data_x = train_data[0]
    train_data_y = train_data[1]
    test_data_x  = test_data[0]
    test_data_y  = test_data[1]
    
    model.set_params(**params)
    nEmissionDim = len(train_data_x)

    scale = 1.0
    cov_mult = [1.0]*(nEmissionDim**2)
    for key, value in six.iteritems(params): 
        if key is 'cov':
            cov_mult = [value]*(nEmissionDim**2)
        if key is 'scale':
            scale = value
            
    ret = model.fit(train_data_x*scale, cov_mult=cov_mult)
    if ret == 'Failure':
        return 0.0, params
    else:
        score = model.score(test_data_x*scale, test_data_y)    
        return score, params

def cross_validate_local(idx, processed_data_path, model, params):
    '''
    
    '''
    
    dim   = 4
    for key, value in six.iteritems(params): 
        if key is 'dim':
            dim = value

    # Load data
    AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')
    d = ut.load_pickle(AE_proc_data)

    pooling_param_dict  = {'dim': dim} # only for AE

    # dim x sample x length
    normalTrainData, pooling_param_dict = dm.variancePooling(d['normTrainData'], \
                                                             pooling_param_dict)
    abnormalTrainData,_                 = dm.variancePooling(d['abnormTrainData'], pooling_param_dict)
    normalTestData,_                    = dm.variancePooling(d['normTestData'], pooling_param_dict)
    abnormalTestData,_                  = dm.variancePooling(d['abnormTestData'], pooling_param_dict)

    trainSet = [normalTrainData, [1.0]*len(normalTrainData) ]

    testData_x = np.vstack([ np.swapaxes(normalTestData, 0, 1), np.swapaxes(abnormalTestData, 0, 1) ])
    testData_x = np.swapaxes(testData_x, 0, 1)
    testData_y = [1.0]*len(normalTestData[0]) + [-1.0]*len(abnormalTestData[0])    
    testSet    = [testData_x, testData_y ]

    return cross_validate(trainSet, testSet, model, params)

