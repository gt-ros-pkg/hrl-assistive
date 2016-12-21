import copy
import numpy as np


def rnd_():
    
    # random sampling?
    ## nSample     = 30
    window_size = 30
    window_step = 5
    X_train = []
    Y_train = []
    ml_dict = {}
    for i in xrange(len(x[0])): # per sample

        s_l = np.arange(startIdx, len(x[0][i])-int(window_size*1.5), window_step)            
        ## s_l = np.random.randint(startIdx, len(x[0][i])-window_size*2, nSample)

        for j in s_l:
            block = x[:,i,j:j+window_size]

            # zero mean to resolve signal displacements
            block -= np.mean(block, axis=1)[:,np.newaxis]

            X_train.append( block )
            Y_train.append( label[i] )
    
def feature_omp(x, label, D0=None, n_iter=1000, sp_ratio=0.1):
    ''' Feature-wise omp '''
    from ksvd import KSVD, KSVD_Encode

    # train feature-wise omp
    X_ = []
    for i in xrange(len(x[0])): # per sample
        X_.append(x[:,i,:]) #-np.mean(x[:,i,:], axis=1)[:, np.newaxis])
    Y_ = copy.copy(label)

    dimension = len(X_[0][0]) #window_size
    dict_size = int(dimension*2) ##10, 1.5)
    n_examples = len(X_)
    target_sparsity = int(sp_ratio*dimension)

    gs = None
    Ds = {}
    X_ = np.array(X_)
    for i in xrange(len(X_[0])): # per feature
        print i, ' / ', len(X_[0])

        if D0 is None:
            # X \simeq g * D
            # D is the dictionary with `dict_size` by `dimension`
            # g is the code book with `n_examples` by `dict_size`
            D, g = KSVD(X_[:,i,:], dict_size, target_sparsity, n_iter,
                            print_interval = 25,
                            enable_printing = True, enable_threading = True)
            Ds[i] = D
        else:
            g = KSVD_Encode(X_[:,i,:], D0[i], target_sparsity)

        if gs is None:
            gs = g
        else:
            gs = np.hstack([gs, g])

    if D0 is None: return Ds, gs, Y_
    else:          return D0, gs, Y_


def m_omp(x, label, D0=None, n_iter=1000, sp_ratio=0.1):
    ''' Multichannel OMP '''
    from ksvd import KSVD, KSVD_Encode

    # train multichannel omp?
    X_ = []
    Y_ = []
    for i in xrange(len(x[0])): # per sample
        for j in xrange(len(x)): # per feature
            X_.append(x[j,i,:]) #-np.mean(x[:,i,j])) 
            ## Y_.append(label[i])
    Y_ = copy.copy(label)

    n_features = len(x)
    dimension  = len(X_[0]) 
    dict_size  = int(dimension*10)
    n_examples = len(X_)
    target_sparsity = int(sp_ratio*dimension)

    gs = None
    X_ = np.array(X_)

    if D0 is None:
        # X \simeq g * D
        # D is the dictionary with `dict_size` by `dimension`
        # g is the code book with `n_examples` by `dict_size`
        D, g = KSVD(X_, dict_size, target_sparsity, n_iter,
                        print_interval = 25,
                        enable_printing = True, enable_threading = True)
    else:        
        g = KSVD_Encode(X_, D0, target_sparsity)        

    # Stacking?
    for i in xrange(len(x[0])): # per sample

        single_g = g[i*n_features:(i+1)*n_features,:].flatten()

        if gs is None: gs = single_g
        else: gs = np.vstack([gs, single_g])

    if D0 is None: return D, gs, Y_
    else:          return D0, gs, Y_


def w_mp(x, label, D0=None, n_iter=1000, sp_ratio=0.05):
    ''' Multichannel OMP with random wavelet dictionary'''
    from sklearn.decomposition import SparseCoder

    def ricker_function(resolution, center, width):
        """Discrete sub-sampled Ricker (Mexican hat) wavelet"""
        x = np.linspace(0, resolution - 1, resolution)
        x = ((2 / ((np.sqrt(3 * width) * np.pi ** 1 / 4)))
             * (1 - ((x - center) ** 2 / width ** 2))
             * np.exp((-(x - center) ** 2) / (2 * width ** 2)))
        return x


    def ricker_matrix(width, resolution, n_components):
        """Dictionary of Ricker (Mexican hat) wavelets"""
        centers = np.linspace(0, resolution - 1, n_components)
        D = np.empty((n_components, resolution))
        for i, center in enumerate(centers):
            D[i] = ricker_function(resolution, center, width)
        D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
        return D


    # train multichannel omp?
    X_ = []
    Y_ = []
    for i in xrange(len(x[0])): # per sample
        for j in xrange(len(x)): # per feature
            X_.append(x[j,i,:]) 
    Y_ = copy.copy(label)

    n_features = len(x)
    dimension  = len(X_[0]) 
    n_examples = len(X_)
    target_sparsity = int(sp_ratio*dimension)
    n_components = 60 #dimension / 2
    w_list = np.logspace(0, np.log10(dimension), dimension/8).astype(int)
    ## w_list = np.linspace(1, dimension-1, dimension/2)

    # Compute a wavelet dictionary
    if D0 is None:
        D = np.r_[tuple(ricker_matrix(width=w, resolution=dimension,
                                      n_components=n_components )
                                      for w in w_list)]
    else:
        D = D0

    gs = None
    X_ = np.array(X_)

    # X \simeq g * D
    coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=target_sparsity,
                        transform_alpha=None, transform_algorithm='omp', n_jobs=1)
    g = coder.transform(X_)

    ## X_est = np.ravel(np.dot(g, D))
    ## squared_error = np.sum((X_est - np.ravel(X_)) ** 2)
    ## print squared_error
    
    # Stacking?
    for i in xrange(len(x[0])): # per sample

        single_g = g[i*n_features:(i+1)*n_features,:].flatten()

        if gs is None: gs = single_g
        else: gs = np.vstack([gs, single_g])

    return D, gs, Y_


    
def time_wise_omp():
    
    # train time-wise omp
    X_ = []
    Y_ = []
    for i in xrange(len(x[0])): # per sample
        for j in xrange(len(x[0][i])): # per time
            X_.append(x[:,i,j]-np.mean(x[:,i,j])) 
            Y_.append(label[i])

    dimension = len(X_[0]) 
    dict_size = int(dimension*10)
    n_examples = len(X_)
    target_sparsity = int(0.1*dimension)

    window_size = 20 # for max pooling?
    window_step = 5  # for max pooling?
    gs = None

    # X \simeq g * D
    # D is the dictionary with `dict_size` by `dimension`
    # g is the code book with `n_examples` by `dict_size`
    D, g = KSVD(X_, dict_size, target_sparsity, 1000,
                    print_interval = 25,
                    enable_printing = True, enable_threading = True)

    # Max pooling?
    for i in xrange(len(x[0])): # per sample
        for j in xrange(window_size, len(x[0][i]), window_step): # per time

            max_pool = np.amax( g[i+j-window_size:i+j,:], axis=0 )

            if gs is None: gs = max_pool
            else: gs = np.vstack([gs, max_pool])
