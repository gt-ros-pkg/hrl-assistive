import sys
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
    gs = None
    for i in xrange(len(x[0])): # per sample

        single_g = g[i*n_features:(i+1)*n_features,:].flatten()

        if gs is None: gs = single_g
        else: gs = np.vstack([gs, single_g])

    if D0 is None: return D, gs, Y_
    else:          return D0, gs, Y_


def w_omp(x, label, D0=None, n_iter=1000, sp_ratio=0.05):
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
    ## for i in xrange(len(g)):
    ##     print np.shape(D), np.shape(g), np.shape(np.dot(g[i:i+1,:],D))
    ##     plot_decoder(X_[i], np.dot(g[i:i+1,:],D)[0] )    
    
    # Stacking?
    for i in xrange(len(x[0])): # per sample

        single_g = g[i*n_features:(i+1)*n_features,:].flatten()

        if gs is None: gs = single_g
        else: gs = np.vstack([gs, single_g])

    return D, gs, Y_


    
def time_omp(x, label, D0=None, n_iter=500, sp_ratio=0.1):
    ''' Time-sample OMP with max pooling and contrast normalization'''
    from ksvd import KSVD, KSVD_Encode
    
    # train time-wise omp
    X_ = []
    Y_ = []
    for i in xrange(len(x[0])): # per sample
        for j in xrange(len(x[0][i])): # per time
            X_.append(x[:,i,j]) #-np.mean(x[:,i,j])) 
            ## Y_.append(label[i])

    dimension  = len(X_[0]) 
    n_examples = len(X_)
    dict_size  = int(dimension*2)
    target_sparsity = int(sp_ratio*dict_size)

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


    # Fixed-size Max pooling?
    window_size = 90 # for max pooling?
    window_step = 10  # for max pooling?
    gs = None
    for i in xrange(len(x[0])): # per sample
        g_per_sample = g[i*len(x[0][i]):(i+1)*len(x[0][i]),:]
        
        for j in xrange(window_size, len(x[0][i]), window_step): # per time

            max_pool = np.amax( g_per_sample[j-window_size:j,:], axis=0 )
            ## max_pool /= np.linalg.norm(max_pool+1e-6)
            
            if gs is None: gs = max_pool
            else: gs = np.vstack([gs, max_pool])
            Y_.append(label[i])

    if D0 is None: return D, gs, Y_
    else:          return D0, gs, Y_



def plot_decoder(x1,x2):
    # visualization
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x1, 'b-')
    plt.plot(x2, 'r-', linewidth=3.0)
    plt.show()

    return
