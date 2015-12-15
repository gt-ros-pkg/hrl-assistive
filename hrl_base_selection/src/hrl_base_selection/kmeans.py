#!/usr/bin/env python
# kmeans.py using any of the 20-odd metrics in scipy.spatial.distance
# kmeanssample 2 pass, first sample sqrt(N)
# Source: http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikits-learn-k-means

from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
# http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py
# HRL library
import roslib; roslib.load_manifest('hrl_base_selection')
import hrl_lib.util as ut
import hrl_lib.quaternion as qt
import copy

__date__ = "2011-11-17 Nov denis"
__date__ = "2014-07-07 July Daehyung Park"
# X sparse, any cdist metric: real app ?
# centres get dense rapidly, metrics in high dim hit distance whiteout
# vs unsupervised / semi-supervised svm

#...............................................................................
def kmeans( X, centres, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1 ):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
        is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
        "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
        or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
        see also: kmeanssample below, class Kmeans below.
    """
    
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    if verbose:
        print "kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s" % (
            X.shape, centres.shape, delta, maxiter, metric)
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):

        # Compute distance
        if metric.find('quaternion') >= 0:
            D = qt.quat_angles( X, centres )  # |X| x |centres|           
        else:
            D = cdist_sparse( X, centres, metric=metric, p=p )  # |X| x |centres|

        xtoc = D.argmin(axis=1)  # X -> nearest centre        
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?

        if verbose >= 2:
            print "kmeans: av |X - nearest centre| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:

                # Compute average
                if metric.find('quaternion') >= 0:                    
                    centres[jc] = qt.quat_avg( X[c] )
                else:
                    centres[jc] = X[c].mean( axis=0 )
                                                                      
    if verbose:
        print "kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc)
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans: cluster 50 % radius", r50.astype(int)
        print "kmeans: cluster 90 % radius", r90.astype(int)
        # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances

#...............................................................................
def kmeanssample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centres
    """
    # merge w kmeans ? mttiw
    # v large N: sample N^1/2, N^1/2 of that
    # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centres = randomsample( X, int(k) )
    samplecentres = kmeans( Xsample, pass1centres, **kwargs )[0]
    return kmeans( X, samplecentres, **kwargs )

def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
    # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    return d


def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]

def nearestcentres( X, centres, metric="euclidean", p=2 ):
    """ each X -> nearest centre, any metric
        euclidean2 (~ withinss) is more sensitive to outliers,
        cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()
     

def AreQuaternionClose(q1,q2):
    
    dot = np.sum(q1*q2)
    
    if dot < 0.0:
        return False
    else:
        return True


    
#...............................................................................
class Kmeans:
    """ km = Kmeans( X, k= or centres=, ... )
        in: either initial centres= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centres, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centres[jcentre]
                J indexes e.g. X[J], classes[J]
    """
    def __init__( self, X, k=0, centres=None, nsample=0, **kwargs ):
        self.X = X
        if centres is None:
            self.centres, self.Xtocentre, self.distances = kmeanssample(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centres, self.Xtocentre, self.distances = kmeans(
                X, centres, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)

#...............................................................................
if __name__ == "__main__":
    import sys
    from time import time
    
    N = 100
    dim = 4
    ncluster = 10
    kmsample = 100  # 0: random centres, > 0: kmeanssample
    kmdelta = .001
    kmiter = 10
    seed = 1

    if False:
        metric = "cityblock"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
        
        exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...
        np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True )
        np.random.seed(seed)
        random.seed(seed)

        print "N %d  dim %d  ncluster %d  kmsample %d  metric %s" % (
            N, dim, ncluster, kmsample, metric)
        X = np.random.exponential( size=(N,dim) )
        
    else:
        metric = "quaternion"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
        X = qt.quat_random( N )

        
    # cf scikits-learn datasets/
    ## t0 = time()
    if kmsample > 0:
        centres, xtoc, dist = kmeanssample( X, ncluster, nsample=kmsample,
                                            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    else:
        randomcentres = randomsample( X, ncluster )
        centres, xtoc, dist = kmeans( X, randomcentres,
                                      delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )


        
    ## print "%.0f msec" % ((time() - t0) * 1000)    
    # also ~/py/np/kmeans/test-kmeans.py
