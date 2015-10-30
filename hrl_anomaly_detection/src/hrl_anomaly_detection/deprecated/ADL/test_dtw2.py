# -*- coding: utf-8 -*-
# An pure python implemetation of Dynamic Time Warpping
# http://en.wikipedia.org/wiki/Dynamic_time_warping

# This code is a copy of https://gist.github.com/socrateslee/3265694
# and modified for multidimensional DTW implementation by Daehyung Park
import sys
import numpy as np

## class distance():
##     def __init__(self, dist_type='euclidean_distance'):
##         self.dist_type = dist_type
        
##     def __call__(self, seq1, seq2):
##         if self.dist_type == 'euclidean_distance':
##             return euclidean_distance(seq1, seq2)
##         else:
##             return 0

##     def euclidean_distance(self, seq1, seq2):
##         return np.norm(seq1-seq2)


def fast_wdist(A, B, W):
    """
    Compute the weighted euclidean distance between two arrays of points:

    D{i,j} = 
    sqrt( ((A{0,i}-B{0,j})/W{0,i})^2 + ... + ((A{k,i}-B{k,j})/W{k,i})^2 )
    
    inputs:
    A is an (k, m) array of coordinates
    B is an (k, n) array of coordinates
    W is an (k, m) array of weights
    
    returns:
    D is an (m, n) array of weighted euclidean distances
    """
    
    # compute the differences and apply the weights in one go using
    # broadcasting jujitsu. the result is (n, k, m)
    wdiff = (A[np.newaxis,...] - B[np.newaxis,...].T) / W[np.newaxis,...]
        
    # square and sum over the second axis, take the sqrt and transpose. the
    # result is an (m, n) array of weighted euclidean distances
    D = np.sqrt((wdiff*wdiff).sum(1)).T
    
    return D


class Dtw(object):
    def __init__(self, seq1, seq2, distance_weights=[1.0, 1.0], distance_func=None):
        '''
        seq1, seq2 are two lists,
        distance_func is a function for calculating
        the local distance between two elements.
        '''
        self._seq1 = seq1
        self._seq2 = seq2
        ## self._distance_func = distance_func if distance_func else lambda: 0
        self._map = {(-1, -1): 0.0}
        self._distance_matrix = {}
        self._path = [] 

        self.w = distance_weights

        if len(seq1)*len(seq2) < 200000:
            sys.setrecursionlimit(200000)
        else:
            print "Need to increase recursion limit!!!!!!!!!!!!!"
            sys.exit()

    def get_distance(self, i1, i2):
        ret = self._distance_matrix.get((i1, i2))
        if not ret:
            ## ret = self._distance_func(self._seq1[i1], self._seq2[i2])
            ## ret = np.linalg.norm(self._seq1[i1]-self._seq2[i2])            

            dist = self._seq1[i1] - self._seq2[i2]
            ret = np.sqrt( np.sum(dist*dist*np.array(self.w)) )
            
            self._distance_matrix[(i1, i2)] = ret
        return ret

    def calculate_backward(self, i1, i2):
        '''
        Calculate the dtw distance between
        seq1[:i1 + 1] and seq2[:i2 + 1]
        '''
        if self._map.get((i1, i2)) is not None:
            return self._map[(i1, i2)]

        if i1 == -1 or i2 == -1:
            self._map[(i1, i2)] = float('inf')
            return float('inf')

        min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                             key=lambda x: self.calculate_backward(*x))

        self._map[(i1, i2)] = self.get_distance(i1, i2) + \
          self.calculate_backward(min_i1, min_i2)

        return self._map[(i1, i2)] 


    def get_path(self):
        '''
        Calculate the path mapping.
        Must be called after calculate()
        '''
        i1, i2 = (len(self._seq1) - 1, len(self._seq2) - 1)
        while (i1, i2) != (-1, -1):
            self._path.append((i1, i2))
            min_i1, min_i2 = min((i1 - 1, i2), (i1, i2 - 1), (i1 - 1, i2 - 1),
                                 key=lambda x: self._map[x[0], x[1]])
            i1, i2 = min_i1, min_i2
        return self._path[::-1]

    def calculate(self):

        return self.calculate_backward(len(self._seq1) - 1,
                                       len(self._seq2) - 1) 
