#!/usr/local/bin/python

from ghmm import *

print "\n\n\n *** Gaussian Model ***"
F = Float()               
m2 = HMMFromMatrices(F,GaussianDistribution(F),
                     [[0.0,1.0,0],[0.5,0.0,0.5],[0.3,0.3,0.4]],
                     [[0.0,1.0],[-1.0,0.5], [1.0,0.2]],
                     [1.0,0,0])

m4 = HMMFromMatrices(F,GaussianDistribution(F),
                     [[0.0,1.0,0],[0.5,0.0,0.5],[0.3,0.3,0.4]],
                     [[0.0,1.3],[-1.0,0.1], [1.0,0.6]],
                     [1.0,0,0])

print m2
trans = m2.getTransition(2,0)
print "a[2,0] = " + str(trans)

print "\nSample:"
cs1 = m2.sample(4,15)                         
print str(cs1) + "\n"

print "\nSampleSingle:"
cs2 = m2.sampleSingle(10)                         
print str(cs2) + "\n"

print "\nget subset"
cs3 = cs1.getSubset([0,2])
print cs3

print "\nViterbi"
spath = m2.viterbi(cs1)
print str(spath) + "\n"

print "\nForward"
logp = m2.loglikelihood(cs1)    
print "logp = " + str(logp) + "\n"

print "\nForward matrices"
(salpha,sscale) = m2.forward(cs2)
print "alpha:\n" + str(salpha) + "\n"
print "scale = " + str(sscale) + "\n"

print "\nBackward matrix"
beta = m2.backward(cs2,sscale)
print "beta = \n " + str(beta) + "\n"

print "Reading SequenceSet from .sqd file"
l = SequenceSetOpen(F,"seq_test.sqd")
print l

print "Model distances (continous):"
d = m2.distance(m4,1000)
print "distance= " + str(d)
