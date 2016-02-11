
import sys
sys.path.insert(0, '/home/dpark/git/cssvm/python')
#for i in xrange(2):
## from cssvm import *
import cssvmutil as cssvm 
import numpy as np

y, x = cssvm.svm_read_problem('/home/dpark/git/cssvm/datasets/heart_scale')

new_x = []
for i in xrange(len(x)):
    new_x.append([x[i][2],x[i][3],x[i][4]])

m = cssvm.svm_train(y[:200], new_x[:200], '-c 4 -C 1 -w1 1 -w-1 1')
p_label, p_acc, p_val = cssvm.svm_predict(y[200:], new_x[200:], m)

## print np.shape(y), np.shape(new_x)
#print new_x
## print p_label
## print y[200:]
## print p_label
## print p_val
