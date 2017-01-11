import sys
import numpy as np
from sklearn.utils import check_array
import matplotlib.pyplot as plt

sys.path.insert(0, '/usr/lib/pymodules/python2.7')
import svmutil as svm
commands = '-q -s 4 -t 1'


x = np.linspace(0, np.pi, 100).tolist()
x = np.vstack([x,x,x]).T.tolist()

y = np.cos(x).T[0].tolist()

print np.shape(x), np.shape(y)

dt = svm.svm_train(y, x, commands )

# p_labels, (ACC, MSE, SCC), p_vals = svm.svm_predict(y, x, dt)
p_labels, (ACC, MSE, SCC), p_vals = svm.svm_predict([0]*len(x), x, dt)
p_vals = np.array(p_vals)

plt.figure()
plt.plot(np.array(x)[:,0], y)
plt.plot(np.array(x)[:,0], p_vals, 'r-')
plt.plot(np.array(x)[:,0], p_vals-0.5, 'r--')
plt.show()
