#!/usr/bin/python


import numpy as np

import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

# Set up our R namespaces
R = rpy2.robjects.r
DTW = importr('dtw')

# Generate our data
idx = np.linspace(0, 2*np.pi, 100)
template1 = np.cos(idx)
template2 = np.sin(idx)
query1 = np.sin(idx) + np.array(R.runif(100))/10
query2 = np.cos(idx) + np.array(R.runif(100))/10


## import matplotlib.pyplot as pp
## pp.figure(1)
## ax = pp.subplot(111)
## pp.plot(template)
## pp.plot(query)
## pp.show()

# Calculate the alignment vector and corresponding distance
alignment = R.dtw(query.tolist(), template.tolist(), keep=True)

print alignment

dist = alignment.rx('distance')[0][0]

print(dist)
