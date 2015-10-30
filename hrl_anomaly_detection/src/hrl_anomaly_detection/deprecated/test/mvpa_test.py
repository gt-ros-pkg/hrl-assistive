#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import numpy as np

# ROS library
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut

from mvpa2.datasets.base import Dataset
from mvpa2.generators.partition import NFoldPartitioner

if __name__ == '__main__':

    m_ds = Dataset(np.random.random((3, 4, 2, 3)))


    print m_ds
