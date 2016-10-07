
import numpy as np
import random

anomaly_list     = np.array(range(13))+1
non_anomaly_list = [0]*20

id_list = anomaly_list.tolist()+non_anomaly_list

random.shuffle(id_list)
print id_list
