from cloud_search import CloudSearch
from sklearn.svm import SVC
import time
cloud = CloudSearch('/root/.starcluster/ipcluster/SecurityGroup:@sc-freecluster-us-east-1.json', '/root/.ssh/mykey.rsa', 'freecluster', 'sgeadmin')
model = SVC()
cloud.run_with_local_data(model, {'C':[1.0, 0.1], 'gamma':[1.0, .1]}, 4898, 10, '/scratch/sgeadmin/winequality-white')
print cloud.get_completed_results()
time.sleep(20)
print cloud.get_completed_results()
