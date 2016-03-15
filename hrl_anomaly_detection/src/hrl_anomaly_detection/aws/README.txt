Cloud Searching using AWS service

Install and setup starcluster following instructions on Setting Starcluster pdf

following files need to be in all nodes:
model		- whichever model library that it is going to run
grab_data.py 	- function to get data and format it locally
cross.py	- runs the test

starcluster automatically shares all files in /home/ directory with NFS.
If worried about writing/reading at same time use /scratch/user_name/
which is local to each node

following files need to be in master node:
/home/user_name/start_cli.py	- starts client if something caused failure in starting ipcluster



using CloudSearch class

initializing:
	path_json	- json file to connect to cluster so tasks could be sent.
			- usually located in ~/.starcluster/ipcluster
			- and has name 'SecurityGroup:@sc-clust_name-zone.json'
	path_key	- file path to key location
	clust_name	- name of cluster to run tasks on
	user_name	- name of user in cluster. Default is 'sgeadmin'

running:
	run_with_data		- give model, param, number of instance, fold, data (input) and target (label) 
				- # of instance is used to do seperation in locally
				- folds the data by indicated fold
	run_with_local_data	- give model, param, num of inst, fold, path_file
				- each node reads local data and runs model test

retrieving:
	get_completed_results() - returns results of all tasks appended.


Running sample:
	install winequality-white from UCI dataset
	put winequality-white in either /scratch/user_name/ or /home/uer_name/ of all node
	install scikit-learn on all node by either:
		1. $starcluster runplugin machine-learn-installer clust_name
		2. having node img already have sklearn at initialization 
	change path_json & key clust & user_name accoding
	$ python test_wine.py
