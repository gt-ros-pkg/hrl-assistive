#requirements to run:
#	starcluster config has been set to include ipcluster plugin
#	grab_data and model is in PYTHNOPATH of each instances
#	starcluster has been started with starcluster with ipcluster plugin at least once
#


from IPython.parallel import Client
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import ShuffleSplit 
from starcluster.config import StarClusterConfig
from starcluster.cluster import ClusterManager
from starcluster import exception
import time
import dill

class CloudSearch():
    def __init__(self, path_json, path_key, clust_name, user_name):
        self.all_tasks = []
        self.path_json = path_json
        self.path_key = path_key
        self.clust_name = clust_name
        self.user_name = user_name

        #connect to aws and start cluster if it is not up already
        self.cfg = StarClusterConfig()
        self.cfg.load()
        self.clust_manager = ClusterManager(self.cfg)
        self.clust = self.clust_manager.get_cluster(self.clust_name)
        if (not self.clust.is_cluster_up()):
            try:
                self.clust.start(create=False)
            except Exception,e:
                print str(e)

        #connect directly to master to start client 
        #as it often fails to start IPcluster plugin
        self.clust.ssh_to_master(user=self.user_name, command="python start_cli.py")

        #connect to cluster nodes to distribute work
        self.client = Client(self.path_json, sshkey=self.path_key)
        self.client[:].use_dill()
        self.lb_view = self.client.load_balanced_view()
        pass

    #stops clusters. It doesn't save any results.
    def stop(self):
        self.flush()
        self.clust.stop_cluster(force=True)

    def terminate(self):
        self.flush()
        self.clust.terminate_cluster(force=True)

    #runs shell command for all nodes at specific location
    def sync_run_shell(self, path_shell):

        #self.clust.ssh_to_master(command=path_shell)
        for node in self.clust.running_nodes:
            self.clust.ssh_to_node(node.id, command=path_shell)

    #deletes all the function assigned
    def flush(self):
        all_tasks = []
        self.lb_view.spin()
        self.client[:].spin()
	
    #run model given data. The local computer sends the data to each node every time it is given
    def run_with_data(self, model, params, n_inst, cv, data, target):
        from cross import cross_validate
        splited = self.split(n_inst, cv)
        all_param = list(ParameterGrid(params))
        for param in all_param:
            for train, test in splited:
                trainSet = (data[train], target[train])
                testSet = (data[test], target[test])
                task = self.lb_view.apply(cross_validate, trainSet, testSet, model, param)
                self.all_tasks.append(task)
        return self.all_tasks

    #run model from data in cloud.
    #each node grabs file from their local path and runs the model
    #requires grab_data to be implemented correctly
    #n_inst is to create a fold. the way it generates fold can be changed
    def run_with_local_data(self, model, params, n_inst, cv, path_file):
        from cross import cross_validate_local
        #from cross import grab_data
        splited = self.split(n_inst, cv)
        all_param = list(ParameterGrid(params))
        for param in all_param:
            for train, test in splited:
                task = self.lb_view.apply(cross_validate_local, train, test, path_file, model, param)
                self.all_tasks.append(task)
        return self.all_tasks


    #for debugging purposes if we are running out of local memories
    ## def get_engines_memory(self):
    ##     """Gather the memory allocated by each engine in MB"""
    ##     def memory_mb():
    ##         import os
    ##         import psutil
    ##         return psutil.Process(os.getpid()).memory_info().rss / 1e6    
    ##     return self.client[:].apply(memory_mb).get_dict()

    #splits the data for cross validation
    def split(self, n_inst, iter_num):
        """splits data to cv values"""
        a = 1/float(iter_num)
        return ShuffleSplit(n_inst, n_iter=iter_num)


	#adds to all client a local path(s) for external libraries
	#default location where local program is run is /home/user/ of the cluster
	def set_up(self, path_libs):
		import sys
		sys.path[:] = sys.path[:] + path_libs
		tasks = self.client[:].apply(syncing, path_libs)
		return tasks.get()

    #returns tasks that has been assigned, including completed, working, and pending
    def get_all_tasks(self):
        return self.all_tasks

    #returns the number of tasks that has been assigned, including completed, working, and pending
    def get_num_all_tasks(self):
        return len(self.all_tasks)

    #returns completed tasks. may include tasks that has caused errors
    def get_tasks_completed(self):
        completed_tasks = []
        for task in self.all_tasks:
            if task.ready():
                completed_tasks.append(task)
        return completed_tasks

    #returns the number of completed tasks. may include tasks that has caused errors
    def get_num_tasks_completed(self):
        return len(self.get_tasks_completed())

    #returns completed tasks's result. Prints error if there was a remote error
    def get_completed_results(self):
        results = []
        completed_tasks = self.get_tasks_completed()
        for task in completed_tasks:
            try:
                results.append(task.get())
            except Exception, e:
                print "some kind of error occured while working on task"
                print str(e)
        return results


    
