#requirements to run:
#	starcluster config has been set to include ipcluster plugin
#	grab_data and model is in PYTHNOPATH of each instances
#	starcluster has been started with starcluster with ipcluster plugin at least once
#


from IPython.parallel import Client
from IPython.parallel import error
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import ShuffleSplit 
from starcluster.config import StarClusterConfig
from starcluster.cluster import ClusterManager
from starcluster.plugins import ipcluster as star_ip
from starcluster import exception
from starcluster import static
from starcluster import deathrow
import time
import dill
import inspect

class CloudSearch():
    def __init__(self, path_json, path_key, clust_name, user_name, auth=False):
        self.all_tasks = []
        self.path_json = path_json
        self.path_key = path_key
        self.clust_name = clust_name
        self.user_name = user_name
        self.auth=auth

        #connect to aws and start cluster if it is not up already
        self.uses_profile=False
        self.cfg = StarClusterConfig()
        self.cfg.load()
        self.clust_manager = ClusterManager(self.cfg)
        self.clust = self.clust_manager.get_cluster(self.clust_name)
        if (not self.clust.is_cluster_up()):
            try:
                self.clust.start(create=False)
            except Exception,e:
                print str(e)
                print 'hello'

        self.client = None
        self.lb_view= None

        #connect directly to master to start client 
        #as it often fails to start IPcluster plugin
        self.use_profile(self.user_name)
        self.start_ipcluster()

        ## for node in self.clust.running_nodes:
        ##     node.ssh.execute(command='apt-get install python-libsvm')


        pass

    #in case you just want to stop ipcluster
    def stop_ipcluster(self):
        master_ssh = self.clust.master_node.ssh
        orig_user = master_ssh.get_current_user()
        master_ssh.switch_user(self.user_name)
        self._revoke_ipcluster()
        master_ssh.switch_user(orig_user)

    #stops clusters. It doesn't save any results.
    def stop(self):
        if self.uses_profile:
            self.stop_profile(self.user_name)
        self._revoke_ipcluster()
        self.flush()
        self.clust.stop_cluster(force=True)

    def terminate(self):
        self.stop()
        self.flush()
        self.clust.terminate_cluster(force=True)

    #runs bash command for all nodes
    def sync_run_shell(self, path_shell):

        #self.clust.ssh_to_master(command=path_shell)
        for node in self.clust.running_nodes:
            node.ssh.execute(command=path_shell)

    #deletes all the function assigned
    def flush(self):
        all_tasks = []
        self.lb_view.spin()
        self.client[:].spin()

    def stop_all_tasks(self):
        for task in all_tasks:
            task.abort()

    def use_profile(self, user=None):
        self.uses_profile=True
        if user and user is not 'root':
            file_path= '/home/' + user + '/.bashrc'
            self.copy_bashrc(file_path, self.clust.master_node)
            """
            self.clust.ssh_to_master(command='mv {0} {1}'.format(file_path, file_path +'_cp'))
            file = self.clust.master_node.remote_file(file_path, 'w')
            file2 = self.clust.master_node.remote_file(file_path, 'r')
            line = file2.readline()
            while line is not '':
                if 'case $- in' in line:
                    line = file2.readline()
                    case_counter = 1
                    while case_counter > 0:
                        if 'case' in line:
                            case_counter = case_counter + 1
                        if 'esac' in line:
                            case_counter = case_counter - 1
                        line = file2.read_line()
                if '[ -z "$PS1" ] && return' in line:
                    line = file2.readline()
                else:
                    file.write(line)
            """
            for node in self.clust.running_nodes:
                file_path = '/root/.bashrc'
                self.copy_bashrc(file_path, node)
        else:
            self.use_profile(user=self.user_name)
    
    def copy_bashrc(self, file_path, node):
        if not node.ssh.path_exists(file_path + '_cp'):
            node.ssh.execute('mv {0} {1}'.format(file_path, file_path +'_cp'))
        file = node.ssh.remote_file(file_path, 'w')
        file2 = node.ssh.remote_file(file_path + '_cp', 'r')
        line = file2.readline()
        while len(line) is not 0:
            if 'case $- in' in line:
                line = file2.readline()
                case_counter = 1
                while case_counter > 0:
                    if 'case' in line:
                        case_counter = case_counter + 1
                    if 'esac' in line:
                        case_counter = case_counter - 1
                        line = file2.readline()
            if '[ -z "$PS1" ] && return' in line:
                line = file2.readline()
            else:
                file.write(line)
                line = file2.readline()

    def stop_profile(self, user=None):
        if self.uses_profile:
            if user and user is not 'root':
                file_path= '/home/' + user + '/.bashrc'
                try:
                    self.clust.master_node.ssh.execute('mv {0} {1}'.format(file_path+'_cp', file_path))
                except Exception,e:
                    print str(e)
                file_path='/root/.bashrc'
                for node in self.clust.running_nodes:
                    orig_user=node.ssh.get_current_user()
                    node.ssh.switch_user('root')
                    node.ssh.execute('mv {0} {1}'.format(file_path+'_cp', file_path))
                    node.ssh.switch_user(orig_user)
            else:
                self.stop_profile(self.user_name)
            self.uses_profile=False                
    
    def start_ipcluster(self):
        master_ssh = self.clust.master_node.ssh
        connection = self.clust.master_node.ec2.conn
        orig_user= master_ssh.get_current_user()
        plugs = [self.cfg.get_plugin('ipcluster')]
        plug = deathrow._load_plugins(plugs)[0]
        self.clust.run_plugin(plug, method_name="on_shutdown")
        time.sleep(1)
        try:
            master_ssh.switch_user(self.user_name)
            master_ssh.execute("ipcluster stop", silent=False)
        except:
            print "ipcluster wasn't stoped. It is likely it was not running"
        time.sleep(1)
        self._revoke_ipcluster()
        time.sleep(1)
        master_ssh.switch_user('root')
        try:
            self.clust.run_plugin(plug)
        except:
            print "run_plugin error, but ignored.."
        time.sleep(10)
        
        if not self.auth:
            self.clust.ssh_to_master(user=self.user_name, command="echo 'hello world'")
        master_ssh.switch_user(self.user_name)
        print master_ssh.get_current_user()
        master_ssh.execute("python -c 'from IPython.parallel import Client; client = Client()'", silent=False)
        #connect to cluster nodes to distribute work
        self.client = Client(self.path_json, sshkey=self.path_key)
        self.client[:].use_dill()
        self.lb_view = self.client.load_balanced_view()
        #try:
        #    master_ssh.switch_user(orig_user)
        #except:
        #    print "failed to change back to original user"
        #self.clust.ssh_to_master(user=self.user_name, command="python -c 'from IPython.parallel import Client; client = Client()'")
        #import os
        #os.system('starcluster runplugin ipcluster ' + self.clust_name)
        #self.clust.run_plugin(plugin=plug)#, method_name="on_shutdown")
        #set_command = 'bash /home/' + self.user_name + '/.profile;env;ipcluster start --daemon'
        #self.clust.ssh_to_master(user=self.user_name, command=set_command)
        #json_file= '/home/' + self.user_name + '/.ipython/profile_default/security/ipcontroller-client.json'
        #print json_file
        #self.clust.master_node.ssh.get(json_file, self.path_json)
    
    def _revoke_ipcluster(self):
        import os
        if os.path.isdir(self.path_json):
            connection_params = json.load(open(self.path_json, 'rb'))
            for channel in star_ip.CHANNEL_NAMES:
                port = connection_params.get(channel)
                if port is not None:
                    self._revoke_port(master, port, channel)
                    channel_authorized = True

    def _revoke_port(self, node, port, service_name, protocol='tcp'):
        group = node.cluster_group[0]
        world_cidr='0.0.0.0/0'
        if isinstance(port, tuple):
            port_min, port_max = port
        else:
            port_min, port_max = port, port
        port_open = node.ec2.has_permission(group, protocol, port_min, port_max, world_cidr)
        if port_open:
            node.ec2.conn.revoke_security_group(group_id=group.id, ip_protocol=protocol, from_port=port_min, to_port=port_max, cidr_ip=world_cidr)
	
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
                if isinstance(e, error.RemoteError):
                    e.print_traceback()
        return results

    #get method with error catching showing trace back of remote error
    #returns None if task is not assigned through CloudSearch class
    def get_task_results(self, task):
        if task in self.all_tasks:
            try:
                return task.get()
            except Exception, e:
                print str(e)
                if isinstance(e, error.RemoteError):
                    e.print_traceback()
        return None

def syncing(path_libs):
    import sys
    if type(path_libs) is str:
        sys.path[:] = sys.path[:] + [path_libs]
    elif type(path_libs) is list:
        sys.path[:] = sys.path[:] + path_libs
    return sys.path[:]

def check_sys_path():
    import sys
    return sys.path

def set_env(var, paths):
    import os

    print "aaaaaa"
    ## return os.environ[var]
    ## for path in paths:
    ##     if var not in os.environ:
    ##         os.environ[var] = path
    ##     elif path not in os.environ.get(var):
    ##         os.environ[var] += ':'+path            
    ## return os.environ.get(var)
