import subprocess
import os
import rospkg

if __name__ == '__main__':
    #needed?
    os.setsid()
    rospack = rospkg.RosPack()
    file_name = rospack.get_path('hrl_manipulation_task')
    file_name = os.path.join(file_name, 'launch/record_topics.launch')
    #print file_name
    #print rospack.get_path('roslaunch')
    if not os.path.isfile(file_name):
        print -1
        sys.exit(0)
    devnull = open(os.devnull, 'wb')
    p = subprocess.Popen(['/opt/ros/indigo/bin/roslaunch', file_name], stdout=devnull, stderr=devnull)
    print p.pid
    
