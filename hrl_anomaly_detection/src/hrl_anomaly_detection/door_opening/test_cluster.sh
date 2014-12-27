#!/bin/bash                                                                                                      

su dpark -c 'source ~/.profile ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &' 
sleep 3.0 ; 

su dpark -c 'source ~/.profile ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &' 
sleep 3.0 ; 

su dpark -c 'source ~/.profile ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &' 
sleep 3.0 ; 

su dpark -c 'source ~/.profile ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &' 
sleep 3.0 ; 
       
#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0
