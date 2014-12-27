#!/bin/bash                                                                                                      

echo pidi5252 | sudo dpark -c 'source ~/.profile ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv > nohup_1.txt &;' 
sleep 3.0 ; 

echo pidi5252 | sudo dpark -c 'source ~/.profile ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv > nohup_1.txt &;' 
sleep 3.0 ; 

echo pidi5252 | sudo dpark -c 'source ~/.profile ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv > nohup_1.txt &;' 
sleep 3.0 ; 

echo pidi5252 | sudo dpark -c 'source ~/.profile ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv > nohup_1.txt &;' 
sleep 3.0 ; 
       
#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0
