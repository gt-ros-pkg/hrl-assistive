#!/bin/bash                                                                                                      

su dpark -c 'source ~/.profile ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv > nohup_1.txt &; sleep 3.0 ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv > nohup_2.txt &; sleep 3.0 ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv > nohup_3.txt &; sleep 3.0 ; nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv > nohup_4.txt &; sleep 3.0 ;'         
       
#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0

#nohup rosrun hrl_anomaly_detection test_anomaly_door_opening.py --cv &
#sleep 3.0
