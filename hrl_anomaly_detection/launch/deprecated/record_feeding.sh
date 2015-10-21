#!/bin/bash -x


FILE_NAME="feeding_hrl"
roslaunch hrl_anomaly_detection record_feeding.launch fname:=$FILE_NAME


#max=4
#complete=0
#i="0"

#while [ $i -lt $max ]; do

#    FILE_NAME="feeding_hrl_"
#    FILE_EXT=".bag"
#    FILE_NAME+=$i
#    FILE_NAME+=$FILE_EXT

#    if [ $complete -eq 1 ]; then
#        exit 0
#    fi

#    if [ ! -f "$FILE_NAME" ]; then
#        complete=1
#        echo "Start to record : $FILE_NAME"
#        roslaunch hrl_anomaly_detection record_feeding.launch fname:=$FILE_NAME
#    fi

#    i=$[$i+1]
#done

