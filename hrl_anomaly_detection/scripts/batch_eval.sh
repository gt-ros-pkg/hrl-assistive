#!/bin/bash

for ((I=2;I<=5;I++)); do
    echo ${I}
    if [ $I -eq 5 ]; then 
        if ["$1" = "scooping"] || ["$1" = "feeding"]; then
            return
        fi 
    fi

    python ../src/hrl_anomaly_detection/rss2016test.py --task $1 --dim $I --ea --np  \;

done
