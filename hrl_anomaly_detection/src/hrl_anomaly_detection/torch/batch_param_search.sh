#!/bin/bash -x




timewindow_list=(2 4 6)

for i in "${timewindow_list[@]}"
do

    FILENAME=$timewindow_list

    timeout 10s th run.lua -model three -midoutputsize 20 -midoutput2size 10 -outputsize 5 -lambda 0.5 -eta 1e-3 -etadecay 1e-5 -batchsize 16 -timewindow 4 >> $FILENAME.log

done
