#!/bin/bash -x


eta_list=(1e-3 1e-4 1e-5 1e-6)
etadecay_list=(1e-5 1e-6 1e-7)
lambda_list=(0.5 1.0 1.5 2.0)
timewindow_list=(2 4 6)


for i in "${eta_list[@]}"
do
    for j in "${etadecay_list[@]}"
    do
        for k in "${lambda_list[@]}"
        do
            for l in "${timewindow_list[@]}"
            do

                FILENAME=./log/E_${i}_ED_${j}_L_${k}_TW_${l}.log

                timeout 180s th run.lua -model three -midoutputsize 20 -midoutput2size 10 -outputsize 5 -lambda ${k} -eta ${i} -etadecay ${j} -batchsize 16 -timewindow ${l} >> $FILENAME

            done
        done
    done
done
