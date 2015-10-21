#!/bin/bash -x

method=ronsimmthdp
datagen=data_gen

category=(0,1,2,3)
task=(0,1,2)


for i in "${category[@]}"
do
    for j in "${task[@]}"
    do
        echo 'run task ' $i $j
        python ./test_multi_modality.py --$method --c "$i" --t "$j" --$datagen \;
        
    done
done