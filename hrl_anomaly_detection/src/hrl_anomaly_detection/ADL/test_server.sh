#!/bin/bash -x

method=ronsimmthdp

category=(0,1,2,3)
task=(0,1,2)


for i in "${alpha[@]}"
do
    for j in "${beta[@]}"
    do
        python ./test_multi_modality.py $method --c $i --t $j
        
    done
done