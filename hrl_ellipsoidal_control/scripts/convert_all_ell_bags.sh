#!/bin/bash

for f in *.bag
do
    rosrun hrl_ellipsoidal_control convert_ell_params.py $f
done
