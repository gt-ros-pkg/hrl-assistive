#!/bin/bash 
dir=`rospack find hrl_head_tracking`/bag
rosrun hrl_head_tracking clickable_pc bag/kelsey_face_$1.bag bag/kelsey_face_$1.click
