#!/bin/bash

#first launch the thermal camera
echo "Launching Thermal Camera"
roslaunch hrl_thermal_camera thermal_camera.launch &

sleep 1
#next launch the kinect, NOT CALIBRATED
echo "Launching the Kinect"
roslaunch kinect2_bridge kinect2_bridge.launch &
