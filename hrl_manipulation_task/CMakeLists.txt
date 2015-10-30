cmake_minimum_required(VERSION 2.8.3)
project(hrl_manipulation_task)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package(
    DEPENDS rospy
    CATKIN_DEPENDS hrl_lib tf tf_conversions hark_msgs ar_track_alvar_msgs geometry_msgs pr2_msgs visualization_msgs pr2_controllers_msgs hrl_haptic_manipulation_in_clutter_msgs std_msgs sandbox_dpark_darpa_m3 hrl_anomaly_detection
    INCLUDE_DIRS 
    LIBRARIES 
)


