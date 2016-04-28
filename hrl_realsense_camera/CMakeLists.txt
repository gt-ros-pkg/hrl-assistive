cmake_minimum_required(VERSION 2.8.3)
project(hrl_realsense_camera)
find_package(catkin REQUIRED COMPONENTS message_generation)

catkin_python_setup()



catkin_package(
    DEPENDS roscpp rospy
    CATKIN_DEPENDS 
    INCLUDE_DIRS 
    LIBRARIES 
)

