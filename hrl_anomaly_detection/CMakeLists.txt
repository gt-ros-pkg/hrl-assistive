cmake_minimum_required(VERSION 2.8.3)
project(hrl_anomaly_detection)
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

add_message_files(
  DIRECTORY msg
  FILES MultiModality.msg
)

generate_messages(
    DEPENDENCIES std_msgs
)

catkin_package(
    DEPENDS roscpp rospy
    CATKIN_DEPENDS message_runtime std_msgs hrl_lib geometry_msgs hrl_srvs sound_play 
    INCLUDE_DIRS 
    LIBRARIES 
)

