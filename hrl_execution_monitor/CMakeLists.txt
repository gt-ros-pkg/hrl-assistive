cmake_minimum_required(VERSION 2.8.3)
project(hrl_execution_monitor)
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

catkin_python_setup()

add_message_files(
  DIRECTORY msg
  FILES MultiModality.msg pclChange.msg audio.msg
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

