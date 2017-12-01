cmake_minimum_required(VERSION 2.8.3)
project(hrl_multimodal_prediction)
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

catkin_python_setup()

add_message_files(
  DIRECTORY msg
  FILES audio.msg pub_relpos.msg pub_mfcc.msg plot_pub.msg
)

generate_messages(
    DEPENDENCIES std_msgs
)


catkin_package(
    DEPENDS roscpp rospy
    CATKIN_DEPENDS message_runtime std_msgs visualization_msgs ar_track_alvar_msgs
    INCLUDE_DIRS  
    LIBRARIES 
)

