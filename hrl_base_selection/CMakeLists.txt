cmake_minimum_required(VERSION 2.8.3)
project(hrl_base_selection)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs geometry_msgs tf tf_conversions ar_track_alvar hrl_lib hrl_msgs sensor_msgs visualization_msgs trajectory_msgs message_generation)

catkin_python_setup()

#uncomment if you have defined services
add_service_files(
    DIRECTORY srv  
    FILES BaseMove.srv 
    FILES BaseMove_multi.srv
)

generate_messages(
    DEPENDENCIES geometry_msgs std_msgs
)

catkin_package(
    DEPENDS rospy std_msgs geometry_msgs tf tf_conversions ar_track_alvar hrl_lib hrl_msgs sensor_msgs visualization_msgs trajectory_msgs message_runtime
    CATKIN_DEPENDS # TODO
    INCLUDE_DIRS # TODO include
    LIBRARIES # TODO
)

#catkin_install_python(PROGRAMS scripts/executable1 
#    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}) 

