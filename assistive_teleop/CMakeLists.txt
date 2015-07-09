cmake_minimum_required(VERSION 2.8.3)
project(assistive_teleop)
find_package(catkin REQUIRED COMPONENTS geometry_msgs std_msgs message_generation rospy tf image_rotate pr2_msgs pr2_controllers_msgs control_msgs pr2_common_action_msgs arm_navigation_msgs kinematics_msgs topic_tools sound_play pixel_2_3d rosbridge_server mjpeg_server message_runtime)

add_service_files(
  DIRECTORY srv
  FILES LookatIk.srv
)

generate_messages(
    DEPENDENCIES geometry_msgs std_msgs
)

catkin_package(
    DEPENDS 
    CATKIN_DEPENDS rospy tf std_msgs geometry_msgs image_rotate pr2_msgs pr2_controllers_msgs control_msgs pr2_common_action_msgs arm_navigation_msgs kinematics_msgs topic_tools sound_play pixel_2_3d rosbridge_server mjpeg_server message_runtime
    INCLUDE_DIRS
    LIBRARIES
)
