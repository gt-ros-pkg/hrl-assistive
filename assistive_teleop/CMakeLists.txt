cmake_minimum_required(VERSION 2.8.3)
project(assistive_teleop)
find_package(catkin REQUIRED COMPONENTS geometry_msgs std_msgs message_generation rospy tf image_rotate pr2_msgs pr2_controllers_msgs control_msgs pr2_common_action_msgs topic_tools sound_play pixel_2_3d rosbridge_server web_video_server)

catkin_python_setup()

add_service_files(
  DIRECTORY srv
  FILES LookatIk.srv
)

generate_messages(
    DEPENDENCIES geometry_msgs std_msgs
)

catkin_package(
    DEPENDS 
    CATKIN_DEPENDS rospy tf std_msgs geometry_msgs image_rotate pr2_msgs pr2_controllers_msgs control_msgs pr2_common_action_msgs topic_tools sound_play pixel_2_3d rosbridge_server web_video_server message_runtime
    INCLUDE_DIRS
    LIBRARIES
)


catkin_install_python(PROGRAMS 
    scripts/clicked_pose_relay
    scripts/contact
    scripts/twist_to_pose
    scripts/lookat_ik
    scripts/ik_move
    scripts/speech_intermediary
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

install(DIRECTORY www/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/www)

install(DIRECTORY vci-www/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/vci-www)
