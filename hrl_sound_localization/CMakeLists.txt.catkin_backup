cmake_minimum_required(VERSION 2.8.3)
project(hrl_sound_localization)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS 
  LIBRARIES
  CATKIN_DEPENDS rospy tf std_msgs hark_msgs hrl_common_code_darpa_m3
  DEPENDS 
)

catkin_install_python(
  PROGRAMS scripts/displaySource
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)