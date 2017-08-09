cmake_minimum_required(VERSION 2.8.3)
project(hrl_dressing)

find_package(catkin REQUIRED COMPONENTS message_generation rospy std_msgs geometry_msgs hrl_msgs netft_rdt_driver zenither)

catkin_python_setup()

add_message_files(
  DIRECTORY msg
  FILES Viz.msg
)

generate_messages(
  DEPENDENCIES std_msgs
)

# catkin_package parameters: http://ros.org/doc/groovy/api/catkin/html/dev_guide/generated_cmake_api.html#catkin-package
# TODO: fill in what other packages will need to use this package
catkin_package(
    DEPENDS rospy std_msgs geometry_msgs hrl_msgs netft_rdt_driver zenither
    CATKIN_DEPENDS message_runtime # TODO
    INCLUDE_DIRS # TODO include
    LIBRARIES # TODO
)
