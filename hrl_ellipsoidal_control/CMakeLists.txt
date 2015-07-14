cmake_minimum_required(VERSION 2.8.3)
project(hrl_ellipsoidal_control)
find_package(catkin REQUIRED COMPONENTS std_msgs geometry_msgs actionlib_msgs message_generation actionlib hrl_pr2_arms interactive_markers tf_conversions pcl_ros pykdl_utils sensor_msgs)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED)

catkin_python_setup()

add_action_files(
  DIRECTORY action/
  FILES EllipsoidMove.action
)

#uncomment if you have defined messages
add_message_files(
  DIRECTORY msg/
  FILES EllipsoidParams.msg
)
#uncomment if you have defined services
add_service_files(
  DIRECTORY srv
  FILES LoadEllipsoidParams.srv
)

generate_messages(
    DEPENDENCIES geometry_msgs std_msgs actionlib_msgs
)

include_directories(include/${PACKAGE_NAME} ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
add_executable(interactive_ellipsoid src/interactive_ellipsoid.cpp)
add_executable(ellipsoid_visualizer src/ellipsoid_visualizer.cpp)
add_dependencies(ellipsoid_visualizer ${catkin_EXPORTED_TARGETS})
add_dependencies(interactive_ellipsoid ${catkin_EXPORTED_TARGETS})
target_link_libraries(ellipsoid_visualizer ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(interactive_ellipsoid ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES})

catkin_package(
    DEPENDS 
    CATKIN_DEPENDS std_msgs geometry_msgs sensor_msgs actionlib_msgs message_runtime actionlib hrl_pr2_arms interactive_markers tf_conversions pcl_ros pykdl_utils
    INCLUDE_DIRS include
    LIBRARIES 
)

catkin_install_python(PROGRAMS
                      scripts/head_tool_pose_server
                      scripts/head_markers
                      scripts/ellipsoidal_interface_backend
                      scripts/cartesian_interface_backend
                      scripts/arm_cart_control_interface
                      scripts/cartesian_controller
                      scripts/ellipsoid_controller
                      scripts/ellipsoid_space
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(TARGETS
        ellipsoid_visualizer
        interactive_ellipsoid
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

install(DIRECTORY params/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/params)
