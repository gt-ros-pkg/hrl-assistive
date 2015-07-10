cmake_minimum_required(VERSION 2.8.3)
project(hrl_head_registration)
find_package(catkin REQUIRED COMPONENTS rospy std_msgs std_srvs tf roscpp pcl_ros eigen_conversions cv_bridge image_transport image_geometry hrl_geom hrl_lib)
find_package(OpenCV)
find_package(PCL)

catkin_python_setup()

add_service_files(
  DIRECTORY srv
  FILES 
  ConfirmRegistration.srv 
  HeadRegistration.srv 
)

generate_messages(
    DEPENDENCIES geometry_msgs std_msgs
)


catkin_package(
    DEPENDS pcl opencv2
    CATKIN_DEPENDS rospy std_msgs std_srvs tf roscpp pcl_ros eigen_conversions cv_bridge image_transport image_geometry hrl_geom hrl_lib 
    INCLUDE_DIRS include
    LIBRARIES pcl_basic
)

include_directories(include 
                    ${catkin_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS})

add_library(pcl_basic src/pcl_basic.cpp)
target_link_libraries(pcl_basic ${catkin_LIBRARIES} ${pcl_LIBRARIES})
add_library(head_registration_lib src/head_registration.cpp)
target_link_libraries(head_registration_lib pcl_basic ${catkin_LIBRARIES} ${pcl_LIBRARIES})

add_executable(head_registration src/head_registration.cpp)
add_executable(head_registration_srv src/head_registration_srv.cpp)
add_dependencies(head_registration_srv ${catkin_EXPORTED_TARGETS})
add_executable(pub_pc src/pub_pc.cpp)
add_executable(clickable_pc src/clickable_pc.cpp)
add_executable(capture_pc src/capture_pc.cpp)
add_executable(head_alignment_confirm src/head_alignment_confirm.cpp)
#add_executable(head_bowl_alignment_confirm src/head_bowl_alignment_confirm.cpp)
target_link_libraries(head_registration_srv head_registration_lib ${catkin_LIBRARIES})
target_link_libraries(head_registration head_registration_lib ${catkin_LIBRARIES})
target_link_libraries(pub_pc pcl_basic)
target_link_libraries(clickable_pc head_registration_lib)
target_link_libraries(capture_pc head_registration_lib)
target_link_libraries(head_alignment_confirm ${catkin_LIBRARIES})

install(TARGETS pcl_basic head_registration_lib
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(TARGETS
        pub_pc
        clickable_pc 
        capture_pc 
        head_registration 
        head_registration_srv
        head_alignment_confirm
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

catkin_install_python(PROGRAMS scripts/registration_loader
                    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
