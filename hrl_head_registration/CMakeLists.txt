cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()

rosbuild_add_boost_directories()
rosbuild_add_library(${PROJECT_NAME} src/pcl_basic.cpp)
rosbuild_add_library(${PROJECT_NAME} src/head_registration.cpp)

#rosbuild_add_executable(test_face_detect src/test_face_detect.cpp)
rosbuild_add_executable(head_registration src/head_registration.cpp)
rosbuild_add_executable(head_registration_srv src/head_registration_srv.cpp)
rosbuild_add_executable(head_training src/head_training.cpp)
rosbuild_add_executable(pub_pc src/pub_pc.cpp)
rosbuild_add_executable(clickable_pc src/clickable_pc.cpp)
rosbuild_add_executable(capture_pc src/capture_pc.cpp)
rosbuild_add_executable(head_alignment_confirm src/head_alignment_confirm.cpp)
rosbuild_add_executable(head_bowl_alignment_confirm src/head_bowl_alignment_confirm.cpp)
target_link_libraries(head_registration ${PROJECT_NAME})
target_link_libraries(head_registration_srv ${PROJECT_NAME})
target_link_libraries(head_training ${PROJECT_NAME})
target_link_libraries(pub_pc ${PROJECT_NAME})
target_link_libraries(clickable_pc ${PROJECT_NAME})
target_link_libraries(capture_pc ${PROJECT_NAME})
