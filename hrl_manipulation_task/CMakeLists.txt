cmake_minimum_required(VERSION 2.8.3)
project(hrl_manipulation_task)

find_package(catkin REQUIRED COMPONENTS roscpp pcl_ros kdl_parser)
include_directories(include ${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})
#find_package(Boost REQUIRED COMPONENTS thread)

catkin_python_setup()

find_package(orocos_kdl REQUIRED)
if (orocos_kdl_FOUND)
  message("--------- User Print Out-----------")
  message("orocos_kdl include: ${orocos_kdl_INCLUDE_DIRS}")
  message("orocos_kdl lib: ${orocos_kdl_LIBRARIES}")
  message("-----------------------------------")
  include_directories(${orocos_kdl_INCLUDE_DIRS})
endif (orocos_kdl_FOUND)

set(CMAKE_PREFIX_PATH /usr/local)
find_package(PCL REQUIRED)
if (PCL_FOUND)
  message("--------- User Print Out-----------")
  message("PCL include: ${PCL_INCLUDE_DIRS}")
  message("PCL lib: ${PCL_LIBRARIES}")
  message("-----------------------------------")
  include_directories(${PCL_INCLUDE_DIRS})
endif (PCL_FOUND)

find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)
find_package(image_geometry REQUIRED)
find_package(cv_bridge REQUIRED)

catkin_package(
    DEPENDS rospy
    CATKIN_DEPENDS hrl_lib tf tf_conversions hark_msgs ar_track_alvar_msgs geometry_msgs pr2_msgs visualization_msgs pr2_controllers_msgs hrl_haptic_manipulation_in_clutter_msgs std_msgs sandbox_dpark_darpa_m3 hrl_anomaly_detection hrl_srvs sensor_msgs pcl_conversions
    INCLUDE_DIRS include
    LIBRARIES 
)

# 

#add_library(robot src/robot.cpp)
#target_link_libraries(robot ${orocos_kdl_LIBRARIES} ${catkin_LIBRARIES} ${Boost_LIBRARIES} )
##rosbuild_link_boost(sift_extractor thread signals)

add_executable(Overlay src/Kinect2Overlayer.cpp)
target_link_libraries(Overlay ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${image_geometry_LIBRARIES} ${image_transport_LIBRARIES} ${cv_bridge_LIBRARIES})

#add_executable(change_detector src/octree_change_detection.cpp src/util.cpp)
#target_link_libraries(change_detector ${PCL_LIBRARIES} robot ${catkin_LIBRARIES} ${Boost_LIBRARIES} )

# Camera Filter
#add_executable(sift_extractor src/pcl_sift_extractor.cpp)
#target_link_libraries(sift_extractor ${PCL_LIBRARIES} robot ${catkin_LIBRARIES} ${Boost_LIBRARIES} )

# Upsampling Filter
add_executable(pcl_upsample src/pcl_upsample.cpp)
target_link_libraries(pcl_upsample ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${Boost_LIBRARIES} )


#install(TARGETS sift_extractor
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
