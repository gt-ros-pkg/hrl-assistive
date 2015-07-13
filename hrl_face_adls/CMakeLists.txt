cmake_minimum_required(VERSION 2.8.3)
project(hrl_face_adls)
find_package(catkin REQUIRED COMPONENTS std_msgs geometry_msgs message_generation)

catkin_python_setup()

add_message_files(
  DIRECTORY msg
  FILES

)
#uncomment if you have defined services
add_service_files(
  DIRECTORY srv
  FILES 
  PointMirror.srv
  EnableFaceController.srv
  EnableCartController.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES geometry_msgs std_msgs
)

catkin_package(
    DEPENDS 
    CATKIN_DEPENDS std_msgs geometry_msgs message_runtime
    INCLUDE_DIRS 
    LIBRARIES
)

catkin_install_python(PROGRAMS 
                      scripts/cartesian_manager
                      scripts/mirrir_pointer
                      scripts/razor_runstop
                      scripts/visualize_poses
                      scripts/record_ell_poses
                      scripts/face_adls_manager
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

install(DIRECTORY params/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/params)
