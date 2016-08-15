cmake_minimum_required(VERSION 2.8.3)
project(hrl_task_planning)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

catkin_python_setup()

## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg
  FILES
  DomainList.msg
  PDDLProblem.msg
  PDDLSolution.msg
  PDDLState.msg
  PDDLPlanStep.msg
)

add_service_files(DIRECTORY srv
  FILES
  PDDLPlanner.srv
  PreemptTask.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES 
  CATKIN_DEPENDS rospy std_msgs message_runtime
  DEPENDS
)

catkin_install_python(PROGRAMS
   scripts/pddl_planner
   scripts/gripper_sensor_grasp_monitor
   scripts/grasping_monitor
   scripts/preprocess_problems
   scripts/task_smacher
   scripts/test_client
   scripts/domain_state_aggregator
   scripts/proximity_monitor
   scripts/parameter_monitor
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY params/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/params
)
