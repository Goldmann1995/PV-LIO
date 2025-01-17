cmake_minimum_required(VERSION 3.10)
project(pv_lio)

#SET(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
# if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 5)
    math(EXPR PROC_NUM "${N} - 2")
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=${PROC_NUM})
    add_definitions(-DOPENBLAS_NUM_THREADS=${PROC_NUM})
    message("core for MP:  ${PROC_NUM}")
  elseif(N GREATER 3)
    math(EXPR PROC_NUM "${N} - 1")
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=${PROC_NUM})
    add_definitions(-DOPENBLAS_NUM_THREADS=${PROC_NUM})
    message("core for MP:  ${PROC_NUM}")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()

find_package(Boost REQUIRED COMPONENTS timer)

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

# find_package(OpenBLAS)
# if (OpenBLAS_FOUND)
#   message("OpenBLAS Found")
#   set(OpenBLAS_LIBRARIES openblas)
#   add_definitions(-DEIGEN_USE_BLAS)
#   add_definitions(-DEIGEN_USE_LAPACKE)
# endif()

# find_package(PythonLibs REQUIRED)
# find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
#find_package(GTSAM REQUIRED QUIET)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenBLAS_INCLUDE_DIRS}
  # ${PYTHON_INCLUDE_DIRS}
  include)

#add_message_files(
#  FILES
#  Pose6D.msg
#)
add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
)
generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_executable(pv_lio_node 
  src/voxelMapping.cpp 
  src/preprocess.cpp
  src/IMU_Processing.cpp)
target_link_libraries(pv_lio_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenBLAS_LIBRARIES})
# ${PYTHON_LIBRARIES})
# target_include_directories(pv_lio_node PRIVATE ${PYTHON_INCLUDE_DIRS})

# add_executable(pandar_to_velodyne src/converter/pandar_to_velodyne.cpp)
# target_link_libraries(pandar_to_velodyne ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
# target_include_directories(pandar_to_velodyne PRIVATE ${PYTHON_INCLUDE_DIRS})

# add_executable(rs_to_velodyne src/converter/rs_to_velodyne.cpp)
# target_link_libraries(rs_to_velodyne ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
# target_include_directories(rs_to_velodyne PRIVATE ${PYTHON_INCLUDE_DIRS})
