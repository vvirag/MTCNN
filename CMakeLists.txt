cmake_minimum_required(VERSION 2.6)
project(mtcnn_cpp)

# C++ additional flags
if (CMAKE_COMPILER_IS_GNUCXX)
  message(STATUS "GCC detected, adding compile flags")
  set(OP_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -m64 -Wpedantic -Wall -Wextra -Wfatal-errors")
endif (CMAKE_COMPILER_IS_GNUCXX)

add_definitions(-std=c++11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

## FIND CUDA
set(CUDA_ARCH Auto CACHE STRING "Select target NVIDIA GPU achitecture.")
set_property(CACHE CUDA_ARCH PROPERTY STRINGS Auto All Manual)
set(USE_CUDNN ON)
include(cmake/Cuda.cmake)
#find_package(CuDNN)
# Set CUDA Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    
## FIND BOOST
find_package(Boost 1.66 REQUIRED COMPONENTS filesystem system)
message(STATUS "Boost information:") 
message(STATUS "  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}") 
message(STATUS "  Boost_LIBRARIES: ${Boost_LIBRARIES}") 
message(STATUS "  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}") 

## FIND OPENCV
if (OpenCV_CONFIG_FILE)
    include (${OpenCV_CONFIG_FILE})
  elseif (OpenCV_INCLUDE_DIRS AND OpenCV_LIBS_DIR)
    file(GLOB_RECURSE OpenCV_LIBS "${OpenCV_LIBS_DIR}*.so")
    set(OpenCV_FOUND 1)
  else ()
    find_package(OpenCV)
endif (OpenCV_CONFIG_FILE)
message(STATUS "OpenCV information:") 
message(STATUS "  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}") 
message(STATUS "  OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}") 
message(STATUS "  OpenCV_LIBRARY_DIRS: ${OpenCV_LIBRARY_DIRS}") 
  
## FIND CAFFE  
include(cmake/FindCaffe.cmake)

# Check if the user specified caffe paths
message(STATUS "\${Caffe_INCLUDE_DIRS} set by the user to " ${Caffe_INCLUDE_DIRS})
message(STATUS "\${Caffe_LIBS} set by the user to " ${Caffe_LIBS})


include_directories(
    include 
    ${OpenCV_INCLUDE_DIRS}
    ${Caffe_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    )
    
    
SET(SOURCES
    src/MTCNN/FaceDetector.cpp
)

SET(HEADERS
    include/MTCNN/FaceDetector.hpp
)

add_library( ${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
  

install (TARGETS ${PROJECT_NAME} DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/MTCNN/)
  
add_subdirectory(example)

