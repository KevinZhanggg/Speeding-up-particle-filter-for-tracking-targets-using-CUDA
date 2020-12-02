cmake_minimum_required(VERSION 3.7)
project(find-the-bombs)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(GSL REQUIRED)
find_package(CUDA QUIET REQUIRED)

set(INCLUDE_DIRS
        ${PROJET_SOURCE_DIR}
        ${OPENCV_INCLUDE_DIR}
        ${GSL_INCLUDE_DIRS}
        )
set(LIBS
        ${OpenCV_LIBS}
        ${GSL_LIBRARIES}
        )
#set(CMAKE_BUILD_TYPE Profile)

#SET(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS_PROFILE "-pg")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

# nvcc flags
set(CUDA_NVCC_FLAGS -O3)

#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


INCLUDE_DIRECTORIES (/usr/local/cuda-10.0/include/ )
INCLUDE_DIRECTORIES (/home/jinhua/NVIDIA_CUDA-10.0_Samples/common/inc/)

include_directories(${INCLUDE_DIRS})
set(SRCS
        main.cpp       
        commander_d.cpp
        dog_d.cu
	dog.cpp
        police_d.cu       
        utils.cu 
 	colorhistogram.cpp       
        )

cuda_add_executable(find-the-bombs 
        main.cpp
        commander_d.cpp
	dog.cpp
        dog_d.cu
        police_d.cu     
        utils.cu 
        colorhistogram.cpp    
  )

target_link_libraries(find-the-bombs ${LIBS})
target_link_libraries(find-the-bombs gflags)
