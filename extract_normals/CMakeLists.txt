cmake_minimum_required(VERSION 3.10)
project(extract_normals)
set(CMAKE_BUILD_TYPE Release)

# PCL
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -fPIC -Wno-deprecated -Wenum-compare -std=c++14")

# Eigen library
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(extract_normals extract_normals.cpp)

target_link_libraries(extract_normals
        ${PCL_LIBRARIES}
        ${EIGEN_LIBRARIES})
