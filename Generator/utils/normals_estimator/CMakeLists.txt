cmake_minimum_required(VERSION 3.10)
project(normals_estimator)
set(CMAKE_BUILD_TYPE Release)

# PCL
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -fPIC -Wno-deprecated -Wenum-compare -std=c++14")

# Eigen library
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(normals_estimator normals_estimator.cpp)

target_link_libraries(normals_estimator
        ${PCL_LIBRARIES}
        ${EIGEN_LIBRARIES})
