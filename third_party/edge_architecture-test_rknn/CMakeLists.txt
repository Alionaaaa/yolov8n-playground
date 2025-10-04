cmake_minimum_required(VERSION 3.14)
project(combined_modules LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

# Флаги оптимизации
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# OCSort module
file(GLOB OCSORT_SRC_LIST tracker_src/*.cpp)

pybind11_add_module(ocsort_module 
    tracker_cpp/tracker.cpp
    ${OCSORT_SRC_LIST}
)

target_include_directories(ocsort_module PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/tracker_include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}  
)

target_link_libraries(ocsort_module PUBLIC 
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

