cmake_minimum_required(VERSION 2.8)
project(CV_Project)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}
        ../main.cpp
        ../main_header.h
        ../segment_plates.cpp
        ../find_histogram.cpp
        ../files_manager.cpp
        ../metrics.cpp
        ../results.cpp
        ../archive/sift_matching.cpp
        ../predict_categories.cpp)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )




