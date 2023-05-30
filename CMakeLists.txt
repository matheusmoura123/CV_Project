cmake_minimum_required(VERSION 2.8)
project(CV_Project)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ../main.cpp ../segment_plates.cpp ../segment_plates.h ../sift_matching.cpp ../sift_matching.h)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
