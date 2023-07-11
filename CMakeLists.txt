cmake_minimum_required(VERSION 2.8)
project(CV_Project)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ../main.cpp ../segment_plates.cpp ../find_histogram.cpp ../main_header.h ../file_writer.cpp ../metrics.cpp)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

add_executable(meanshift ../meanshift.cpp ../segment_plates.cpp ../find_histogram.cpp ../main_header.h ../file_writer.cpp ../metrics.cpp)
target_link_libraries(meanshift ${OpenCV_LIBS} )

add_executable(identification ../identification.cpp ../segment_plates.cpp ../find_histogram.cpp  ../main_header.h ../file_writer.cpp ../metrics.cpp)
target_link_libraries(identification ${OpenCV_LIBS} )