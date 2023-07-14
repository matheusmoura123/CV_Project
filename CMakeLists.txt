cmake_minimum_required(VERSION 2.8)
project(CV_Project)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ../main.cpp ../main_header.h ../segment_plates.cpp ../find_histogram.cpp ../files_manager.cpp ../metrics.cpp ../results.cpp)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

add_executable(meanshift ../meanshift.cpp ../main_header.h ../segment_plates.cpp ../find_histogram.cpp ../files_manager.cpp ../metrics.cpp ../results.cpp)
target_link_libraries(meanshift ${OpenCV_LIBS} )

add_executable(identification ../identification.cpp ../main_header.h ../segment_plates.cpp ../find_histogram.cpp ../files_manager.cpp ../metrics.cpp ../results.cpp)
target_link_libraries(identification ${OpenCV_LIBS} )

add_executable(final_segmentation ../final_segmentation.cpp ../main_header.h ../segment_plates.cpp ../find_histogram.cpp ../files_manager.cpp ../metrics.cpp ../results.cpp)
target_link_libraries(final_segmentation ${OpenCV_LIBS} )


