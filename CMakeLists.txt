cmake_minimum_required(VERSION 2.8)
project(CV_Project)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ../main.cpp ../segment_plates.cpp ../sift_matching.cpp ../find_histogram.cpp ../predict_categories.cpp ../main_header.h)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

add_executable(bow ../bow.cpp ../segment_plates.cpp ../sift_matching.cpp ../find_histogram.cpp ../predict_categories.cpp ../main_header.h)
target_link_libraries(bow ${OpenCV_LIBS} )

add_executable(img_seg ../img_seg.cpp ../segment_plates.cpp ../sift_matching.cpp ../find_histogram.cpp ../main_header.h)
target_link_libraries(img_seg ${OpenCV_LIBS} )

add_executable(meanshift ../meanshift.cpp ../segment_plates.cpp ../sift_matching.cpp ../find_histogram.cpp ../main_header.h)
target_link_libraries(meanshift ${OpenCV_LIBS} )

add_executable(identification ../segment_plates.cpp ../sift_matching.cpp ../find_histogram.cpp ../identification.cpp ../main_header.h)
target_link_libraries(identification ${OpenCV_LIBS} )