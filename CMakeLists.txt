cmake_minimum_required(VERSION 2.8)
project(CV_Project)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} ../main.cpp ../segment_plates.cpp ../segment_plates.h ../sift_matching.cpp ../sift_matching.h ../find_histogram.h ../find_histogram.cpp ../predict_categories.cpp ../predict_categories.h)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

add_executable(bow ../bow.cpp ../segment_plates.cpp ../segment_plates.h ../sift_matching.cpp ../sift_matching.h ../find_histogram.h ../find_histogram.cpp ../predict_categories.cpp ../predict_categories.h)
target_link_libraries(bow ${OpenCV_LIBS} )

add_executable(img_seg ../img_seg.cpp ../segment_plates.cpp ../segment_plates.h ../sift_matching.cpp ../sift_matching.h ../find_histogram.h ../find_histogram.cpp)
target_link_libraries(img_seg ${OpenCV_LIBS} )