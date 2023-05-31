#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#ifndef CV_PROJECT_SEGMENT_PLATES_H
#define CV_PROJECT_SEGMENT_PLATES_H

Mat segment_plates(const Mat& img);

Mat get_contours(const Mat& img);

int segment_hsv(const Mat& src, Mat &dst, int T_hue, int T_sat, int T_value);

Mat contour_hsv(const Mat& img);

#endif //CV_PROJECT_SEGMENT_PLATES_H
