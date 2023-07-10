#ifndef CV_PROJECT_MAIN_HEADER_H
#define CV_PROJECT_MAIN_HEADER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <cmath>
#include <utility>
#include <sys/stat.h>

array<int,4> find_histogram(const cv::Mat& src);
Mat mean_histogram(const vector<cv::Mat>& src);
Mat mean_histogram2(const vector<cv::Mat>& src);
array<double, 2> compare_histogram(const Mat& src_hist, const vector<Mat>& categories_hist);


vector<Mat> segment_plates(const Mat& img);

Mat get_contours(const Mat& img);

int segment_hsv(const Mat& src, Mat &dst, int T_hue, int T_sat, int T_value);

Mat contour_hsv(const Mat& img);

int sift_matching(const cv::Mat& img1, const cv::Mat& img2);


#endif //CV_PROJECT_MAIN_HEADER_H
