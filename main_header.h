#ifndef CV_PROJECT_MAIN_HEADER_H
#define CV_PROJECT_MAIN_HEADER_H

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <utility>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

class food {
public:
    string className;
    int imageNumbers;
    int classLable;
    food(string x, int y, int z) { // Constructor with parameters
        className = std::move(x);
        imageNumbers = y;
        classLable = z;
    }
};

class box {
public:
    int ID;
    int p0x;
    int p0y;
    int width;
    int height;
    box(int a, int b, int c, int d, int e) { // Constructor with parameters
        ID = a;
        p0x = b;
        p0y = c;
        width = d;
        height = e;
    }
};

const string TRAY_PATH = "../Food_leftover_dataset/tray";
const string CATEGORIES_PATH = "../FoodCategories/";
const string RESULTS_PATH = "../FoodResults/tray";
const string IMAGE_EXT = ".jpg";
const int NUMBER_TRAYS = 8;

array<int,4> find_histogram(const cv::Mat& src);

Mat mean_histogram(const vector<cv::Mat>& src);

Mat mean_histogram2(const vector<cv::Mat>& src);

array<double, 2> compare_histogram(const Mat& src_hist, const vector<Mat>& categories_hist);

vector<box> segment_plates(const Mat& img, vector<Mat>& dst);

Mat get_contours(const Mat& img);

Mat segment_hsv(const Mat& src, int T_hue, int T_sat, int T_value);

Mat contour_hsv(const Mat& img);

int sift_matching(const cv::Mat& img1, const cv::Mat& img2);

void predict_categories(vector<Mat> images_to_predict, vector<food> categories, vector<string>& predicted_classNames);

int box_file_writer (const vector<box>& boxes, const string& path);

double boxes_IoU (const box& box1, const box& box2);

#endif //CV_PROJECT_MAIN_HEADER_H
