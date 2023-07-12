#ifndef CV_PROJECT_MAIN_HEADER_H
#define CV_PROJECT_MAIN_HEADER_H

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <utility>
#include <numeric>
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
    double conf;
    Mat img;
    box(int a, int b, int c, int d, int e, double f, const Mat& g) { // Constructor with parameters
        ID = a;
        p0x = b;
        p0y = c;
        width = d;
        height = e;
        conf = f;
        img = g.clone();
    }
    [[nodiscard]] bool is_inside(const int& y, const int& x) const {
        return((y >= p0y && y <= p0y+height) && (x >= p0x && x <= p0x+width));
    }
};


const string TRAY_PATH = "../Food_leftover_dataset/tray";
const string CATEGORIES_PATH = "../FoodCategories/";
const string RESULTS_PATH = "../FoodResults/tray";
const string IMAGE_EXT = ".jpg";
const int NUMBER_TRAYS = 8;

//find_histogram.cpp
array<int,4> find_histogram(const cv::Mat& src);
Mat mean_histogram2(const vector<cv::Mat>& src);
Mat mean_histogram(const vector<cv::Mat>& src);
array<double, 3> compare_histogram(const Mat& src_hist, const vector<Mat>& categories_hist);

//segment_plates.cpp
vector<box> segment_plates(const Mat& img, vector<Mat>& dst);
Mat get_contours(const Mat& img);
Mat segment_hsv(const Mat& src, int T_hue, int T_sat, int T_value);
Mat contour_hsv(const Mat& img);

//files_manager.cpp
int box_file_writer (const vector<box>& boxes, const string& path);
int box_file_reader (vector<box>& boxes, const string& path);

//metrics.cpp
double boxes_IoU (const box& box1, const box& box2);
double img_mAp (const vector<box>& boxes_truth, const vector<box>& boxes_result);
double masks_mIoU (const Mat& mask1, const Mat& mask2);
vector<vector<double>> leftover_ratio (const Mat& mask_before, const Mat& mask_after);

//results.cpp
int food_localization();
int food_segmentation ();
int food_leftover();

//archive
int sift_matching(const cv::Mat& img1, const cv::Mat& img2);
void predict_categories(vector<Mat> images_to_predict, vector<food> categories, vector<string>& predicted_classNames);

#endif //CV_PROJECT_MAIN_HEADER_H
