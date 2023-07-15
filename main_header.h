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

class Interface
{
public:
    [[nodiscard]] virtual Interface* clone() const = 0;
};


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

class box: public Interface {
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
    [[nodiscard]] Interface* clone() const override { return new box(*this); }
};

const vector<food> pastaCategories{
        {"pesto", 4, 1},
        {"pomodoro", 5, 2},
        {"ragu", 2, 3},
        {"pasta_clams", 8, 4},
        {"rice", 5, 5}
};
const vector<food> foodCategories{
        //{className, numberOfImgs, ID},
        //{"plate", 9, 0},
        {"pasta", 13, 19},
        {"pesto", 4, 1},
        {"pomodoro", 5, 2},
        {"ragu", 2, 3},
        {"pasta_clams", 8, 4},
        {"rice", 6, 5},
        {"pork", 9, 6},
        {"fish", 10, 7},
        {"rabbit", 12, 8},
        {"seafood", 5, 9},
        {"beans", 14, 10},
        {"potato", 7, 11},
        //{"salad", 15, 12},
        //{"bread", 18, 13},
        //{"carrot", 6, 14},
        //{"pepper", 2, 15},
        //{"tomato", 10, 16},
        //{"lettuce", 15, 17},
        //{"plate_salad", 3, 18},
};

const string TRAY_PATH = "../Food_leftover_dataset/tray";
const string CATEGORIES_PATH = "../FoodCategories/";
const string RESULTS_PATH = "../FoodResults/tray";
const string IMAGE_EXT = ".jpg";
const string MASK_EXT = ".png";
const int NUMBER_TRAYS = 8;

//find_histogram.cpp
array<int,3> find_histogram(const cv::Mat& src);
Mat mean_histogram2(const vector<cv::Mat>& src);
array<double, 3> compare_histogram(const Mat& src_hist, const vector<Mat>& categories_hist);
void plot_histogram(Mat histogram);

//segment_plates.cpp
vector<box> segment_plates(const Mat& img, vector<Mat>& dst);
Mat get_contours(const Mat& img);
Mat segment_rgb_hsv(const Mat& src, int hue, int sat, int val, int T_hue, int T_sat, int T_value, bool is_hsv=false);
Mat contour_hsv(const Mat& img);
Mat meanshift(Mat img, int spatial, int color);
Mat otsu_segmentation(Mat gray_img, int num_grid);
box crop_image(const Mat& img, const box& plate_box);
box segment_food(const box& plate_box);
Mat K_means(Mat img, int num_of_clusters);
vector<box> separate_food(Mat food_box);



//files_manager.cpp
int box_file_writer (const vector<box>& boxes, const string& path);
int box_file_reader (vector<box>& boxes, const string& path);
Mat mask_img_builder (const vector<box>& boxes);
int mask_file_writer (const vector<box>& boxes, const string& path);

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
void predict_categories(const vector<Mat>& images_to_predict, vector<food> categories, vector<string>& predicted_classNames);

#endif //CV_PROJECT_MAIN_HEADER_H
