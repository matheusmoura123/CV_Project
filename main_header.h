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
    int classLabel;
    food(string x, int y, int z) { // Constructor with parameters
        className = std::move(x);
        imageNumbers = y;
        classLabel = z;
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

const vector<food> foodCategories{
        //{className, numberOfImgs, ID},
        {"pasta", 13, 19},
        {"pesto", 11, 1},
        {"pomodoro", 5, 2},
        {"ragu", 2, 3},
        {"pasta_clams", 8, 4},
        {"rice", 6, 5},
        {"pork", 15, 6},
        {"fish", 14, 7},
        {"rabbit", 11, 8},
        {"seafood", 8, 9},
        {"beans", 13, 10},
        {"potato", 10, 11}
};

const string TRAY_PATH = "../Food_leftover_dataset/tray";
const string CATEGORIES_PATH = "../FoodCategories/";
const string RESULTS_PATH = "../FoodResults/tray";
const string IMAGE_EXT = ".jpg";
const string MASK_EXT = ".png";
const int NUMBER_TRAYS = 8;

//find_histogram.cpp
vector<Mat> categories_histogram (const vector<int>& categories);
array<double, 3> compare_histogram(const Mat& src_hist, const vector<Mat>& categories_hist);
Mat mean_histogram2(const vector<cv::Mat>& src);
array<int,3> find_histogram(const cv::Mat& src);
void plot_histogram(Mat histogram);
bool sort_bigger_area (const box& i,const box& j);
bool sort_ID (const box& i,const box& j);
bool sort_ID_bigger (const box& i,const box& j);
bool sort_num_match (const vector<int>& i,const vector<int>& j);


//segment_plates.cpp
vector<box> segment_plates(const Mat& img, vector<Mat>& dst);
Mat get_contours(const Mat& img);
Mat segment_rgb_hsv(const Mat& src, int hue, int sat, int val, int T_hue, int T_sat, int T_value, bool is_hsv=false);
Mat meanshift(const Mat& img, int spatial, int color);
Mat otsu_segmentation(const Mat& gray_img, int num_grid);
box crop_image(const Mat& img, const box& plate_box);
box segment_food(const box& plate_box);
Mat K_means(const Mat& img, int num_of_clusters);
vector<box> separate_food(const box& food_box) ;

//predict_categories.cpp
void predict_categories(const vector<Mat>& images_to_predict, vector<food> categories, vector<int>& pred_IDs, vector<double>& pred_strengths);
void write_kmeans(vector<food> categories);

//sift_matching.cpp
int sift_matching(const cv::Mat& img1, const cv::Mat& img2);
int surf_matching(const Mat& img1, const Mat& img2);
void compare_plates(const vector<box> food_plate, vector<box>& leftover_plate);

//files_manager.cpp
void draw_rectangles_masks (Mat img, const vector<box>& boxes, int tray_num, const string& file_name);
int box_file_writer (const vector<box>& boxes, const string& path);
int box_file_reader (vector<box>& boxes, const string& path);
Mat mask_img_builder (const vector<box>& boxes);
Mat mask_img_builder_color (const vector<box>& boxes);
int mask_file_writer (const vector<box>& boxes, const string& path);
int save_boxes_masks_at_tray_stage (const vector<box>& boxes, int tray_num, const string& file_name);

//metrics.cpp
double boxes_IoU (const box& box1, const box& box2);
double img_mAp (const vector<box>& boxes_truth, const vector<box>& boxes_result);
double masks_mIoU (const Mat& mask1, const Mat& mask2);
vector<vector<double>> leftover_ratio (const Mat& mask_before, const Mat& mask_after);

//results.cpp
int food_localization();
int food_segmentation ();
int food_leftover();

#endif //CV_PROJECT_MAIN_HEADER_H
