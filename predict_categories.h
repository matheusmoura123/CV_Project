#ifndef CV_PROJECT_PREDICT_CATEGORIES_H
#define CV_PROJECT_PREDICT_CATEGORIES_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <cmath>
#include <utility>
#include <sys/stat.h>

#include "segment_plates.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

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

void predict_categories(vector<food> categories, vector<Mat> images_to_predict, vector<string>& predicted_classNames);

#endif

