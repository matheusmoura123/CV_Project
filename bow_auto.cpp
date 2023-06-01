#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdio>

#include "segment_plates.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

const string DATASET_PATH = "../FoodCategories/";
const string IMAGE_EXT = ".png";
const int NUMBER_CLASSES = 17;
const int DICT_SIZE = 80*NUMBER_CLASSES;	//80 word per class
const int TESTING_PERCENT_PER = 7;

class food {
    public:
        string className;
        int imageNumbers;
        int classLable;
        food(string x, int y, int z) { // Constructor with parameters
            className = x;
            imageNumbers = y;
            classLable = z;
        }
};

const vector<food> foodCategories{
                         {"plate", 4, 0},
				         {"pasta", 12, 17},
                         {"pesto", 4, 1},
                         {"beans", 13, 10},
                         {"pomodoro", 5, 2},
                         {"ragu", 2, 3},
                         {"pasta_clams", 8, 4},
                         {"rice", 7, 5},
                         {"pork", 9, 6},
                         {"fish", 10, 7},
                         {"rabbit", 12, 8},
                         {"seafood", 5, 9},
                         {"potato", 13, 11},
                         {"bread", 12, 13},
                         {"carrot", 6, 14},
                         {"lettuce", 15, 12},
                         {"pepper", 2, 15},
                         {"tomato", 10, 16}
                    };

Mat allDescriptors;
vector<Mat> allDescPerImg;
vector<int> allClassPerImg;
int allDescPerImgNum = 0;
void readDetectComputeimage(const string& className, int imageNumbers, int classLable) {
    for (int i = 1; i <= imageNumbers; i++) {
        Mat grayimg;
        Ptr<SIFT> siftptr = SIFT::create();
        cvtColor(imread(DATASET_PATH + className + "/" + className + to_string(i) + IMAGE_EXT), grayimg, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat descriptors;
        siftptr->detect(grayimg, keypoints);
        siftptr->compute(grayimg, keypoints, descriptors);

        allDescriptors.push_back(descriptors);
        allDescPerImg.push_back(descriptors);
        allClassPerImg.push_back(classLable);
        allDescPerImgNum++;
    }
}

Mat kCenters, kLabels;
Mat getDataVector(Mat descriptors) {
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors, kCenters, matches);

    //Make a Histogram of visual words
    Mat datai = Mat::zeros(1, DICT_SIZE, CV_32F);
    int index = 0;
    for (auto j = matches.begin(); j < matches.end(); j++, index++) {
        datai.at<float>(0, matches.at(index).trainIdx) = datai.at<float>(0, matches.at(index).trainIdx) + 1;
    }
    return datai;
}

Mat inputData;
Mat inputDataLables;
void getHistogram(const string& className, int imageNumbers, int classLable) {
    for (int i = 1; i <= imageNumbers; i++) {
        Mat grayimg;
        Ptr<SIFT> siftptr = SIFT::create();
        cvtColor(imread(DATASET_PATH + className + "/" + className + to_string(i) + IMAGE_EXT), grayimg, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat descriptors;
        siftptr->detect(grayimg, keypoints);
        siftptr->compute(grayimg, keypoints, descriptors);

        inputData.push_back(getDataVector(descriptors));
        inputDataLables.push_back(Mat(1, 1, CV_32SC1, classLable));
    }
}

void getHistogramFast() {
        for (int i = 0; i < allDescPerImgNum; i++) {
            Mat dvec = getDataVector(allDescPerImg[i]);

            inputData.push_back(dvec);
            inputDataLables.push_back(Mat(1, 1, CV_32SC1, allClassPerImg[i]));
        }
}

Ptr<SVM> svm;
double testData(const string& className, int imageNumbers, int classLable) {
    int allTests = 0;
    int correctTests = 0;
    for (int i = TESTING_PERCENT_PER; i <= imageNumbers; i += TESTING_PERCENT_PER) {
        float r = 0;
        Mat grayimg;
        Ptr<SIFT> siftptr = SIFT::create();
        cvtColor(imread(DATASET_PATH + className + "/" + className + to_string(i) + IMAGE_EXT), grayimg, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat descriptors;
        siftptr->detect(grayimg, keypoints);
        siftptr->compute(grayimg, keypoints, descriptors);

        Mat dvector = getDataVector(descriptors);
        allTests++;
        if (svm->predict(dvector) == classLable) {
            correctTests++;
        }
    }
    return (double)correctTests / allTests;
}

void predictImg(const Mat& img) {
    Mat grayimg;
    Ptr<SIFT> siftptr = SIFT::create();
    cvtColor(img, grayimg, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints;
    Mat descriptors;
    siftptr->detect(grayimg, keypoints);
    siftptr->compute(grayimg, keypoints, descriptors);

    Mat dvector = getDataVector(descriptors);
    float prediction = svm->predict(dvector);
    cout << "Predicted category: " << prediction << endl;
    for (const auto & foodCategorie : foodCategories) {
        if (prediction == float(foodCategorie.classLable)){
            cout << "The plate has " << foodCategorie.className << "." << endl;
        }
    };

    Mat out;
    drawKeypoints(img, keypoints, out);
    namedWindow("Predicted Img");
    imshow("Predicted Img",out);
    waitKey(0);
}

int main(int argc, char **argv)
{

    for (int i = 0; i < NUMBER_CLASSES; ++i) {
        readDetectComputeimage(foodCategories[i].className, foodCategories[i].imageNumbers, foodCategories[i].classLable);
    };

    FileStorage storage("kmeans_FoodCategories.yml", FileStorage::READ);
    storage["kLabels"] >> kLabels;
    storage["kCenters"] >> kCenters;
    storage.release();

    getHistogramFast();

    svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
    Ptr<TrainData> td = TrainData::create(inputData, ROW_SAMPLE, inputDataLables);
    svm->train(td);

    string tray_num, image_num, plate_num;
    cout << "Tray: ";
    cin >> tray_num;
    if (tray > 0){
        if (atoi(argv[2]) > dishes.size()) index = dishes.size();
        else index = atoi(argv[2]);
    }

    Mat img1, dst;
    vector<Mat> dishes;
    img1 = imread(argv[1]);
    dishes = segment_plates(img1);
    int index = 1;
    if (argc > 2){
        if (atoi(argv[2]) > dishes.size()) index = dishes.size();
        else index = atoi(argv[2]);
    }
    dst = dishes[index-1].clone();
    predictImg(dst);

    return(0);
}