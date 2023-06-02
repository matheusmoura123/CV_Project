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
#include <sys/stat.h>

#include "segment_plates.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

const string DATASET_PATH = "../FoodCategories/";
const string TRAY_PATH = "../Food_leftover_dataset/tray";
const string IMAGE_EXT = ".jpg";
const int TESTING_PERCENT_PER = 7;
bool EXIT = false;
const int NUMBER_TRAYS = 8;

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
                         {"pesto", 4, 1},
                         {"pomodoro", 5, 2},
                         {"ragu", 2, 3},
                         {"pasta_clams", 8, 4},
                         {"rice", 7, 5},
                         {"pork", 9, 6},
                         {"fish", 10, 7},
                         {"rabbit", 12, 8},
                         {"seafood", 5, 9},
                         {"beans", 13, 10},
                         {"potato", 13, 11},
                         {"lettuce", 15, 12},
                         {"bread", 12, 13},
                         {"carrot", 6, 14},
                         {"pepper", 2, 15},
                         {"tomato", 10, 16},
                         {"pasta", 12, 17},
                    };

const int NUMBER_CLASSES = int(foodCategories.size());
const int DICT_SIZE = 80*NUMBER_CLASSES;	//80 word per class

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

void predictImg(const Mat& img, Mat& dst, string& className) {
    Mat grayimg;
    Ptr<SIFT> siftptr = SIFT::create();
    cvtColor(img, grayimg, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints;
    Mat descriptors;
    siftptr->detect(grayimg, keypoints);
    siftptr->compute(grayimg, keypoints, descriptors);

    Mat dvector = getDataVector(descriptors);
    string predicted_dish;
    float prediction = svm->predict(dvector);
    for (const auto & foodCategorie : foodCategories) {
        if (prediction == float(foodCategorie.classLable)){
            predicted_dish = foodCategorie.className;
        }
    };

    Mat out;
    drawKeypoints(img, keypoints, out);
    dst = out.clone();
    className = predicted_dish;
}

int main(int argc, char **argv)
{
    clock_t sTime = clock();
    for (int i = 0; i < NUMBER_CLASSES; ++i) {
        readDetectComputeimage(foodCategories[i].className, foodCategories[i].imageNumbers, foodCategories[i].classLable);
    };

    try {
        const char* dir = "./kmeans_FoodCategories.yml";
        struct stat sb{};
        if (stat(dir, &sb) == 0){
            FileStorage storage("kmeans_FoodCategories.yml", FileStorage::READ);
            storage["kLabels"] >> kLabels;
            storage["kCenters"] >> kCenters;
            storage.release();
        }
        else{
            cout << "The Kmeans haven't been calculated yet!" << endl;
            throw 505;
        }
    }
    catch(int err) {
        int clusterCount = DICT_SIZE, attempts = 5, iterationNumber = 1e4;
        sTime = clock();
        cout << "Running kmeans..." << endl;
        kmeans(allDescriptors, clusterCount, kLabels, TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, iterationNumber, 1e-4), attempts, KMEANS_PP_CENTERS, kCenters);
        cout << "-> kmeans run in " << (clock() - sTime) / double(CLOCKS_PER_SEC) << " Second(s)." << endl;
        cout << "---------------------------------------------------" << endl;

        FileStorage storage("kmeans_FoodCategories.yml", FileStorage::WRITE);
        storage << "kLabels" << kLabels << "kCenters" << kCenters;
        storage.release();

    }

    getHistogramFast();

    svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
    Ptr<TrainData> td = TrainData::create(inputData, ROW_SAMPLE, inputDataLables);
    svm->train(td);

    int option = 2;
    cout << "[1] Run all trays" << endl;
    cout << "[2] Run specific trays" << endl;
    cout << "[3] Run specific tray with no plate segmentation" << endl;
    cout << "Option: ";
    cin >> option;
    cout << "---------------------------------------------------" << endl;

    if (option == 1){
        cout << "Press [ANY] key to keep going or [ESC] to exit." << endl;
        for (int i = 0; i < NUMBER_TRAYS; ++i) {
            for (int j = 0; j < 4; ++j) {
                int key;
                string file_name;
                switch(j) {
                    case 0:
                        file_name = "food_image";
                        break;
                    case 1:
                        file_name = "leftover1";
                        break;
                    case 2:
                        file_name = "leftover2";
                        break;
                    case 3:
                        file_name = "leftover3";
                        break;
                    default:
                        file_name = "food_image";
                }

                Mat img;
                //img1 = imread(argv[1]);
                img = imread(TRAY_PATH + to_string(i+1) + "/" + file_name + IMAGE_EXT);

                string window_name_img = "Tray " + to_string(i+1) + " " + file_name;
                namedWindow(window_name_img, WINDOW_NORMAL);
                resizeWindow(window_name_img, 600, 400);
                imshow(window_name_img, img);
                key = waitKeyEx(0);
                if (key == 1048603) return(0);

                Mat img_to_predict, dst;
                vector<Mat> predictions, dishes;
                string predicted;
                vector<string> predicted_classes;
                dishes = segment_plates(img);
                int k=0;
                for (const auto & dishe : dishes) {
                    k++;
                    img_to_predict = dishe.clone();
                    predictImg(img_to_predict, dst, predicted);
                    predictions.push_back(dst);
                    predicted_classes.push_back(predicted);

                    string window_name = to_string(k) + ": This image have " + predicted;
                    namedWindow(window_name, WINDOW_NORMAL);
                    resizeWindow(window_name, 400, 400);
                    imshow(window_name, dst);
                    key = waitKeyEx(0);
                    if (key == 1048603) return(0);
                }
                destroyAllWindows();
            }

        }

    }
    else if(option == 2){
        while (!EXIT) {
            int key;
            int tray_num, image_num;
            cout << "Tray: ";
            cin >> tray_num;
            if (tray_num <= 0 or tray_num > 8) tray_num = 1;
            cout << "Image Number {0, 1, 2, 3}: ";
            cin >> image_num;
            cout << "Press [ANY] key to keep going or [ESC] to exit." << endl;
            cout << "---------------------------------------------------" << endl;
            string file_name;
            switch(image_num) {
                case 0:
                    file_name = "food_image";
                    break;
                case 1:
                    file_name = "leftover1";
                    break;
                case 2:
                    file_name = "leftover2";
                    break;
                case 3:
                    file_name = "leftover3";
                    break;
                default:
                    file_name = "food_image";
            }

            Mat img;
            //img1 = imread(argv[1]);
            img = imread(TRAY_PATH + to_string(tray_num) + "/" + file_name + IMAGE_EXT);

            string window_name_img = "Tray " + to_string(tray_num) + " " + file_name;
            namedWindow(window_name_img, WINDOW_NORMAL);
            resizeWindow(window_name_img, 600, 400);
            imshow(window_name_img, img);
            moveWindow(window_name_img, 500, 500);
            key = waitKeyEx(0);
            if (key == 1048603) return(0); //if press ESC end program


            Mat img_to_predict, dst;
            vector<Mat> predictions, dishes;
            string predicted;
            vector<string> predicted_classes;
            dishes = segment_plates(img);
            int i=0;
            for (const auto & dishe : dishes) {
                i++;
                img_to_predict = dishe.clone();
                predictImg(img_to_predict, dst, predicted);
                predictions.push_back(dst);
                predicted_classes.push_back(predicted);

                string window_name = to_string(i) + ": This image have " + predicted;
                namedWindow(window_name, WINDOW_NORMAL);
                resizeWindow(window_name, 400, 400);
                imshow(window_name, dst);
                moveWindow(window_name, 500, 500);
                key = waitKeyEx(0);
                if (key == 1048603) return(0);
            }
            destroyAllWindows();
        }
    }
    else if(option == 3){
        while (!EXIT) {
            int key;
            int tray_num, image_num;
            cout << "Tray: ";
            cin >> tray_num;
            if (tray_num <= 0 or tray_num > 8) tray_num = 1;
            cout << "Image Number {0, 1, 2, 3}: ";
            cin >> image_num;
            cout << "Press [ANY] key to keep going or [ESC] to exit." << endl;
            cout << "---------------------------------------------------" << endl;
            string file_name;
            switch(image_num) {
                case 0:
                    file_name = "food_image";
                    break;
                case 1:
                    file_name = "leftover1";
                    break;
                case 2:
                    file_name = "leftover2";
                    break;
                case 3:
                    file_name = "leftover3";
                    break;
                default:
                    file_name = "food_image";
            }

            Mat img;
            //img1 = imread(argv[1]);
            img = imread(TRAY_PATH + to_string(tray_num) + "/" + file_name + IMAGE_EXT);

            string window_name_img = "Tray " + to_string(tray_num) + " " + file_name;
            /*
            namedWindow(window_name_img, WINDOW_NORMAL);
            resizeWindow(window_name_img, 600, 400);
            imshow(window_name_img, img);
            moveWindow(window_name_img, 500, 500);
            key = waitKeyEx(0);
            if (key == 1048603) return(0); //if press ESC end program
            */

            Mat img_to_predict, dst;
            vector<Mat> predictions, dishes(1);
            string predicted;
            vector<string> predicted_classes;
            dishes[0] = img;
            int i=0;
            for (const auto & dishe : dishes) {
                i++;
                img_to_predict = dishe.clone();
                predictImg(img_to_predict, dst, predicted);
                predictions.push_back(dst);
                predicted_classes.push_back(predicted);

                string window_name = to_string(i) + ": This image have " + predicted;
                namedWindow(window_name, WINDOW_NORMAL);
                resizeWindow(window_name, 600, 400);
                imshow(window_name, dst);
                moveWindow(window_name, 500, 500);
                key = waitKeyEx(0);
                if (key == 1048603) return(0);
            }
            destroyAllWindows();
        }
    }
    else return(0);

    return(0);
}