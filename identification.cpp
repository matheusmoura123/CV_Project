#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "segment_plates.h"
#include "find_histogram.h"

using namespace cv;
using namespace std;

const string DATASET_PATH = "../FoodCategories/";
const string IMAGE_EXT = ".jpg";

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

const vector<food> pastaCategories{
        {"pesto", 4, 1},
        {"pomodoro", 5, 2},
        {"ragu", 2, 3},
        {"pasta_clams", 8, 4},
};
const vector<food> foodCategories{
        //{"plate", 9, 0},
        {"rice", 7, 5},
        {"pork", 9, 6},
        {"fish", 10, 7},
        {"rabbit", 12, 8},
        {"seafood", 5, 9},
        {"beans", 13, 10},
        {"potato", 13, 11},
        //{"lettuce", 15, 12},
        {"bread", 18, 13},
        //{"carrot", 6, 14},
        //{"pepper", 2, 15},
        //{"tomato", 10, 16},
        {"pasta", 20, 17},
        {"salad", 15, 18},
        //{"plate_salad", 3, 19},
};


int main(int argc, char **argv) {
    string path;
    if (argc == 2) {
        path = argv[1];
    }

    string className = "potato";
    int index=0;
    for (index; index < foodCategories.size(); ++index) {
        if (className == foodCategories[index].className) break;
    }

    for (int i = 1; i <= foodCategories[index].imageNumbers; ++i) {
        Mat img = imread(DATASET_PATH + className + "/" + className + to_string(i) + IMAGE_EXT);
        find_histogram(img);
    }

}