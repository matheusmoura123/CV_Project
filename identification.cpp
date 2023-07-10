#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "segment_plates.h"
#include "find_histogram.h"

using namespace cv;
using namespace std;

const string TRAY_PATH = "../Food_leftover_dataset/tray";
const string DATASET_PATH_CATEGORIES = "../FoodCategories/";
const string IMAGE_EXT = ".jpg";
const int NUMBER_TRAYS = 8;


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
        //{"bread", 18, 13},
        //{"carrot", 6, 14},
        //{"pepper", 2, 15},
        //{"tomato", 10, 16},
        {"pasta", 20, 17},
        //{"salad", 15, 18},
        //{"plate_salad", 3, 19},
};


int main(int argc, char **argv) {

    //Read all trays and segment plates
    for (int i = 0; i < NUMBER_TRAYS; ++i) {
        for (int j = 0; j < 4; ++j) {
            int key;
            string file_name;
            switch (j) {
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
            img = imread(TRAY_PATH + to_string(i + 1) + "/" + file_name + IMAGE_EXT);

            vector<Mat> dishes;
            dishes = segment_plates(img);
            for (const auto & dishe : dishes) {
                string window_name_img = "Tray " + to_string(i + 1) + " " + file_name;
                namedWindow(window_name_img);
                imshow(window_name_img, dishe);
                cout << dishe.size() << endl;
                waitKey(0);
            }
        }
    }


    //Find the index of className
    string className = "potato";
    int index=0;
    for (index; index < foodCategories.size(); ++index) {
        if (className == foodCategories[index].className) break;
    }
    //Load all reference imgs
    vector<Mat> categories_hist;
    for (auto &food: foodCategories ) {
        vector<Mat> imgs;
        for (int i = 1; i <= food.imageNumbers; ++i) {
            Mat img = imread(DATASET_PATH_CATEGORIES + food.className + "/" + food.className + to_string(i) + IMAGE_EXT);
            //find_histogram(img);
            imgs.push_back(img);
        }
        categories_hist.push_back(mean_histogram2(imgs));
    }

    string path = "../FoodCategories/pure/pure_potato1.jpg";
    if (argc == 2) {
        path = argv[1];
    }
    vector<Mat> test_img = {imread(path)};
    Mat test_hist = mean_histogram2(test_img);

    array<double, 2> values;
    values = compare_histogram(test_hist, categories_hist);

    cout << "--------------------------------------" << endl;
    cout << values[0] << endl;
    cout << foodCategories[int(values[1])].className << endl;

}