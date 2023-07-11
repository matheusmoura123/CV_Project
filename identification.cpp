#include "main_header.h"

const vector<food> pastaCategories{
        {"pesto", 4, 1},
        {"pomodoro", 5, 2},
        {"ragu", 2, 3},
        {"pasta_clams", 8, 4},
};
const vector<food> foodCategories{
        //{className, numberOfImgs, ID},
        //{"plate", 9, 0},
        {"rice", 7, 5},
        {"pork", 9, 6},
        {"fish", 10, 7},
        {"rabbit", 12, 8},
        {"seafood", 5, 9},
        {"beans", 13, 10},
        {"potato", 13, 11},
        {"pasta", 20, 19},
        //{"salad", 15, 12},
        //{"bread", 18, 13},
        //{"carrot", 6, 14},
        //{"pepper", 2, 15},
        //{"tomato", 10, 16},
        //{"lettuce", 15, 17},
        //{"plate_salad", 3, 18},
};


int main(int argc, char **argv) {

    vector<Mat> dishes;
    vector<int> dishes_areas;
    vector<box> boxes;
    //Read all trays and segment plates
    //for (int i = 0; i < NUMBER_TRAYS; ++i) {
    for (int i = 1; i < 2; ++i) {
        //for (int j = 0; j < 4; ++j) {
        for (int j = 0; j < 1; ++j) {
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
            imshow("img", img);

            boxes = segment_plates(img, dishes);
            for (const auto & dishe : dishes) {
                string window_name_img = "Tray " + to_string(i + 1) + " " + file_name;
                namedWindow(window_name_img);
                imshow(window_name_img, dishe);
                dishes_areas.push_back(int(dishe.cols*dishe.rows));
                waitKey();
            }

            //1. Identifying Salad
            sort(dishes_areas.begin(), dishes_areas.end(), greater<int>());
            if (dishes.size() > 2) {
                for(int k = 0; k < boxes.size(); ++k) {
                    if (boxes[k].height*boxes[k].width == dishes_areas[2]) {
                        boxes[k].ID = 12;
                    }
                }
            }


            //Print the boxes
            for(const auto box:boxes) {
                cout << box.ID << " " << box.p0x << " " << box.p0y << " " << box.width << " " << box.height << endl;
            }

            //Creating bounding_box result files
            string result_box_path = RESULTS_PATH + to_string(i + 1) + "/bounding_boxes/" + file_name + "_result_box.txt";
            box_file_writer(boxes, result_box_path);

            //Read box from file
            vector<box> truth_boxes;
            string true_box_path = TRAY_PATH + to_string(i + 1) + "/bounding_boxes/" + file_name + "_bounding_box.txt";
            box_file_reader(truth_boxes, true_box_path);
        }
    }



    /*
    //Find the index of className
    string className = "potato";
    int index=0;
    for (index; index < foodCategories.size(); ++index) {
        if (className == foodCategories[index].className) break;
    }
    */

    //Load all reference imgs
    vector<Mat> categories_hist;
    for (auto &food: foodCategories ) {
        vector<Mat> imgs;
        for (int i = 1; i <= food.imageNumbers; ++i) {
            Mat img = imread(CATEGORIES_PATH + food.className + "/" + food.className + to_string(i) + IMAGE_EXT);
            //find_histogram(img);
            imgs.push_back(img);
        }
        categories_hist.push_back(mean_histogram2(imgs));
    }

    //Test histogram comparison
    string path = "../FoodCategories/pure/pure_beans1.jpg";
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