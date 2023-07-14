#include "main_header.h"

bool sort_bigger_area (const box& i,const box& j) {
    return(i.width*i.height>j.width*j.height);
}

int main(int argc, char **argv) {

    vector <Mat> dishes;
    vector <box> boxes;

    //Calculate categories histogram
    vector <Mat> categories_hist;
    for (auto &food: foodCategories) {
        vector <Mat> imgs;
        for (int i = 1; i <= food.imageNumbers; ++i) {
            Mat img = imread(CATEGORIES_PATH + food.className + "/" + food.className + to_string(i) + IMAGE_EXT);
            //find_histogram(img);
            imgs.push_back(img);
        }
        categories_hist.push_back(mean_histogram2(imgs));
    }

    //Calculate pasta sauce histogram
    vector <Mat> pasta_sauce_hist;
    for (auto &food: pastaCategories) {
        vector <Mat> imgs;
        for (int i = 1; i <= food.imageNumbers; ++i) {
            Mat img = imread(CATEGORIES_PATH + food.className + "/" + food.className + to_string(i) + IMAGE_EXT);
            //find_histogram(img);
            imgs.push_back(img);
        }
        pasta_sauce_hist.push_back(mean_histogram2(imgs));
    }

    //Calculate pasta histogram
    vector <Mat> pasta_hist, imgs_pasta;
    //Find the index of ID
    int ID = 19, index = 0;
    for (index; index < foodCategories.size(); ++index) {
        if (ID == foodCategories[index].classLable) break;
    }
    for (int i = 1; i <= foodCategories[index].imageNumbers; ++i) {
        Mat img = imread(CATEGORIES_PATH + foodCategories[index].className + "/" + foodCategories[index].className + to_string(i) + IMAGE_EXT);
        //find_histogram(img);
        resize(img, img, Size(500, 500));
        imgs_pasta.push_back(img);
    }
    pasta_hist.push_back(mean_histogram2(imgs_pasta));

    //Go through all trays and imgs
    for (int i = 0; i < NUMBER_TRAYS; ++i) {
    //for (int i = 0; i < 1; ++i) {
        cout << "----- TRAY " << to_string(i+1) << " -----" << endl;
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

            //Load img
            Mat img;
            img = imread(TRAY_PATH + to_string(i + 1) + "/" + file_name + IMAGE_EXT);
            //imshow("img", img);

            //Extract each box from img
            boxes = segment_plates(img, dishes);

            /*
            //Print the boxes
            cout << "from segment plates" << endl;
            for (const auto box: boxes) {
                cout << box.ID << " " << box.p0x << " " << box.p0y << " " << box.width << " " << box.height << endl;
            }
             */

            //Cascade Classification
            //1. Which one is salad
            sort(boxes.begin(), boxes.end(), sort_bigger_area);
            if (boxes.size() > 2) {
                boxes[2].ID = 12;
                boxes[2].conf = 1;
            }

            // 2. Which one is pasta
            array<double, 3> values_max{};
            vector<box> copy(boxes);
            double max_value = 0;
            int max_index;
            for (int k = 0; k < boxes.size(); ++k) {
                if (boxes[k].ID == -1) {
                    boxes[k] = segment_food(copy[k]);
                    //Calculate histogram for box img
                    waitKey();
                    vector<Mat> box_hist_img = {boxes[k].img.clone()};
                    Mat box_hist = mean_histogram2(box_hist_img);

                    //Compare with the pasta_hist
                    array<double, 3> values{};
                    values = compare_histogram(box_hist, pasta_hist);
                    if (values[0] >= max_value) {
                        max_value = values[0];
                        values_max[0] = values[0];
                        values_max[1] = values[1];
                        values_max[2] = values[2];
                        max_index = k;
                    }
                }
            }
            //Save to box that is pasta
            boxes[max_index].ID = foodCategories[int(values_max[1])].classLable;
            boxes[max_index].conf = values_max[2];

            //Show the plates
            for (int k = 0; k < boxes.size(); ++k) {
                string window_name_img = "Tray" + to_string(i + 1) + " " + file_name + " Food" + to_string(boxes[k].ID);
                namedWindow(window_name_img);
                imshow(window_name_img, boxes[k].img);
            }

            /*
            //2. Histogram Comparison
            for (int k = 0; k < boxes.size(); ++k) {
                if (boxes[k].ID == -1) {
                    //Calculate histogram for box img
                    vector <Mat> box_hist_img = {boxes[k].img};
                    Mat box_hist = mean_histogram2(box_hist_img);

                    //Compare with the categories hist
                    array<double, 3> values{};
                    values = compare_histogram(box_hist, categories_hist);

                    //3. Type of pasta
                    if (foodCategories[int(values[1])].className == "pasta"){

                    }

                    //Save to boxes
                    boxes[k].ID = foodCategories[int(values[1])].classLable;
                    boxes[k].conf = values[2];
                }
            }
             */



            //Print the boxes
            //cout << "after classification" << endl;
            for (const auto box: boxes) {
                cout << box.ID << " " << box.p0x << " " << box.p0y << " " << box.width << " " << box.height << " " << box.conf << endl;
            }

        }
    }
    waitKey();
    return 0;
}


    /*
    //Find the index of ID
    int ID = 12;
    int index1=0, index2=0;
    for (index1; index1 < boxes.size(); ++index1) {
        if (ID == boxes[index1].ID) break;
    }
    for (index2; index2 < truth_boxes.size(); ++index2) {
        if (ID == truth_boxes[index2].ID) break;
    }

    //Compare two boxes
    double salad_IoU = boxes_IoU(truth_boxes[index2], boxes[index1]);
    cout << "Salad IoU = " << salad_IoU << endl;
    */

    /*
    //Test histogram comparison
    string path = "../FoodCategories/pure/pure_beans1.jpg";
    if (argc == 2) {
        path = argv[1];
    }
    vector<Mat> test_img = {imread(path)};
    Mat test_hist = mean_histogram2(test_img);

    array<double, 3> values{};
    values = compare_histogram(test_hist, categories_hist);

    cout << "--------------------------------------" << endl;
    cout << values[0] << endl;
    cout << foodCategories[int(values[1])].className << endl;
    cout << "Confidence: " << values[2] << endl;
    */