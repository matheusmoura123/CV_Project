#include "main_header.h"

bool sort_bigger_area (const box& i,const box& j) {
    return(i.width*i.height>j.width*j.height);
}

vector<Mat> categories_histogram (int begin, int final) {
    vector <Mat> categories_hist;
    for (int j = begin; j <= final ; ++j) {
        vector <Mat> imgs;
        for (int i = 1; i <= foodCategories[j].imageNumbers; ++i) {
            Mat img = imread(CATEGORIES_PATH + foodCategories[j].className + "/" + foodCategories[j].className + to_string(i) + IMAGE_EXT);
            //find_histogram(img);
            imgs.push_back(img);
        }
        categories_hist.push_back(mean_histogram2(imgs));
    }
    return categories_hist;
}

int main(int argc, char **argv) {

    vector <Mat> dishes;
    vector <box> boxes;

    //Calculate pasta histogram
    vector <Mat> pasta_hist = categories_histogram(0, 0);
    vector<Mat> rice_hist = categories_histogram(5, 5);

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

            // Cascade Classification
            // 1. Which one is salad
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
                    vector<Mat> box_hist_img = {boxes[k].img.clone()};
                    Mat box_hist = mean_histogram2(box_hist_img);
                    //Compare with the pasta_hist
                    array<double, 3> values{};
                    cout << "pasta: ";
                    values = compare_histogram(box_hist, pasta_hist);
                    cout << "rice: ";
                    compare_histogram(box_hist, rice_hist);
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
            boxes[max_index].ID = 19;
            boxes[max_index].conf = values_max[2];

            /*
            // 3.Rice or pasta?
            vector<Mat> box_hist_img = {boxes[max_index].img.clone()};
            Mat box_hist = mean_histogram2(box_hist_img);
            cout << box_hist.size << endl;
            if (compare_histogram(box_hist, rice_hist) > compare_histogram(box_hist, pasta_hist)) {
                boxes[max_index].ID = 5;
            }
             */


            // 4.Contorno
            for (int k = 0; k < boxes.size(); ++k) {
                if (boxes[k].ID == -1) {
                    boxes[k] = segment_food(copy[k]);
                    //Calculate histogram for box img
                    vector<Mat> box_hist_img = {boxes[k].img.clone()};
                    Mat box_hist = mean_histogram2(box_hist_img);
                    //Compare with the pasta_hist
                    array<double, 3> values{};
                    cout << "pasta: ";
                    values = compare_histogram(box_hist, pasta_hist);
                    cout << "rice: ";
                    compare_histogram(box_hist, rice_hist);
                    if (values[0] >= max_value) {
                        max_value = values[0];
                        values_max[0] = values[0];
                        values_max[1] = values[1];
                        values_max[2] = values[2];
                        max_index = k;
                    }
                }
            }


            //Show the plates
            for (int k = 0; k < boxes.size(); ++k) {
                string window_name_img = "Tray" + to_string(i + 1) + " " + file_name + " Food" + to_string(k+1) + " ID:" + to_string(boxes[k].ID);
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
        waitKey();
        destroyAllWindows();
    }
    return 0;
}
