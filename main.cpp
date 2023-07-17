#include "main_header.h"

int main(int argc, char** argv) {

    //Calculate pasta histogram
    vector<Mat> pasta_hist = categories_histogram({0, 5});
    vector<Mat> rice_hist = categories_histogram({5});
    vector<Mat> contorno_hist = categories_histogram({10, 11});

    //kmeans for pasta or rice
    vector<food> cat_19_5;
    cat_19_5.push_back(foodCategories[0]);
    cat_19_5.push_back(foodCategories[5]);
    //write_kmeans(cat_19_5);

    //kmeans for meats
    vector<food> cat_6_9;
    cat_6_9.push_back(foodCategories[6]);
    cat_6_9.push_back(foodCategories[7]);
    cat_6_9.push_back(foodCategories[8]);
    cat_6_9.push_back(foodCategories[9]);
    //write_kmeans(cat_6_9);

    int tray;
    cout << "Tray(1-8) or 100 for metrics: ";
    cin >> tray;
    int i = tray-1;
    //Go through all trays and imgs
    //for (int i = 0; i < NUMBER_TRAYS; ++i) {
    if (0 <= i && i <= 7) {
        vector<box> boxes;
        cout << "\n" << "------- TRAY " << to_string(i + 1) << " -------" << endl;
        for (int j = 0; j < 4; ++j) {
            string file_name;
            if (j == 0) {
                file_name = "food_image";
                cout << "--- " << file_name << " ---" << endl;
                vector<Mat> dishes;

                //Load img
                Mat img;
                img = imread(TRAY_PATH + to_string(i + 1) + "/" + file_name + IMAGE_EXT);
                //imshow("img", img);

                //Extract each  plate box from img
                boxes = segment_plates(img, dishes);

                // Cascade Classification
                // 1. Which one is salad
                sort(boxes.begin(), boxes.end(), sort_bigger_area);
                if (boxes.size() > 2) {
                    boxes[2].ID = 12;
                    boxes[2].conf = 1;
                }

                // 2.Which one is pasta
                array<double, 3> values_max{};
                vector<box> copy(boxes);
                double max_value = 0;
                int max_index;
                for (int k = 0; k < boxes.size(); ++k) {
                    boxes[k] = segment_food(copy[k]);
                    if (boxes[k].ID == -1) {
                        //Calculate histogram for box img
                        vector<Mat> box_hist_img = {boxes[k].img.clone()};
                        Mat box_hist = mean_histogram2(box_hist_img);
                        //Compare with the pasta_hist
                        array<double, 3> values{};
                        //cout << "pasta: ";
                        values = compare_histogram(box_hist, pasta_hist);
                        //cout << "rice: ";
                        //compare_histogram(box_hist, rice_hist);
                        if (values[0] >= max_value) {
                            max_value = values[0];
                            values_max[0] = values[0];
                            values_max[1] = values[1];
                            values_max[2] = values[2];
                            max_index = k;
                        }
                    }
                }
                boxes[max_index].ID = 19;
                boxes[max_index].conf = values_max[2];

                // 3.Rice or pasta?
                vector<Mat> box_img3 = {boxes[max_index].img.clone()};
                vector<int> predicted_IDs3;
                vector<double> predicted_strengths3;
                predict_categories(box_img3, cat_19_5, predicted_IDs3, predicted_strengths3);
                boxes[max_index].ID = predicted_IDs3[0];

                // 4.Type of Pasta
                int neighborhood = 10;
                if (boxes[max_index].ID == 19) {
                    float mean_hue = 0.0;
                    for(int n = 0; n < 3; ++n){
                        Rect roi(boxes[max_index].img.rows/2-10+n*neighborhood, boxes[max_index].img.cols/2-10+n*neighborhood, 100, 100);
                        Mat img_crop = boxes[max_index].img(roi);
                        array<int,3> max_hue_values = find_histogram(img_crop);
                        //cout << max_hue_values[0] << endl << max_hue_values[1] << endl << max_hue_values[2] << endl;
                        float mean_hue_tmp;
                        for(int a = 0; a < 3; ++a){
                            mean_hue_tmp = mean_hue_tmp + (float)max_hue_values[a];
                        }
                        mean_hue_tmp = mean_hue_tmp/3;
                        //cout << mean_hue_tmp << endl;
                        mean_hue = mean_hue + mean_hue_tmp;
                    }
                    mean_hue = mean_hue/3;
                    //cout << mean_hue << endl;
                    if(mean_hue <= 16){
                        boxes[max_index].ID = 4;
                    }
                    else if(mean_hue > 16 && mean_hue <= 19){
                        boxes[max_index].ID = 2;
                    }
                    else if(mean_hue > 19 && mean_hue <= 23){
                        boxes[max_index].ID = 3;
                    }
                    else{
                        boxes[max_index].ID = 1;
                    }
                }

                // 5. Is contorno? Which one?
                vector<box> foods = {};
                for (int k = 0; k < boxes.size(); ++k) {
                    if (boxes[k].ID == -1) {
                        foods = separate_food(boxes[k]);
                        boxes[k] = foods[0];
                    }
                }
                if (foods.size() > 1) {
                    for (int l = 1; l < foods.size(); ++l) {
                        boxes.push_back(foods[l]);
                    }
                    sort(boxes.begin(), boxes.end(), sort_ID);
                    array<double, 3> values_max2{};
                    double max_value2 = 0;
                    int max_index2;
                    for (int k = 0; k < foods.size(); ++k) {
                        vector<Mat> box_hist_img = {boxes[k].img.clone()};
                        Mat box_hist = mean_histogram2(box_hist_img);
                        array<double, 3> values{};
                        values = compare_histogram(box_hist, contorno_hist);
                        if (values[0] >= max_value2) {
                            max_value2 = values[0];
                            values_max2[0] = values[0];
                            values_max2[1] = values[1];
                            values_max2[2] = values[2];
                            max_index2 = k;
                        }
                    }
                    boxes[max_index2].ID = foodCategories[int(10+values_max2[1])].classLabel;
                }
                sort(boxes.begin(), boxes.end(), sort_ID);

                // 6.Which meat?
                if (boxes[0].ID == -1) {
                    vector<int> predicted_IDs6;
                    vector<double> predicted_strengths6;
                    vector<Mat> box_img6 = {boxes[0].img};
                    predict_categories(box_img6, cat_6_9, predicted_IDs6, predicted_strengths6);
                    boxes[0].ID = predicted_IDs6[0];
                }

                //Show the plates
                sort(boxes.begin(), boxes.end(), sort_ID);
                for (int k = 0; k < boxes.size(); ++k) {
                    string window_name_img = "Tray" + to_string(i + 1) + " " + file_name + " Food" + to_string(k + 1) + " ID:" + to_string(boxes[k].ID);
                    namedWindow(window_name_img);
                    imshow(window_name_img, boxes[k].img);
                }
                //Print the boxes
                for (const auto box: boxes) {
                    cout << box.ID << " " << box.p0x << " " << box.p0y << " " << box.width << " " << box.height << " " << box.conf << endl;
                }

                draw_rectangles_masks(img, boxes, i, file_name);
                save_boxes_masks_at_tray_stage(boxes, i, file_name);
                waitKey();
                destroyAllWindows();
            }
            else {
                file_name = "leftover" + to_string(j);
                cout << "--- " << file_name << " ---" << endl;

                vector<Mat> dishes;
                vector<box> boxes_left;

                //Load img
                Mat img;
                img = imread(TRAY_PATH + to_string(i + 1) + "/" + file_name + IMAGE_EXT);
                //imshow("img", img);

                //Extract each  plate box from img
                boxes_left = segment_plates(img, dishes);

                // Cascade Classification
                // 1. Which one is salad
                sort(boxes_left.begin(), boxes_left.end(), sort_bigger_area);
                if (boxes_left.size() > 2) {
                    boxes_left[2].ID = 12;
                    boxes_left[2].conf = 1;
                }

                // 2. Identify pasta from sift matching
                vector<box> box_pasta;
                for (const auto& box: boxes) {
                    if (1 <= box.ID && box.ID <= 5)
                        box_pasta.push_back(box);
                }
                compare_plates(box_pasta, boxes_left);

                // 3. Segment food
                vector<box> copy(boxes_left);
                for (int k = 0; k < boxes_left.size(); ++k) {
                    boxes_left[k] = segment_food(copy[k]);
                }

                // 4. Separate food
                vector<box> foods_left = {};
                vector<box> copy2(boxes_left);
                for (int k = 0; k < boxes_left.size(); ++k) {
                    if (boxes_left[k].ID == -1) {
                        foods_left = separate_food(copy2[k]);
                        boxes_left[k] = foods_left[0];
                    }
                }
                if (foods_left.size() > 1) {
                    for (int l = 1; l < foods_left.size(); ++l) {
                        boxes_left.push_back(foods_left[l]);
                    }
                }

                // 5. Compare last dish
                vector<box> box_rest;
                for (const auto& box: boxes) {
                    if (6 <= box.ID && box.ID <= 11)
                        box_rest.push_back(box);
                }
                compare_plates(box_rest, boxes_left);

                sort(boxes_left.begin(), boxes_left.end(), sort_ID_bigger);

                if (boxes_left.size() > boxes.size()) {
                    for (int k = 0; k < boxes_left.size(); ++k) {
                        if (boxes_left[k].ID == -1) {
                            boxes_left.pop_back();
                        }
                    }
                }

                //Show the plates
                sort(boxes_left.begin(), boxes_left.end(), sort_ID);
                for (int k = 0; k < boxes_left.size(); ++k) {
                    string window_name_img = "Tray" + to_string(i + 1) + " " + file_name + " Food" + to_string(k + 1) + " ID:" + to_string(boxes_left[k].ID);
                    namedWindow(window_name_img);
                    imshow(window_name_img, boxes_left[k].img);
                }
                //Print the boxes
                for (const auto box: boxes_left) {
                    cout << box.ID << " " << box.p0x << " " << box.p0y << " " << box.width << " " << box.height << " " << box.conf << endl;
                }
                draw_rectangles_masks(img, boxes_left, i, file_name);
                save_boxes_masks_at_tray_stage(boxes_left, i, file_name);
                waitKey();
                destroyAllWindows();
            }
        }
        //waitKey();
        destroyAllWindows();
        cout << "All boxes and masks from this tray saved." << endl;
    } else if (tray == 100) {
        food_localization();
        food_segmentation ();
        food_leftover();
    } else cout << "Invalid input." << endl;
    return 0;
}
