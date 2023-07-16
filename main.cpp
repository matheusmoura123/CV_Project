#include "main_header.h"

int main(int argc, char** argv) {

    //Calculate pasta histogram
    vector<Mat> pasta_hist = categories_histogram({0, 5});
    vector<Mat> rice_hist = categories_histogram({5});

    //Go through all trays and imgs
    for (int i = 0; i < NUMBER_TRAYS; ++i) {
        //for (int i = 4; i < 5; ++i) {
        cout << "----- TRAY " << to_string(i + 1) << " -----" << endl;
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

            vector<box> boxes;
            vector<Mat> dishes;

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
                    //cout << "pasta: ";
                    values = compare_histogram(box_hist, pasta_hist);
                    //cout << "rice: ";
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

            // 3.Rice or pasta?
            vector<Mat> box_hist_img = {boxes[max_index].img.clone()};
            //Mat box_hist = mean_histogram2(box_hist_img);
            vector<int> predicted_IDs;
            vector<food> cat_19_5;
            cat_19_5.push_back(foodCategories[0]);
            cat_19_5.push_back(foodCategories[5]);
            predict_categories(box_hist_img, cat_19_5, predicted_IDs);
            boxes[max_index].ID = predicted_IDs[0];

            /*
            if (compare_histogram(box_hist, rice_hist) > compare_histogram(box_hist, pasta_hist)) {
                boxes[max_index].ID = 5;
            }
            */

            // 4.Type of Pasta

            // 5. Is contorno? Which contorno?
            vector<box> foods;
            for (int k = 0; k < boxes.size(); ++k) {
                if (boxes[k].ID == -1) {
                    foods = separate_food(boxes[k]);
                    /*
                    for (int l = 0; l < foods.size(); ++l) {
                        imshow("food:" + to_string(l + 1) + "ID:" + to_string(foods[l].ID), foods[l].img);
                    }
                    */

                    cout << "debug1" << endl;
                    cout << foods.size() << endl;
                    cout << foods[0].p0x << endl;
                    boxes[k] = foods[0];
                    cout << "debug1" << endl;
                }
            }
            if (foods.size() > 1) {
                cout << "debug1" << endl;
                for (int l = 1; l < foods.size(); ++l) {
                    cout << "debug1" << endl;
                    boxes.push_back(foods[l]);
                    cout << "debug1" << endl;
                }
            }

            cout << "debug1" << endl;
            //Show the plates
            for (int k = 0; k < boxes.size(); ++k) {
                cout << "debug1" << endl;
                string window_name_img = "Tray" + to_string(i + 1) + " " + file_name + " Food" + to_string(k + 1) + " ID:" + to_string(boxes[k].ID);
                namedWindow(window_name_img);
                imshow(window_name_img, boxes[k].img);
                /*
                if (boxes[k].ID == 19) {
                    vector<box> foods;
                    foods = separate_food(boxes[k]);
                    for (int l = 0; l < foods.size(); ++l) {
                        imshow("food:" + to_string(l + 1) + "ID:" + to_string(foods[l].ID), foods[l].img);
                    }
                }
                 */
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
            for (const auto box: boxes) {
                cout << box.ID << " " << box.p0x << " " << box.p0y << " " << box.width << " " << box.height << " " << box.conf << endl;
            }

        }
        waitKey();
        destroyAllWindows();
    }
    return 0;
}

    /*
    //test for masks_mIoU
    Mat mask_truth = imread("../Food_leftover_dataset/tray5/masks/food_image_mask_result.png");
    Mat mask_result = imread("../Food_leftover_dataset/tray5/masks/leftover1_result.png");
    cout << masks_mIoU(mask_truth, mask_result) << endl;
    */

    /*
    //test for img_mAp
    vector<box> boxes_truth, boxes_results;
    box_file_reader(boxes_truth, "../Food_leftover_dataset/tray2/bounding_boxes/food_image_bounding_box.txt");
    box_file_reader(boxes_results, "../Food_leftover_dataset/tray2/bounding_boxes/food_image_bounding_box_test.txt");
    double map = img_mAp(boxes_truth, boxes_results);
    cout << "img_mAp = " <<  map << endl;
    */

    /*
    //test for leftover_ratio
    Mat mask_before = imread("../Food_leftover_dataset/tray5/masks/food_image_mask_result.png");
    Mat mask_after = imread("../Food_leftover_dataset/tray5/masks/leftover3_result.png");
    leftover_ratio (mask_before, mask_after);
     */

    /*
    //test results.cpp
    food_localization();
    food_segmentation ();
    food_leftover();
    */
