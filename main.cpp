#include "main_header.h"

int main(int argc, char** argv) {
    vector<string> file_name = {"leftover1", "leftover2", "leftover3"};
    vector<Mat> dishes1, dishes2;
    //for(int k = 0; k < 8; ++k){
        Mat img1 = imread(TRAY_PATH + to_string(0 + 1) + "/" + "food_image" + IMAGE_EXT);
        vector<box> boxes1;
        boxes1 = segment_plates(img1, dishes1);

        vector<vector<Mat>> best_matches;
        for(int i = 0; i < file_name.size(); ++i){
            Mat img2 = imread(TRAY_PATH + to_string(0 + 1) + "/" + file_name[i] + IMAGE_EXT);
            vector<box> boxes2;
            boxes2 = segment_plates(img2, dishes2);


            vector<vector<int>> all_matches;
            for(int j = 0; j < boxes1.size(); ++j){
                for(int l = 0; l < boxes2.size(); ++l){
                    vector<int> matches;
                    int matches_tmp = sift_matching(boxes1[j].img, boxes2[l].img);
                    matches.push_back(matches_tmp);
                    matches.push_back(j);
                    matches.push_back(l);
                    all_matches.push_back(matches);
                }
            }
            sort(all_matches.begin(), all_matches.end(), sort_num_match);

            vector<Mat> img_matches;

            cout << all_matches.size();

            for(int j = 0; j < 4; ++j){
                int ind1 = all_matches[j][1];
                int ind2 = all_matches[j][2];

                img_matches.push_back(boxes1[ind1].img);
                img_matches.push_back(boxes2[ind2].img);
                imshow("match1", img_matches[0]);
                imshow("match2", img_matches[1]);
                waitKey();

            }

            best_matches.push_back(img_matches);

      //  }


    }







    /*

    //Calculate pasta histogram
    vector<Mat> pasta_hist = categories_histogram({0, 5});
    vector<Mat> rice_hist = categories_histogram({5});

    //Go through all trays and imgs
    for (int i = 0; i < NUMBER_TRAYS; ++i) {
        //for (int i = 4; i < 5; ++i) {
        //if (i==4) continue;
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
            vector<Mat> box_img3 = {boxes[max_index].img.clone()};
            //Mat box_hist = mean_histogram2(box_hist_img);
            vector<int> predicted_IDs3;
            vector<double> predicted_strengths3;
            vector<food> cat_19_5;
            cat_19_5.push_back(foodCategories[0]);
            cat_19_5.push_back(foodCategories[5]);
            write_kmeans(cat_19_5);
            predict_categories(box_img3, cat_19_5, predicted_IDs3, predicted_strengths3);
            boxes[max_index].ID = predicted_IDs3[0];

            // 4.Type of Pasta
            int neighborhood = 10;
            if (boxes[max_index].ID == 19){
                K_Means(boxes[max_index].img, 3);
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

            // 5. Is contorno? Which contorno?
            vector<box> foods;
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
            }
            sort(boxes.begin(), boxes.end(), sort_ID);
            vector<int> predicted_IDs5;
            vector<double> predicted_strengths5;
            vector<food> cat_10_11;
            vector<Mat> box_img5;
            cat_10_11.push_back(foodCategories[10]);
            cat_10_11.push_back(foodCategories[11]);
            for (int k = 0; k < foods.size(); ++k) {
                    box_img5.push_back(boxes[k].img);
            }
            write_kmeans(cat_10_11);
            predict_categories(box_img5, cat_10_11, predicted_IDs5, predicted_strengths5);
            double stronger_pred = 0;
            int stronger_index;
            for (int k = 0; k < box_img5.size(); ++k) {
                if (fabs(predicted_strengths5[k]) > stronger_pred) {
                    stronger_pred = fabs(predicted_strengths5[k]);
                    stronger_index = k;
                }
            }
            boxes[stronger_index].ID = predicted_IDs5[stronger_index];



            sort(boxes.begin(), boxes.end(), sort_ID);
            //Show the plates
            for (int k = 0; k < boxes.size(); ++k) {
                string window_name_img = "Tray" + to_string(i + 1) + " " + file_name + " Food" + to_string(k + 1) + " ID:" + to_string(boxes[k].ID);
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
                    boxes[k].ID = foodCategories[int(values[1])].classLabel;
                    boxes[k].conf = values[2];
                }
            }



            //Print the boxes
            for (const auto box: boxes) {
                cout << box.ID << " " << box.p0x << " " << box.p0y << " " << box.width << " " << box.height << " " << box.conf << endl;
            }

        }
        waitKey();
        destroyAllWindows();
    }
                 */



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
