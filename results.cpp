#include "main_header.h"

int food_localization () {
    try {
        cout << "Writing food_localization results contents to file..." << endl;
        //open file for writing
        string path = "../FoodResults/food_localization_results.txt";
        ofstream fw(path, std::ofstream::out);
        //check if file was successfully opened for writing
        if (fw.is_open())
        {
            //Read all boxes and comupte mAp
            for (int i = 0; i < NUMBER_TRAYS; ++i) {
                for (int j = 0; j < 3; ++j) {
                    string file_name;
                    switch (j) {
                        case 0: file_name = "food_image"; break;
                        case 1: file_name = "leftover1"; break;
                        case 2: file_name = "leftover2"; break;
                        case 3: file_name = "leftover3"; break;
                        default: file_name = "food_image";
                    }
                    vector<box> boxes_result;
                    string box_result_path = RESULTS_PATH + to_string(i + 1) + "/bounding_boxes/" + file_name + "_result_box.txt";
                    box_file_reader(boxes_result, box_result_path);

                    vector<box> boxes_truth;
                    string box_truth_path = TRAY_PATH + to_string(i + 1) + "/bounding_boxes/" + file_name + "_bounding_box.txt";
                    box_file_reader(boxes_truth, box_truth_path);
                    double image_mAp = img_mAp(boxes_truth, boxes_result);
                    fw << "Tray" << to_string(i + 1) << "." << file_name << " mAp = " << image_mAp << "\n";
                }
            }
            //store results contents to text file
            fw.close();
        }
        else cout << "Problem with opening file" << endl;
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    cout << "Done!" << endl;
    return 0;
}

int food_segmentation () {
    try {
        cout << "Writing food_segmentation results contents to file..." << endl;
        //open file for writing
        string path = "../FoodResults/food_segmentation_results.txt";
        ofstream fw(path, std::ofstream::out);
        //check if file was successfully opened for writing
        if (fw.is_open())
        {
            //Read all boxes and comupte mAp
            for (int i = 0; i < NUMBER_TRAYS; ++i) {
                for (int j = 0; j < 3; ++j) {
                    string file_name;
                    switch (j) {
                        case 0: file_name = "food_image_mask"; break;
                        case 1: file_name = "leftover1"; break;
                        case 2: file_name = "leftover2"; break;
                        case 3: file_name = "leftover3"; break;
                        default: file_name = "food_image";
                    }
                    Mat mask_result;
                    string mask_result_path = RESULTS_PATH + to_string(i + 1) + "/masks/" + file_name + "_result.png";
                    mask_result = imread(mask_result_path);

                    Mat mask_truth;
                    string mask_truth_path = TRAY_PATH + to_string(i + 1) + "/masks/" + file_name + ".png";
                    mask_truth = imread(mask_truth_path);

                    double mask_mean_Iou = masks_mIoU(mask_truth, mask_result);
                    fw << "Tray" << to_string(i + 1) << "." << file_name << " mIoU = " << mask_mean_Iou << "\n";
                }
            }
            //store results contents to text file
            fw.close();
        }
        else cout << "Problem with opening file" << endl;
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    cout << "Done!" << endl;
    return 0;
}

int food_leftover () {
    try {
        cout << "Writing food_leftover results contents to file..." << endl;
        //open file for writing
        string path = "../FoodResults/food_leftover_results.txt";
        ofstream fw(path, std::ofstream::out);
        //check if file was successfully opened for writing
        if (fw.is_open())
        {
            //Read all boxes and comupte mAp
            for (int i = 0; i < NUMBER_TRAYS; ++i) {
                for (int j = 0; j < 4; ++j) {
                    string file_name;
                    switch (j) {
                        case 0: file_name = "food_image_mask"; break;
                        case 1: file_name = "leftover1"; break;
                        case 2: file_name = "leftover2"; break;
                        case 3: file_name = "leftover3"; break;
                        default: file_name = "food_image";
                    }
                    Mat mask_result;
                    string mask_result_path = RESULTS_PATH + to_string(i + 1) + "/masks/" + file_name + "_result.png";
                    mask_result = imread(mask_result_path);

                    Mat mask_truth;
                    string mask_truth_path = TRAY_PATH + to_string(i + 1) + "/masks/" + file_name + ".png";
                    mask_truth = imread(mask_truth_path);

                    double mask_mean_Iou = masks_mIoU(mask_truth, mask_result);
                    fw << "Tray" << to_string(i + 1) << "." << file_name << " mIoU = " << mask_mean_Iou << "\n";
                }
            }
            //store results contents to text file
            fw.close();
        }
        else cout << "Problem with opening file" << endl;
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    cout << "Done!" << endl;
    return 0;
}