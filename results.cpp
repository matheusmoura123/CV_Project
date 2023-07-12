#include "main_header.h"

int food_localization (int argc, char **argv) {
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
            //for (int i = 1; i < 2; ++i) {
                //for (int j = 0; j < 4; ++j) {
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
                    fw << "Tray" << to_string(i + 1) << " - " << file_name << ": mAp = " << image_mAp << "\n";
                }
            }
            //store results contents to text file
            fw.close();
        }
        else cout << "Problem with opening file";
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    cout << "Done!" << endl;

    return 0;
}