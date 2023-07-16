#include "main_header.h"

int box_file_writer (const vector<box>& boxes, const string& path) {
    try {
        cout << "Writing box contents to file..." << endl;
        //open file for writing
        ofstream fw(path, std::ofstream::out);
        //check if file was successfully opened for writing
        if (fw.is_open())
        {
            //store array contents to text file
            for (const auto box: boxes) {
                fw << "ID: " << box.ID << "; [" << box.p0x << ", " <<  box.p0y << ", " << box.width << ", " <<  box.height << "]\n";
            }
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

int box_file_reader (vector<box>& boxes, const string& path) {
    try {
        cout << "Reading box contents from file..." << endl;
        string line;
        //open file for reading
        ifstream fw(path, ifstream::in);
        //check if file was successfully opened for writing
        if (fw.is_open())
        {
            while (getline(fw, line)) {
                string str;
                vector<string> vec;
                stringstream ss(line);
                // Use while loop to check the getline() function condition.
                while (getline(ss, str, ' ')) {
                    vec.push_back(str);
                }
                boxes.emplace_back(
                        stoi(vec[1].substr(0, vec[1].size()-1)),
                        stoi(vec[2].substr(1, vec[2].size()-2)),
                        stoi(vec[3].substr(0, vec[3].size()-1)),
                        stoi(vec[4].substr(0, vec[4].size()-1)),
                        stoi(vec[5].substr(0, vec[5].size()-1)),
                        -1,
                        Mat::zeros(stoi(vec[5].substr(0, vec[5].size()-1)),
                                   stoi(vec[4].substr(0, vec[4].size()-1)),
                                   CV_8UC3)
                        );
            }
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

Mat mask_img_builder (const vector<box>& boxes) {
    int img_cols = 1280;
    int img_rows = 960;
    Mat img_out = Mat::zeros(img_rows, img_cols, CV_8UC1);
    for(const auto box:boxes) {
        Mat gray_img;
        cvtColor(box.img, gray_img, COLOR_BGR2GRAY);
        string name = "gray" + to_string(box.p0y);
        //imshow(name, gray_img );
        for (int y = 0; y < box.height; ++y) {
            for (int x = 0; x < box.width; ++x) {
                if (gray_img.at<uchar>(y, x) != 0) {
                    img_out.at<uchar>(y+box.p0y, x+box.p0x) = box.ID;
                }
            }
        }
    }
    return img_out;
}

int mask_file_writer (const vector<box>& boxes, const string& path) {
    try {
        cout << "Creating mask img file..." << endl;
        Mat img = mask_img_builder(boxes);
        bool check = imwrite(path, img);
        if (check) {}
        else cout << "Problem with saving img file" << endl;
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    cout << "Done!" << endl;
    return 0;
}

int save_all_boxes_masks_at_tray (vector<vector<box>>& all_boxes, int tray_num) {
        for (int j = 0; j < all_boxes.size(); ++j) {
            string file_name;
            switch (j) {
                case 0: file_name = "food_image"; break;
                case 1: file_name = "leftover1"; break;
                case 2: file_name = "leftover2"; break;
                case 3: file_name = "leftover3"; break;
                default: file_name = "food_image";
            }
            string box_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/bounding_boxes/" + file_name + "_result_box.txt";
            string mask_result_path;
            if (j == 0) mask_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/masks/food_image_mask_result.png";
            else mask_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/masks/" + file_name + "_result.png";
            box_file_writer(all_boxes[j], box_result_path);
            mask_file_writer(all_boxes[j], mask_result_path);
        }
    return 0;
}