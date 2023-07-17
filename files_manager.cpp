#include "main_header.h"

Scalar color_ID (int ID) {
    if (ID == -1) return {255, 255, 255};
    if (ID == 0) return {0, 0, 0};
    if (ID == 1) return {255, 0, 0};
    if (ID == 2) return {0, 255, 0};
    if (ID == 3) return {0, 0, 255};
    if (ID == 4) return {255, 255, 0};
    if (ID == 5) return {255, 0, 255};
    if (ID == 6) return {0, 255, 255};
    if (ID == 7) return {124, 255, 255};
    if (ID == 8) return {255, 124, 255};
    if (ID == 9) return {255, 255, 124};
    if (ID == 10) return {124, 124, 255};
    if (ID == 11) return {124, 255, 124};
    if (ID == 12) return {255, 124, 124};
    if (ID == 13) return {124, 124, 124};
    return {0, 0, 0};
}

void draw_rectangles_masks (Mat img, const vector<box>& boxes, int tray_num, const string& file_name) {
    for (int i = 0; i < boxes.size(); ++i) {
        Point p1(boxes[i].p0x, boxes[i].p0y);
        // Bottom Right Corner
        Point p2(boxes[i].p0x+boxes[i].width, boxes[i].p0y+boxes[i].height);
        Point pt(boxes[i].p0x+3, boxes[i].p0y+boxes[i].height-3);
        int thickness = 2;
        Scalar color = color_ID((boxes[i].ID));
        // Drawing the Rectangle
        rectangle(img, p1, p2, color,thickness, LINE_8);

        Point text_position(80, 80);//Declaring the text position//
        int font_size = 1;//Declaring the font size//
        int font_weight = 2;//Declaring the font weight//
        string text = "ID: " + to_string(boxes[i].ID);
        putText(img, text, pt ,FONT_HERSHEY_COMPLEX, font_size ,color, font_weight);
    }
    //imshow("img with boxes" + file_name, img);

    string img_result_path;
    if (file_name == "food_image") img_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/"  + file_name + "_result.jpg";
    else img_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/" + file_name + "_result.jpg";
    imwrite(img_result_path, img);

    Mat mask_out = mask_img_builder_color(boxes);
    string mask_result_path;
    if (file_name == "food_image") mask_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/"  + file_name + "_color_mask_result.jpg";
    else mask_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/" + file_name + "_color_mask_result.jpg";
    imwrite(mask_result_path, mask_out);
}

int box_file_writer (const vector<box>& boxes, const string& path) {
    try {
        //cout << "Writing box contents to file..." << endl;
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
        else cout << "Problem saving box file" << endl;
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    //cout << "Done!" << endl;
    return 0;
}

int box_file_reader (vector<box>& boxes, const string& path) {
    try {
        //cout << "Reading box contents from file..." << endl;
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
        else cout << "Problem reading box file" << endl;
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    //cout << "Done!" << endl;
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

Mat mask_img_builder_color (const vector<box>& boxes) {
    int img_cols = 1280;
    int img_rows = 960;
    Mat img_out = Mat::zeros(img_rows, img_cols, CV_8UC3);
    for(const auto box:boxes) {
        Mat sup_img = box.img.clone();
        //string name = "gray" + to_string(box.p0y);
        //imshow(name, gray_img );
        for (int y = 0; y < box.height; ++y) {
            for (int x = 0; x < box.width; ++x) {
                if (sup_img.at<Vec3b>(y, x)[0] != 0 && sup_img.at<Vec3b>(y, x)[1] != 0 && sup_img.at<Vec3b>(y, x)[2] != 0) {
                    img_out.at<Vec3b>(y+box.p0y, x+box.p0x) = {(uchar)color_ID(box.ID)[0],(uchar)color_ID(box.ID)[1],(uchar)color_ID(box.ID)[2]};
                }
            }
        }
    }
    return img_out;
}

int mask_file_writer (const vector<box>& boxes, const string& path) {
    try {
        //cout << "Creating mask img file..." << endl;
        Mat img = mask_img_builder(boxes);
        bool check = imwrite(path, img);
        if (check) {}
        else cout << "Problem with saving img file" << endl;
    }
    catch (const char* msg) {
        cerr << msg << endl;
    }
    //cout << "Done!" << endl;
    return 0;
}

int save_boxes_masks_at_tray_stage (const vector<box>& boxes, int tray_num, const string& file_name) {
    string box_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/bounding_boxes/" + file_name + "_result_box.txt";
    string mask_result_path;

    if (file_name == "food_image") mask_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/masks/"  + file_name + "_mask_result.png";
    else mask_result_path = RESULTS_PATH + to_string(tray_num + 1) + "/masks/" + file_name + "_result.png";

    box_file_writer(boxes, box_result_path);
    mask_file_writer(boxes, mask_result_path);
    return 0;
}