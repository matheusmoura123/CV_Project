#include "main_header.h"

const string TRAY_PATH = "../Food_leftover_dataset/tray";
const string IMAGE_EXT = ".jpg";
bool EXIT = false;
const int NUMBER_TRAYS = 8;


vector<food> categories_left;
const vector<food> pastaCategories{
        {"pesto", 4, 1},
        {"pomodoro", 5, 2},
        {"ragu", 2, 3},
        {"pasta_clams", 8, 4},
};
const vector<food> foodCategories{
        //{"plate", 9, 0},
        {"rice", 7, 5},
        {"pork", 9, 6},
        {"fish", 10, 7},
        {"rabbit", 12, 8},
        {"seafood", 5, 9},
        {"beans", 13, 10},
        {"potato", 13, 11},
        //{"lettuce", 15, 12},
        {"bread", 18, 13},
        //{"carrot", 6, 14},
        //{"pepper", 2, 15},
        //{"tomato", 10, 16},
        {"pasta", 31, 17},
        {"salad", 15, 18},
        //{"plate_salad", 3, 19},
};

int main(int argc, char **argv)
{
    int option = 2;
    if(argc == 2) {
        cout << "A path was passed as argument." << endl;
        cout << "Press [ESC] to exit." << endl;
        cout << "---------------------------------------------------" << endl;
        option = 4;
    }
    else {
        cout << "[1] Run all trays" << endl;
        cout << "[2] Run specific trays" << endl;
        cout << "[3] Run specific tray with no plate segmentation" << endl;
        cout << "[4] Run from a given path" << endl;
        cout << "[5] Run full tray mode" << endl;
        cout << "Option: ";
        cin >> option;
    }

    if (option == 1){
        cout << "Press [ANY] key to keep going or [ESC] to exit." << endl;
        for (int i = 0; i < NUMBER_TRAYS; ++i) {
            for (int j = 0; j < 4; ++j) {
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
                //img1 = imread(argv[1]);
                img = imread(TRAY_PATH + to_string(i + 1) + "/" + file_name + IMAGE_EXT);

                string window_name_img = "Tray " + to_string(i + 1) + " " + file_name;
                namedWindow(window_name_img, WINDOW_NORMAL);
                resizeWindow(window_name_img, 600, 400);
                imshow(window_name_img, img);
                key = waitKeyEx(0);
                if (key == 1048603) return (0);

                vector<Mat> dishes;
                vector<string> predicted_classes;
                dishes = segment_plates(img);
                predict_categories(dishes, foodCategories, predicted_classes);
                for (int k = 0; k < predicted_classes.size(); ++k) {
                    if (predicted_classes[k] == "pasta"){
                        vector<Mat> new_dish{dishes[k]};
                        vector<string> new_class;
                        predict_categories(new_dish, pastaCategories, new_class);
                        predicted_classes[k] = new_class[0];
                    }
                    //Show Image with keypoints
                    string window_name = to_string(k+1) + ": This image have " + predicted_classes[k];
                    namedWindow(window_name, WINDOW_NORMAL);
                    resizeWindow(window_name, 400, 400);
                    imshow(window_name, dishes[k]);
                    key = waitKeyEx(0);
                    if (key == 1048603) return(0);
                }
                destroyAllWindows();
            }
        }

    }
    else if(option == 2){
        while (!EXIT) {
            int key;
            int tray_num, image_num;
            cout << "Tray: ";
            cin >> tray_num;
            if (tray_num <= 0 or tray_num > 8) tray_num = 1;
            cout << "Image Number {0, 1, 2, 3}: ";
            cin >> image_num;
            cout << "Press [ANY] key to keep going or [ESC] to exit." << endl;
            cout << "---------------------------------------------------" << endl;
            string file_name;
            switch(image_num) {
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
            //img1 = imread(argv[1]);
            img = imread(TRAY_PATH + to_string(tray_num) + "/" + file_name + IMAGE_EXT);

            string window_name_img = "Tray " + to_string(tray_num) + " " + file_name;
            namedWindow(window_name_img, WINDOW_NORMAL);
            resizeWindow(window_name_img, 600, 400);
            imshow(window_name_img, img);
            moveWindow(window_name_img, 500, 500);
            key = waitKeyEx(0);
            if (key == 1048603) return(0); //if press ESC end program


            vector <Mat> dishes;
            vector <string> predicted_classes;
            dishes = segment_plates(img);
            predict_categories(dishes, foodCategories, predicted_classes);
            destroyAllWindows();
        }
    }
    else if(option == 3){
        while (!EXIT) {
            int key;
            int tray_num, image_num;
            cout << "Tray: ";
            cin >> tray_num;
            if (tray_num <= 0 or tray_num > 8) tray_num = 1;
            cout << "Image Number {0, 1, 2, 3}: ";
            cin >> image_num;
            cout << "Press [ANY] key to keep going or [ESC] to exit." << endl;
            cout << "---------------------------------------------------" << endl;
            string file_name;
            switch(image_num) {
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
            //img1 = imread(argv[1]);
            img = imread(TRAY_PATH + to_string(tray_num) + "/" + file_name + IMAGE_EXT);

            string window_name_img = "Tray " + to_string(tray_num) + " " + file_name;

            vector <Mat> dishes;
            vector <string> predicted_classes;
            dishes = segment_plates(img);
            predict_categories(dishes, foodCategories, predicted_classes);
            destroyAllWindows();
        }
    }
    else if(option == 4){
        while (!EXIT) {
            int key;
            string path;
            if(argc == 2) {
                path = argv[1];
            } else {
                cout << "Path of the image: ";
                cin >> path;
                cout << "Press [ANY] key to keep going or [ESC] to exit." << endl;
                cout << "---------------------------------------------------" << endl;
            }

            Mat img;
            //img1 = imread(argv[1]);
            img = imread(path);
            string window_name_img = path;
            namedWindow(window_name_img, WINDOW_NORMAL);
            resizeWindow(window_name_img, 600, 400);
            imshow(window_name_img, img);
            key = waitKeyEx(0);
            if (key == 1048603) return (0);

            vector<Mat> img_sections;
            int rect_size = 200;
            for (int y=0; y<img.rows; y+=rect_size) {
                for (int x=0; x<img.cols; x+=rect_size) {
                    //CROPPING Section
                    int rowf_y, colf_x;
                    rowf_y = y + rect_size;
                    colf_x = x + rect_size;
                    if (rowf_y >= img.rows) {
                        rowf_y = img.rows-1;
                        if (rowf_y - y < 100) continue;
                    }
                    if (colf_x >= img.cols) {
                        colf_x = img.cols-1;
                        if (colf_x - x < 100) continue;
                    }
                    img_sections.push_back(img(Range(y, rowf_y), Range(x, colf_x)));
                }
            }
            /*
            string crp_img_name = to_string(y) + ":" + to_string(x) + "Img";
            namedWindow(crp_img_name, WINDOW_NORMAL);
            resizeWindow(crp_img_name, 400, 400);
            imshow(crp_img_name, img_section);
            key = waitKeyEx(0);
            if (key == 1048603) return (0);
             */

            vector <string> predicted_classes;
            predict_categories(img_sections, foodCategories, predicted_classes);
            for (const auto & predicted: predicted_classes) {
                cout << predicted << endl;
            }
            destroyAllWindows();
        }
    }
    else if(option == 5){
        while (!EXIT) {
            int key;
            int tray_num;
            cout << "Tray: ";
            cin >> tray_num;
            if (tray_num <= 0 or tray_num > 8) tray_num = 1;
            cout << "Press [ANY] key to keep going or [ESC] to exit." << endl;
            cout << "---------------------------------------------------" << endl;
            for (int image_num = 0; image_num < 4; ++image_num) {
                string file_name;
                if (image_num == 0){
                    file_name = "food_image";
                    Mat img;
                    img = imread(TRAY_PATH + to_string(tray_num) + "/" + file_name + IMAGE_EXT);

                    string window_name_img = "Tray " + to_string(tray_num) + " " + file_name;
                    namedWindow(window_name_img, WINDOW_NORMAL);
                    resizeWindow(window_name_img, 600, 400);
                    imshow(window_name_img, img);
                    moveWindow(window_name_img, 500, 500);
                    key = waitKeyEx(0);
                    if (key == 1048603) return(0); //if press ESC end program

                    vector<Mat> dishes;
                    vector<string> predicted_classes;
                    dishes = segment_plates(img);
                    predict_categories(dishes,foodCategories, predicted_classes);

                    for (int i = 0; i < predicted_classes.size(); ++i) {
                        for (int j = 0; j < foodCategories.size(); ++j) {
                            if (foodCategories[j].className == predicted_classes[i]) {
                                categories_left.push_back(foodCategories[j]);
                            }
                        }
                    }
                }
                else if (image_num == 1){
                    cout << "Running Leftover1..." << endl;
                    file_name = "leftover1";
                    Mat img;
                    img = imread(TRAY_PATH + to_string(tray_num) + "/" + file_name + IMAGE_EXT);

                    string window_name_img = "Tray " + to_string(tray_num) + " " + file_name;
                    namedWindow(window_name_img, WINDOW_NORMAL);
                    resizeWindow(window_name_img, 600, 400);
                    imshow(window_name_img, img);
                    moveWindow(window_name_img, 500, 500);
                    key = waitKeyEx(0);
                    if (key == 1048603) return(0); //if press ESC end program

                    vector<Mat> dishes;
                    vector<string> predicted_classes;
                    dishes = segment_plates(img);
                    predict_categories(dishes,categories_left, predicted_classes);
                    cout << "---------------------------------------------------" << endl;
                }
                else if (image_num == 2) {
                    cout << "Running Leftover2..." << endl;

                    file_name = "leftover2";
                    Mat img;
                    img = imread(TRAY_PATH + to_string(tray_num) + "/" + file_name + IMAGE_EXT);

                    string window_name_img = "Tray " + to_string(tray_num) + " " + file_name;
                    namedWindow(window_name_img, WINDOW_NORMAL);
                    resizeWindow(window_name_img, 600, 400);
                    imshow(window_name_img, img);
                    moveWindow(window_name_img, 500, 500);
                    key = waitKeyEx(0);
                    if (key == 1048603) return(0); //if press ESC end program

                    vector<Mat> dishes;
                    vector<string> predicted_classes;
                    dishes = segment_plates(img);
                    predict_categories(dishes,categories_left, predicted_classes);
                    cout << "---------------------------------------------------" << endl;
                }
                else if (image_num == 3) {
                    cout << "Running Leftover3..." << endl;

                    file_name = "leftover3";
                    Mat img;
                    img = imread(TRAY_PATH + to_string(tray_num) + "/" + file_name + IMAGE_EXT);

                    string window_name_img = "Tray " + to_string(tray_num) + " " + file_name;
                    namedWindow(window_name_img, WINDOW_NORMAL);
                    resizeWindow(window_name_img, 600, 400);
                    imshow(window_name_img, img);
                    moveWindow(window_name_img, 500, 500);
                    key = waitKeyEx(0);
                    if (key == 1048603) return(0); //if press ESC end program

                    vector<Mat> dishes;
                    vector<string> predicted_classes;
                    dishes = segment_plates(img);
                    predict_categories(dishes,categories_left, predicted_classes);
                    cout << "---------------------------------------------------" << endl;
                }
                else return(0);
            }
            key = waitKeyEx(0);
            if (key == 1048603) return(0);
            destroyAllWindows();
            return(0);
        }
    }
    else return(0);

    return(0);
}