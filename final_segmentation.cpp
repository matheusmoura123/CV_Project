#include "main_header.h"

int main(int argc, char **argv) {
    string path;
    if (argc == 2) {
        path = argv[1];
    }

    //Read image
    Mat rgb_img_original, hsv_img, gray_img, mean_img;
    Mat img = imread(path);

    //Separate dishes
    vector<Mat> foods;
    vector<box> dishes;
    vector<box> plates_boxes = segment_plates(img, foods);
    for (const auto& plate_box: plates_boxes) {
        box food_box = segment_food(plate_box);
        //imshow("food", food_box.img);
        dishes = separate_food(food_box);
        //waitKey();
    }





    /*

    //Let's segment the mask that we've got

    Mat bi_img_new;
    bilateralFilter(rgb_img, bi_img_new, 5, 150, 50);

    Mat mean_img_new, rgb_segment;

    mean_img_new = meanshift(bi_img_new, 0, 30);

    imshow("New meanshift", mean_img_new);

    rgb_segment = segment_rgb_hsv(mean_img_new, 110, 110, 110, 50, 50, 50);
    imshow("New segment", rgb_segment);
    waitKey();


    Mat image_3c_new;
    Mat in_new[3] = {rgb_segment, rgb_segment, rgb_segment};
    merge(in_new, 3, image_3c_new);


    cvtColor(rgb_segment, rgb_segment, COLOR_BGR2GRAY);
     */

    /*
    // Morphological Closing
    int morph_size_new = 2;
    Mat element_new = getStructuringElement(MORPH_CROSS, Size(2 * morph_size_new + 1, 2 * morph_size_new + 1), Point(morph_size_new, morph_size_new));
    Mat morph_img_new;
    morphologyEx(img_out, morph_img_new, MORPH_CLOSE, element_new, Point(-1, -1), 6);
    imshow("morph", morph_img_new);

    Mat morph_rgb_new;
    Mat chan_new[3] = {morph_img_new, morph_img_new, morph_img_new};
    merge(chan_new, 3, morph_rgb_new);

    Mat final_new = get_contours(morph_rgb_new);
    cvtColor(final_new, final_new, COLOR_BGR2GRAY);


    for (int y = 0; y < rgb_img.rows; ++y) {
        for (int x = 0; x < rgb_img.cols; ++x) {
            if(final.at<uchar>(y, x) == 0 ) {
                rgb_img.at<Vec3b>(y,x)[0] = 0;
                rgb_img.at<Vec3b>(y,x)[1] = 0;
                rgb_img.at<Vec3b>(y,x)[2] = 0;
            }
        }
    }
    imshow("mask", rgb_img);
    waitKey();
     */




}

