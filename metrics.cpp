#include "main_header.h"

double boxes_IoU (const box& box1, const box& box2) {
    vector<double> c{0, 0, 0};
    int y_min = min(box1.p0y, box2.p0y);
    int y_max = max(box1.p0y+box1.height, box2.p0y+box2.height);
    int x_min = min(box1.p0x, box2.p0x);
    int x_max = max(box1.p0x+box1.width, box2.p0x+box2.width);
    for (int y = y_min; y <= y_max; ++y) {
        for (int x = x_min; x <= x_max; ++x) {
            if ((y >= box1.p0y && y <= box1.p0y+box1.height) && (x >= box1.p0x && x <= box1.p0x+box1.width)) {
                if ((y >= box2.p0y && y <= box2.p0y+box2.height) && (x >= box2.p0x && x <= box2.p0x+box2.width)) {
                    ++c[0];
                } else ++c[1];
            } else if ((y >= box2.p0y && y <= box2.p0y+box2.height) && (x >= box2.p0x && x <= box2.p0x+box2.width)) {
                ++c[2];
            }
        }
    }
    return c[0]/(c[0]+c[1]+c[2]);
}

double masks_mIoU (const Mat& mask_truth, const Mat& mask_result) {
    Mat gray_truth, gray_result;
    cvtColor(mask_truth, gray_truth, COLOR_BGR2GRAY);
    cvtColor(mask_result, gray_result, COLOR_BGR2GRAY);
    int number_of_categories = 14; //including background
    vector<double> overlap_cat(number_of_categories, 0);
    vector<double> truth_cat(number_of_categories, 0);
    vector<double> result_cat(number_of_categories, 0);
    for (int y = 0; y < gray_truth.rows; ++y) {
        for (int x = 0; x < gray_truth.cols; ++x) {
            truth_cat[gray_truth.at<uchar>(y, x)] += 1;
            result_cat[gray_result.at<uchar>(y, x)] += 1;
            if (gray_truth.at<uchar>(y, x) == gray_result.at<uchar>(y, x)) {
                overlap_cat[gray_truth.at<uchar>(y, x)] += 1;
            }
        }
    }
    double sum_IoU = 0, truth_categories = 0;
    for (int k = 0; k < number_of_categories; ++k) {
        if (truth_cat[k] != 0) {
            sum_IoU += overlap_cat[k]/(truth_cat[k]+result_cat[k]-overlap_cat[k]);
            ++truth_categories;
        }
    }
    /*
    //Just showing each category count
    cout << "truth result overlap" << endl;
    for (int k = 0; k < number_of_categories; ++k) {
        cout << truth_cat[k] << " " << result_cat[k] << " " << overlap_cat[k] << endl;
    }
    cout << truth_categories << endl;
    */
    return sum_IoU/truth_categories;
}