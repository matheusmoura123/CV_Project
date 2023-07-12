#include "main_header.h"

double boxes_IoU (const box& box1, const box& box2) {
    vector<double> c{0, 0, 0};
    int y_min = min(box1.p0y, box2.p0y);
    int y_max = max(box1.p0y+box1.height, box2.p0y+box2.height);
    int x_min = min(box1.p0x, box2.p0x);
    int x_max = max(box1.p0x+box1.width, box2.p0x+box2.width);
    for (int y = y_min; y <= y_max; ++y) {
        for (int x = x_min; x <= x_max; ++x) {
            if (box1.is_inside(y,x)) {
                if (box2.is_inside(y, x)) {
                    ++c[0];
                } else ++c[1];
            } else if (box2.is_inside(y, x)) {
                ++c[2];
            }
        }
    }
    return c[0]/(c[0]+c[1]+c[2]);
}

double img_mAp (const vector<box>& boxes_truth, const vector<box>& boxes_result) {
    //The confidence doesn't matter because we have only one box for each class
    double iou_tresh = 0.5;
    double total_truth = boxes_truth.size();
    vector<double> TP(boxes_result.size(), 0);
    vector<double> FP(boxes_result.size(), 0);
    vector<double> AP(boxes_result.size(), 0);
    //vector<double> FN(boxes_result.size(), 0);
    for (int i = 0; i < boxes_result.size(); ++i) {
        double precision = 0, recall = 0, round_recall;
        int find = 0;
        for (int j = 0; j < boxes_truth.size(); ++j) {
            if (boxes_result[i].ID == boxes_truth[j].ID){
                if (boxes_IoU(boxes_result[i], boxes_truth[j]) >= iou_tresh) {
                    TP[i] += 1;
                } else FP[i] += 1;
                ++find;
            }
        }
        if(!find) FP[i] += 1;
        precision = TP[i]/(TP[i] + FP[i]);
        recall = TP[i]/total_truth;
        //round_recall= floor(recall*10)+1;
        //AP[i] = precision*round_recall/11;
        AP[i] = precision;
        //cout << AP[i] << endl;
    }
    //cout << "ac = " << accumulate(AP.begin(),AP.end(), 0.0) << endl;
    //cout << "size = " << boxes_result.size() << endl;
    return std::accumulate(AP.begin(),AP.end(), 0.0)/boxes_result.size();
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

vector<vector<double>> leftover_ratio (const Mat& mask_before, const Mat& mask_after) {
    Mat gray_before, gray_after;
    cvtColor(mask_before, gray_before, COLOR_BGR2GRAY);
    cvtColor(mask_after, gray_after, COLOR_BGR2GRAY);
    //parameters for histogram
    int cat_bins = 14;
    int histSize = cat_bins;
    float cat_ranges[] = {0, 14};
    const float *ranges[] = {cat_ranges};
    bool uniform = true;
    bool accumulate = false;
    Mat hist_before, hist_after;
    calcHist(&gray_before, 1, nullptr, Mat(), hist_before, 1, &histSize, ranges, uniform, accumulate);
    calcHist(&gray_after, 1, nullptr, Mat(), hist_after, 1, &histSize, ranges, uniform, accumulate);

    vector<vector<double>> leftover_ratio;
    for (int i = 1; i < histSize; ++i) {
        if (double(hist_before.at<float>(i)) != 0)
            leftover_ratio.push_back({double(i), (hist_after.at<float>(i))/double(hist_before.at<float>(i))});
    }

    cout << "ratio" << endl;
    for (int i = 0; i < leftover_ratio.size(); ++i) {
        cout << leftover_ratio[i][0] << ", " << leftover_ratio[i][1] << endl;
    }

    return leftover_ratio;
}