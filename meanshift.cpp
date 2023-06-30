#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "segment_plates.h"
#include "find_histogram.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    string path;
    if(argc == 2) {
        path = argv[1];
    }

    Mat rgb_img, hsv_img, gray_img, dst;
    // take first frame of the video
    //capture >> frame;
    Mat img = imread(path);
    string name = "Original Img";
    namedWindow(name, WINDOW_NORMAL);
    imshow(name, img);
    vector<Mat> dishes;
    vector<string> predicted_classes;
    dishes = segment_plates(img);
    rgb_img = dishes[2];
    cvtColor(rgb_img, hsv_img, COLOR_BGR2HSV);
    TermCriteria  termcrit = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 5, 1);
    vector<int>  sp{50};
    vector<int>  sr{ 70};
    vector<int>  p{0, 1, 2, 3};
    for (auto & sp_i : sp) {
        for (auto & sr_i : sr) {
            pyrMeanShiftFiltering(rgb_img, dst, sp_i, sr_i, 3, termcrit);
            string w_name = to_string(sp_i) + "-" + to_string(sr_i) + " Level 3";
            cout << w_name << endl;
            namedWindow(w_name, WINDOW_NORMAL);
            imshow(w_name, dst);
        }
        
    }
    cvtColor(dst, gray_img, COLOR_BGR2GRAY);
    find_histogram(gray_img);
    Mat otu_img;
    threshold(gray_img, otu_img, 0, 255,  THRESH_BINARY | THRESH_OTSU);

    string o_name = " Otus Method";
    namedWindow(o_name, WINDOW_NORMAL);
    imshow(o_name, otu_img);
    waitKey(0);


    /*
    Ptr<MSER> mser = MSER::create(5, 1000, 1440000, 0.25, 0.2, 200, 1.01, 0.003, 1);
    vector<Rect> bboxes;
    vector<vector<Point>> msers;
    mser->detectRegions(dst, msers, bboxes);

    for (auto & box : bboxes) {
        rectangle(frame, box, 255, 2);
    }
    string b_name = "mser";
    namedWindow(b_name, WINDOW_NORMAL);
    imshow(b_name, dst);
    waitKey(0);
    */

}