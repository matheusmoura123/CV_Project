#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include "segment_plates.h"
using namespace cv;
using namespace std;
int main(int argc, char **argv)
{
    /*
    const string about =
            "This sample demonstrates the meanshift algorithm.\n"
            "The example file can be downloaded from:\n"
            " https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4";
    const string keys =
            "{ h help | | print this help message }"
            "{ @image |<none>| path to image file }";
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string filename = parser.get<string>("@image");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    VideoCapture capture(filename);
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }
    */

    string path;
    if(argc == 2) {
        path = argv[1];
    }

    Mat frame, roi, hsv_roi, mask, dst;
    // take first frame of the video
    //capture >> frame;
    Mat img = imread(path);
    vector <Mat> dishes;
    vector <string> predicted_classes;
    dishes = segment_plates(img);
    frame = dishes[0];
    TermCriteria  termcrit = TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 5, 1);
    vector<int>  sp{50};
    vector<int>  sr{ 70};
    vector<int>  p{0, 1, 2, 3};
    for (auto & sp_i : sp) {
        for (auto & sr_i : sr) {
            pyrMeanShiftFiltering(frame, dst, sp_i, sr_i, 3, termcrit);
            string w_name = to_string(sp_i) + "-" + to_string(sr_i) + " Level 3";
            cout << w_name << endl;
            namedWindow(w_name, WINDOW_NORMAL);
            imshow(w_name, dst);
        }
        
    }
    Ptr<MSER> mser = MSER::create(5, 1000, 1440000, 0.25, 0.2, 200, 1.01, 0.003, 1);
    vector<Rect> bboxes;
    vector<vector<Point>> msers;
    mser->detectRegions(dst, msers, bboxes);

    for (auto & box : bboxes) {
        rectangle(frame, box, 255, 2);
    }
    string b_name = "mser";
    namedWindow(b_name, WINDOW_NORMAL);
    imshow(b_name, frame);
    waitKey(0);
    /*
    // setup initial location of window
    Rect track_window(300, 200, 100, 50); // simply hardcoded the values
    // set up the ROI for tracking
    roi = frame(track_window);
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
    inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);
    float range_[] = {0, 180};
    const float* range[] = {range_};
    Mat roi_hist;
    int histSize[] = {180};
    int channels[] = {0};
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);
    // Setup the termination criteria, either 10 iteration or move by at least 1 pt
    TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);
    while(true){
        Mat hsv, dst;
        capture >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        calcBackProject(&hsv, 1, channels, roi_hist, dst, range);
        // apply camshift to get the new location
        RotatedRect rot_rect = CamShift(dst, track_window, term_crit);
        // Draw it on image
        Point2f points[4];
        rot_rect.points(points);
        for (int i = 0; i < 4; i++)
            line(frame, points[i], points[(i+1)%4], 255, 2);
        imshow("img2", frame);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
     */
}