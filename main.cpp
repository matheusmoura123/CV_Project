#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "segment_plates.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    Mat img, seg;
    img = imread(argv[1]);

    seg = segment_plates(img);

   // imshow("Original", img);
//    waitKey(0);


    return 0;
}
