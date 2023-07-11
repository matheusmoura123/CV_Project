#include "main_header.h"

int main(int argc, char** argv) {
    Mat img1,img2;
    vector<Mat> seg;
    img1 = imread(argv[1]);
   //img2 = imread(argv[2]);

    //sift_matching(img1,img2);

    segment_plates(img1, seg);


   // imshow("Original", img);
   //waitKey(0);


    return 0;
}
