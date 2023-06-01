#include "segment_plates.h"
#include "sift_matching.h"


int main(int argc, char** argv) {
    Mat img1,img2;
    vector<Mat> seg;
    img1 = imread(argv[1]);
   //img2 = imread(argv[2]);

    //sift_matching(img1,img2);

    seg = segment_plates(img1);


   // imshow("Original", img);
   //waitKey(0);


    return 0;
}
