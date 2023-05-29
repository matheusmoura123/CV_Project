#include "segment_plates.h"


int main(int argc, char** argv) {
    Mat img, seg;
    img = imread(argv[1]);

    seg = segment_plates(img);

   // imshow("Original", img);
//    waitKey(0);


    return 0;
}
