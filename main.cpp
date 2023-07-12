#include "main_header.h"

int main(int argc, char** argv) {

    /*
    //test for masks_mIoU
    Mat mask_truth = imread("../Food_leftover_dataset/tray5/masks/food_image_mask.png");
    Mat mask_result = imread("../Food_leftover_dataset/tray5/masks/leftover1.png");

    cout << masks_mIoU(mask_truth, mask_result) << endl;
     */

    // testing for histogram
    Mat img = imread(argv[1]);


    plot_histogram(img);
    find_histogram(img);


    return 0;
}
