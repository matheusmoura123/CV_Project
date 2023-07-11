#include "main_header.h"

int main(int argc, char** argv) {

    //test for masks_mIoU
    Mat mask_truth = imread("../Food_leftover_dataset/tray5/masks/food_image_mask.png");
    Mat mask_result = imread("../Food_leftover_dataset/tray5/masks/leftover1.png");

    cout << masks_mIoU(mask_truth, mask_result) << endl;

    return 0;
}
