#include "main_header.h"

int main(int argc, char** argv) {

    /*
    //test for masks_mIoU
    Mat mask_truth = imread("../Food_leftover_dataset/tray5/masks/food_image_mask.png");
    Mat mask_result = imread("../Food_leftover_dataset/tray5/masks/leftover1.png");
    cout << masks_mIoU(mask_truth, mask_result) << endl;
    */

    //test for img_mAp
    vector<box> boxes_truth, boxes_results;
    box_file_reader(boxes_truth, "../Food_leftover_dataset/tray2/bounding_boxes/food_image_bounding_box.txt");
    box_file_reader(boxes_results, "../Food_leftover_dataset/tray2/bounding_boxes/food_image_bounding_box_test.txt");
    double map = img_mAp(boxes_truth, boxes_results);
    cout << "img_mAp = " <<  map << endl;

    return 0;
}
