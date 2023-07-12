#include "main_header.h"

int main(int argc, char **argv) {
    string path;
    if (argc == 2) {
        path = argv[1];
    }

    //Read image
    Mat rgb_img, hsv_img, gray_img, mean_img;
    Mat img = imread(path);
    string name = "Original Img";
    //namedWindow(name, WINDOW_NORMAL);
    //imshow(name, img);

    //Separate dishes
    vector<Mat> dishes;
    vector<string> predicted_classes;
    segment_plates(img, dishes);
    rgb_img = dishes[0];
    imshow("Plate", rgb_img);
    //cvtColor(rgb_img, hsv_img, COLOR_BGR2HSV);

    //Bilateral filter to try to create sharper edges
    Mat bi_img;
    bilateralFilter(rgb_img, bi_img, 5, 150, 50);
    //imshow("bilateral filter", bi_img);

    mean_img = meanshift(bi_img, 70, 50);

    //cvtColor(mean_img, gray_img, COLOR_HSV2BGR);
    //cvtColor(mean_img, gray_img, COLOR_BGR2GRAY);
    //find_histogram(gray_img);



    //array<int, 3> hue_val = find_histogram(mean_img);
    //
    // cvtColor(mean_img,mean_img,COLOR_BGR2HSV);
    Mat hsv_segment = segment_hsv(mean_img, 160, 160, 160, 80, 80, 80);

    // here was otsu



    Mat hsv_image;
    Mat in[3] = {hsv_segment, hsv_segment, hsv_segment};
    merge(in, 3, hsv_image);



    for (int i = 0; i < hsv_segment.rows; ++i)
    {
        for (int j = 0; j < hsv_segment.cols; ++j)
        {
            int blue_temp = int(rgb_img.at<Vec3b>(i,j)[0]);
            int green_temp = int(rgb_img.at<Vec3b>(i,j)[1]);
            int red_temp = int(rgb_img.at<Vec3b>(i,j)[2]);

            if(blue_temp == 0 && green_temp == 0 && red_temp == 0) {
                hsv_image.at<Vec3b>(i,j)[0] = 0;
                hsv_image.at<Vec3b>(i,j)[1] = 0;
                hsv_image.at<Vec3b>(i,j)[2] = 0;
            }
        }
    }

    Mat img_out = get_contours(hsv_image);

    namedWindow("konture");
    imshow("konture", img_out);

    cvtColor(img_out, img_out, COLOR_BGR2GRAY);


    //cvtColor(otsu_rgb, otsu_rgb, COLOR_BGR2GRAY);
    // Morphological Closing
    int morph_size = 2;
    Mat element = getStructuringElement(MORPH_CROSS, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
    Mat morph_img;
    morphologyEx(img_out, morph_img, MORPH_OPEN, element, Point(-1, -1), 6);
    imshow("morph", morph_img);

    Mat morph_rgb;
    Mat chan[3] = {morph_img, morph_img, morph_img};
    merge(chan, 3, morph_rgb);

    Mat final = get_contours(morph_rgb);
    cvtColor(final, final, COLOR_BGR2GRAY);

    //Using Otu as mask
    for (int y = 0; y < rgb_img.rows; ++y) {
        for (int x = 0; x < rgb_img.cols; ++x) {
            if(final.at<uchar>(y, x) == 0 ) {
                rgb_img.at<Vec3b>(y,x)[0] = 0;
                rgb_img.at<Vec3b>(y,x)[1] = 0;
                rgb_img.at<Vec3b>(y,x)[2] = 0;
            }
        }
    }
    imshow("mask", rgb_img);
    waitKey();

}




//---------------GRAVEYARD OF DEAD IDEAS_______________________________________

    /*
    //Log Transform to deal with the shadows
    Mat log_img;
    rgb_img.convertTo(log_img, CV_32F);
    log_img = log_img + 1;
    log(log_img, log_img);
    normalize(log_img, log_img, 0, 255, NORM_MINMAX);
    convertScaleAbs(log_img, log_img);
    imshow("log transform", log_img);
    */

    /*
    //CANNY TRY
    Mat src_gray;
    Mat dst_c, detected_edges;
    int lowThreshold = 20;
    const int max_lowThreshold = 100;
    const int ratio = 3;
    const int kernel_size = 3;
    const char* window_name = "Edge Map";
    dst_c.create( dst.size(), dst.type() );
    cvtColor( dst, src_gray, COLOR_BGR2GRAY );
    namedWindow( window_name, WINDOW_AUTOSIZE );
    //createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    dst_c = Scalar::all(0);
    dst.copyTo( dst_c, detected_edges);
    imshow( window_name, dst_c );
     */


    /*
    //MERGING CLUSTERS
    float colorThreshold = 70.0;     // Adjust as needed
    float distanceThreshold = 100.0;  // Adjust as needed
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    Mat result;
    cv::cvtColor(dst, result, cv::COLOR_HSV2BGR);
    cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
    int numLabels = cv::connectedComponentsWithStats(result, labels, stats, centroids);
    std::vector<cv::Vec3b> mergedCentroids;
    std::vector<int> mergedLabels(numLabels, -1);
    for (int label = 1; label < numLabels; ++label) {
        // Extract centroid of current cluster
        cv::Vec3b centroid = centroids.at<cv::Vec3b>(label);

        // Check for color and distance similarity with merged clusters
        bool merged = false;
        for (int mergedLabel = 0; mergedLabel < mergedCentroids.size(); ++mergedLabel) {
            // Compute color and distance difference
            double colorDiff = norm(centroid, mergedCentroids[mergedLabel]);
            double distanceDiff = norm(centroid - mergedCentroids[mergedLabel]);

            // Merge if color and distance differences are below thresholds
            if (colorDiff < colorThreshold && distanceDiff < distanceThreshold) {
                mergedLabels[label] = mergedLabel;
                merged = true;
                break;
            }
        }
        // Create a new merged cluster if not merged with existing ones
        if (!merged) {
            mergedLabels[label] = mergedCentroids.size();
            mergedCentroids.push_back(centroid);
        }
    }
    cv::Mat mergedImage(result.size(), CV_8UC3);
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            int label = labels.at<int>(i, j);
            int mergedLabel = mergedLabels[label];
            mergedImage.at<cv::Vec3b>(i, j) = mergedCentroids[mergedLabel];
        }
    }
    cv::imshow("Merged Result", mergedImage);
    cv::waitKey(0);
     */


    /*
    Mat adap_img;
    adaptiveThreshold(gray_img,adap_img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,11,2);
    string a_name = " Adaptive Method";
    namedWindow(a_name, WINDOW_NORMAL);
    imshow(a_name, otu_img);
    waitKey(0);
    */

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



