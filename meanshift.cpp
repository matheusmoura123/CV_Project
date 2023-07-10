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
    namedWindow(name, WINDOW_NORMAL);
    imshow(name, img);

    //Separate dishes
    vector<Mat> dishes;
    vector<string> predicted_classes;
    dishes = segment_plates(img);
    rgb_img = dishes[0];
    //cvtColor(rgb_img, hsv_img, COLOR_BGR2HSV);

    //Bilateral filter to try to create sharper edges
    Mat bi_img;
    bilateralFilter(rgb_img, bi_img, 5, 150, 50);
    imshow("bilateral filter", bi_img);

    //MeanShift
    TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 20, 1);
    vector<int> sp{50}; //Spatial window radius
    vector<int> sr{70}; //Color window radius
    vector<int> p{0, 1, 2, 3};
    for (auto &sp_i: sp) {
        for (auto &sr_i: sr) {
            cout << "Calculating meanshift..." << endl;
            pyrMeanShiftFiltering(bi_img, mean_img, sp_i, sr_i, 2, termcrit);
            string w_name = to_string(sp_i) + "-" + to_string(sr_i) + " Level 3";
            cout << w_name << endl;
            namedWindow(w_name, WINDOW_NORMAL);
            imshow(w_name, mean_img);
        }

    }

    //cvtColor(mean_img, gray_img, COLOR_HSV2BGR);
    cvtColor(mean_img, gray_img, COLOR_BGR2GRAY);
    find_histogram(gray_img);

    //Divide Img in multiple sections num_grid X num_grid and perform Otu separately in each one
    Mat otu_img;
    gray_img.copyTo(otu_img);
    int width = otu_img.cols;
    int height = otu_img.rows;
    int num_grid = 1;
    int step_SIZE_x = floor(width / num_grid);
    int step_SIZE_y = floor(height / num_grid);
    vector<Rect> mCells;
    for (int y = 0; y < height; y += step_SIZE_y) {
        for (int x = 0; x < width; x += step_SIZE_x) {
            int GRID_SIZE_x = floor(width / num_grid);
            int GRID_SIZE_y = floor(height / num_grid);
            if (x + GRID_SIZE_x >= width) {
                GRID_SIZE_x = width - x - 1;
            }
            if (y + GRID_SIZE_y >= height) {
                GRID_SIZE_y = height - y - 1;
            }
            Rect grid_rect(x, y, GRID_SIZE_x, GRID_SIZE_y);
            //cout << grid_rect << endl;
            mCells.push_back(grid_rect);
            //rectangle(otu_img, grid_rect, Scalar(0, 255, 0), 1);
            //imshow("otu_img", otu_img);
            threshold(otu_img(grid_rect), otu_img(grid_rect), 0, 255, THRESH_BINARY | THRESH_OTSU);
            //imshow(format("grid%d%d",y, x), otu_img(grid_rect));
            //imshow("otu with grid", otu_img);
            //waitKey();
        }
    }

    Mat img_out = get_contours(bi_img);
    imshow("konture", img_out);
    waitKey();
   //invert otu
    otu_img = 255- otu_img;
    imshow("inv otu with grid", otu_img);

    // Morphological Closing
    int morph_size = 1;
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
    Mat morph_img;
    morphologyEx(otu_img, morph_img, MORPH_CLOSE, element, Point(-1, -1), 10);
    imshow("morph", morph_img);

    //Using Otu as mask
    for (int y = 0; y < rgb_img.rows; ++y) {
        for (int x = 0; x < rgb_img.cols; ++x) {
            if(morph_img.at<uchar>(y, x) == 0 ) {
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



