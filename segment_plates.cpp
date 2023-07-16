#include "main_header.h"

vector<box> segment_plates(const Mat& img, vector<Mat>& dst) {

    Mat gray, Gaus, binaryImg;
    vector<Mat> theCircles;
    vector<box> plates_boxes;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, Gaus,Size(3,3), 1.5, 1.5); //original

    // DETECTING CIRCLES (PLATES)
    vector<Vec3f> circles;
    HoughCircles(Gaus, circles, HOUGH_GRADIENT, 1, Gaus.cols/3, 100, 80, 160, 500); //get all the plates right and nothing else. PERFECT!
    vector<Mat> plates;

    for(size_t i = 0; i < circles.size(); i++)
    {
        Mat mask(img.rows, img.cols,CV_8UC3, Scalar(0,0,0));
        Vec3i c = circles[i];
        int radius = c[2];

        for (int j = 0; j < mask.rows; ++j) {
            for (int k= 0; k < mask.cols; ++k) {
                if((pow((int(c[1])-j),2) + pow((int(c[0])-k),2)) <= (pow((int(radius)),2))) {
                    mask.at<Vec3b>(j,k)[0] = int(img.at<Vec3b>(j,k)[0]);
                    mask.at<Vec3b>(j,k)[1] = int(img.at<Vec3b>(j,k)[1]);
                    mask.at<Vec3b>(j,k)[2] = int(img.at<Vec3b>(j,k)[2]);
                }
            }
        }
        plates.push_back(mask);

        Mat segmented = mask.clone();

        //CROPPING PLATES
        int row0_y, rowf_y, col0_x, colf_x;
        if (int(c[1])-int(radius) <= 0) row0_y = 0;
        else row0_y = int(c[1])-int(radius);
        if (int(c[1])+int(radius) >= mask.rows) rowf_y = mask.rows;
        else rowf_y = int(c[1])+int(radius);
        if (int(c[0])-int(radius) <= 0) col0_x = 0;
        else col0_x = int(c[0])-int(radius);
        if (int(c[0])+int(radius) >= mask.cols) colf_x = mask.cols;
        else colf_x = int(c[0])+int(radius);

        Mat cropped_img = mask(Range(row0_y, rowf_y), Range(col0_x, colf_x));
        theCircles.push_back(cropped_img);
        plates_boxes.emplace_back(-1, col0_x, row0_y, colf_x-col0_x, rowf_y-row0_y, -1, cropped_img); //ID: -1 indicates that the box is not yet identified
    }

    dst.assign(theCircles.begin(), theCircles.end());
    return plates_boxes;
}

Mat get_contours(const Mat& img) {
    Mat img_out, gray;
    img_out = img.clone();
    cvtColor(img, gray, COLOR_BGR2GRAY);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(img_out, contours, -1, Scalar(0, 0, 255), 1);

    vector<vector<Point>> conPoly(contours.size());
    vector<double> area;

    for (int i = 0; i < contours.size(); i++) {
        double area_tmp = contourArea(contours[i]);
        area.push_back(area_tmp);
        //drawContours(img_out, contours, i, Scalar(255, 0, 255), 2);
    }

    sort(area.begin(), area.end(), greater<>());

    for (int j = 0; j < contours.size(); j++) {
        double area_tmp = contourArea(contours[j]);
        if(area_tmp == area[0]){
            fillPoly(img_out, contours[j], Scalar(255, 255, 255));
        }
        else{
            fillPoly(img_out, contours[j], Scalar(0, 0, 0));
        }
    }
    return img_out;
}

Mat segment_rgb_hsv(const Mat& src, int hue, int sat, int val, int T_hue, int T_sat, int T_value, bool is_hsv) {
    Mat img_hsv, dst;
    dst = src.clone();
    int neighborhood = 0;

    // subtract one from the rows and cols
    for (int i = neighborhood; i < dst.rows-neighborhood; ++i) {
        for (int j = neighborhood; j < dst.cols-neighborhood; ++j) {

            int huee = int(src.at<Vec3b>(i,j)[0]);
            int saturation = int(src.at<Vec3b>(i,j)[1]);
            int value = int(src.at<Vec3b>(i,j)[2]);

            if(abs(huee - hue) < T_hue && abs(saturation - sat) < T_sat   && abs(value - val) < T_value) {
                dst.at<Vec3b>(i,j)[0] = 255;
                dst.at<Vec3b>(i,j)[1] = 255;
                dst.at<Vec3b>(i,j)[2] = 255;
            }
            else {
                dst.at<Vec3b>(i,j)[0] = 0;
                dst.at<Vec3b>(i,j)[1] = 0;
                dst.at<Vec3b>(i,j)[2] = 0;
            }
        }
    }

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    erode(dst, dst, kernel);
    dilate(dst, dst, kernel);

    Mat final;
    if (is_hsv) cvtColor(dst, dst, COLOR_HSV2BGR);

    cvtColor(dst, final, COLOR_BGR2GRAY);
    return final;
}

Mat meanshift(const Mat& img, int spatial, int color){
    Mat mean_img;
    mean_img = img.clone();
    //MeanShift
    TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 1);
    vector<int> sp{spatial}; //Spatial window radius
    vector<int> sr{color}; //Color window radius
    vector<int> p{0, 1, 2, 3};
    for (auto &sp_i: sp) {
        for (auto &sr_i: sr) {
            cout << "Calculating meanshift..." << endl;
            pyrMeanShiftFiltering(img, mean_img, sp_i, sr_i, 1, termcrit);
            string w_name = to_string(sp_i) + "-" + to_string(sr_i) + " Level 3";
            //cout << w_name << endl;
            //namedWindow(w_name, WINDOW_NORMAL);
            //imshow(w_name, mean_img);
        }
    }
    return mean_img;
}

Mat otsu_segmentation(Mat gray_img, int num_grid) {
    Mat otsu_img = gray_img.clone();

    //Divide Img in multiple sections num_grid X num_grid and perform Otu separately in each one
    gray_img.copyTo(otsu_img);
    int width = otsu_img.cols;
    int height = otsu_img.rows;
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
            threshold(otsu_img(grid_rect), otsu_img(grid_rect), 1, 255, THRESH_BINARY | THRESH_OTSU);
            //imshow(format("grid%d%d",y, x), otu_img(grid_rect));
            //imshow("otu with grid", otu_img);
            //waitKey();
        }
    }
    //invert otu
    otsu_img = 255- otsu_img;

    return otsu_img;

}

box crop_image(const Mat& img, const box& plate_box){
    //Crop only food
    Mat final_image = img.clone();
    int row0_y=0, rowf_y=0, col0_x=0, colf_x=0;
    Mat gray_result;
    cvtColor(img, gray_result, COLOR_BGR2GRAY);
    //vector<int> rows, columns;
    for (int y = 0; y < gray_result.rows; ++y) {
        bool done = false;
        for (int x = 0; x < gray_result.cols; ++x){
            if(gray_result.at<uchar>(y, x) != 0) {
                row0_y = y;
                done = true;
                break;}
        }
        if(done) break;
    }
    for (int x = 0; x < gray_result.cols; ++x){
        bool done = false;
        for (int y = 0; y < gray_result.rows; ++y) {
            if(gray_result.at<uchar>(y, x) != 0) {
                col0_x = x;
                done = true;
                break;}
        }
        if(done) break;
    }
    for (int y = gray_result.rows-1; y > 0; --y) {
        int done = false;
        for (int x = 0; x < gray_result.cols; ++x){
            if(gray_result.at<uchar>(y, x) != 0) {
                rowf_y = y;
                done = true;
                break;}
        }
        if(done) break;
    }
    for (int x = gray_result.cols-1; x > 0; --x){
        int done = false;
        for (int y = 0; y < gray_result.rows; ++y) {
            if(gray_result.at<uchar>(y, x) != 0) {
                colf_x = x;
                done = true;
                break;}
        }
        if(done) break;
    }

    Mat g = Mat::zeros(Size(0,0), CV_8UC1);
    box food_box = {-1, 0, 0, 0, 0, 0, g};

    Mat cropped_img = final_image(Range(row0_y, rowf_y), Range(col0_x, colf_x));
    food_box = {-1, plate_box.p0x+col0_x, plate_box.p0y+row0_y, colf_x-col0_x, rowf_y-row0_y, -1, g}; //ID: -1 indicates that the box is not yet identified
    food_box.img = cropped_img.clone();

    return food_box;
}

box segment_food(const box& plate_box) {
    Mat g = Mat::zeros(Size(0,0), CV_8UC1);
    box food_box = {-1, 0, 0, 0, 0, 0, g};

    Mat rgb_img = plate_box.img.clone();
    Mat final_image = rgb_img.clone();

    Mat bi_img_new, gray_new;

    //Let's segment the mask that we've got
    cvtColor(rgb_img, gray_new, COLOR_BGR2GRAY);
    Canny(gray_new, bi_img_new, 20, 60);

    int morph_size_new = 2;
    Mat element_new = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size_new + 1, 2 * morph_size_new + 1), Point(morph_size_new, morph_size_new));
    Mat morph_img_new;
    morphologyEx(bi_img_new, morph_img_new, MORPH_CLOSE, element_new, Point(-1, -1), 3);
    morphologyEx(morph_img_new, morph_img_new, MORPH_OPEN, element_new, Point(-1, -1), 2);

    Mat morph_3c;
    Mat in_morph[3] = {morph_img_new, morph_img_new, morph_img_new};
    merge(in_morph, 3, morph_3c);

    Mat img_contour = get_contours(morph_3c);

    cvtColor(img_contour, img_contour, COLOR_BGR2GRAY);

    for (int y = 0; y < final_image.rows; ++y) {
        for (int x = 0; x < final_image.cols; ++x) {
            if(img_contour.at<uchar>(y, x) == 0 ) {
                final_image.at<Vec3b>(y,x)[0] = 0;
                final_image.at<Vec3b>(y,x)[1] = 0;
                final_image.at<Vec3b>(y,x)[2] = 0;
            }
        }
    }
    food_box = crop_image(final_image, plate_box);

    return food_box;
}

Mat K_means(const Mat& src, int num_of_clusters) {

    Mat samples(src.rows * src.cols, 5, CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            samples.at<float>(y + x * src.rows, 0) = (float)x / (float)src.cols;
            samples.at<float>(y + x * src.rows, 1) = (float)y / (float)src.rows;
            for (int z = 0; z < src.channels(); z++) {
                samples.at<float>(y + x * src.rows, z+2) = (float)src.at<Vec3b>(y, x)[z];
            }
        }
    }
    Mat labels;
    int attempts = 7;
    Mat centers;
    int K = num_of_clusters;
    TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, 100, 0.01);
    kmeans(samples, K, labels, criteria, attempts, KMEANS_PP_CENTERS, centers);

    //cout << labels.size << endl;
    //cout << src.rows*src.cols << endl;

    int colors[K];
    for(int i=0; i<K; i++) {
        colors[i] = 252/(i+1);
    }

    Mat clustered = Mat(src.rows, src.cols, CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * src.rows, 0);
            //cout << (float)(colors[cluster_idx]) << endl;
            clustered.at<float>(y, x) = (float)(colors[cluster_idx]);
        }
    }
    Mat output;
    clustered.convertTo(output, CV_8UC1);
    return output;
}

vector<box> separate_food(const box& food_box) {
    Mat gray_img, dst;
    vector<box> dishes;
    cvtColor(food_box.img, gray_img, COLOR_BGR2GRAY);
    // cvtColor(food_box, gray_img, COLOR_BGR2GRAY);
    Mat mean_img = meanshift(food_box.img ,70, 110);
    Mat kmeans_img = K_means(mean_img, 3);
    //cvtColor(kmeans_img, kmeans_img, COLOR_BGR2GRAY);

    int num_of_foods = 2;
    int background_int = int(kmeans_img.at<uchar>(0,0));
    vector<int> intensities = {252, 252/2, 252/3};
    intensities.erase(std::remove(intensities.begin(), intensities.end(), background_int), intensities.end());
    for(int k = 0; k < num_of_foods; ++k){
        Mat final = kmeans_img.clone();
        for (int y = 0; y < kmeans_img.rows; ++y) {
            for (int x = 0; x < kmeans_img.cols; ++x) {
                int intensity = int(kmeans_img.at<uchar>(y,x));
                if(intensity == background_int) {
                    final.at<uchar>(y,x) = 0;
                }
                else if(intensity == intensities[k]) {
                    final.at<uchar>(y,x) = 255;
                }
                else if(intensity == intensities[1-k]){
                    final.at<uchar>(y,x) = 0;
                }
            }
        }

        for (int y = 0; y < kmeans_img.rows; ++y) {
            for (int x = 0; x < kmeans_img.cols; ++x) {
                if(gray_img.at<uchar>(y, x) == 0 ) {
                    final.at<uchar>(y,x) = 0;
                }
            }
        }

        int morph_size_new = 2;
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size_new + 1, 2 * morph_size_new + 1), Point(morph_size_new, morph_size_new));
        Mat morph_gray;
        morphologyEx(final, morph_gray, MORPH_OPEN, element, Point(-1, -1), 1);

        Mat morph;
        Mat in[3] = {morph_gray, morph_gray, morph_gray};
        merge(in, 3, morph);

        Mat contour = get_contours(morph);

        cvtColor(contour, contour, COLOR_BGR2GRAY);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        findContours(contour, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double area_tmp = contourArea(contours[0]);

        if(area_tmp < 10000) break;

        Mat food_tmp = food_box.img.clone();

        for (int y = 0; y < contour.rows; ++y) {
            for (int x = 0; x < contour.cols; ++x) {
                if(contour.at<uchar>(y, x) == 0 ) {
                    food_tmp.at<Vec3b>(y,x)[0] = 0;
                    food_tmp.at<Vec3b>(y,x)[1] = 0;
                    food_tmp.at<Vec3b>(y,x)[2] = 0;
                }
            }
        }

        box tmp_box = crop_image(food_tmp, food_box);

        dishes.push_back(tmp_box);
    }

    return dishes;
}


//---------------GRAVEYARD OF DEAD IDEAS--------------------
// hahahahahahahahha

/*  FOR SEGMENTING INTRA FOOD CLASSES, CANNY
GaussianBlur(gray_img, gray_img, Size(3,3), 1.5, 1.5);
Canny(gray_img, dst, 70, 100);


int morph_size_new = 2;
Mat element_new = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size_new + 1, 2 * morph_size_new + 1), Point(morph_size_new, morph_size_new));
Mat morph_img_new;
morphologyEx(dst, morph_img_new, MORPH_CLOSE, element_new, Point(-1, -1), 2);
morphologyEx(morph_img_new, morph_img_new, MORPH_OPEN, element_new, Point(-1, -1), 3);


Mat morph_3c;
Mat in_morph[3] = {morph_img_new, morph_img_new, morph_img_new};
merge(in_morph, 3, morph_3c);

Mat img_contour = get_contours(morph_3c);

cvtColor(img_contour, img_contour, COLOR_BGR2GRAY);

imshow("morph", img_contour);

 */



//find_histogram(mask);

//GaussianBlur(mask, mask,Size(5,5), 5, 5);

/*
Mat mask2 = mask.clone();

bilateralFilter(mask, mask2, 7, 500, 1);

*/

//cvtColor(mask, mask, COLOR_BGR2GRAY);


//GaussianBlur(mask, mask,Size(5,5), 5, 5);

/*

normalize(mask, mask, 0, 255, NORM_MINMAX,-1, noArray());


threshold(mask, binaryImg, 0, 255, THRESH_OTSU);
 */

/*

    cvtColor(mask, mask, COLOR_BGR2GRAY);
    */

/*
    Ptr<MSER> mserEXTR = MSER::create(2,50,10000,1.5);

    vector<Rect> FoodCategories;
    vector<vector<cv::Point>> mserContours;
    mserEXTR ->detectRegions(mask,mserContours,FoodCategories);

    for (vector<cv::Point> v : mserContours){
        for (cv::Point p : v){
            mask.at<Vec3b>(p.y, p.x)[0] = 0;
            mask.at<Vec3b>(p.y, p.x)[1] = 0;
            mask.at<Vec3b>(p.y, p.x)[2] = 0;
        }
    }

    GaussianBlur(mask, mask,Size(5,5), 2, 2);
    normalize(mask, mask, 0, 1023, NORM_MINMAX,-1, noArray());

    */

/*
    Mat mask2 = segment_hsv(mask);

    erode(mask2, mask2, Mat());

    dilate(mask2, mask2, Mat());
    dilate(mask2, mask2, Mat());
    dilate(mask2, mask2, Mat());
    */

// Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
//   dilate(mask, mask, kernel);




// Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
//   dilate(mask, mask, kernel);


//Canny(mask, mask, 100, 150);


/*
GaussianBlur(mask, mask,Size(5,5), 5, 5);
normalize(mask, mask, 0, 512, NORM_MINMAX,-1, noArray());
Mat mask2 = segment_hsv(mask);

 */



/*  MATHEUS SEGMENTATION
//Bilateral filter to try to create sharper edges
Mat bi_img, mean_img;
bilateralFilter(plate_box.img, bi_img, 5, 150, 50);

//First separating the darker foods
mean_img = meanshift(bi_img, 50, 70);
cvtColor(mean_img, mean_img, COLOR_BGR2HSV);

Mat hsv_segment = segment_rgb_hsv(mean_img, 10, 150, 100, 30, 150, 150, true);

Mat image_3c;
Mat in[3] = {hsv_segment, hsv_segment, hsv_segment};
merge(in, 3, image_3c);
Mat rgb_img = plate_box.img.clone();
Mat final_image =  box.img.clone();
for (int i = 0; i < hsv_segment.rows; ++i) {
    for (int j = 0; j < hsv_segment.cols; ++j) {
        int blue_temp = int(rgb_img.at<Vec3b>(i,j)[0]);
        int green_temp = int(rgb_img.at<Vec3b>(i,j)[1]);
        int red_temp = int(rgb_img.at<Vec3b>(i,j)[2]);
        if(blue_temp == 0 && green_temp == 0 && red_temp == 0) {
            image_3c.at<Vec3b>(i,j)[0] = 0;
            image_3c.at<Vec3b>(i,j)[1] = 0;
            image_3c.at<Vec3b>(i,j)[2] = 0;
        }
    }
}
Mat img_out = get_contours(image_3c);
cvtColor(img_out, img_out, COLOR_BGR2GRAY);


// Morphological Closing
int morph_size = 2;
Mat element = getStructuringElement(MORPH_CROSS, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
Mat morph_img;
morphologyEx(img_out, morph_img, MORPH_CLOSE, element, Point(-1, -1), 6);

Mat morph_rgb;
Mat chan[3] = {morph_img, morph_img, morph_img};
merge(chan, 3, morph_rgb);

Mat final = get_contours(morph_rgb);
cvtColor(final, final, COLOR_BGR2GRAY);
for (int y = 0; y < rgb_img.rows; ++y) {
    for (int x = 0; x < rgb_img.cols; ++x) {
        if(final.at<uchar>(y, x) == 0 ) {
            rgb_img.at<Vec3b>(y,x)[0] = 0;
            rgb_img.at<Vec3b>(y,x)[1] = 0;
            rgb_img.at<Vec3b>(y,x)[2] = 0;
        }
    }
}
//dishes.push_back(rgb_img);
//imshow("mask", rgb_img);
//waitKey();
 */

/*
   Mat new_image(src.size(), src.type());
   for (int y = 0; y < src.rows; y++) {
       for (int x = 0; x < src.cols; x++) {
           int cluster_idx = labels.at<int>(y + x * src.rows, 0);
           for (int i = 0; i < src.channels(); i++) {
               new_image.at<Vec3b>(y, x)[i] = (uchar)centers.at<float>(cluster_idx, i);
           }
       }
   }
   //imshow("clustered image", new_image);
   return new_image;
    */

/*
Mat p = Mat::zeros(src.cols*src.rows, 5, CV_32F);
Mat bestLabels, centers, clustered;
vector<Mat> bgr;
cv::split(src, bgr);

for(int i=0; i<src.cols*src.rows; i++) {
    p.at<float>(i,0) = (i/src.cols) / src.rows;
    p.at<float>(i,1) = (i%src.cols) / src.cols;
    p.at<float>(i,2) = bgr[0].data[i] / 255.0;
    p.at<float>(i,3) = bgr[1].data[i] / 255.0;
    p.at<float>(i,4) = bgr[2].data[i] / 255.0;
}

int K = num_of_clusters;
TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, 100, 0.01);
cv::kmeans(p, K, bestLabels, criteria, 7, KMEANS_PP_CENTERS, centers);

int colors[K];
for(int i=0; i<K; i++) {
    colors[i] = 255/(i+1);
}

clustered = Mat(src.rows, src.cols, CV_32F);
for(int i=0; i<src.cols*src.rows; i++) {
    clustered.at<float>(i/src.cols, i%src.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
}

Mat output;
output.convertTo(clustered, CV_8UC1);
return output;
 */

