#include "main_header.h"

vector<box> segment_plates(const Mat& img, vector<Mat>& dst) {

    Mat gray, Gaus, binaryImg;
    vector<Mat> theCircles;
    vector<box> plates_boxes;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, Gaus,Size(3,3), 1.5, 1.5);

    // DETECTING CIRCLES (PLATES)
    /*
    these parameters are good for every image except food_image tray 4, 6 and 7
    HoughCircles(Gaus, circles, HOUGH_GRADIENT, 1, Gaus.cols/3, 200, 50, 160, 500); cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, Gaus,Size(3,3), 1.5, 1.5);
     */

    vector<Vec3f> circles;
    HoughCircles(Gaus, circles, HOUGH_GRADIENT, 1, Gaus.cols/3, 200, 50, 160, 500);
    vector<Mat> plates;

    for(size_t i = 0; i < circles.size(); i++)
    {
        Mat mask(img.rows, img.cols,CV_8UC3, Scalar(0,0,0));
        Vec3i c = circles[i];
        //Point center = Point(c[0], c[1]);
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
    vector<int> area;

    for (int i = 0; i < contours.size(); i++) {

        int area_tmp = contourArea(contours[i]);

        area.push_back(area_tmp);

//        drawContours(img_out, contours, i, Scalar(255, 0, 255), 2);


    }

    sort(area.begin(), area.end(), greater<int>());

    for (int j = 0; j < contours.size(); j++) {

        int area_tmp = contourArea(contours[j]);


        if(area_tmp == area[0]){
            fillPoly(img_out, contours[j], Scalar(255, 255, 255));
        }
        else{
            fillPoly(img_out, contours[j], Scalar(0, 0, 0));
        }

    }


    cout << area.size() << endl;


    return img_out;

}


Mat segment_rgb_hsv(const Mat& src, int hue, int sat, int val, int T_hue, int T_sat, int T_value) {
    Mat img_hsv, dst;

    dst = src.clone();


    for (int i = 0; i < dst.rows; ++i)
    {
        for (int j = 0; j < dst.cols; ++j)
        {

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
    try {
        cvtColor(dst, dst, COLOR_HSV2BGR);
    }
    catch (string error) {
        error = "BGR image";

        cout << error << endl;
    }


    cvtColor(dst, final, COLOR_BGR2GRAY);

    imshow("hsv img", final);
    waitKey(0);


    return final;

}



Mat contour_hsv(const Mat& input) {

    Mat hsv;
    cvtColor(input,hsv,COLOR_BGR2HSV);

    vector<Mat> channels;
    split(hsv, channels);

    Mat H = channels[0];
    Mat S = channels[1];
    Mat V = channels[2];

    Mat shiftedH = H.clone();
    int shift = 25; // in openCV hue values go from 0 to 180 (so have to be doubled to get to 0 .. 360) because of byte range from 0 to 255
    for(int j=0; j<shiftedH.rows; ++j)
        for(int i=0; i<shiftedH.cols; ++i)
        {
            shiftedH.at<unsigned char>(j,i) = (shiftedH.at<unsigned char>(j,i) + shift)%180;
        }

    Mat cannyH, cannyS;

    Canny(shiftedH, cannyH, 100, 50);
    Canny(S, cannyS, 200, 100);


    // extract contours of the canny image:
    vector<std::vector<cv::Point> > contoursH;
    vector<cv::Vec4i> hierarchyH;
    findContours(cannyH,contoursH, hierarchyH, RETR_TREE , CHAIN_APPROX_SIMPLE);

// draw the contours to a copy of the input image:
    Mat outputH = input.clone();
    for( int i = 0; i< contoursH.size(); i++ )
    {
        drawContours( outputH, contoursH, i, cv::Scalar(0,0,255), 2, 8, hierarchyH, 0);
    }

    dilate(cannyH, cannyH, Mat());
    dilate(cannyH, cannyH, Mat());
    dilate(cannyH, cannyH, Mat());

    Mat final = input.clone();
    for( int i = 0; i< contoursH.size(); i++ )
    {
        if(cv::contourArea(contoursH[i]) < 20) continue; // ignore contours that are too small to be a patty
        if(hierarchyH[i][3] < 0) continue;  // ignore "outer" contours

        cv::drawContours( final, contoursH, i, cv::Scalar(0,0,255), 2, 8, hierarchyH, 0);
    }


    return outputH;
}

Mat meanshift(Mat img, int spatial, int color){
    Mat mean_img;
    mean_img = img.clone();
    //MeanShift
    TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 20, 1);
    vector<int> sp{spatial}; //Spatial window radius
    vector<int> sr{color}; //Color window radius
    vector<int> p{0, 1, 2, 3};
    for (auto &sp_i: sp) {
        for (auto &sr_i: sr) {
            cout << "Calculating meanshift..." << endl;
            pyrMeanShiftFiltering(img, mean_img, sp_i, sr_i, 1, termcrit);
            string w_name = to_string(sp_i) + "-" + to_string(sr_i) + " Level 3";
            cout << w_name << endl;
            namedWindow(w_name, WINDOW_NORMAL);
            imshow(w_name, mean_img);
        }
    }

    return mean_img;
}

Mat otsu_segmentation(Mat gray_img, int num_grid){
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
            threshold(otsu_img(grid_rect), otsu_img(grid_rect), 0, 255, THRESH_BINARY | THRESH_OTSU);
            //imshow(format("grid%d%d",y, x), otu_img(grid_rect));
            //imshow("otu with grid", otu_img);
            //waitKey();
        }
    }
    //invert otu
    otsu_img = 255- otsu_img;

    return otsu_img;
}


vector<box> segment_food(const Mat& img, vector<Mat>& dst){
    vector<Mat> dishes;
    vector<box> box_plates = segment_plates(img, dishes);
    vector<box> box_food;

    for (const auto &box: box_plates){
        //Bilateral filter to try to create sharper edges
        Mat bi_img, mean_img;
        bilateralFilter(box.img, bi_img, 5, 150, 50);

        //First separating the darker foods
        mean_img = meanshift(bi_img, 50, 70);
        cvtColor(mean_img, mean_img, COLOR_BGR2HSV);

        Mat hsv_segment = segment_rgb_hsv(mean_img, 10, 150, 100, 30, 150, 150);


        Mat image_3c;
        Mat in[3] = {hsv_segment, hsv_segment, hsv_segment};
        merge(in, 3, image_3c);

        Mat rgb_img = box.img.clone();


        for (int i = 0; i < hsv_segment.rows; ++i)
        {
            for (int j = 0; j < hsv_segment.cols; ++j)
            {
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

        dst.push_back(rgb_img);

        imshow("mask", rgb_img);
        waitKey();

        cvtColor(rgb_img, rgb_img, COLOR_BGR2GRAY);

        int row0_y, rowf_y, col0_x, colf_x;
        //vector<int> rows, columns;
        for (int y = 0; y < rgb_img.rows; ++y) {
            for (int x = 0; x < rgb_img.cols; ++x){
                if(rgb_img.at<uchar>(y, x) != 0){
                    row0_y=y;
                    break;
                }
            }
        }
        for (int x = 0; x < rgb_img.cols; ++x){
            for (int y = 0; y < rgb_img.rows; ++y) {
                if(rgb_img.at<uchar>(y, x) != 0) {
                    col0_x=x;
                    break;
                }
            }
        }
        for (int y = rgb_img.rows; y > 0; --y) {
            for (int x = 0; x < rgb_img.cols; ++x){
                if(rgb_img.at<uchar>(y, x) != 0){
                    rowf_y=y;
                    break;
                }
            }
        }
        for (int x = rgb_img.cols; x > 0; --x){
            for (int y = 0; y < rgb_img.rows; ++y) {
                if(rgb_img.at<uchar>(y, x) != 0) {
                    colf_x=x;
                    break;
                }
            }
        }
        Mat cropped_img = rgb_img(Range(row0_y, rowf_y), Range(col0_x, colf_x));

        box_food.emplace_back(-1, col0_x, row0_y, colf_x-col0_x, rowf_y-row0_y, -1, cropped_img); //ID: -1 indicates that the box is not yet identified
    }
    return box_food;
}





//---------------GRAVE OF DEAD IDEAS--------------------
// hahahahahahahahha

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

