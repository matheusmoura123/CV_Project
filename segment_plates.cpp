#include "main_header.h"

vector<box> segment_plates(const Mat& img, vector<Mat>& dst) {

    Mat gray, Gaus, binaryImg;
    vector<Mat> theCircles;
    vector<box> plates_boxes;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, Gaus,Size(3,3), 1.5, 1.5);

    //Mat img(200, 200, CV_8UC3);


    // DETECTING CIRCLES (PLATES)

    /*these parameters are good for every image except food_image tray 4, 6 and 7
     * HoughCircles(Gaus, circles, HOUGH_GRADIENT, 1,
                 Gaus.cols/3, 200, 50, 160, 500);
     *     cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, Gaus,Size(3,3), 1.5, 1.5);
     */

    vector<Vec3f> circles;
    HoughCircles(Gaus, circles, HOUGH_GRADIENT, 1,
                 Gaus.cols/3, 200, 50, 160, 500);


    vector<Mat> plates;
    vector<array<int,4>> histograms;

    for(size_t i = 0; i < circles.size(); i++)
    {
        Mat mask(img.rows, img.cols,CV_8UC3, Scalar(0,0,0));
        Vec3i c = circles[i];
        //Point center = Point(c[0], c[1]);
        int radius = c[2];


        for (int j = 0; j < mask.rows; ++j)
        {
            for (int k= 0; k < mask.cols; ++k)
            {

                if((pow((int(c[1])-j),2) + pow((int(c[0])-k),2)) <= (pow((int(radius)),2))) {
                    mask.at<Vec3b>(j,k)[0] = int(img.at<Vec3b>(j,k)[0]);
                    mask.at<Vec3b>(j,k)[1] = int(img.at<Vec3b>(j,k)[1]);
                    mask.at<Vec3b>(j,k)[2] = int(img.at<Vec3b>(j,k)[2]);
                }
            }
        }
        array<int,4> histogram = find_histogram(mask);
        plates.push_back(mask);
        histograms.push_back(histogram);

        Mat segmented = mask.clone();

        /*
        for (int l = 0; l < 4; ++l){
            segment_hsv(mask, segmented,histogram[l] , 200, 100);
            //imshow("Segmented", segmented);
            //waitKey();
        }
         */


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

        theCircles.push_back(mask(Range(row0_y, rowf_y), Range(col0_x, colf_x)));
        //imshow("Detected circles", mask);
        //waitKey();
    }

   // Mat histogram = find_histogram(plates[0]);

    //imshow("Detected circles", plates[0]);
    //waitKey();

    dst.assign(theCircles.begin(), theCircles.end());
    return plates_boxes;
};


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


Mat segment_hsv(const Mat& src, int T_hue, int T_sat, int T_value) {
    Mat img_hsv, dst;

    cvtColor(src, img_hsv, COLOR_BGR2HSV);
    dst = img_hsv.clone();


    for (int i = 0; i < dst.rows; ++i)
    {
        for (int j = 0; j < dst.cols; ++j)
        {

            int hue = int(img_hsv.at<Vec3b>(i,j)[0]);
            int saturation = int(img_hsv.at<Vec3b>(i,j)[1]);
            int value = int(img_hsv.at<Vec3b>(i,j)[2]);


            if(abs(hue - T_hue) < 50 && abs(saturation - T_sat) < 60   && abs(value - T_value) < 150) {
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
    cvtColor(dst, dst, COLOR_HSV2BGR);
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

