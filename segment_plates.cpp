#include "segment_plates.h"
#include "sift_matching.h"
#include "find_histogram.h"


vector<Mat> segment_plates(const Mat& img) {

    Mat gray, Gaus, binaryImg;
    vector<Mat> theCircles;
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
    vector<Mat> histograms;

    for(size_t i = 0; i < circles.size(); i++)
    {
        Mat mask(img.rows, img.cols,CV_8UC3, Scalar(0,0,0));
        Vec3i c = circles[i];
        //Point center = Point(c[0], c[1]);
        int radius = c[2];


        for (int j = 0; j < mask.rows-1; ++j)
        {
            for (int k= 0; k < mask.cols - 1; ++k)
            {

                if((pow((int(c[1])-j),2) + pow((int(c[0])-k),2)) <= (pow((int(radius)),2))) {
                    mask.at<Vec3b>(j,k)[0] = int(img.at<Vec3b>(j,k)[0]);
                    mask.at<Vec3b>(j,k)[1] = int(img.at<Vec3b>(j,k)[1]);
                    mask.at<Vec3b>(j,k)[2] = int(img.at<Vec3b>(j,k)[2]);
                }
            }
        }
        find_histogram(mask);

        Mat segmented = mask.clone();

        segment_hsv(mask, segmented,10, 200, 100);

        plates.push_back(mask);
        //histograms.push_back(histogram);

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


    return theCircles;

};


Mat get_contours(const Mat& img) {
    Mat img_out, gray;
    img_out = img.clone();
    cvtColor(img, gray, COLOR_BGR2GRAY);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(img, contours, 0, Scalar(0, 0, 255), 1);

    /*
    vector<vector<Point>> conPoly(contours.size());

    float peri = arcLength(contours[0], false);
    approxPolyDP(contours[0], conPoly[0], 0.02*peri, false);
    drawContours(img_out, conPoly, 0, Scalar(0, 0, 255), 1);
    //line(img, conPoly[0][1], conPoly[0][3],Scalar(0,0,255), 1, LINE_AA);
     */


    return img_out;

}


int segment_hsv(const Mat& src, Mat &dst, int T_hue, int T_sat, int T_value) {
    Mat img_hsv;

    cvtColor(src, img_hsv, COLOR_BGR2HSV);


    for (int i = 0; i < dst.rows; ++i)
    {
        for (int j = 0; j < dst.cols; ++j)
        {

            int hue = int(img_hsv.at<Vec3b>(i,j)[0]);
            int saturation = int(img_hsv.at<Vec3b>(i,j)[1]);
            int value = int(img_hsv.at<Vec3b>(i,j)[2]);


            if(abs(hue - T_hue) < 30 && abs(saturation - T_sat) < 60   && abs(value - T_value) < 150) {
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

    //imshow("hsv img", dst);
    //waitKey(0);


    return 0;

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

