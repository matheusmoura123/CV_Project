#include "segment_plates.h"
#include "sift_matching.h"


Mat segment_plates(const Mat& img) {

    Mat gray, Gaus, theCircles, binaryImg;

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



    Mat mask(img.rows, img.cols,CV_8UC3, Scalar(0,0,0));

    for(size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
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

    }


    
    //cvtColor(mask, mask, COLOR_BGR2GRAY);


    //GaussianBlur(mask, mask,Size(5,5), 5, 5);

    /*

    normalize(mask, mask, 0, 255, NORM_MINMAX,-1, noArray());


    threshold(mask, binaryImg, 0, 255, THRESH_OTSU);
     */





/*
    Mat imgcopy = img.clone();
    Ptr<MSER> mserEXTR = MSER::create(5,10,14400,1.5, 0.5, 200, 1.01, 0.003, 3);

    vector<Rect> food;
    vector<vector<cv::Point>> mserContours;
    mserEXTR ->detectRegions(mask,mserContours,food);

    for (vector<cv::Point> v : mserContours){
        for (cv::Point p : v){
            mask.at<uchar>(p.y, p.x) = 255;
           // mask.at<Vec3b>(p.y, p.x)[1] = 255;
           // mask.at<Vec3b>(p.y, p.x)[2] = 255;
        }
    }

   // Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
 //   dilate(mask, mask, kernel);
*/

/*
    Mat imgcopy = img.clone();
    Ptr<MSER> mserEXTR = MSER::create(5,10,14400,1.5, 0.5, 200, 1.01, 0.003, 3);

    vector<Rect> food;
    vector<vector<cv::Point>> mserContours;
    mserEXTR ->detectRegions(mask,mserContours,food);

    for (vector<cv::Point> v : mserContours){
        for (cv::Point p : v){
            mask.at<uchar>(p.y, p.x) = 255;
           // mask.at<Vec3b>(p.y, p.x)[1] = 255;
           // mask.at<Vec3b>(p.y, p.x)[2] = 255;
        }
    }
    */

   // Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
 //   dilate(mask, mask, kernel);


    //Canny(mask, mask, 100, 150);


    imshow("Detected circles", mask);
    waitKey();


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

