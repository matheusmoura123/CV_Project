#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/core/types_c.h>

using namespace cv;
using namespace std;

Mat mean_histogram(const vector<Mat>& vector_src) {
    int histSize = 255;

    float range[] = {1, 255};
    const float *histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    // Draw the histograms for B, G and R
    int hist_w = 256;
    int hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    Mat new_img;
    vector<Mat> hsv_planes;

    if (vector_src[0].channels() == 1) {
        Mat mean_hist = Mat::zeros(histSize, 1, CV_8UC1);
        Mat gray_hist;

        int k = 0;
        for (auto &src: vector_src) {
            calcHist(&src, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);
            add(mean_hist, gray_hist, mean_hist, noArray(), 5);
            k++;
        }
        mean_hist = mean_hist/k;
        normalize(mean_hist, mean_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        for (int i = 1; i < histSize; i++) {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(mean_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(mean_hist.at<float>(i))),
                 Scalar(255, 0, 0), 2, 8, 0);
        }
        namedWindow("calcHist Demo");
        imshow("calcHist Demo", histImage);
        waitKey(0);

        return mean_hist;

    } else {
        Mat zero = Mat::zeros(histSize, 1, CV_8UC1);
        Mat mean_hist[3] = {zero, zero, zero};
        Mat b_hist, g_hist, r_hist;

        int k = 0;
        for (auto &src: vector_src) {
            cvtColor(src, new_img, COLOR_BGR2HSV);

            split(new_img, hsv_planes);

            calcHist(&hsv_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
            calcHist(&hsv_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
            calcHist(&hsv_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

            add(mean_hist[0], b_hist, mean_hist[0], noArray(), 5);
            add(mean_hist[1], g_hist, mean_hist[1], noArray(), 5);
            add(mean_hist[2], r_hist, mean_hist[2], noArray(), 5);

            k++;
        }
        mean_hist[0] = mean_hist[0]/k;
        mean_hist[1] = mean_hist[1]/k;
        mean_hist[2] = mean_hist[2]/k;

        normalize(mean_hist[0], mean_hist[0], 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(mean_hist[1], mean_hist[1], 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(mean_hist[2], mean_hist[2], 0, histImage.rows, NORM_MINMAX, -1, Mat());

        for (int i = 1; i < histSize; i++) {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(mean_hist[0].at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(mean_hist[0].at<float>(i))),
                 Scalar(255, 0, 0), 2, 8, 0);

            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(mean_hist[1].at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(mean_hist[1].at<float>(i))),
                 Scalar(0, 255, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(mean_hist[2].at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(mean_hist[2].at<float>(i))),
                 Scalar(0, 0, 255), 2, 8, 0);
        }
        namedWindow("calcHist Demo");
        imshow("calcHist Demo", histImage);
        waitKey(0);

        Mat m;
        merge(mean_hist, 3, m);
        return m;
    }
}

array<int,4> find_histogram(const Mat& src) {
    int histSize = 256;

    float range[] = { 0, 255 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    // Draw the histograms for B, G and R
    int hist_w = 256; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat new_img;
    vector<Mat> hsv_planes;
    Mat gray_hist;
    Mat b_hist, g_hist, r_hist;

    //vector<float> max_values;
    array <int, 4> max_values{0, 0, 0, 0};
    float max = 0.0;
    int position;

    float max1 = 0.0;
    float max2 = 0.0;
    float max3 = 0.0;
    float max4 = 0.0;

    if (src.channels() == 1){

        calcHist( &src, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate );
        normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(gray_hist.at<float>(i-1)) ) ,
                  Point( bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i)) ),
                  Scalar( 255, 0, 0), 2, 8, 0  );

            /*
            if (b_hist.at<float>(i) > max) {
                max = b_hist.at<float>(i);
                position = i;
                max_values[4] = max_values[3];
                max_values[3] = max_values[2];
                max_values[2] = max_values[1];
                max_values[1] = max_values[0];
                max_values[0] = position;
            }
            //max_values.push_back(r_hist.at<float>(i));
     */
        }


        for( int i = 1; i < histSize/4; i++ ){
            if (gray_hist.at<float>(i) > max1) {
                max_values[0] = i;
            }

            if (gray_hist.at<float>(2*i) > max2) {
                max_values[1] = 2*i;
            }

            if (gray_hist.at<float>(3*i) > max3) {
                max_values[2] = 3*i;
            }

            if (gray_hist.at<float>(4*i) > max4) {
                max_values[3] = 4*i;
            }


        }
    }
    else{

        cvtColor(src, new_img, COLOR_BGR2HSV);

        split( new_img, hsv_planes );

        calcHist( &hsv_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &hsv_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &hsv_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );


        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );



        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                  Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                  Scalar( 255, 0, 0), 2, 8, 0  );

            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                  Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                  Scalar( 0, 255, 0), 2, 8, 0  );
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                  Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                  Scalar( 0, 0, 255), 2, 8, 0  );


            /*
            if (b_hist.at<float>(i) > max) {
                max = b_hist.at<float>(i);
                position = i;
                max_values[4] = max_values[3];
                max_values[3] = max_values[2];
                max_values[2] = max_values[1];
                max_values[1] = max_values[0];
                max_values[0] = position;
            }
            //max_values.push_back(r_hist.at<float>(i));
     */
        }

        for( int i = 1; i < histSize/4; i++ ){
            if (b_hist.at<float>(i) > max1) {
                max_values[0] = i;
            }

            if (b_hist.at<float>(2*i) > max2) {
                max_values[1] = 2*i;
            }

            if (b_hist.at<float>(3*i) > max3) {
                max_values[2] = 3*i;
            }

            if (b_hist.at<float>(4*i) > max4) {
                max_values[3] = 4*i;
            }


        }
    }


    cout << "---------------------------" << endl;
    cout << max_values[0] << endl <<  max_values[1] << endl <<  max_values[2] << endl <<  max_values[3] << endl;


    // sort(max_values.begin(), max_values.end(), greater<>());
   // cout << max_values[0] << endl;

    namedWindow("calcHist Demo");
    imshow("calcHist Demo", histImage);
    waitKey(0);

    return max_values;

}
