#include "main_header.h"

vector<Mat> categories_histogram (const vector<int>& categories) {
    vector <Mat> categories_hist;
    for (const int& j: categories) {
        vector <Mat> imgs;
        //cout << foodCategories[j].className << endl;
        for (int i = 1; i <= foodCategories[j].imageNumbers; ++i) {
            Mat img = imread(CATEGORIES_PATH + foodCategories[j].className + "/" + foodCategories[j].className + to_string(i) + IMAGE_EXT);
            //find_histogram(img);
            imgs.push_back(img);
        }
        categories_hist.push_back(mean_histogram2(imgs));
    }
    return categories_hist;
}

array<double, 3> compare_histogram(const Mat& src_hist, const vector<Mat>& categories_hist) {

    double comp_values = 0;
    double confidence = 0;
    double max_value = 0, sec_max = 0,  max_index = 0;
    int k = 0;
    for(auto &type_hist: categories_hist) {
        comp_values = compareHist(src_hist, type_hist, 0);
        //cout << comp_values << endl;
        if (comp_values >= max_value) {
            sec_max = max_value;
            max_value = comp_values;
            max_index = k;
        } else if (comp_values < 0) max_value = 0.00001;
        k++;
    }
    //cout << "---------------------" << endl;
    confidence = max_value/(max_value+sec_max);
    return {max_value, max_index, confidence};
}

Mat mean_histogram2(const vector<Mat>& vector_src) {

    int h_bins = 180, s_bins = 256, v_bins = 246;
    int histSize[] = { h_bins, s_bins, v_bins};
    // hue varies from 0 to 179, saturation and value from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    float v_ranges[] = { 10, 256 };
    const float* ranges[] = { h_ranges, s_ranges, v_ranges};
    // Use all channels
    int channels[] = { 0, 1, 2};

    bool uniform = true;
    bool accumulate = false;

    Mat mean_hist = Mat::zeros(3, histSize, CV_8UC1);
    Mat img_hist;

    int k = 0;
    for (auto &src: vector_src) {
        cvtColor(src, src, COLOR_BGR2HSV);

        calcHist(&src, 1, channels, Mat(), img_hist, 3, histSize, ranges, uniform, accumulate);
        normalize(img_hist, img_hist, 0, 1, NORM_MINMAX, -1, Mat());
        //cout << "img_hist" << img_hist.size << " " << img_hist.channels() << " " << img_hist.type() << endl;
        add(mean_hist, img_hist, mean_hist, noArray(), 5);
        k++;
    }
    mean_hist = mean_hist/k;
    return mean_hist;
}

array<int,3> find_histogram(const Mat& src) {
    int histSize = 256;

    float range[] = { 0, 255 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    // Draw the histograms for B, G and R
    int hist_w = 256; int hist_h = 100;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat new_img;
    vector<Mat> hsv_planes;
    Mat gray_hist;
    Mat b_hist, g_hist, r_hist;

    //vector<float> max_values;
    array <int, 3> max_values{0, 0, 0};
    int position;
    int neighborhood = 5;

    if (src.channels() == 1){
        float max = 0.0;

        calcHist( &src, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate );
        normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );


        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(gray_hist.at<float>(i-1)) ) ,
                  Point( bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i)) ),
                  Scalar( 255, 0, 0), 2, 8, 0  );

            if (gray_hist.at<float>(i) > max) {
                max = gray_hist.at<float>(i);
                position = i;
                max_values[0] = position;
            }
            //max_values.push_back(r_hist.at<float>(i));

        }
    }

    else{

        cvtColor(src, new_img, COLOR_BGR2HSV);

        split(new_img, hsv_planes);

        calcHist( &hsv_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &hsv_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &hsv_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );


        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );


        for(int j = 0; j < 3; j++){
            float max = 0.0;

            for(int i = 1; i < 180; i++){
                if (b_hist.at<float>(i) > max) {
                    max = b_hist.at<float>(i);
                    position = i;
                    max_values[j] = position;
                }
                //cout << b_hist.at<float>(i) << endl;
            }

            //cout << "-----------------------------------------" << endl;

            for(int i = position - neighborhood; i < position + neighborhood; i++){
                b_hist.at<float>(i) = 0;
            }
        }
    }

    /*
    cout << "---------------------------" << endl;
    cout << max_values[0] << endl <<  max_values[1] << endl <<  max_values[2] << endl <<  max_values[3] << endl;


    // sort(max_values.begin(), max_values.end(), greater<>());
   // cout << max_values[0] << endl;

    namedWindow("calcHist Demo");
    imshow("calcHist Demo", histImage);
    waitKey(0);
     */

    return max_values;
}

bool sort_bigger_area (const box& i,const box& j) {
    return(i.width*i.height>j.width*j.height);
}

bool sort_ID (const box& i,const box& j) {
    return(i.ID < j.ID);
}

bool sort_ID_bigger (const box& i,const box& j) {
    return(i.ID > j.ID);
}

bool sort_num_match (const vector<int>& i,const vector<int>& j) {
    return(i[0] > j[0]);
}


//-------------------------------------------------------------------------


/*
void plot_histogram(Mat histogram){

    int h_bins = 180, s_bins = 256, v_bins = 256, gray_bins = 256;
    int histSize[] = { h_bins, s_bins, v_bins };
    // hue varies from 0 to 179, saturation and value from 0 to 255

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w/gray_bins);
    int bin_rgb = cvRound((double) hist_w/h_bins);


    Mat grayscale_hist(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));
    Mat rgb_hist(hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );


    if (histogram.channels() == 1){

        for( int i = 1; i < gray_bins; i++ ) {
            line( grayscale_hist, Point( bin_w*(i-1), hist_h - cvRound(histogram.at<float>(i-1)) ),
                  Point( bin_w*(i), hist_h - cvRound(histogram.at<float>(i)) ),
                  Scalar( 255, 255, 255), 1, 8, 0  );
        }

        namedWindow("Grayscale image histogram");
        imshow("Grayscale image histogram", grayscale_hist);
        waitKey();
        }

    else{
        for (int i = 1; i < h_bins; i++) {

            line( rgb_hist, Point( bin_rgb*(i-1), hist_h - cvRound(histogram.at<float>(i-1, 0, 0)) ),
                  Point( bin_rgb*(i), hist_h - cvRound(histogram.at<float>(i-1, 0, 0)) ),
                  Scalar( 255, 0, 0), 1, 8, 0  );
            line( rgb_hist, Point( bin_rgb*(i-1), hist_h - cvRound(histogram.at<float>(0, i-1, 0))),
                  Point( bin_rgb*(i), hist_h - cvRound(histogram.at<float>(0, i-1, 0)) ),
                  Scalar( 0, 255, 0), 1, 8, 0  );

            line( rgb_hist, Point( bin_rgb*(i-1), hist_h - cvRound(histogram.at<float>(0, 0, i-1))),
                  Point( bin_rgb*(i), hist_h - cvRound(histogram.at<float>(0, 0, i-1))),
                  Scalar( 0, 0, 255), 1, 8, 0  );



        }
        namedWindow("Image histogram");
        imshow("Image histogram", rgb_hist);
        waitKey();
    }
}
*/