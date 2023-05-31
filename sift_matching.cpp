#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include "sift_matching.h"

using namespace std;
using namespace cv;

int sift_matching(const Mat& img1, const Mat& img2){

    Mat out1, out2, desc1, desc2;

    Ptr<SIFT> sift = SIFT::create(0, 3, 0.04, 10, 1.6);
    vector<cv::KeyPoint> keypoints1, keypoints2;

    sift->detect(img1, keypoints1);
    sift->detect(img2, keypoints2);

    sift->compute(img1, keypoints1, desc1);
    sift->compute(img2, keypoints2, desc2);

    drawKeypoints(img1, keypoints1, out1);
    drawKeypoints(img2, keypoints2, out2);

    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(desc1, desc2, matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;

    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }

    /*
    cout << keypoints1.size()<< endl << keypoints2.size() << endl;
    cout << good_matches.size() << endl;

    float kp = (keypoints1.size() + keypoints2.size())/2;
    float percentage = (100 * good_matches.size())/kp;

    cout << percentage << endl;
    */

    Mat imageMatches;

    drawMatches(img1, keypoints1, img2, keypoints2,
                good_matches, imageMatches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


    namedWindow("SIFT features, Brute-Force Matcher");
    imshow("SIFT features, Brute-Force Matcher",imageMatches);
    waitKey(0);

    return good_matches.size();
}
