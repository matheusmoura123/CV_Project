#include "main_header.h"

using namespace cv::xfeatures2d;

int sift_matching(const Mat& img1, const Mat& img2){
    Mat out1, out2, desc1, desc2;

    Ptr<SIFT> sift = SIFT::create(0, 5, 0.01, 10, 1.6);
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
    const float ratio_thresh = 0.8f;
    std::vector<DMatch> good_matches;

    for (auto & matche : matches){
        if (matche[0].distance < ratio_thresh * matche[1].distance){
            good_matches.push_back(matche[0]);
        }
    }
/*
    Mat imageMatches;

    drawMatches(img1, keypoints1, img2, keypoints2,
                good_matches, imageMatches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


    namedWindow("SIFT features, Brute-Force Matcher");
    imshow("SIFT features, Brute-Force Matcher",imageMatches);
    waitKey(0);
    */

    return good_matches.size();
}

int surf_matching(const Mat& img1, const Mat& img2){

    Mat out1, out2, desc1, desc2;

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    Ptr<SURF> surf = SURF::create(minHessian);

    vector<KeyPoint> keypoints1, keypoints2;

    surf->detect(img1, keypoints1);
    surf->detect(img2, keypoints2);

    surf->compute(img1, keypoints1, desc1);
    surf->compute(img2, keypoints2, desc2);

    drawKeypoints(img1, keypoints1, out1);
    drawKeypoints(img2, keypoints2, out2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch> > matches;
    matcher->knnMatch(desc1, desc2, matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;

    for (size_t i = 0; i < matches.size(); i++){
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance && matches[i][0].distance < ratio_thresh * matches[i][2].distance){
            good_matches.push_back(matches[i][0]);
        }
    }

    /*
    Mat imageMatches;

    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, imageMatches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    namedWindow("SURF features, FLANN Matcher");
    imshow("SURF features, FLANN Matcher",imageMatches);
    waitKey(0);
     */

    return good_matches.size();
}

void compare_plates(vector<box> food_plate, vector<box>& leftover_plate){
    vector<vector<int>> best_matches;
    vector<vector<int>> all_matches;
    for(int i = 0; i < leftover_plate.size(); ++i){
        for(int j = 0; j < food_plate.size(); ++j){
            vector<int> matches;
            int matches_tmp = sift_matching(food_plate[j].img, leftover_plate[i].img);
            matches.push_back(matches_tmp);
            matches.push_back(i);
            matches.push_back(j);
            all_matches.push_back(matches);
        }
        sort(all_matches.begin(), all_matches.end(), sort_num_match);

    }
    for(int i = 0; i < all_matches.size(); ++i){
        if(leftover_plate[all_matches[i][1]].ID == -1 && food_plate[all_matches[i][2]].ID != -1){
            leftover_plate[all_matches[i][1]].ID = food_plate[all_matches[i][2]].ID;
            food_plate[all_matches[i][2]].ID = -1;
            /*
            imshow("leftover", leftover_plate[all_matches[i][1]].img);
            imshow("food_plate", food_plate[all_matches[i][1]].img);
            waitKey();
             */
        }
    }
}