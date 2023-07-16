#include "main_header.h"

using namespace cv::ml;

//int number_classes = 0;
//int dict_size = 0;

//Mat k_centers, k_labels;
//Mat input_data;
//Mat input_data_labels;
//Mat all_desc;
//vector<Mat> all_desc_per_img;
//vector<int> all_class_per_img;
//int all_desc_per_img_num = 0;

void write_kmeans(vector<food> categories) {
    Mat all_desc;
    vector<Mat> all_desc_per_img;
    vector<int> all_class_per_img;
    int all_desc_per_img_num = 0;
    Mat k_centers, k_labels;
    int num_class = int(categories.size());
    int d_size = 40 * num_class;

    for (int i = 0; i < num_class; ++i) {
        for (int j = 1; j <= categories[i].imageNumbers; j++) {
            Mat gray_img;
            Ptr<SIFT> sift_ptr = SIFT::create(0, 3, 0.04, 20, 1.6);
            cvtColor(imread(CATEGORIES_PATH + categories[i].className + "/" + categories[i].className + to_string(j) + IMAGE_EXT), gray_img, COLOR_BGR2GRAY);
            //gray_img = imread(CATEGORIES_PATH + categories[i].className + "/" + categories[i].className + to_string(j) + IMAGE_EXT);

            vector<KeyPoint> keypoints;
            Mat descriptors;
            sift_ptr->detect(gray_img, keypoints);
            sift_ptr->compute(gray_img, keypoints, descriptors);

            all_desc.push_back(descriptors);
            //all_desc_per_img.push_back(descriptors);
            //all_class_per_img.push_back(categories[i].classLabel);
            //all_desc_per_img_num++;
        }
    }

    string name_categories;
    for (const auto & category : categories) {
        name_categories += to_string(category.classLabel) + "_";
    }
    string path_to_k = "../kmeans/kmeans_" + name_categories + ".yml";
    int clusterCount = d_size, attempts = 5, iterationNumber = 1000;
    cout << "Running kmeans..." << endl;
    TermCriteria crit = TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, iterationNumber, 0.001);
    kmeans(all_desc, clusterCount, k_labels, crit, attempts, KMEANS_PP_CENTERS, k_centers);

    FileStorage storage(path_to_k, FileStorage::WRITE);
    storage << "k_labels" << k_labels << "k_centers" << k_centers;
    storage.release();
    cout << "Kmeans file created with name: " << path_to_k << endl;
    cout << "---------------------------------------------------" << endl;
}

/*
void read_detect_compute_img(const string& className, int imageNumbers, int classLabel) {
    for (int i = 1; i <= imageNumbers; i++) {
        Mat gray_img;
        Ptr<SIFT> sift_ptr = SIFT::create(0, 3, 0.04, 20, 1.6);
        //cvtColor(imread(CATEGORIES_PATH + className + "/" + className + to_string(i) + IMAGE_EXT), gray_img, COLOR_BGR2GRAY);
        gray_img = imread(CATEGORIES_PATH + className + "/" + className + to_string(i) + IMAGE_EXT);

        vector<KeyPoint> keypoints;
        Mat descriptors;
        sift_ptr->detect(gray_img, keypoints);
        sift_ptr->compute(gray_img, keypoints, descriptors);

        all_desc.push_back(descriptors);
        all_desc_per_img.push_back(descriptors);
        all_class_per_img.push_back(classLabel);
        all_desc_per_img_num++;
    }
}
 */

Mat get_data_vector(const Mat& descriptors, const Mat& k_centers, int dict_size) {
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptors, k_centers, matches, 2);

    const float ratio_thresh = 0.8f;
    vector<DMatch> good_matches;

    for (auto & matche : matches) {
        if (matche[0].distance < ratio_thresh * matche[1].distance) {
            good_matches.push_back(matche[0]);
        }
    }
    //Make a Histogram of visual words
    Mat data_i = Mat::zeros(1, dict_size, CV_32F);
    int index = 0;
    for (auto j = good_matches.begin(); j < good_matches.end(); j++, index++) {
        data_i.at<float>(0, good_matches.at(index).trainIdx) = data_i.at<float>(0, good_matches.at(index).trainIdx) + 1;
    }
    return data_i;
}

/*
void get_histogram() {
    for (int i = 0; i < all_desc_per_img_num; i++) {
        Mat data_vector = get_data_vector(all_desc_per_img[i]);
        input_data.push_back(data_vector);
        input_data_labels.push_back(Mat(1, 1, CV_32SC1, all_class_per_img[i]));
    }
}
 */

void predict_img(const Mat& img, Mat& img_keypoints, int& pred_ID, double& pred_strength, const Ptr<SVM>& svm, const Mat& k_centers, int dict_size) {
    Mat gray_img;
    Ptr<SIFT> sift_ptr = SIFT::create(0, 3, 0.04, 20, 1.6);
    //cvtColor(img, gray_img, COLOR_BGR2GRAY);
    gray_img = img.clone();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    sift_ptr->detect(gray_img, keypoints);
    sift_ptr->compute(gray_img, keypoints, descriptors);

    Mat data_vector = get_data_vector(descriptors, k_centers, dict_size);
    Mat results;
    svm->predict(data_vector, results, 1);
    pred_strength = results.at<float>(0);
    svm->predict(data_vector, results);
    pred_ID = int(results.at<float>(0));
    drawKeypoints(img, keypoints, img_keypoints);
}

void predict_categories(const vector<Mat>& images_to_predict, vector<food> categories, vector<int>& pred_IDs, vector<double>& pred_strengths) {

    //number_classes = 0;
    //dict_size = 0;
    //all_desc.release();
    //k_centers.release();
    //k_labels.release();
    //input_data.release();
    //input_data_labels.release();
    //all_desc_per_img.clear();
    //all_class_per_img.clear();
    //all_desc_per_img_num = 0;

    Mat k_centers, k_labels;

    Mat all_desc;
    vector<Mat> all_desc_per_img;
    vector<int> all_class_per_img;
    int all_desc_per_img_num = 0;

    Mat input_data;
    Mat input_data_labels;

    Ptr<SVM> svm;

    int num_class = int(categories.size());
    int d_size = 40 * num_class;

    /*
    for (int i = 0; i < number_classes; ++i) {
        read_detect_compute_img(categories[i].className, categories[i].imageNumbers, categories[i].classLabel);
    }
     */

    for (int i = 0; i < num_class; ++i) {
        for (int j = 1; j <= categories[i].imageNumbers; j++) {
            Mat gray_img;
            Ptr<SIFT> sift_ptr = SIFT::create(0, 3, 0.04, 20, 1.6);
            cvtColor(imread(CATEGORIES_PATH + categories[i].className + "/" + categories[i].className + to_string(j) + IMAGE_EXT), gray_img, COLOR_BGR2GRAY);
            //gray_img = imread(CATEGORIES_PATH + categories[i].className + "/" + categories[i].className + to_string(j) + IMAGE_EXT);

            vector<KeyPoint> keypoints;
            Mat descriptors;
            sift_ptr->detect(gray_img, keypoints);
            sift_ptr->compute(gray_img, keypoints, descriptors);

            all_desc.push_back(descriptors);
            all_desc_per_img.push_back(descriptors);
            all_class_per_img.push_back(categories[i].classLabel);
            all_desc_per_img_num++;
        }
    }

    string name_categories;
    for (const auto &category: categories) {
        name_categories += to_string(category.classLabel) + "_";
    }
    string path_to_k = "../kmeans/kmeans_" + name_categories + ".yml";

    const char *direc = path_to_k.c_str();
    struct stat sub{};
    if (stat(direc, &sub) == 0) {
        FileStorage storage_k(path_to_k, FileStorage::READ);
        storage_k["k_labels"] >> k_labels;
        storage_k["k_centers"] >> k_centers;
        storage_k.release();

        /*
    try {
        const char* direc = path_to_k.c_str();
        struct stat sub{};
        if (stat(direc, &sub) == 0){
            FileStorage storage_k(path_to_k, FileStorage::READ);
            storage_k["k_labels"] >> k_labels;
            storage_k["k_centers"] >> k_centers;
            storage_k.release();
        }
        else{
            cout << "The Kmeans haven't been calculated yet!" << endl;
            throw 444;
        }
    }
    catch(int err) {

    }
         */

        //get_histogram();

        for (int i = 0; i < all_desc_per_img_num; i++) {
            Mat data_vector = get_data_vector(all_desc_per_img[i], k_centers, d_size);
            input_data.push_back(data_vector);
            input_data_labels.push_back(Mat(1, 1, CV_32SC1, all_class_per_img[i]));
        }

        svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1e4, 1e-6));
        Ptr<TrainData> trained = TrainData::create(input_data, ROW_SAMPLE, input_data_labels);
        svm->train(trained);

        int k = 0;
        for (const auto &img: images_to_predict) {
            k++;
            Mat dst;
            string predicted;
            int pred_ID;
            double pred_strength;
            predict_img(img, dst, pred_ID, pred_strength, svm, k_centers, d_size);
            /*
            string crp_img_name = to_string(k) + "- ID:" + to_string(pred_ID);
            namedWindow(crp_img_name, WINDOW_NORMAL);
            resizeWindow(crp_img_name, 400, 400);
            imshow(crp_img_name, dst);
             */
            //Add predictions to output of function
            cout << pred_ID << " " << pred_strength << endl;
            pred_IDs.push_back(pred_ID);
            pred_strengths.push_back(pred_strength);
        }
    }
}