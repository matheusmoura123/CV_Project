#include "predict_categories.h"

int NUMBER_CLASSES = 0;
int DICT_SIZE = 0;
Mat allDescriptors;
Mat kCenters, kLabels;
Mat inputData;
Mat inputDataLables;
vector<Mat> allDescPerImg;
vector<int> allClassPerImg;
int allDescPerImgNum = 0;


void readDetectComputeimage(const string& className, int imageNumbers, int classLable) {
    for (int i = 1; i <= imageNumbers; i++) {
        Mat grayimg;
        Ptr<SIFT> siftptr = SIFT::create();
        cvtColor(imread(DATASET_PATH + className + "/" + className + to_string(i) + IMAGE_EXT), grayimg, COLOR_BGR2GRAY);

        vector<KeyPoint> keypoints;
        Mat descriptors;
        siftptr->detect(grayimg, keypoints);
        siftptr->compute(grayimg, keypoints, descriptors);

        allDescriptors.push_back(descriptors);
        allDescPerImg.push_back(descriptors);
        allClassPerImg.push_back(classLable);
        allDescPerImgNum++;
    }
}
Mat getDataVector(Mat descriptors) {
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptors, kCenters, matches, 2);

    const float ratio_thresh = 0.7f;
    vector<DMatch> good_matches;

    for (auto & matche : matches)
    {
        if (matche[0].distance < ratio_thresh * matche[1].distance)
        {
            good_matches.push_back(matche[0]);
        }
    }
    //Make a Histogram of visual words
    Mat datai = Mat::zeros(1, DICT_SIZE, CV_32F);
    int index = 0;
    for (auto j = good_matches.begin(); j < good_matches.end(); j++, index++) {
        datai.at<float>(0, good_matches.at(index).trainIdx) = datai.at<float>(0, good_matches.at(index).trainIdx) + 1;
    }
    return datai;
}
void getHistogramFast() {
    for (int i = 0; i < allDescPerImgNum; i++) {
        Mat dvec = getDataVector(allDescPerImg[i]);

        inputData.push_back(dvec);
        inputDataLables.push_back(Mat(1, 1, CV_32SC1, allClassPerImg[i]));
    }
}
void predictImg(const Mat& img, Mat& img_keypoints, string& className, vector<food> categories, Ptr<SVM> svm) {
    Mat grayimg;
    Ptr<SIFT> siftptr = SIFT::create();
    cvtColor(img, grayimg, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints;
    Mat descriptors;
    siftptr->detect(grayimg, keypoints);
    siftptr->compute(grayimg, keypoints, descriptors);

    Mat dvector = getDataVector(descriptors);
    Mat results;
    svm->predict(dvector, results, StatModel::RAW_OUTPUT);
    for (const auto & category : categories) {
        if (int(results.at<float>(0)) == category.classLable){
            className = category.className;
        }
    }
    drawKeypoints(img, keypoints, img_keypoints);
}

void predict_categories(vector<Mat> images_to_predict,
                        vector<food> categories,
                        vector<string>& predicted_classNames) {

    NUMBER_CLASSES = 0;
    DICT_SIZE = 0;
    allDescriptors.release();
    kCenters.release();
    kLabels.release();
    inputData.release();
    inputDataLables.release();
    allDescPerImg.clear();
    allClassPerImg.clear();
    allDescPerImgNum = 0;

    Ptr<SVM> svm;

    NUMBER_CLASSES = int(categories.size());
    DICT_SIZE = 200*NUMBER_CLASSES;

    for (int i = 0; i < NUMBER_CLASSES; ++i) {
        readDetectComputeimage(categories[i].className, categories[i].imageNumbers, categories[i].classLable);
    }

    string name_categories;
    for (const auto & category : categories) {
        name_categories += to_string(category.classLable) + "_";
    }
    string k_path = "./kmeans_" + name_categories + ".yml";
    string svm_path = "./svm_" + name_categories + ".yml";

    try {
        const char* dir = k_path.c_str();
        struct stat sb{};
        if (stat(dir, &sb) == 0){
            FileStorage storage(k_path, FileStorage::READ);
            storage["kLabels"] >> kLabels;
            storage["kCenters"] >> kCenters;
            storage.release();
        }
        else{
            cout << "The Kmeans haven't been calculated yet!" << endl;
            throw 505;
        }
    }
    catch(int err) {
        int clusterCount = DICT_SIZE, attempts = 5, iterationNumber = 1e4;
        cout << "Running kmeans..." << endl;
        kmeans(allDescriptors, clusterCount, kLabels, TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, iterationNumber, 1e-4), attempts, KMEANS_PP_CENTERS, kCenters);

        FileStorage storage(k_path, FileStorage::WRITE);
        storage << "kLabels" << kLabels << "kCenters" << kCenters;
        storage.release();
        cout << "Kmeans file created with name: " << k_path << endl;
        cout << "---------------------------------------------------" << endl;
    }

    getHistogramFast();

    svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria( TermCriteria::MAX_ITER, 1e4, 1e-6));
    Ptr<TrainData> td = TrainData::create(inputData, ROW_SAMPLE, inputDataLables);
    svm->train(td);

    int k=0;
    int key;
    for (const auto & img : images_to_predict) {
        k++;
        Mat dst;
        string predicted;
        predictImg(img, dst,  predicted, categories, svm);
        string crp_img_name = to_string(k) + ":" + predicted;
        namedWindow(crp_img_name, WINDOW_NORMAL);
        resizeWindow(crp_img_name, 400, 400);
        imshow(crp_img_name, dst);
        //Add predictions to output of function
        predicted_classNames.push_back(predicted);
    }
    key = waitKeyEx(0);
    if (key == 1048603) return;
}