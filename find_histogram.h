
#ifndef COMPUTERVISION_FIND_HISTOGRAM_H
#define COMPUTERVISION_FIND_HISTOGRAM_H

array<int,4> find_histogram(const cv::Mat& src);
Mat mean_histogram(const vector<cv::Mat>& src);
Mat mean_histogram2(const vector<cv::Mat>& src);
array<double, 2> compare_histogram(const Mat& src_hist, const vector<Mat>& categories_hist);

#endif //COMPUTERVISION_FIND_HISTOGRAM_H
