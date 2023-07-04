
#ifndef COMPUTERVISION_FIND_HISTOGRAM_H
#define COMPUTERVISION_FIND_HISTOGRAM_H

array<int,4> find_histogram(const cv::Mat& src);
Mat mean_histogram(const vector<cv::Mat>& src);

#endif //COMPUTERVISION_FIND_HISTOGRAM_H
