#include "colorhistogram.h"

#include "opencv2/core/cuda.hpp"
#include <device_launch_parameters.h>
#include <cuda_runtime.h> 
#include <curand_kernel.h>
ColorHistogram::ColorHistogram() {
    histSize[0] = 256;
    histSize[1] = 256;
    histSize[1] = 256;

    hranges[0] = 0.0;
    hranges[1] = 256.0;

    ranges[0] = hranges;
    ranges[1] = hranges;
    ranges[2] = hranges;

    channels[0] = 0;
    channels[1] = 1;
    channels[2] = 2;
}

cv::Mat ColorHistogram::getHistogram(const cv::Mat &image) {
    cv::Mat hist;

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSize, ranges);

    return hist;
}

cv::Mat *ColorHistogram::getHueHistogram(const cv::Mat &image, int minSaturation) {
    cv::Mat *hist = new cv::Mat();
    cv::Mat hsv;
    cv::cvtColor(image, hsv, CV_BGR2HSV);

    cv::Mat mask;

    if (minSaturation > 0) {
        std::vector<cv::Mat> v;
        cv::split(hsv, v);

        cv::threshold(v[1], mask, minSaturation, 255, cv::THRESH_BINARY);

		
    }

	cv::calcHist(&hsv, 1, channels, mask, *hist, 1, histSize, ranges);
	

    cv::Mat *normal_hist = new cv::Mat();
    cv::normalize(*hist, *normal_hist, 1.0);

    return normal_hist;
}
