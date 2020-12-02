#ifndef COLORHISTOGRAM_H
#define COLORHISTOGRAM_H

/* from OpenCV library */
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ColorHistogram {
public:
    ColorHistogram();

    cv::Mat getHistogram(const cv::Mat &image);

    cv::Mat *getHueHistogram(const cv::Mat &image, int minSaturation = 0);

private:
    int histSize[3];
    float hranges[2];
    const float *ranges[3];
    int channels[3];
};

#endif
