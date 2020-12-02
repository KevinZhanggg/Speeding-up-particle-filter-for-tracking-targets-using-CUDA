#ifndef DOG_H
#define DOG_H

/* from local library */
#include "colorhistogram.h"

/* from OpenCV library */
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/* from gsl library */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* from gflags library */
//#include <gflags/gflags.h>

// use console flag
//DECLARE_double(std);

class Dog {
public:
    Dog();

    Dog(const int fw, const int fh, const cv::Rect &rect, cv::Mat *hist);

    void run(cv::Mat &frame, gsl_rng *rng);
	void run2(cv::Mat &frame, gsl_rng *rng);
	void run3(cv::Mat &frame, gsl_rng *rng);
    //bool operator<(const Dog &p) { return weight < p.weight; }

    /* coefficient for normalizing */
    static const int LAMBDA;

    /* autoregressive dynamics parameters for transition model */
    static const float A1;
    static const float A2;
    static const float B0;

    /* standard deviations for gaussian sampling in transition model */
    static const double TRANS_X_STD;
    static const double TRANS_Y_STD;
    static const double TRANS_S_STD;

    int fw;
    int fh;
    float x0;
    float y0;
    float xp;
    float yp;
    float x;
    float y;
    float sp;
    float s;
    float width;
    float height;
    cv::Mat *hist;
    float weight;

private:
    static ColorHistogram *pHist;

    float _likelihood(const cv::Mat &frame);

    float _histo_dist_sq(cv::Mat *h1, cv::Mat *h2);
};

#endif
