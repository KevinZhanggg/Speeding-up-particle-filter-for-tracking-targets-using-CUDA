#include "dog.h"
#include <gsl/gsl_sf.h>
#include "opencv2/core/cuda.hpp"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>  
#include <curand_kernel.h>

#define FLAGS_std 10.0
const int Dog::LAMBDA = 20;

/* autoregressive dynamics parameters for transition model */
const float Dog::A1 = 2.0f;
const float Dog::A2 = -1.0f;
const float Dog::B0 = 1.0f;

/* standard deviations for gaussian sampling in transition model */
const double Dog::TRANS_X_STD = FLAGS_std;
const double Dog::TRANS_Y_STD = FLAGS_std;
const double Dog::TRANS_S_STD = 0.0;

ColorHistogram *Dog::pHist = new ColorHistogram();

Dog::Dog() {

}

Dog::Dog(const int fw, const int fh, const cv::Rect &rect, cv::Mat *hist) {
    this->fw = fw;
    this->fh = fh;
    this->x0 = this->xp = this->x = rect.x + rect.width / 2;
    this->y0 = this->yp = this->y = rect.y + rect.height / 2;
    this->sp = 1.0;
    this->s = 1.0;
    this->width = rect.width;
    this->height = rect.height;
    this->hist = hist;
    this->weight = 0;

}

void Dog::run(cv::Mat &frame, gsl_rng *rng) {
    float x, y, s;

    x = A1 * (this->x - this->x0) + A2 * (this->xp - this->x0)
        + B0 * (float) gsl_ran_gaussian(rng, TRANS_X_STD) + this->x0;
    y = A1 * (this->y - this->y0) + A2 * (this->yp - this->y0)
        + B0 * (float) gsl_ran_gaussian(rng, TRANS_Y_STD) + this->y0;
    s = A1 * (this->s - 1.0f) + A2 * (this->sp - 1.0f)
        + B0 * (float) gsl_ran_gaussian(rng, TRANS_S_STD) + 1.0f;

    this->xp = this->x;
    this->yp = this->y;
    this->sp = this->s;

    float hw = this->width * this->s / 2 + 1.0f;
    float hh = this->height * this->s / 2 + 1.0f;

    /* restrict dogs in the frame */
    this->x = std::max(hw, std::min((float) this->fw - hw, x));
    this->y = std::max(hh, std::min((float) this->fh - hh, y));
    this->s = std::max(0.2f, s);

    this->weight = _likelihood(frame);
}
void Dog::run2(cv::Mat &frame, gsl_rng *rng) {
	float x, y, s;

	x = A1 * (this->x - this->x0) + A2 * (this->xp - this->x0)
		+ B0 * (float)gsl_ran_gaussian(rng, TRANS_X_STD) + this->x0;
	y = A1 * (this->y - this->y0) + A2 * (this->yp - this->y0)
		+ B0 * (float)gsl_ran_gaussian(rng, TRANS_Y_STD) + this->y0;
	s = A1 * (this->s - 1.0f) + A2 * (this->sp - 1.0f)
		+ B0 * (float)gsl_ran_gaussian(rng, TRANS_S_STD) + 1.0f;

	this->xp = this->x;
	this->yp = this->y;
	this->sp = this->s;

	float hw = this->width * this->s / 2 + 1.0f;
	float hh = this->height * this->s / 2 + 1.0f;

	/* restrict dogs in the frame */
	this->x = std::max(hw, std::min((float) this->fw - hw, x));
	this->y = std::max(hh, std::min((float) this->fh - hh, y));
	this->s = std::max(0.2f, s);

}
void Dog::run3(cv::Mat &frame, gsl_rng *rng) {
	this->weight = _likelihood(frame);
}
float Dog::_likelihood(const cv::Mat &frame) {
    int c = cvRound(this->x);
    int r = cvRound(this->y);
    int w = cvRound(this->width * this->s);
    int h = cvRound(this->height * this->s);

    cv::Mat imgROI = frame(cv::Rect(c - w / 2, r - h / 2, w, h));

    cv::Mat *hist = pHist->getHueHistogram(imgROI, 40);

    float d_sq = _histo_dist_sq(this->hist,hist);
    

    return std::exp(-LAMBDA * d_sq);
}

float Dog::_histo_dist_sq(cv::Mat *h1, cv::Mat *h2) {
    float sum = 0.0f;
   
    for (int i = 0; i < 256; i++) {
        sum += std::sqrt(h1->at<float>(i) * h2->at<float>(i));
    }

    return 1.0f - sum;
}





	
        
      

      
