#ifndef Dog_d_H
#define Dog_d_H

/* from local library */
#include "colorhistogram.h"

/* from OpenCV library */
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/* from gsl library */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <helper_cuda.h>
#include <cuda_runtime.h>  
#include "opencv2/core/cuda.hpp"
#include "dog.h"

namespace GPU = cv::cuda;
class DogArr
{
public:
	DogArr();
	~DogArr();
	void iniDogArr(int num);
	void init_iDogArr(int i,const int fw, const int fh, const cv::Rect &rect, cv::Mat *hist);
	void fresh_iDog(int i, Dog *idog);
	void getResult_iDog(int i, Dog *idog);
public:
	void iniDogArr_d(int num, DogArr *tDogArr);
	void freshDogArr_d(int num, DogArr *tDogArr);

	void getResultDogArr_d(int num, DogArr *tDogArr);
public:
	bool is_GPU = false;
	int n;
	int *fw;
	int *fh;
	float *x0;
	float *y0;
	float *xp;
	float *yp;
	float *x;
	float *y;
	float *sp;
	float *s;
	float *width;
	float *height;
	float *weight;

	float *hist;
	float *hist_d;
};

__global__ void  dog_Run_d(int threadNum, int n, float *frame, int rows, int cols,
	float *x, float *y, float *width, float *height, float *s, float *hist, float * weight
	);

#define LAMBDA  20
//#define maxRow  40//!! 
#define maxRow 100
class ColorHistogram_d {
public:
	int histSize[3];
	float hranges[2];
	const float *ranges[3];
	int channels[3];
};





#endif
