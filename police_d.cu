#include "police_d.h"

/* from stl */
#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <helper_cuda.h>
#include <helper_string.h>
#include <time.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>  
#include <curand_kernel.h>
#include "dog_d.h"
#include "dog.h"

//cudaError_t mStatus;
#include "opencv2/core/cuda.hpp"

namespace GPU = cv::cuda;

Police_d::Police_d(const cv::Mat &frame, const cv::Rect &rect, const int num_Dog_ds) {
    ColorHistogram ch;
    cv::Mat imgROI = frame(rect);
    cv::Mat *hist = ch.getHueHistogram(imgROI, 40);

	for (int i = 0; i < num_Dog_ds; i++) 
	{
		Dog *pd = new Dog(frame.cols, frame.rows, rect, hist);
		_dogs.push_back(pd);
	}

	_dogArr = new DogArr();
	_dogArr->iniDogArr(num_Dog_ds);

    for (int i = 0; i < num_Dog_ds; i++) 
	{
		_dogArr->init_iDogArr(i, frame.cols, frame.rows, rect, hist);
    }
	_num_dogs = num_Dog_ds;

	_dogArr_d = new DogArr();
	_dogArr_d->iniDogArr_d(num_Dog_ds,_dogArr);

	f = NULL;
	f_d = NULL;
}
Police_d::~Police_d()
{
	delete _dogArr;
	delete _dogArr_d;
	for (int i = 0; i < _dogs.size(); i++)
	{
		Dog *pp = _dogs[i];
		delete pp;
	}

	if (f == NULL)
	{
		delete f;
		cudaFree(f_d);
	}
}
void Police_d::beginCuda()
{
	for (int i = 0; i < _num_dogs; i++)
	{
		_dogArr->fresh_iDog(i,_dogs[i]);
	}

	_dogArr_d->freshDogArr_d(_num_dogs, _dogArr);
}
void Police_d::endCuda()
{
	_dogArr_d->getResultDogArr_d(_num_dogs, _dogArr);

	for (int i = 0; i < _num_dogs; i++)
	{
		(*(_dogs[i])).weight = _dogArr->weight[i];
	}
}

void Police_d::mat2float(cv::Mat &mat, float *f)
{
	                                                    
	for (int i = 0; i < mat.rows; i++) 
	{
		for (int j = 0; j < mat.cols; j++)  
		{
			uchar3 v = mat.at<uchar3>(i, j);

			f[(i*mat.cols + j) * 3 + 0] = (float)v.x;
			f[(i*mat.cols + j) * 3 + 1] = (float)v.y;
			f[(i*mat.cols + j) * 3 + 2] = (float)v.z;
		}
	}

}

void Police_d::find(cv::Mat &frame, gsl_rng *rng) 
{
	//clock_t T0 = clock();
    std::vector<Dog *>::iterator it;
	for (it = _dogs.begin(); it != _dogs.end(); it++) {
        (*it)->run2(frame, rng);
    }
	//clock_t T1 = clock();
	beginCuda();
	//clock_t T2 = clock();

	cudaError_t mStatus;
	if (f==NULL)
	{
		f = new float[frame.rows*frame.cols * 3];
		cudaMalloc(&f_d, frame.rows*frame.cols * 3 * sizeof(float));
	}
	mat2float(frame, f);
	//clock_t T3 = clock();
	mStatus = cudaMemcpy(f_d, f, frame.rows*frame.cols * 3 * sizeof(float), cudaMemcpyHostToDevice);
	//clock_t T4 = clock();
	if (mStatus != cudaSuccess)
	{
		printf("Error 5");
		return;
	}

	{       //CUDA kernels 
		int threadNum = 64; //64;
		int blockNum = _num_dogs / threadNum + 1;

		dog_Run_d << < blockNum, threadNum >> > (threadNum, _num_dogs, f_d, frame.rows, frame.cols,
			_dogArr_d->x, _dogArr_d->y, _dogArr_d->width, _dogArr_d->height, _dogArr_d->s, _dogArr_d->hist_d, _dogArr_d->weight
			);
		cudaDeviceSynchronize();
	}
	//clock_t T5 = clock();
	endCuda();
	// T6 = clock();

	//printf("t1 :%f \n", (double)(T1 - T0) / CLOCKS_PER_SEC);
	//printf("t2 :%f \n", (double)(T2 - T1) / CLOCKS_PER_SEC);
	//printf("t3 :%f \n", (double)(T3 - T2) / CLOCKS_PER_SEC);
	//printf("t4 :%f \n", (double)(T4 - T3) / CLOCKS_PER_SEC);
	//printf("t5 :%f \n", (double)(T5 - T4) / CLOCKS_PER_SEC);
	//printf("t6 :%f \n", (double)(T6 - T5) / CLOCKS_PER_SEC);

}

void Police_d::normalize() {
	float sum = 0.0f;

	std::vector<Dog *>::iterator it;
	for (it = _dogs.begin(); it != _dogs.end(); it++) {
		sum += (*it)->weight;
	}
	for (it = _dogs.begin(); it != _dogs.end(); it++) {
		(*it)->weight /= sum;
	}
}

void Police_d::reassign() {
	std::vector<Dog *> tmp_dogs(_dogs);
	int k = 0;

	/* clear _polices */
	std::vector<Dog *>().swap(_dogs);

	std::vector<Dog *>::iterator it;
	for (it = tmp_dogs.begin(); it != tmp_dogs.end(); it++) {
		int np = cvRound((*it)->weight * _num_dogs);
		for (int i = 0; i < np; i++) {
			Dog *pd = new Dog(**it);
			pd->weight = 0.0f;
			_dogs.push_back(pd);
			k++;

			if (k == _num_dogs) {
				return;
			}
		}
	}

	while (k < _num_dogs) {
		Dog *pd = new Dog(*tmp_dogs.front());
		pd->weight = 0.0f;
		_dogs.push_back(pd);
		k++;
	}
}

void Police_d::sort_dogs() {
	std::sort(_dogs.begin(), _dogs.end(),
		[](Dog *const d1, Dog *const d2) -> bool {
		return d1->weight > d2->weight;
	});
}

void Police_d::report_best(cv::Mat &frame, const cv::Scalar color) {
	Dog *bd = _dogs.front();

	int x1 = cvRound(bd->x - 0.5 * bd->s * bd->width);
	int y1 = cvRound(bd->y - 0.5 * bd->s * bd->height);
	int x2 = cvRound(bd->s * bd->width) + x1;
	int y2 = cvRound(bd->s * bd->height) + y1;

	cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
		color, 1, 8, 0);
}

void Police_d::report_all(cv::Mat &frame, const cv::Scalar color) {
	cv::Point center;

	std::vector<Dog *>::iterator it;
	for (it = _dogs.begin(); it != _dogs.end(); it++) {
		center.x = cvRound((*it)->x);
		center.y = cvRound((*it)->y);

		cv::circle(frame, center, 2, color, -1);
	}
}
