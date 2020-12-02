#include "dog_d.h"

#include <gsl/gsl_sf.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <time.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>  
#include <curand_kernel.h>


#include <cuda.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/cuda.hpp"
#include <math.h>

namespace GPU = cv::cuda;

DogArr::DogArr() 
{

}
void DogArr::iniDogArr(int num)
{
	n = num;
	fw = new int[n];
	fh = new int[n];
	x0 = new float[n];
	y0 = new float[n];
	xp = new float[n];
	yp = new float[n];
	x = new float[n];
	y = new float[n];
	sp = new float[n];
	s = new float[n];
	width = new float[n];
	height = new float[n];
	weight = new float[n];

	hist = new float[256];
}

void DogArr::init_iDogArr(int i, const int fw, const int fh, const cv::Rect &rect, cv::Mat *hist)
{
	
	for (int i = 0; i < 256; i++)
	{
		this->hist[i] = hist->at<float>(i);
	}
	this->weight[i] = 0;

}

void DogArr::iniDogArr_d(int num, DogArr *tDogArr)
{
	cudaError_t mStatus;

	n = num;
	is_GPU = true;
	cudaMalloc(&fw, n * sizeof(int)); 
	cudaMalloc(&fh, n * sizeof(int)); 

	cudaMalloc(&x0, n * sizeof(float)); 
	cudaMalloc(&y0, n * sizeof(float));
	cudaMalloc(&xp, n * sizeof(float));
	cudaMalloc(&yp, n * sizeof(float));

	cudaMalloc(&x, n * sizeof(float));
	cudaMalloc(&y, n * sizeof(float));
	cudaMalloc(&sp, n * sizeof(float));
	cudaMalloc(&s, n * sizeof(float));

	cudaMalloc(&width, n * sizeof(float));
	cudaMalloc(&height, n * sizeof(float));
	cudaMalloc(&weight, n * sizeof(float));

	mStatus = cudaMalloc(&hist_d, 256 * sizeof(float));
	if (mStatus != cudaSuccess)
	{
		printf("Error 1");
		return;
	}
}

void DogArr::freshDogArr_d(int num, DogArr *tDogArr)
{
	cudaError_t mStatus;

	n = num;
	is_GPU = true;
	cudaMemcpy(x, tDogArr->x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y, tDogArr->y, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(s, tDogArr->s, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(width, tDogArr->width, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(height, tDogArr->height, n * sizeof(float), cudaMemcpyHostToDevice);

	mStatus = cudaMemcpy(hist_d, tDogArr->hist, 256 * sizeof(float), cudaMemcpyHostToDevice);
	if (mStatus != cudaSuccess)
	{
		printf("Error 5");
		return;
	}
}
void DogArr::getResultDogArr_d(int num, DogArr *tDogArr)
{
	n = num;
	cudaMemcpy(tDogArr->weight,weight, n * sizeof(float), cudaMemcpyDeviceToHost);
}
void DogArr::fresh_iDog(int i, Dog *idog)
{
	fw[i] = (*idog).fw;
	fh[i] = (*idog).fh;
	x0[i] = (*idog).x0;
	y0[i] = (*idog).y0;
	xp[i] = (*idog).xp;
	yp[i] = (*idog).yp;
	x[i] = (*idog).x;
	y[i] = (*idog).y;
	sp[i] = (*idog).sp;
	s[i] = (*idog).s;

	width[i] = (*idog).width;
	height[i] = (*idog).height;
	weight[i] = (*idog).weight;
}
DogArr::~DogArr()
{
	if (!is_GPU)
	{
		delete fw, fh, x0, y0, xp, yp, x, y, sp, s, width, height, weight;
		delete hist;
	}
	else
	{
		cudaFree(fw);
		cudaFree(fh);

		cudaFree(x0);
		cudaFree(y0);
		cudaFree(xp);
		cudaFree(yp);

		cudaFree(x);
		cudaFree(y);
		cudaFree(sp);
		cudaFree(s);

		cudaFree(width);
		cudaFree(height);
		cudaFree(weight);

		cudaFree(hist_d);
	}

}

__device__ void ColorHistogram_d_Init(ColorHistogram_d &data)
{
	data.histSize[0] = 256;
	data.histSize[1] = 256;
	data.histSize[1] = 256;

	data.hranges[0] = 0.0;
	data.hranges[1] = 256.0;

	data.ranges[0] = data.hranges;
	data.ranges[1] = data.hranges;
	data.ranges[2] = data.hranges;

	data.channels[0] = 0;
	data.channels[1] = 1;
	data.channels[2] = 2;

}

__device__  int bgr2hsv(float b, float g, float r, float &h, float &s, float &v)
{
	b /= 255;
	g /= 255;
	r /= 255;

	
	float max = b;
	if (g > max) max = g;
	if (r > max) max = r;

	float min = b;
	if (g < min) min = g;
	if (r < min) min = r;

	if (r == max)
		h = (g - b) / (max - min);
	if (g == max)
		h = 2.0 + (b - r) / (max - min);
	if (b == max)
		h = 4.0 + (r - g) / (max - min);
	h = h * 60.0;
	if (h < 0)
		h = h + 360.0;
	v = max;

	s = (v - min) / v;
	if (v == 0) s = 0;// S = (V - min) * 255 / V, result: s:0~1
	return 0;
}
__device__ void BGR2HSV(float image[maxRow*maxRow * 3], int rows, int cols)
{
	float b; float g; float r; float h; float s; float v;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			b = image[(i*cols + j) * 3 + 0];
			g = image[(i*cols + j) * 3 + 1];
			r = image[(i*cols + j) * 3 + 2];
			bgr2hsv(b, g, r, h, s, v);
			//image[(i*cols + j) * 3 + 0] = ((h / 2.0));// the result have one with cv RGB2hsv
			//image[(i*cols + j) * 3 + 0] = ((h / 2.0)) *255.0 / 180.0;
			image[(i*cols + j) * 3 + 0] = ((h / 2.0));
			image[(i*cols + j) * 3 + 1] = ((s * 255));
			image[(i*cols + j) * 3 + 2] = ((v * 255));
		}
	}
	return;
}

__device__  void ColorHistogram_d_getHueHistogram(ColorHistogram_d &data, float image[maxRow*maxRow * 3], int rows, int cols, int minSaturation, float normal_hist0[256])
{
	BGR2HSV(image, rows, cols);

	int mask[maxRow*maxRow];
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				int v = image[(i*cols + j) * 3 + 1];

				if (v>minSaturation)
				{
					mask[i*cols + j] = 1;
				}
				else
				{
					mask[i*cols + j] = 0;
				}
			}
		}
	}

	float max = 1e-10;
	

	for (int iH = data.ranges[0][0]; iH < data.ranges[0][1]; iH++)
	{
		normal_hist0[iH] = 0.0;
	}

	int v4 = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			v4 = (int)image[(i*cols + j) * 3 + 0];
			normal_hist0[v4] += mask[i*cols + j];
		}
	}


	for (int iH = data.ranges[0][0]; iH < data.ranges[0][1]; iH++)
	{
		max += normal_hist0[iH] * normal_hist0[iH];
	}
	max = sqrt(max);
	for (int iH = data.ranges[0][0]; iH < data.ranges[0][1]; iH++)
	{
		normal_hist0[iH] /= max;
	}
    return;
}

__device__ float BhattacharyyaDistance(ColorHistogram_d &pHist2, float *H1, float H2[256])
{
	float H1Avg = 0.0;
	float H2Avg = 0.0;
	float d_sq = 0.0;
	float Num = pHist2.ranges[0][1] - pHist2.ranges[0][0] + 1.0f;
	for (int iH = pHist2.ranges[0][0]; iH <= pHist2.ranges[0][1]; iH++)//for (int i = 0; i < 256; i++)
	{
		H1Avg += H1[iH];
		H2Avg += H2[iH];
		d_sq += sqrt(H1[iH] * H2[iH]);
	}
	H1Avg /= Num;
	H2Avg /= Num;

	float dis = 1.0;
	if (d_sq>1e-10)
	{
		H1Avg = 1.0f - 1.0f / sqrt(H1Avg*H2Avg) / Num*d_sq;
		if (H1Avg>0)
		{
			dis = sqrt(H1Avg);
		}
		else
		{
			dis = 1.0;
		}
	}
	return dis;
}

__device__ float _likelihood_d(int ith, float *frame, int rows, int cols,
	float *x, float *y, float *width, float *height, float *s, float *hist, float * weight)
{
	int c = round(x[ith]);
	int r = round(y[ith]);
	int wCol = round(width[ith] * s[ith]);
	int hRow = round(height[ith] * s[ith]);

	float imgROI[maxRow*maxRow * 3];

	if (c + wCol / 2 > cols)c = cols - wCol / 2;
	if (c - wCol / 2 < 0)c = wCol / 2;
	if (r + hRow / 2 > rows)r = rows - hRow / 2;
	if (r - hRow / 2 < 0)r = hRow / 2;

	int iRow, iCol,iD;
	for (int i = 0; i < hRow; i++)
	{
		for (int j = 0; j < wCol; j++)
		{
			iCol = c - wCol / 2 + j;
			iRow = r - hRow / 2 + i;
			iD = (iRow*cols + iCol) * 3;

			{
				imgROI[(i*wCol + j) * 3 + 0] = frame[iD + 0];
				imgROI[(i*wCol + j) * 3 + 1] = frame[iD + 1];
				imgROI[(i*wCol + j) * 3 + 2] = frame[iD + 2];
			}
		}
	}

	ColorHistogram_d pHist2;
	ColorHistogram_d_Init(pHist2);

	float normal_hist0[256];
	ColorHistogram_d_getHueHistogram(pHist2, imgROI, hRow,wCol, 40, normal_hist0);

/*	float d_sq = 0.0;
	for (int iH = pHist2.ranges[0][0]; iH < pHist2.ranges[0][1]; iH++)
	{
		d_sq += sqrt(hist[iH] * normal_hist0[iH]);
	}
	d_sq = 1.0f - d_sq;
	d_sq = exp(-LAMBDA * d_sq);

	return d_sq;
*/   
        float distance = 1.0f - BhattacharyyaDistance(pHist2, hist, normal_hist0);
        distance = exp(100.0f * distance);

 	return distance;
}
__global__ void dog_Run_d(int threadNum, int n, float *frame, int rows, int cols,
	float *x, float *y, float *width, float *height, float *s, float *hist, float * weight
	)
{
	const int i = blockIdx.x;
	const int j = threadIdx.x;

	int ith = i*threadNum + j;
	if (ith >= n)
	{
		return;
	}

	weight[ith] = _likelihood_d(ith, frame, rows, cols, x, y, width, height, s, hist, weight);
}












      
