#ifndef POLICE_d_H
#define POLICE_d_H

/* from local library */
#include "dog_d.h"

/* from stl */
#include <vector>

class Police_d {
public:
	Police_d(const cv::Mat &frame, const cv::Rect &rect, const int num_dogs);

	~Police_d();

    void find(cv::Mat &frame, gsl_rng *rng);

    void normalize();

    void reassign();

    void report_best(cv::Mat &frame, const cv::Scalar color);

    void report_all(cv::Mat &frame, const cv::Scalar color);

    void sort_dogs();
public:
	void mat2float(cv::Mat &mat, float *f);
	void beginCuda();
	void endCuda();

	
private:
	DogArr *_dogArr;
	DogArr *_dogArr_d;
    std::vector<Dog *> _dogs;
    int _num_dogs;

private:
	int rows;
	int cols;
	float *f;
	float *f_d;
};

#endif
