#include "commander_d.h"
#include "police_d.h"

/* from stl */
#include <iostream>

cv::Scalar *Commander_d::_colors = new cv::Scalar[3]{
        CV_RGB(255, 0, 0),
        CV_RGB(0, 255, 0),
        CV_RGB(0, 0, 255)
};

Commander_d::Commander_d(const int num_dogs_per_police) : NUM_DOGS_PER_POLICE(num_dogs_per_police)
{
}
Commander_d::~Commander_d()
{
	for (int i = 0; i < _polices.size(); i++)
	{
		Police_d *pp = _polices[i];
		delete pp;
	}
}

void Commander_d::initialize(const cv::Mat &frame, const std::vector<cv::Rect> &pv) 
{
    std::vector<cv::Rect>::const_iterator it;
    for (it = pv.begin(); it != pv.end(); it++)
	{
		Police_d *pp = new Police_d(frame, *it, NUM_DOGS_PER_POLICE);
        _polices.push_back(pp);
    }
}

void Commander_d::transit(cv::Mat &frame, gsl_rng *rng) 
{
	std::vector<Police_d *>::iterator it;
    for (it = _polices.begin(); it != _polices.end(); it++) 
	{
        (*it)->find(frame, rng);
    }
}

void Commander_d::normalize_weights() 
{
    float sum = 0.0f;

	std::vector<Police_d *>::iterator it;
    for (it = _polices.begin(); it != _polices.end(); it++) 
	{
        (*it)->normalize();
    }
}

void Commander_d::resample() 
{
	std::vector<Police_d *>::iterator it;
    for (it = _polices.begin(); it != _polices.end(); it++) 
	{
        (*it)->reassign();
    }
}

void Commander_d::show_best(cv::Mat &frame) 
{
	std::vector<Police_d *>::iterator it;
    for (it = _polices.begin(); it != _polices.end(); it++) 
	{
        (*it)->report_best(frame, _colors[it - _polices.begin()]);
    }
}

void Commander_d::show_all(cv::Mat &frame) 
{
	std::vector<Police_d *>::iterator it;
    for (it = _polices.begin(); it != _polices.end(); it++) 
	{
        (*it)->report_all(frame, _colors[it - _polices.begin()]);
    }
}

void Commander_d::sort_all_dogs() 
{
	std::vector<Police_d *>::iterator it;
    for (it = _polices.begin(); it != _polices.end(); it++) 
	{
        (*it)->sort_dogs();
    }
}
