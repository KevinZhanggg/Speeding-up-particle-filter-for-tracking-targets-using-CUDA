//from local library//
#include "colorhistogram.h"
#include "police_d.h"
#include "utils.h"
#include "commander_d.h"
#include "time.h"
/* from standard C library */
#include <iostream>
#include <unistd.h>
#include <gflags/gflags.h>

#define MAX_OBJECTS 10

/* data types */
typedef struct params {
    cv::Point loc1[MAX_OBJECTS];
    cv::Point loc2[MAX_OBJECTS];
    cv::Mat *img;
    int num;
} params;

/* function prototypes */
void init_gsl_rng(); // initialize gsl rng
int set_bombs(cv::Mat &mat, std::vector<cv::Rect> &pvRects);

void on_mouse(int event, int x, int y, int flags, void *ustc);

/* global variables */
gsl_rng *rng;

/* console flags */
DEFINE_string(file, "../Ball.avi", "file to track");
DEFINE_int32(ndpp, 300, "number of dogs per police");
DEFINE_double(std, 10.0, "standard deviation of transition");
DEFINE_bool(showall, false, "whether to show all the dogs");

bool mIsOk = false;
int main(int argc, char **argv) 
{ 
	gflags::SetUsageMessage("Usage: ./find-the-bombs");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

	clock_t startTime_main, endTime_main;

    startTime_main = clock();

    if (!InitCUDA()) {
        return 0;
   	}
  	printf("CUDA initialized.\n");

	/*calculate the execution time*/
	clock_t startTime_stream, endTime_stream;

	/*local variables */
	cv::Mat frame;
	int num_objects = 0;
	std::vector<cv::Rect> pvRects;
	Commander_d cmder(FLAGS_ndpp);

	/* initialize rng */
	init_gsl_rng();

    startTime_stream = clock();

	/* open video */
	cv::VideoCapture capture(FLAGS_file);
	//cv::VideoCapture capture(0);

	if (!capture.isOpened()) {
		std::cout << "couldn't open video file" << std::endl;
		return 1;
	}

	/* get basic video info */
	//double num_frame = capture.get(CV_CAP_PROP_FRAME_COUNT);
	double rate = capture.get(CV_CAP_PROP_FPS);
    std::cout << "FPS: " << rate << std::endl;

	bool stop(false);
	cv::namedWindow("Video");

	int delay = (int)(1000 / rate);

	//set parameters for saving videos
	cv::VideoWriter writer("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, cv::Size(1280, 720), true);

	int index = 0;
	while (!stop) {
		if (!capture.read(frame))
			break;

		if (index == 0) 
		{

			mIsOk = false;
			
			writer << frame;

			while (num_objects == 0) 
			{
				std::cout << "Select object(s) to track ..." << std::endl;
				num_objects = set_bombs(frame, pvRects);
				std::cout << "num of objects: " << num_objects << std::endl;
			}
                        for (size_t i = 0; i < num_objects; i++)
			{
				if (pvRects[i].width>maxRow)
				{
					std::cout << "width>maxRow " << std::endl;
					pvRects[i].width = maxRow;
				}
				if (pvRects[i].height>maxRow)
				{
					pvRects[i].height = maxRow;
					std::cout << "height>maxRow " << std::endl;
				}
				std::cout << "pvRects " <<pvRects[i] << std::endl;
				std::cout << "width: " << pvRects[i].width << std::endl;
				std::cout << "height: " << pvRects[i].height << std::endl;
			}
			mIsOk = true;
	
			/* compute reference histograms and distribute particles */
			cmder.initialize(frame, pvRects);
		}
		else 
		{

			/* perform prediction and measurement for each particle */
			cmder.transit(frame, rng);//

			/* sort all dogs */
			cmder.sort_all_dogs();

			/* display */
			if (FLAGS_showall) {
				cmder.show_all(frame);
			}
			else {

				cmder.show_best(frame);
			}

			/* normalize weights and resample a set of unweighted particles */
			cmder.normalize_weights();
			cmder.resample();

			cv::imshow("Video", frame);
		}

		if (cv::waitKey(delay) == 32)
			stop = true;
	
		index++;


	}

   	std::cout << "frame_count:" << index << std::endl;

	capture.release();
	writer.release();
	gflags::ShutDownCommandLineFlags();
        
	endTime_main = clock();
        
	//printf("Totle Time of :%f  \n", (double)(endTime_main - startTime_main) / CLOCKS_PER_SEC);

	return 0;
}

void init_gsl_rng() {
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));
}

int set_bombs(cv::Mat &mat, std::vector<cv::Rect> &pvRects) {
    params p;

    p.img = &mat;
    p.num = 0;

    cv::namedWindow("Video");
    cv::imshow("Video", mat);
    cv::setMouseCallback("Video", &on_mouse, &p);
    cv::waitKey(0);

    if (p.num == 0) {
        return 0;
    }

    for (int i = 0; i < p.num; i++) {
        int x1 = std::min(p.loc1[i].x, p.loc2[i].x);
        int x2 = std::max(p.loc1[i].x, p.loc2[i].x);
        int y1 = std::min(p.loc1[i].y, p.loc2[i].y);
        int y2 = std::max(p.loc1[i].y, p.loc2[i].y);
        int w = x2 - x1;
        int h = y2 - y1;

        // ensure odd width and height
        w = (w % 2) ? w : w + 1;
        h = (h % 2) ? h : h + 1;

        pvRects.push_back(cv::Rect(x1, y1, w, h));
    }

    return p.num;
}

void on_mouse(int event, int x, int y, int flags, void *ustc) {

	if (mIsOk)
	{
		return;
	}
	

    params *p = (params *) ustc;
    cv::Mat tmp;
    cv::Point *loc;
    int n = p->num;

    if (n == MAX_OBJECTS)
        return;

    //p->img->copyTo(tmp);
    tmp = (*p->img).clone();

    // paint previous rects
    for (int i = 0; i < n; i++) {
        cv::rectangle(tmp, p->loc1[i], p->loc2[i], CV_RGB(255, 255, 255), 1, 8, 0);
        cv::imshow("Video", tmp);
    }

    if (event == CV_EVENT_LBUTTONDOWN) {
        loc = p->loc1;
        loc[n].x = x;
        loc[n].y = y;
    } else if (event == CV_EVENT_LBUTTONUP) {
        loc = p->loc2;
        loc[n].x = x;
        loc[n].y = y;
        p->num++;

        cv::rectangle(tmp, p->loc1[n], loc[n], CV_RGB(255, 255, 255), 1, 8, 0);
        cv::imshow("Video", tmp);
    } else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)) {
        cv::rectangle(tmp, p->loc1[n], cv::Point(x, y), CV_RGB(255, 255, 255), 1, 8, 0);
        cv::imshow("Video", tmp);
    }
}
