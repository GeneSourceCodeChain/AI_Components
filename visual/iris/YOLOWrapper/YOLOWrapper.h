#ifndef YOLOWRAPPER_H
#define YOLOWRAPPER_H

#include <vector>
#include <map>
#include <string>
#include <boost/tuple/tuple.hpp>
#include <opencv2/opencv.hpp>
#define BLOCK 512
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cudnn.h>
extern "C" {
#include "darknet.h"
}

//#define USE_TINY

using namespace std;
using namespace cv;

class YOLOWrapper {
	static const string cfg_path;
	static const string weight_path;
	static const string coco_names[];
	network *net;
public:
	YOLOWrapper();
	virtual ~YOLOWrapper();
	map<string,vector<Rect> > predict(Mat img,float thresh = 0.5);
};

#endif
