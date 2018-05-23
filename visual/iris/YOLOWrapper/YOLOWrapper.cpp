#include "YOLOWrapper.h" 

#ifdef USE_TINY
const string YOLOWrapper::cfg_path = "darknet/cfg/yolov3-tiny.cfg";
const string YOLOWrapper::weight_path = "models/yolov3-tiny.weights";
#else
const string YOLOWrapper::cfg_path = "darknet/cfg/yolov3.cfg";
const string YOLOWrapper::weight_path = "models/yolov3.weights";
#endif
const string YOLOWrapper::coco_names[] = {
	"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
	"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
	"elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
	"skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
	"tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
	"sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
	"pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
	"cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
	"scissors","teddy bear","hair drier","toothbrush"
};

YOLOWrapper::YOLOWrapper()
{
	cuda_set_device(0);
	net = load_network(const_cast<char*>(cfg_path.c_str()),const_cast<char*>(weight_path.c_str()),0);
	set_batch_network(net,1);
}

YOLOWrapper::~YOLOWrapper()
{
}

map<string,vector<Rect> > YOLOWrapper::predict(Mat img,float thresh)
{
#ifndef NDEBUG
	assert(false == img.empty());
#endif
	map<string,vector<Rect> > retVal;
	//allocate image
	image im = make_image(img.cols,img.rows,3);
	for(int c = 0 ; c < img.channels() ; c++)
		for(int h = 0 ; h < img.rows ; h++)
			for(int w = 0 ; w < img.cols ; w++) {
				int dst_index = w + img.cols * h + img.cols * img.rows * c;
				im.data[dst_index] = static_cast<float>(img.ptr<unsigned char>(h)[w * img.channels() + c] / 255.0);
			}
	image sized = letterbox_image(im,net->w,net->h);
	//detect
	float *X = sized.data;
	network_predict(net,X);
	int nboxes = 0;
	detection *dets = get_network_boxes(net,im.w,im.h,thresh,0.5,0,1,&nboxes);
	layer l = net->layers[net->n-1];
	do_nms_sort(dets,nboxes,l.classes,0.45);
	//extract detection results
	for(int i = 0 ; i < nboxes ; i++) {
		int _class = -1;
		for(int j = 0 ; j < l.classes ; j++) {
			if(dets[i].prob[j] > thresh) if(_class < 0) _class = j;
		}
		if(_class >= 0) {
			box b = dets[i].bbox;
			int left = (b.x - b.w / 2.) * im.w;
			int right = (b.x + b.w / 2.) * im.w;
			int top = (b.y - b.h / 2.) * im.h;
			int bot = (b.y + b.h / 2.) * im.h;
			
			if(left < 0) left = 0;
			if(right > im.w - 1) right = im.w - 1;
			if(top < 0) top = 0;
			if(bot > im.h - 1) bot = im.h - 1;
			
			retVal[coco_names[_class]].push_back(Rect(Point(left,top),Point(right + 1,bot + 1)));
		}
	}
	//free image
	free_detections(dets,nboxes);
	free_image(im);
	free_image(sized);
	return retVal;
}
