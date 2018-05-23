#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/tuple/tuple.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include "YOLOWrapper.h"

#define CROPPED

using namespace std;
using namespace boost::program_options;
using namespace cv;

int main(int argc,char ** argv)
{
	string img_path;
	string output;
	options_description desc;
	desc.add_options()
		("help,h","print current message")
		("input,i",value<string>(&img_path),"input image")
		("output,o",value<string>(&output)->default_value("output.png"),"output image");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == vm.count("help")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}

	if(1 != vm.count("input")) {
		cout<<"input image must be specified"<<endl;
		return EXIT_FAILURE;
	}

	Mat img = imread(img_path);
	if(img.empty()) {
		cout<<"invalid image"<<endl;
		return EXIT_FAILURE;
	}

	YOLOWrapper yolo;
	map<string,vector<Rect> > objects = yolo.predict(img);
	for(auto & object : objects["person"]) {
		rectangle(img,object,Scalar(255,0,0),img.cols / 200);
	}
	cout<<"detected "<<objects["person"].size()<<" persons"<<endl;
	for(auto & object : objects["dog"]) {
		rectangle(img,object,Scalar(0,255,0),img.cols / 200);
	}
	cout<<"detected "<<objects["dog"].size()<<" dogs"<<endl;
	for(auto & object : objects["car"]) {
		rectangle(img,object,Scalar(0,0,255),img.cols / 200);
	}
	cout<<"detected "<<objects["car"].size()<<" cars"<<endl;
	imwrite(output,img);
	return EXIT_SUCCESS;
}
