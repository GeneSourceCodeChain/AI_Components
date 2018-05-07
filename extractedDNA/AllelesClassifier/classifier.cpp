#include <stdexcept>
#include "classifier.h"

Classifier::Classifier(string file)
{
	std::ifstream in(file);
	if(false == in.is_open()) throw runtime_error("invalid model file");
	text_iarchive ia(in);
	ia >> conprobs >> priors;
	classnum = conprobs.size();
#ifndef NDEBUG
	assert(4 == conprobs.begin()->second.size1());
	assert(classnum == priors.size());
#endif
	dimnum = conprobs.begin()->second.size2();
}

Classifier::~Classifier()
{
}

int Classifier::predict(vector<char> & v)
{
#ifndef NDEBUG
	assert(dimnum == v.size());
#endif
	vector<float> scores(classnum,1);
	for(int i = 0 ; i < classnum ; i++) {
		scores[i] *= priors[i];
		for(int j = 0 ; j < dimnum ; j++)
			switch(v[j]) {
				case 'A':case 'a': scores[i] *= conprobs[i](0,j);
				case 'T':case 't': scores[i] *= conprobs[i](1,j);
				case 'C':case 'c': scores[i] *= conprobs[i](2,j);
				case 'G':case 'g': scores[i] *= conprobs[i](3,j);
				default: break;
			}
	}
	vector<float>::iterator max_iter = max_element(scores.begin(),scores.end());
	return max_iter - scores.begin();
}

