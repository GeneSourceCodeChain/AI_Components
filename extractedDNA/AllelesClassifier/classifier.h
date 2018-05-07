#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost;
using namespace boost::serialization;
using namespace boost::archive;
namespace ublas = boost::numeric::ublas;

class Classifier {
	map<int,ublas::matrix<float> > conprobs;
	map<int,float> priors;
	int classnum;
	int dimnum;
public:
	Classifier(string file);
	virtual ~Classifier();
	int predict(vector<char> & v);
};

#endif

