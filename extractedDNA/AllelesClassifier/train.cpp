#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <boost/algorithm/string/trim.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/tokenizer.hpp>
#include <boost/program_options.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace boost;
using namespace boost::serialization;
using namespace boost::archive;
using namespace boost::program_options;
namespace ublas = boost::numeric::ublas;

map<int,vector<vector<char> > > loadList(string file);

int main(int argc,char ** argv)
{
	options_description desc;
	string inputfile;
	string outputfile;
	desc.add_options()
		("help,h","print current usage")
		("input,i",value<string>(&inputfile),"training samples list")
		("output,o",value<string>(&outputfile),"model file");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);

	if(1 == argc || vm.count("help")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}

	if(1 != vm.count("input") || 1 != vm.count("output")) {
		cout<<"the input and output file must be specified"<<endl;
		return EXIT_FAILURE;
	}
	std::ofstream out(outputfile);
	if(false == out.is_open()) {
		cout<<"can't open output file"<<endl;
		return EXIT_FAILURE;
	}

	map<int,vector<vector<char> > > list = loadList(inputfile);
	if(0 == list.size()) {
		cout<<"empty list"<<endl;
		return EXIT_FAILURE;
	}
	int dim = list.begin()->second[0].size();
#ifndef NDEBUG
	cout<<"number of alleles: "<<dim<<endl;
#endif
	map<int,ublas::matrix<float> > probabilities;
	for(auto & _class : list) {
		//accumulate conditional probability of current class
		probabilities[_class.first] = ublas::zero_matrix<float>(4,dim);
		for(auto & sample : _class.second) {
			for(int i = 0 ; i < dim ; i++)
				switch(sample[i]) {
					case 'A':case 'a': probabilities[_class.first](0,i) += 1; break;
					case 'T':case 't': probabilities[_class.first](1,i) += 1; break;
					case 'C':case 'c': probabilities[_class.first](2,i) += 1; break;
					case 'G':case 'g': probabilities[_class.first](3,i) += 1; break;
					default:break;
				}
		}
		probabilities[_class.first] /= _class.second.size();
	}
	map<int,float> priors;
	float sum = 0;
	for(auto & _class : list) {
		priors[_class.first] = _class.second.size();
		sum += _class.second.size();
	}
	for(auto & _class : priors)
		_class.second /= sum;

	text_oarchive oa(out);
	oa << probabilities << priors;

	return EXIT_SUCCESS;
}

map<int,vector<vector<char> > > loadList(string file)
{
	std::ifstream in(file);
	if(false == in.is_open()) throw runtime_error("can't open the training list file");
	char_separator<char> sep(" \t");
	typedef boost::tokenizer<char_separator<char> > tokenizer;
	map<int,vector<vector<char> > > retVal;
	while(false == in.eof()) {
		string line;
		getline(in,line);
		trim(line);
		if("" == line) continue;
		tokenizer tokens(line,sep);
		vector<char> alleles;
		int c;
		for(tokenizer::iterator tok_iter = tokens.begin() ; tok_iter != tokens.end() ; tok_iter++) {
			auto next = tok_iter;
			next++;
			if(next != tokens.end())
				alleles.push_back(lexical_cast<char>(*tok_iter));
			else
				c = lexical_cast<int>(*tok_iter);
		}
#ifndef NDEBUG
		for(auto & a : alleles) {
			if(
				a != 'a' && a != 'A' && a != 't' && a != 'T' &&
				a != 'c' && a != 'C' && a != 'g' && a != 'G'
			) throw logic_error("the value of alleles can only within ATCG");
		}
#endif
		retVal[c].push_back(alleles);
	}
	return retVal;
}

