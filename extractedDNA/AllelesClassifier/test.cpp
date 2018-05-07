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
#include "classifier.h"

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
	string modelfile;
	desc.add_options()
		("help,h","print current usage")
		("input,i",value<string>(&inputfile),"test samples list")
		("model,m",value<string>(&modelfile),"model file");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);

	if(1 == argc || vm.count("help")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}

	if(1 != vm.count("input") || 1 != vm.count("model")) {
		cout<<"the input and model file must be specified"<<endl;
		return EXIT_FAILURE;
	}

	Classifier classifier(modelfile);
	std::ifstream in(inputfile);
	if(false == in.is_open()) {
		cout<<"the test sample file is invalid"<<endl;
		return EXIT_FAILURE;
	}

	map<int,vector<vector<char> > > list = loadList(inputfile);
	int correct = 0,wrong = 0;
	for(auto itr = list.begin() ; itr != list.end() ; itr++)
		for(auto sample = itr->second.begin() ; sample != itr->second.end() ; sample++) {
			int _class = classifier.predict(*sample);
			if(_class == itr->first) correct++;
			else wrong++;
		}
	cout<<"precision: "<<static_cast<float>(correct) / (correct + wrong)<<endl;

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
