#include <cstdlib>
#include <iostream>
#include <boost/filesystem.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/util/blob.h>
#include <caffe2/util/model.h>
#include <caffe2/util/net.h>
#include <cvplot/cvplot.h>

#define NDEBUG
#define WITH_CUDA
#define TRAINSIZE 350418
#define BATCHSIZE 80
#define SEQLENGTH 25
#define CLASSNUM 10

using namespace std;
using namespace boost::filesystem;
using namespace caffe2;
using namespace cvplot;

void setupTrainNet(NetDef & init, NetDef & predict);
void setupSaveNet(NetDef & init, NetDef & save);

unique_ptr<NetBase> predict_net;
unique_ptr<NetBase> save_net;


void atexit_handler()
{
	cout<<"saving params"<<endl;
	remove_all("MobileID_params");
	save_net->Run();
}

int main(int argc,char ** argv)
{
	NetDef init,predict,save;
	setupTrainNet(init,predict);
	setupSaveNet(init,save);
#ifdef WITH_CUDA
	auto device = CUDA;
#else
	auto device = CPU;
#endif
	init.mutable_device_option()->set_device_type(device);
	predict.mutable_device_option()->set_device_type(device);
	save.mutable_device_option()->set_device_type(device);
	Workspace workspace(nullptr);
	workspace.RunNetOnce(init);
	predict_net = CreateNet(predict,&workspace);
	save_net = CreateNet(save,&workspace);
	atexit(atexit_handler);
#ifndef NDEBUG
	//show loss degradation
	cvplot::window("loss revolution");
	cvplot::move("loss",300,300);
	cvplot::resize("loss",500,300);
	cvplot::figure("loss").series("train").color(cvplot::Purple);
#endif
	for(int i = 0 ; ; i++) {
		predict_net->Run();
		cout<<"iter:"<<i<<endl;
		if(i % 100 == 0) {
			cout<<"saving params"<<endl;
			remove_all("MobileID_params");
			save_net->Run();
		}
	}
	return EXIT_SUCCESS;
}

void setupTrainNet(NetDef & init, NetDef & predict)
{
	ModelUtil network(init,predict);
	network.init.AddCreateDbOp("db","lmdb","./dataset");
	network.predict.AddInput("db");
	//data in format of batch_size x seq_length x 4
	network.AddTensorProtosDbInputOp("db","data","label",BATCHSIZE);
	//transpose into format seq_length x batch_size x 4
	network.AddTransposeOp("data","data_transposed",{1,0,2});
	
	network.init.AddConstantIntFillOp({BATCHSIZE},SEQLENGTH,"LSTM1/seq_lengths");
	network.AddLSTMOps("data_transposed","LSTM1","seq_lengths","hidden_init","cell_init","hidden_state","cell_state",4,100,false);
	network.AddCopyOp("LSTM1/hidden_state","LSTM1/hidden_init");
	network.AddCopyOp("LSTM1/cell_state","LSTM1/cell_init");
	
	network.init.AddConstantIntFillOp({BATCHSIZE},SEQLENGTH,"LSTM2/seq_lengths");
	network.AddLSTMOps("LSTM1/hidden_t_all","LSTM2","seq_lengths","hidden_init","cell_init","hidden_state","cell_state",100,100,false);
	network.AddCopyOp("LSTM2/hidden_state","LSTM2/hidden_init");
	network.AddCopyOp("LSTM2/cell_state","LSTM2/cell_init");
	
	network.AddFcOps("LSTM2/hidden_state","fc1",100,50);
	network.AddFcOps("fc1","fc2",50,CLASSNUM);
	network.AddSoftmaxWithLossOp({"fc2","label"},{"softmax","loss"});
	
	network.AddConstantFillWithOp(1.0, "loss", "loss_grad");
	network.predict.AddGradientOps();
	network.AddIterOps();
#ifndef NDEBUG
	network.predict.AddTimePlotOp("loss","iter","loss","train",10);
#endif
	network.AddLearningRateOp("iter", "lr", -0.01,0.9,100*round(static_cast<float>(TRAINSIZE)/BATCHSIZE));
	string optimizer = "adam";
	network.AddOptimizerOps(optimizer);
	//输出网络结构
	network.predict.WriteText("models/lstm_train_predict.pbtxt");
	network.init.WriteText("models/lstm_train_init.pbtxt");
}

void setupSaveNet(NetDef & init, NetDef & save)
{
	NetUtil InitNet(init);
	NetUtil SaveNet(save);
	vector<string> params;
	for(auto & op : InitNet.net.op()) {
		if(op.type() == "CreateDB") continue;
		for(auto & output : op.output())
			params.push_back(output);
	}
	SaveNet.AddSaveOp(params,"lmdb","LSTM_params");
	//output network
	SaveNet.WriteText("models/lstm_train_save.pbtxt");
}
 
