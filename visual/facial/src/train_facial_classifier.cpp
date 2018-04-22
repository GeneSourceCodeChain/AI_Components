#include <cstdlib>
#include <iostream>
#include <string>
#include <caffe2/core/init.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/util/blob.h>
#include <caffe2/util/model.h>
#include <caffe2/util/net.h>
#include <caffe2/zoo/mobilenet.h>

#define NDEBUG
#define WITH_CUDA
#define TRAINSIZE 32000
#define BATCHSIZE 80
#define CLASSNUM 4

using namespace std;
using namespace caffe2;

void setupTrainNet(NetDef & init, NetDef & predict);
void setupSaveNet(NetDef & init, NetDef & save);

unique_ptr<NetBase> predict_net;
unique_ptr<NetBase> save_net;

void atexit_handler()
{
	cout<<"saving params"<<endl;
	remove_all("facial_classifier_params");
	save_net->Run();
}

int main(int argc,char ** argv)
{
	NetDef init,predict,save;
	setupTrainNet(init,predict);
	setupSaveNet(init,save);
	auto device = CUDA;
	init.mutable_device_option()->set_device_type(device);
	predict.mutable_device_option()->set_device_type(device);
	save.mutable_device_option()->set_device_type(device);
	Workspace workspace(nullptr);
	workspace.RunNetOnce(init);
	predict_net = CreateNet(predict,&workspace);
	save_net = CreateNet(save,&workspace);
	atexit(atexit_handler);
	for(int i = 0 ; ; i++) {
		predict_net->Run();
		cout<<"iter:"<<i<<endl;
		if(i % 100 == 0) {
			cout<<"saving params"<<endl;
			remove_all("facial_classifier_params");
			save_net->Run();
		}
	}
	return EXIT_SUCCESS;
}

void setupTrainNet(NetDef & init, NetDef & predict)
{
    ModelUtil MobileNet(init,predict);
    MobileNet.init.AddCreateDbOp("db","lmdb","./dataset");
    MobileNet.predict.AddInput("db");
    MobileNet.addTensorProtosDbInputOp("db","data","label",BATCHSIZE);
    MobileNet.predict.SetName("MobileNet");
    auto input = "data";
    auto n = 0;
    auto alpha = 1.0;
    bool train = true;
    std::string layer = input;
    layer = MobileNet.AddFirst("1", layer, 32, 2, alpha, train)->output(0);
    layer = MobileNet.AddFilter(tos2(n++), layer, 32, 64, 1, alpha, train)->output(0); 
    layer = MobileNet.AddFilter(tos2(n++), layer, 64, 128, 2, alpha, train)->output(0); 
    layer = MobileNet.AddFilter(tos2(n++), layer, 128, 128, 1, alpha, train)->output(0); 
    layer = MobileNet.AddFilter(tos2(n++), layer, 128, 256, 2, alpha, train)->output(0); 
    layer = MobileNet.AddFilter(tos2(n++), layer, 256, 256, 1, alpha, train)->output(0); 
    layer = MobileNet.AddFilter(tos2(n++), layer, 256, 512, 2, alpha, train)->output(0); 
    for (auto i = 0; i < 5; i++) { // 6 - 10 
        layer = MobileNet.AddFilter(tos2(n++), layer, 512, 512, 1, alpha, train)->output(0);
    } 
    layer = MobileNet.AddFilter(tos2(n++), layer, 512, 1024, 2, alpha, train)->output(0); 
    layer = MobileNet.AddFilter(tos2(n++), layer, 1024, 1024, 1, alpha, train)->output(0); 	
    MobileNet.AddAveragePoolOp(layer, "final_avg", 1, 0, 5); 	
    MobileNet.AddFcOps("final_avg", "fc", 1024, 128, train);	
    MobileNet.AddFcOps("fc","logits",128,CLASSNUM,train);
    MobileNet.AddSoftmaxOp("logit","softmax");
    MobileNet.AddCrossEntropyOp({"softmax","label"},"loss");
    MobileNet.AddConstantFillWithOp(1.0, "loss", "loss_grad"); 	
    MobileNet.predict.AddGradientOps(); 	
    MobileNet.AddIterOps(); 
#ifndef NDEBUG 	
    MobileNet.predict.AddTimePlotOp("loss","iter","train",10); 
#endif 	
    MobileNet.AddLearningRateOp("iter", "lr", -0.01,0.9,100*round(static_cast<float>(TRAINSIZE)/BATCHSIZE)); 	
    string optimizer = "adam"; 	
    MobileNet.AddOptimizerOps(optimizer); 	
    //输出网络结构 	
    MobileNet.init.WriteText("models/MobileNet_train_init.pbtxt"); 	
    MobileNet.predict.WriteText("models/MobileNet_train_predict.pbtxt");
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