CAFFE2_PREFIX=/home/xieyi/opt/caffe2
CAFFE2_HELPER_PREFIX=/home/xieyi/opt/caffe2_helper
CXXFLAGS=`pkg-config --cflags opencv dlib-1 eigen3` -I. -I${CAFFE2_PREFIX}/include \
-I${CAFFE2_HELPER_PREFIX}/include -std=c++14 -g2
LIBS= -L${CAFFE2_HELPER_PREFIX}/lib -lcaffe2_cpp -lcaffe2_cpp_gpu \
-L${CAFFE2_PREFIX}/lib -lcaffe2_gpu -lcaffe2 \
`pkg-config --libs opencv dlib-1 eigen3` \
-lglog -lprotobuf -lcudart -lcurand \
-lboost_filesystem -lboost_system -lboost_thread -lboost_regex -lboost_program_options -lpthread -ldl
OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

all: train_LSTM

train_LSTM: src/train_LSTM.o
	$(CXX) $^ -o ${@} $(LIBS)

clean:
	$(RM) train_LSTM $(OBJS)
