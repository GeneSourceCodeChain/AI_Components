CUDA_PREFIX=/usr/local/cuda
CXXFLAGS=-I. `pkg-config --cflags opencv` -I${CUDA_PREFIX}/include -DGPU -DCUDNN -O2 -msse3 -msse4
LIBS=`pkg-config --libs opencv` -Ldarknet -ldarknet -lboost_program_options -lboost_filesystem -lboost_system -L${CUDA_PREFIX}/lib64 -lcudart -lcuda -lcublas -lcurand -lcudnn
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: demo

demo: YOLOWrapper.o main.o
	$(CXX) $^ $(LIBS) -o ${@}

run: demo
	LD_LIBRARY_PATH=darknet:${LD_LIBRARY_PATH} ./demo -i test/1.jpg -o 1.png
	LD_LIBRARY_PATH=darknet:${LD_LIBRARY_PATH} ./demo -i test/2.jpg -o 2.png
	LD_LIBRARY_PATH=darknet:${LD_LIBRARY_PATH} ./demo -i test/3.jpg -o 3.png
	LD_LIBRARY_PATH=darknet:${LD_LIBRARY_PATH} ./demo -i darknet/data/dog.jpg -o dog.png
	
clean:
	$(RM) $(OBJS) demo
