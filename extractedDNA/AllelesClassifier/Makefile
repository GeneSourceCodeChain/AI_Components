CXXFLAGS=-I. -std=c++14
LIBS=-lboost_serialization -lboost_program_options
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: train test

train: train.o
	$(CXX) $^ $(LIBS) -o ${@}

test: test.o classifier.o
	$(CXX) $^ $(LIBS) -o ${@}

clean:
	$(RM) train test $(OBJS)
