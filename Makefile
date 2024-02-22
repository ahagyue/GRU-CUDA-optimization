TARGET=main
OBJ_DIR=./bin
OBJS=main.o util.o namegen.o
OBJECTS=$(patsubst %.o,$(OBJ_DIR)/%.o,$(OBJS))
CONFIGS=config.h

CFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -mno-avx512f -fopenmp -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CFLAGS),-Xcompiler=$(option))

LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS= -lstdc++ -lcudart -lm

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

$(OBJ_DIR)/%.o: %.cpp
	mkdir -p bin
	$(CXX) $(CFLAGS) -c -o $@ $^

$(OBJ_DIR)/%.o: %.cu
	mkdir -p bin
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

