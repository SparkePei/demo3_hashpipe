#NVCC compiler and flags
CUDA_DIR ?= /usr/local/cuda
NVCC = nvcc
NVCCFLAGS   = -O3 --compiler-options '-fPIC' --compiler-bindir=/usr/bin/gcc --shared -Xcompiler -Wall -arch=sm_61 -lrt

# linker options
CUDA_LDFLAGS  = -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib64 -lcudart -lcufft
LFLAGS_PGPLOT = -L/usr/lib64/pgplot -lpgplot -lcpgplot -lX11
HP_LDFLAGS = -L/usr/local/lib -lhashpipe -lhashpipestatus -lrt -lm

NVCC_FLAGS = $(NVCCFLAGS) $(CUDA_LDFLAGS) $(LFLAGS_PGPLOT) $(HP_LDFLAGS)

# HASHPIPE
HP_LIB_TARGET   = demo3_hashpipe.o
HP_LIB_SOURCES  = demo3_net_thread.c \
		      demo3_output_thread.c \
		      demo3_databuf.c
HP_LIB_OBJECTS = $(patsubst %.c,%.o,$(HP_LIB_SOURCES))
HP_LIB_INCLUDES = demo3_databuf.h demo3_gpu_thread.h
HP_TARGET = demo3_hashpipe.so
# GPU
GPU_LIB_TARGET = demo3_gpu_kernels.o
GPU_LIB_SOURCES = demo3_gpu_kernels.cu demo3_gpu_thread.cu
GPU_LIB_INCLUDES =  demo3_gpu_thread.h

#PLOT
GPU_PLOT_TARGET = demo3_plot.o
# Filterbank
FILTERBANK_OBJECT   = filterbank.o

all: $(GPU_LIB_TARGET) $(FILTERBANK_OBJECT) $(GPU_PLOT_TARGET) $(HP_LIB_TARGET) $(HP_TARGET)

$(GPU_LIB_TARGET): $(GPU_LIB_SOURCES)
	$(NVCC) -c $^ $(NVCC_FLAGS)
	
$(FILTERBANK_OBJECT): filterbank.cpp filterbank.h
	$(NVCC) -c $< $(NVCC_FLAGS)

$(GPU_PLOT_TARGET): demo3_plot.cu demo3_plot.h
	$(NVCC) -c $< $(NVCC_FLAGS)

$(HP_LIB_TARGET): $(HP_LIB_SOURCES)
	$(NVCC) -c $^ $(NVCC_FLAGS)

# Link HP_OBJECTS together to make plug-in .so file
$(HP_TARGET): $(GPU_LIB_TARGET) $(FILTERBANK_OBJECT)
	$(NVCC) *.o -o $@ $(NVCC_FLAGS)
tags:
	ctags -R .
clean:
	rm -f $(HP_LIB_TARGET) $(GPU_LIB_TARGET) $(FILTERBANK_OBJECT) $(GPU_PLOT_TARGET) $(HP_TARGET) *.o tags 

prefix=/home/peix/local
LIBDIR=$(prefix)/lib
BINDIR=$(prefix)/bin
install-lib: $(HP_TARGET)
	mkdir -p "$(LIBDIR)"
	install -p $^ "$(LIBDIR)"
install: install-lib

.PHONY: all tags clean install install-lib
# vi: set ts=8 noet :
