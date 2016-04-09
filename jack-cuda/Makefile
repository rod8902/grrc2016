EXE	:= gpujacktest

# CUDA source files (compiled with nvcc)
CUFILES	:= dsp5.cu		
SMFLAGS	:= -arch sm_21 -maxrregcount=32

# CUDA libs
LIBDIR	:= -L/usr/local/cuda-6.5/lib64 -L/usr/local/cuda-6.5/lib
CUINC	:= -I/usr/local/cuda-6.5/include
			
# C/C++ source files (compiled with gcc / c++)
CCFILES	:= 
CCMAIN	:= cudajack_client2.cpp 
CCLIBS	:= -ljack -lcudart -lcuda 

################################################################################
# Rules and targets

all:
	nvcc $(CUINC) -c $(CUFILES) -o cu.o $(SMFLAGS)
	g++ $(CUINC) $(CCMAIN) cu.o -o $(EXE) $(LIBDIR) $(CCLIBS)



#NVCC=nvcc
#CXX=g++
#CC=gcc
#
#all: $(EXE)
#
#$(EXE): $(CUFILES) $(CCFILES)
#	$(CXX) $(CCMAIN) cu.o -o $(EXE) $(LIBDIR) $(CCLIBS)
#
#$(CUFILES):
#	$(NVCC) $(CUINC) $(CUFILES) -o cu.o
#
#$(CCFILES):
#	$(CXX) $(CUINC) $< -o $@
#
#clean:
#	rm -rf *.o $(EXE)
#
