EXE	:= firlpf firhpf	#gpujacktest

# CUDA source files (compiled with nvcc)
CUFILES	:= origin_dsp5.cu		
SMFLAGS	:= -arch sm_21 -maxrregcount=32

# CUDA libs
LIBDIR	:= -L/usr/local/cuda-6.5/lib64 -L/usr/local/cuda-6.5/lib
CUINC	:= -I/usr/local/cuda-6.5/include
			
# C/C++ source files (compiled with gcc / c++)
CCFILES	:= 
CCMAIN	:= origin_cudajack_client2.cpp 
CCLIBS	:= -ljack -lcudart -lcuda 

################################################################################
# Rules and targets

all:
	#nvcc $(CUINC) -c $(CUFILES) -o cu.o $(SMFLAGS)
	#g++ $(CUINC) $(CCMAIN) cu.o -o $(EXE) $(LIBDIR) $(CCLIBS)
	nvcc $(CUINC) -c firlpf.cu -o lpfcu.o $(SMFLAGS)
	g++ $(CUINC) lpf_cudajack_client.cpp lpfcu.o -o firlpf $(LIBDIR) $(CCLIBS)
	nvcc $(CUINC) -c firhpf.cu -o hpfcu.o $(SMFLAGS)
	g++ $(CUINC) hpf_cudajack_client.cpp hpfcu.o -o firhpf $(LIBDIR) $(CCLIBS)



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
clean:
	rm -rf *.o $(EXE)


