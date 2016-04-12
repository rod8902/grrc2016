/* dsp.cu */


#include <stdio.h>
#include <math.h>

#include <jack/jack.h>
#define BLOCK_SIZE 512
#define KERNELTAPS	256	//must be odd value. 
//Freq/Taps = Filter Frequency accuracy. 
//5.4Hz for 44100.

#define THREAD_NUM 512	// executed thread count per block, do not change. 
// shared memory is common in the block.

#define DATAPERCYCLE 64  // data count per loop. do not change

#define BLOCKS 4
#define THREADS 512

#define N  255
/**********************************************************************/
#define NUM_TAPS	51
#define SAMPLERATE	44100
#define FX	2.0

__device__  float coeff_Kernel[KERNELTAPS];

__global__ void calcFIR(const float * g_indata, float * g_outdata, const int nframes)
{
		/************************************************************************************/
		// access Block Width
		//const unsigned int bw = gridDim.x;
		// access Block ID
		//const unsigned int bix = blockIdx.x;

		// access thread id
		//	const unsigned int tid = threadIdx.x;

		//int bx = blockIdx.x;
		//int tx = threadIdx.x;
		//int x = blockIdx.x * blockDim.x + threadIdx.x;
		int x = blockIdx.x*blockDim.x + threadIdx.x;
		/* 
		   for (int i=0; i<nframes; i++) {
		   g_outdata[i]=0.0f;
		   for (int j=0; j<KERNELTAPS; j++) {
		   g_outdata[i]+=h[j]*g_indata[i+j];
		   }
		   }
		 */		//      loop over i and j:
		//	out[i]+=h[j]*in[i+j];
		//__shared__ float buf[N];
		
		float twopioversamplerate = (2*M_PI)/ 44100;	//rod
		float comp;	//rod
		float amountoflast, amountofcurrent;
		int cutoff = 500;

		comp = 2 - cos(twopioversamplerate * cutoff);
		amountoflast = comp - (float)sqrt( comp * comp -1);
		amountofcurrent = 1 - amountoflast;

		g_outdata[x]=0.0f;
		
		if(x+N>nframes) {
			return;
		}
#pragma unroll
		for (int j=0; j<N-1; j++) {
				//buf[j]=g_indata[x];
				if(x+j>nframes) {
						__syncthreads();
						return;
				} else {
						//g_outdata[x] = g_outdata[x] + g_indata[(x+j)]*h[j]/(M_PI);
						//g_outdata[x] = g_outdata[x]*amountoflast + (g_indata[(x+j)]*amountofcurrent)/(M_PI);
						g_outdata[x] = g_outdata[x]*amountoflast + g_indata[(x+j)]*(1-amountofcurrent);
				}
		}
		__syncthreads();
}

/*

//do FIR
//each threads has offseted address to global memory. loop jumps threads*blocks.
for (int index = 0; index < nframes; index = index + THREAD_NUM*bw)
{
float dOut = 0.0;
//x[oldest]=g_indata[index];

//read g_indata to Shared Memory
//cycle is, ex, 8=8192/1024.

for (int j = 0; j < KERNELTAPS/DATAPERCYCLE; j++)
{
shared[tid             ] = g_indata[DATAPERCYCLE*j + THREAD_NUM*bix + index + tid           ];
//__syncthreads();
shared[tid+THREAD_NUM  ] = g_indata[DATAPERCYCLE*j + THREAD_NUM*bix + index + tid + THREAD_NUM];
//__syncthreads();
shared[tid+THREAD_NUM*2] = g_indata[DATAPERCYCLE*j + THREAD_NUM*bix + index + tid + THREAD_NUM*2];
__syncthreads();

#pragma unroll
for(int k = 0; k < DATAPERCYCLE; k = k+1)
{
// dOut += x[(oldest + k+tid)% N] * h[j*DATAPERCYCLE + k];
dOut += shared[k + tid] * h[j*DATAPERCYCLE + k];
//dOut += shared[k + tid] * coeff_Kernel[j*DATAPERCYCLE + k];
__syncthreads();
}
}
__syncthreads();
g_outdata[THREAD_NUM*bix + index + tid] = dOut;
}
}
 */



/*
//do FIR
//each threads has offseted address to global memory. loop jumps threads*blocks.
for (int index = 0; index < CalcSize; index = index + THREAD_NUM*bw)
{
dOut = 0.0;
x[oldest]=g_indata[index];
__syncthreads();

//read g_indata to Shared Memory
//cycle is, ex, 8=8192/1024.

for (int j = 0; j < KERNELTAPS/DATAPERCYCLE; j++)
{
shared[tid             ] = g_indata[DATAPERCYCLE*j + THREAD_NUM*bix + index + tid           ];
__syncthreads();
shared[tid+THREAD_NUM  ] = g_indata[DATAPERCYCLE*j + THREAD_NUM*bix + index + tid + THREAD_NUM];
__syncthreads();
shared[tid+THREAD_NUM*2] = g_indata[DATAPERCYCLE*j + THREAD_NUM*bix + index + tid + THREAD_NUM*2];
__syncthreads();

#pragma unroll 16
for(int k = 0; k < DATAPERCYCLE; k = k+1)
{
dOut += x[(oldest + k+tid)% N] * h[j*DATAPERCYCLE + k];
//dOut += shared[k + tid] * coeff_Kernel[j*DATAPERCYCLE + k];
__syncthreads();
}
}
__syncthreads();
g_outdata[THREAD_NUM*bix + index + tid] = dOut;
}
}
 */

__device__ float xv[4][3], yv[4][3], xv2[4][3], yv2[4][3], gain, a1, a2, a3, b1, b2;

__device__ void zerothem() {

		for(int i=0;i<4;++i) {
				for(int j=0;j<3;++j) {
						xv[i][j]=0.0;
						yv[i][j]=0.0;
						xv2[i][j]=0.0;
						yv2[i][j]=0.0;
				}
		}
}

extern "C" void GPU_INIT() {
		return;
}

__device__ int firstrun=0;

__device__ float mult (float a, float b) {
		return a*b;
}

__device__ void  device_memcpy (jack_default_audio_sample_t *to, jack_default_audio_sample_t *from, int len);

__global__ void GPU_DSP(jack_default_audio_sample_t *ins, jack_default_audio_sample_t *outs, int count)
{

	/**
	 * Gain with 2 threads
	 *

	 int i;
	 for (i=0; i< count/2; i++) {
	 if (threadIdx.x==0) {
	 outs[i]=ins[i];
	 } else if (threadIdx.x==1) {
	 outs[i+count/2]=ins[i+count/2]; 
	 }
	 }
	 */

	/**
	 * GAIN with 256 threads
	 */
	/*
	   for (int i=0; i< count/BLOCKSIZE; i++) {
	   outs[i+threadIdx.x*count/BLOCKSIZE]=0.1*ins[i+threadIdx.x*count/BLOCKSIZE];
	   }
	   }  
	*/

	int n=0;

	if (firstrun==0){ 
		gain=1.513365061e+03;
		a1=1.0;
		a2=2.0;
		a3=1.0;
		b1=-0.9286270861;
		b2=1.9259839697;
		zerothem();
		firstrun=1;
		__syncthreads();
	}

#pragma unroll 8
	for (int i=0; i< count; i++) {
		float intermediate = 0.0;
		xv[n][0] = xv[n][1]; xv[n][1] = xv[n][2];
		xv[n][2] = ins[i] / gain;
		yv[n][0] = yv[n][1]; yv[n][1] = yv[n][2];
		yv[n][2] =   (mult(a1,xv[n][0]) + mult(a3,xv[n][2])) + mult(a2,xv[n][1])
				+ mult(b1,yv[n][0]) + (mult(b2,yv[n][1]));
		__syncthreads();
		intermediate = yv[n][2];

		xv2[n][0] = xv2[n][1]; xv2[n][1] = xv2[n][2];
		xv2[n][2] = intermediate / gain;
		yv2[n][0] = yv2[n][1]; yv2[n][1] = yv2[n][2];
		yv2[n][2] =   (mult(a1,xv2[n][0]) + mult(a3,xv2[n][2])) + mult(a2,xv2[n][1])
				+ (mult(b1,yv2[n][0])) + (mult(b2,yv2[n][1]));
		__syncthreads();
		outs[i]=yv2[n][2];
	}
}


extern "C" void RunGPU_DSP( int grid, jack_default_audio_sample_t *ins, jack_default_audio_sample_t *outs, int count)
{
		/*
		   if(first==0) {
		   for (int j=0; j<N; j++) {
		   x[j]=0.0;
		   accum=0;
		   impulseResponse<<< BLOCKS,THREADS >>>(j);
		   h[j]=A[0]+accum;
		   }
		   first=1;
		   }
		 */
		calcFIR<<< BLOCKS,THREAD_NUM >>>( ins, outs, count );
}

__device__ void device_memcpy (jack_default_audio_sample_t *to, 
								jack_default_audio_sample_t *from, int len)
{
		int i;

		for (i=0; i< len; i++) to[i] = from[i];

}

