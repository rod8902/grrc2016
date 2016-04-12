#include <stdio.h>
#include <math.h>
#include <jack/jack.h>

#define KERNELTAPS	256	//must be odd value. 
						//Freq/Taps = Filter Frequency accuracy. 
						//5.4Hz for 44100.

#define BLOCKS	64	//4
#define THREAD_NUM	32	// executed thread count per block, do not change. 
						// shared memory is common in the block.

#define DATAPERCYCLE 64  // data count per loop. do not change

/****************************/
#define SAMPLERATE	44100
#define NUM_TAPS	32
#define CUTOFF	2.0	//2.0

__device__ float samplerate = SAMPLERATE;
__device__ float m_taps[NUM_TAPS];
__device__ float m_sr[NUM_TAPS];
__device__ float m_lambda;

__global__ void do_sample(const float *g_indata, float *g_outdata)
{
	int i, n, x;

	/*** init() ***/
	for(i = 0; i < NUM_TAPS; i++) m_sr[i] = 0;
	m_lambda = M_PI * CUTOFF / (SAMPLERATE/2);

	/*** designLPF ***/
	
	double mm;

	for(n = 0; n < NUM_TAPS; n++){
		mm = n - (NUM_TAPS - 1.0) / 2.0;
		if( mm == 0.0 ) m_taps[n] = m_lambda / M_PI;
		else m_taps[n] = sin( mm * m_lambda ) / (mm * M_PI);
	}

	x = blockIdx.x * blockDim.x + threadIdx.x;

	/*** do_sample ***/
	float result;
	//short m1, m2;
	//char local_in[4];
	unsigned temp;

	result = g_indata[x];
	temp = *(unsigned *)&result;

	printf("in: %a, %a\n", g_indata[x], temp);
	/*
	for(i = NUM_TAPS - 1; i >= 1; i--){
		m_sr[i] = m_sr[i-1];
	}
	m_sr[0] = m1;

	result = 0;
	for(i = 0; i < NUM_TAPS; i++) 
		result += m_sr[i] * m_taps[i];
	m1 = ((int)result << 16);	

	//printf("result: %f\n",result);
	for(i = NUM_TAPS - 1; i >= 1; i--){
		m_sr[i] = m_sr[i-1];
	}
	m_sr[0] = m2;

	result = 0;
	for(i = 0; i < NUM_TAPS; i++) 
		result += m_sr[i] * m_taps[i];
	m2 = result;	

	g_outdata[x] = 0xff & (m1 | m2); //result;
	*/
	g_outdata[x] = g_indata[x];
	__syncthreads();
}
/*
__global__ void calcFIR(const float * g_indata, float * g_outdata, const int nframes)
{
	// access Block Width
	//const unsigned int bw = gridDim.x;

	// access Block ID
	//const unsigned int bix = blockIdx.x;

	// access thread id
	//	const unsigned int tid = threadIdx.x;

	//int x = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float sharedM[64];	// 32(thread_num) + 32(window_size)
									// shareM은 block 내의 thread 간의 공유 메모리

	//int max_sharedsize = 2048;
	//int loadsize = max_sharedsize/blockDim.x;	// 64
	//int begin = loadsize*threadIdx.x;	
	
	int sharedIdx = 2*threadIdx.x;
	int globalIdx = blockIdx.x*blockDim.x+sharedIdx;

	int begin = threadIdx.x;	
	int end = begin + NUM_TAPS;	// begin + loadsize;

	sharedM[sharedIdx] = g_indata[globalIdx];
	sharedM[sharedIdx+1] = g_indata[globalIdx+1];

	printf("blockIdx.x: %d, threadIdx.x: %d, shreadIdx: %d %d, globalIdx: %d %d, data: %f %f\n", 
			blockIdx.x, threadIdx.x, sharedIdx, sharedIdx+1, globalIdx, globalIdx+1, g_indata[globalIdx], g_indata[globalIdx+1]);

	__syncthreads();
			
	//This is Second algorithm
	int samplerate = 44100;
	float cutoff = 240.0f;
	float RC = 1.0/(cutoff*2*M_PI);
	float dt = 1.0/samplerate;
	float alpha = dt/(RC+dt);
	float temp;

	temp=sharedM[begin];
	for (int j=begin+1; j<end; j++) {
		//buf[j]=g_indata[x];
		//if(x+j>nframes) {
		//	__syncthreads();
		//	return;
		//} else {
			//g_outdata[x] = g_outdata[x] + (alpha * (g_indata[x] - g_outdata[x]));
			//g_outdata[x] = g_indata[x];
		//}
		//temp = temp + (alpha * (sharedM[j] - temp));
	}
	g_outdata[begin] = temp;
	__syncthreads();
	

}
*/
extern "C" void RunGPU_DSP( int grid, jack_default_audio_sample_t *ins, jack_default_audio_sample_t *outs, int count)
{
	// count is nframes, ex) 2048 or 4096
	//calcFIR<<< BLOCKS,THREAD_NUM >>>( ins, outs, count );
	// 64, 32
	do_sample<<<BLOCKS, THREAD_NUM>>>(ins, outs);
}
