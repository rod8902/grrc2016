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

__global__ void do_sample(const float *g_indata, float *g_outdata, float *g_sr, float *g_taps, int g_num_taps)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	float result;
	
	float sr[32];
	int idx = x+1;
	
	// For threads of block 0
	if( x>=0 && x<g_num_taps){
		for(i=0; i< g_num_taps; i++){
			sr[i] = g_sr[i];
		}
	
		for(i = g_num_taps - 1; i >= idx; i--){
			sr[i] = sr[i-idx];
		}

		for(i=0; i < idx; i++){
			sr[i] = g_indata[x-i];
		}
	}else if(x>=32 && x<2048){	// For other blocks
		for(i=0; i<g_num_taps;i++){
			sr[i] = g_indata[x-i];
		}
	}else
	//	printf("a\n");
	
	__syncthreads();

	// calculate
	result = 0;
	for(i = 0; i < g_num_taps; i++) {
		result += sr[i] * g_taps[i];
	}

	g_outdata[x] = result;	//g_indata[x];
	__syncthreads();

}

extern "C" void RunGPU_DSP( jack_default_audio_sample_t *ins, jack_default_audio_sample_t *outs, jack_default_audio_sample_t *sr, jack_default_audio_sample_t *taps, int numtaps)
{
	// count is nframes, ex) 2048 or 4096
	//calcFIR<<< BLOCKS,THREAD_NUM >>>( ins, outs, count );
	// 64, 32 - 2048
	do_sample<<<BLOCKS, THREAD_NUM>>>(ins, outs, sr, taps, numtaps);
}
