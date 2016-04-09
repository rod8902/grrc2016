/* dsp.cu */


#include <stdio.h>
#include <math.h>

#include <jack/jack.h>
#define BLOCK_SIZE	512
#define KERNELTAPS	256	//must be odd value. 
//Freq/Taps = Filter Frequency accuracy. 
//5.4Hz for 44100.

#define THREAD_NUM	512	// executed thread count per block, do not change. 
// shared memory is common in the block.

#define DATAPERCYCLE 64  // data count per loop. do not change

#define BLOCKS	4
#define THREADS 512

#define N  255
#define NFL 255.0

#define M ((N-1)/2.0)
/*
   __device__ float A[N]=
   {
   1.0, 0.9, 0.8, 1.0, 0.6, 1.0, 0.1, 1.0, 1.0, 1.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0
   };
 */

__device__ float h[N]={
	-0.0493254733597,-0.0447872903729,-0.0393994536854,-0.0332411068378,-0.0264058290462,-0.0190002768323,-0.0111425845971,-0.00296054721187,
	0.00541038834992, 0.0138292917627, 0.0221520447997, 0.0302337272125, 0.0379310547361, 0.0451048294826, 0.0516223621892,  0.0573598256355, 
	0.0622044990661 , 0.0660568646325, 0.0688325187041, 0.0704638633709, 0.0709015465243, 0.0701156225398, 0.0680964097194,  0.0648550252437,
	0.0604235833498 , 0.0548550477276, 0.0482227346262, 0.0406194688013, 0.0321564001233, 0.0229614943155, 0.0131777168103, 0.00296093400475,
	-0.00752243881471,-0.0180980090182,-0.0285861596699,-0.0388049425028,-0.0485730663149,-0.0577129339705,-0.0660536796058,  -0.073434156796,
	-0.0797058283755,-0.0847355093131,-0.0884079155425,-0.0906279739124,-0.0913228514356,-0.0904436657294,-0.0879668429255,    -0.0838950943,
	-0.0782579883898,-0.0711121013189,-0.0625407343861,-0.0526531945653,-0.0415836403373,-0.0294895021142,-0.0165494933214,-0.00296123486462,
	0.0110614778744, 0.0252917303025, 0.0394928253115, 0.0534218554965, 0.0668334605377,  0.079483715632, 0.0911340938844,   0.101555443413,
	0.110531918634,  0.117864804812,  0.123376175496,  0.126912323904,  0.128346911709,  0.127583781927,  0.124559386695,   0.119244785671,
	0.111647176399,  0.101810924296, 0.0898180667823, 0.0757882734584, 0.0598782519045, 0.0422805967207, 0.0232220875214, 0.00296144977576,
	-0.0182133995541,-0.0399885917727, -0.062027994285, -0.083977302115, -0.105468482203, -0.126124513729, -0.145564362603,   -0.16340812401,
	-0.17928226353, -0.192824884961, -0.203690951608, -0.211557387432, -0.216127985183, -0.217138050406, -0.214358712997,  -0.207600841817,
	-0.196718502626, -0.181611905284, -0.162229792631, -0.138571230734,  -0.11068676802,-0.0786789392362,-0.0427020989897,-0.00296157872693,
	0.0402878297219, 0.0867440516088,  0.136060546126,  0.187849732079,  0.241686983193,  0.297115145795,  0.353649524067,   0.410783271339,
	0.467993119886,  0.524745376675,  0.580502108391,  0.634727436091,  0.686893857863,  0.736488517077,  0.783019334129,   0.826020921048,
	0.86506020094,  0.899741657881,   0.92971214763,  0.954665205188,  0.974344791832,   0.98854843163,  0.997129695559,              0.0,
	0.997129695559,   0.98854843163,  0.974344791832,  0.954665205188,   0.92971214763,  0.899741657881,   0.86506020094,   0.826020921048,
	0.783019334129,  0.736488517077,  0.686893857863,  0.634727436091,  0.580502108391,  0.524745376675,  0.467993119886,   0.410783271339,
	0.353649524067,  0.297115145795,  0.241686983193,  0.187849732079,  0.136060546126, 0.0867440516088, 0.0402878297219,-0.00296157872693,
	-0.0427020989897,-0.0786789392362,  -0.11068676802, -0.138571230734, -0.162229792631, -0.181611905284, -0.196718502626,  -0.207600841817,
	-0.214358712997, -0.217138050406, -0.216127985183, -0.211557387432, -0.203690951608, -0.192824884961,  -0.17928226353,   -0.16340812401,
	-0.145564362603, -0.126124513729, -0.105468482203, -0.083977302115, -0.062027994285,-0.0399885917727,-0.0182133995541, 0.00296144977576,
	0.0232220875214, 0.0422805967207, 0.0598782519045, 0.0757882734584, 0.0898180667823,  0.101810924296,  0.111647176399,   0.119244785671,
	0.124559386695,  0.127583781927,  0.128346911709,  0.126912323904,  0.123376175496,  0.117864804812,  0.110531918634,   0.101555443413,
	0.0911340938844,  0.079483715632, 0.0668334605377, 0.0534218554965, 0.0394928253115, 0.0252917303025, 0.0110614778744,-0.00296123486462,
	-0.0165494933214,-0.0294895021142,-0.0415836403373,-0.0526531945653,-0.0625407343861,-0.0711121013189,-0.0782579883898,    -0.0838950943,
	-0.0879668429255,-0.0904436657294,-0.0913228514356,-0.0906279739124,-0.0884079155425,-0.0847355093131,-0.0797058283755,  -0.073434156796,
	-0.0660536796058,-0.0577129339705,-0.0485730663149,-0.0388049425028,-0.0285861596699,-0.0180980090182,-0.00752243881471,0.00296093400475,
	0.0131777168103, 0.0229614943155, 0.0321564001233, 0.0406194688013, 0.0482227346262, 0.0548550477276, 0.0604235833498,  0.0648550252437,
	0.0680964097194, 0.0701156225398, 0.0709015465243, 0.0704638633709, 0.0688325187041, 0.0660568646325, 0.0622044990661,  0.0573598256355,
	0.0516223621892, 0.0451048294826, 0.0379310547361, 0.0302337272125, 0.0221520447997, 0.0138292917627,0.00541038834992,-0.00296054721187,
	-0.0111425845971,-0.0190002768323,-0.0264058290462,-0.0332411068378,-0.0393994536854,-0.0447872903729,-0.0493254733597 };

__device__ float accum = 0.0;
/*
   __global__ void impulseResponse(int n) {
   __shared__ float ans[BLOCKS][THREADS];

   int tid = threadIdx.x;
   int bid = blockIdx.x;
   int k = bid*tid;

   if (k < M-1) {
   ans[bid][tid] = 2.0*A[k]*cos(2.0*M_PI*(n-M)*k/NFL);
   } else {
   ans[bid][tid] = 0.0;
   }
   __syncthreads();

   accum += ans[bid][tid]/NFL;

   __syncthreads();
   }
 */
__device__ int first=0;

//__device__ float x[N];

__device__ int oldest=0;

__device__  float coeff_Kernel[KERNELTAPS];

__global__ void calcFIRHJ(const float * g_indata, float * g_outdata, const int nframes)
{

	int max_sharedsize = 2048;

	__shared__ float sharedM[2048];

	int loadsize = max_sharedsize/blockDim.x;//64

	int begin = loadsize*threadIdx.x;
	int end	= begin+loadsize;
	//shared memory 관련//


	int samplerate = 44100;
        double cutoff = 500.0;
        double RC = 1.0/(cutoff*2*M_PI);
        double dt = 1.0/samplerate;
        double alpha = dt/(RC+dt);

	float twopioversamplerate = (2*M_PI)/ 44100; //rod
        float comp;  //rod
        float amountoflast, amountofcurrent;
           //int cutoff = 500;
        bool first;

           comp = 2 - cos(twopioversamplerate * cutoff);
           amountoflast = comp - (float)sqrt( comp * comp -1);
           amountofcurrent = 1 - amountoflast;

	for(int i=begin;i<end;i++){
		sharedM[i]=g_indata[i];	//global memory data -> shared memory
	}

	__syncthreads();

	for(int i=begin;i<end;i++){
		//printf("end-begin:%d\n",end-begin);//64
		g_outdata[i]=g_outdata[i]*amountoflast+ sharedM[i]*(amountofcurrent);

	}
	__syncthreads();

}



__global__ void calcFIR(const float * g_indata, float * g_outdata, const int nframes)
{
	// access Block Width
	//const unsigned int bw = gridDim.x;
	// access Block ID
	//const unsigned int bix = blockIdx.x;

	// access thread id
	//	const unsigned int tid = threadIdx.x;

	int x = blockIdx.x*blockDim.x + threadIdx.x;

	//This is Second algorithm
	int samplerate = 44100;
	double cutoff = 500.0;
	double RC = 1.0/(cutoff*2*M_PI);
	double dt = 1.0/samplerate;
	double alpha = dt/(RC+dt);

	g_outdata[x]=0.0f;
	for (int j=0; j<N-1; j++) {
		//buf[j]=g_indata[x];
		if(x+j>nframes) {
			__syncthreads();
			return;
		} else {
			g_outdata[x] = g_outdata[x] + g_indata[(x+j)]*h[j]/(M_PI);
			//g_outdata[x] = g_outdata[x]*amountoflast + (g_indata[(x+j)]*amountofcurrent)/(M_PI);
			//g_outdata[x] = g_outdata[x]*amountoflast + g_indata[(x+j)]*amountofcurrent;
			//g_outdata[x] = g_indata[x];//g_outdata[x-1] + (alpha * (g_indata[x] - g_outdata[x-1]));
		}
	}
	__syncthreads();

	// This is First algorithm
	/*
	   float twopioversamplerate = (2*M_PI)/ 44100;	//rod
	   float comp;	//rod
	   float amountoflast, amountofcurrent;
	   int cutoff = 500;
	   bool first;

	   printf("nframes = %d\n",nframes);
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
	g_outdata[x] = g_outdata[x]*amountoflast + g_indata[(x+j)]*amountofcurrent;
	}
	}
	 */
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
	calcFIRHJ<<< 256, 8 >>>( ins, outs, count );
}

__device__ void device_memcpy (jack_default_audio_sample_t *to, 
		jack_default_audio_sample_t *from, int len)
{
	int i;

	for (i=0; i< len; i++) to[i] = from[i];

}

