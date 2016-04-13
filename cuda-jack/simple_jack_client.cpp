
/**
 * cuda_jackclient.cpp
 */

#include <time.h>
#include <math.h>
//#include <cuda_runtime.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <jack/jack.h>

#define FRAMES 1000000
typedef struct io_t {
	jack_default_audio_sample_t *p_in;
	jack_default_audio_sample_t *p_out;
	jack_client_t *client;
 	int first_exec;
} IO_T;


jack_port_t *input_port;
jack_port_t *output_port;
jack_client_t *client;
IO_T io;
IO_T* pio=&io;

int m_num_taps;
int m_error_flag;
double m_Fs;
double m_Fx;
double m_lambda;
double *m_taps;
double *m_sr;
int first_exec;
/*
extern "C" void RunGPU_DSP( int grid, jack_default_audio_sample_t *ins, jack_default_audio_sample_t *outs, int count);

extern "C" void GPU_INIT();

int cuda_initialise()
{
        GPU_INIT();
	
	pio->first_exec=0;
	IO_T *ptio=&io;
	int nframes = 10000;
	int deviceCount = 0;
	int multiCount = 1;
	cudaError rc;

	printf( "==== Initialising GPU DSP ====\n" );
	rc = cudaGetDeviceCount( &deviceCount );
	if( rc != cudaSuccess )
	{
		printf( " ! cudaGetDeviceCount() failed: %s\n", cudaGetErrorString( rc ) );
		return 0;
	}

	if( deviceCount == 1 )
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties( &prop, 0 );

		if( prop.major < 1 || prop.minor < 1)   // at least version 1.1 needed
 			deviceCount = 0;
		multiCount = prop.multiProcessorCount;
	}

	if( deviceCount == 0 )
	{
		printf( " CUDA capable device v. 1.1 needed\n" );
//		return 0;
	}
	else
	{
		printf( " Number of CUDA devices present: %d, multiprocessors = %d\n", deviceCount, multiCount );
	}
        printf("init success\nlast error: %d\n",cudaGetLastError());
 	return multiCount;
}
*/
void init()
{
	int i;

	if( m_error_flag != 0 ) return;

	for(i = 0; i < m_num_taps; i++) m_sr[i] = 0;

	return;
}

void designLPF()
{
	int n;
	double mm;

	for(n = 0; n < m_num_taps; n++){
		mm = n - (m_num_taps - 1.0) / 2.0;
		if( mm == 0.0 ) m_taps[n] =	m_lambda / M_PI;
		else m_taps[n] = sin( mm * m_lambda ) / (mm * M_PI);
	}

	return;
}
void filter(int num_taps, double Fs, double Fx)
{
	m_error_flag = 0;
	m_num_taps = num_taps;
	m_Fs = Fs;
	m_Fx = Fx;
	m_lambda = M_PI * Fx / (Fs/2);

	if( Fs <= 0 ) printf("-1\n");;
	if( Fx <= 0 || Fx >= Fs/2 ) printf("-2\n");
	if( m_num_taps <= 0 || m_num_taps > 1000 ) printf("-3\n");

	m_taps = m_sr = NULL;
	m_taps = (double*)malloc( m_num_taps * sizeof(double) );
	m_sr = (double*)malloc( m_num_taps * sizeof(double) );
	if( m_taps == NULL || m_sr == NULL ) printf("-4\n");
	
	init();

	designLPF();

	return;
}

double do_sample(double data_sample)
{
	int i;
	double result;

	//if( m_error_flag != 0 ) return(0);

	for(i = m_num_taps - 1; i >= 1; i--){
		m_sr[i] = m_sr[i-1];
	}	
	m_sr[0] = data_sample;

	result = 0;
	//printf("m_taps: ");
	for(i = 0; i < m_num_taps; i++) {
		result += m_sr[i] * m_taps[i];
		//printf(" %f ", m_taps[i]);
	}
	//printf("\n");

	return result;
}
/**
 * The process callback for this JACK application is called in a
 * special realtime thread once for each audio cycle.
 *
 * This client does nothing more than copy data from its input
 * port to its output port. It will exit when stopped by 
 * the user (e.g. using Ctrl-C on a unix-ish operating system)
 */

static int
_process (jack_nframes_t nframes, void *arg)
{
	io_t *gpuio=(io_t *)arg;
	//cudaError rc;
	
	jack_default_audio_sample_t *in, *out;
	//short *in, *out;

	float samp_dat;
	double out_val;

	in = (jack_default_audio_sample_t*)jack_port_get_buffer (input_port, nframes);
	out = (jack_default_audio_sample_t*)jack_port_get_buffer (output_port, nframes);
	//in = (short*) jack_port_get_buffer(input_port, nframes);
	//out = (short*) jack_port_get_buffer(output_port, nframes);
	if(!first_exec){
		filter(16, 44.1, 2.0);
		first_exec=1;
	}

	for(int i=0; i < nframes; i++){
		
		//samp_dat =  in[i];
		//out_val = do_sample((double) samp_dat);
		//samp_dat =  out_val;
		//out[i] = samp_dat;
		out[i] = (float) do_sample((double) in[i]);
		
		//samp_dat = 0.543543543*in[i];
		//out[i] = samp_dat;
		
		//out[i]=in[i];
	}

/*
	rc = cudaMemcpy( gpuio->p_in, in, sizeof(jack_default_audio_sample_t) * nframes, cudaMemcpyHostToDevice );
	if( rc != cudaSuccess )
	{
		printf( " cudaMemcpy() failed: %s\n", cudaGetErrorString( rc ) );
		
	}
	rc = cudaMemcpy( gpuio->p_out, out, sizeof(jack_default_audio_sample_t) * nframes, cudaMemcpyHostToDevice );
        if( rc != cudaSuccess )
        {
                printf( " ! cudaMemcpy() failed: %s\n", cudaGetErrorString( rc ) );
        }

	RunGPU_DSP( grid, gpuio->p_in, gpuio->p_out, nframes );
	
	rc = cudaMemcpy( in, gpuio->p_in, sizeof(jack_default_audio_sample_t) * nframes, cudaMemcpyDeviceToHost );
	if( rc != cudaSuccess )
	{
		printf( " ! cudaMemcpy() failed: %s\n", cudaGetErrorString( rc ) );
	}
		
	rc = cudaMemcpy( out, gpuio->p_out, sizeof(jack_default_audio_sample_t) * nframes, cudaMemcpyDeviceToHost );
	if( rc != cudaSuccess )
	{
		printf( " ! cudaMemcpy() failed: %s\n", cudaGetErrorString( rc ) );
	}
*/	

	return 0;
}

static void*
jack_thread (void *arg) 
{
	//cudaError rc;

	io_t *gpuio=(io_t *)arg;

	if (!gpuio->first_exec) {
/*
		rc = cudaMalloc( (void **)&(gpuio->p_in), sizeof(jack_default_audio_sample_t) * FRAMES );
		if( rc != cudaSuccess )
		{
			printf( " ! cudaMalloc() failed: %s\n", cudaGetErrorString( rc ) );
			return 0;
		}
		
		rc = cudaMalloc( (void **)&(gpuio->p_out), sizeof(jack_default_audio_sample_t) * FRAMES );
		if( rc != cudaSuccess )
		{
			printf( " ! cudaMalloc() failed: %s\n", cudaGetErrorString( rc ) );
			return 0;
		}
*/		gpuio->first_exec=1;
	}	

	while (1) {
		jack_nframes_t frames = jack_cycle_wait (gpuio->client);
		int status = _process(frames, gpuio);
		jack_cycle_signal (gpuio->client, status);

		// do something after signaling next clients in graph ...
		
		// end condition
		if (status != 0)
			return 0;

	}

	//Not reached
	return 0;

}



/**
 * JACK calls this shutdown_callback if the server ever shuts down or
 * decides to disconnect the client.
 */
void
jack_shutdown (void *arg)
{
	io_t *gpuio=(io_t *)arg;
	//cudaFree( gpuio->p_in );
	//cudaFree( gpuio->p_out );
	exit (1);
}

int
main (int argc, char *argv[])
{
	const char **ports;
	const char *client_name = "CUDA-DSP";
	const char *server_name = NULL;
	jack_options_t options = JackNullOption;
	jack_status_t status;
	
	/* open a client connection to the JACK server */
	first_exec = 0;
	client = jack_client_open (client_name, options, &status, server_name);
	if (client == NULL) {
		fprintf (stderr, "jack_client_open() failed, "
			 "status = 0x%2.0x\n", status);
		if (status & JackServerFailed) {
			fprintf (stderr, "Unable to connect to JACK server\n");
		}
		exit (1);
	}
	if (status & JackServerStarted) {
		fprintf (stderr, "JACK server started\n");
	}
	if (status & JackNameNotUnique) {
		client_name = jack_get_client_name(client);
		fprintf (stderr, "unique name `%s' assigned\n", client_name);
	}

	/* tell the JACK server to call `process()' whenever
	   there is work to be done.
	*/

	pio->client=client;

	//if(jack_set_process_thread (client, jack_thread, &io) < 0)
	if(jack_set_process_callback (client, _process, &io) < 0)
	    exit(1);


	/* tell the JACK server to call `jack_shutdown()' if
	   it ever shuts down, either entirely, or if it
	   just decides to stop calling us.
	*/

	jack_on_shutdown (client, jack_shutdown, &io);

	/* display the current sample rate. 
	 */

	printf ("engine sample rate: %d\n",
		jack_get_sample_rate (client));

	
	/* create two ports */

	input_port = jack_port_register (client, "input",
					 JACK_DEFAULT_AUDIO_TYPE,
					 JackPortIsInput, 0);
	output_port = jack_port_register (client, "output",
					  JACK_DEFAULT_AUDIO_TYPE,
					  JackPortIsOutput, 0);

	if ((input_port == NULL) || (output_port == NULL)) {
		fprintf(stderr, "no more JACK ports available\n");
		exit (1);
	}

	//cuda_initialise();	
		
	/* Tell the JACK server that we are ready to roll.  Our
	 * process() callback will start running now. */

	if (jack_activate (client)) {
		fprintf (stderr, "cannot activate client");
		exit (1);
	}

	/* Connect the ports.  You can't do this before the client is
	 * activated, because we can't make connections to clients
	 * that aren't running.  Note the confusing (but necessary)
	 * orientation of the driver backend ports: playback ports are
	 * "input" to the backend, and capture ports are "output" from
	 * it.
	 */

	ports = jack_get_ports (client, NULL, NULL,
				JackPortIsPhysical|JackPortIsOutput);
	if (ports == NULL) {
		fprintf(stderr, "no physical capture ports\n");
		exit (1);
	}

	if (jack_connect (client, ports[0], jack_port_name (input_port))) {
		fprintf (stderr, "cannot connect input ports\n");
	}

	free (ports);
	
	ports = jack_get_ports (client, NULL, NULL,
				JackPortIsPhysical|JackPortIsInput);
	if (ports == NULL) {
		fprintf(stderr, "no physical playback ports\n");
		exit (1);
	}

	if (jack_connect (client, jack_port_name (output_port), ports[0])) {
		fprintf (stderr, "cannot connect output ports\n");
	}

	free (ports);

	/* keep running until stopped by the user */

	sleep (-1);

	/* this is never reached but if the program
	   had some other way to exit besides being killed,
	   they would be important to call.
	*/

	jack_client_close (client);
	exit (0);
}

