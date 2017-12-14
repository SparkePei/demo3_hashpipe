/*demo3_gpu_thread.c
 *
 * Get two numbers from input databuffer, calculate them and write the sum to output databuffer.
 */
#ifdef __cplusplus
extern "C"{
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include "hashpipe.h"
#include "demo3_databuf.h"
#include "demo3_gpu_thread.h"
#include <cuda.h>
#include <cufft.h>
#include <time.h>
#include <cuda_runtime.h>

/* plotting */
extern float* g_pfSumPowX;
extern float* g_pfSumPowY;
extern float* g_pfSumStokesRe;
extern float* g_pfSumStokesIm;
extern float* g_pfFreq;
extern float g_fFSamp;

int g_iIsDataReadDone = FALSE;
char4* g_pc4InBuf = NULL;
char4* g_pc4InBufRead = NULL;
char4* g_pc4Data_d = NULL;              /* raw data starting address */
char4* g_pc4DataRead_d = NULL;          /* raw data read pointer */
int g_iNFFT = DEF_LEN_SPEC;
dim3 g_dimBPFB(1, 1, 1);
dim3 g_dimGPFB(1, 1);
dim3 g_dimBCopy(1, 1, 1);
dim3 g_dimGCopy(1, 1);
dim3 g_dimBAccum(1, 1, 1);
dim3 g_dimGAccum(1, 1);
float4* g_pf4FFTIn_d = NULL;
float4* g_pf4FFTOut_d = NULL;
cufftHandle g_stPlan = {0};
float4* g_pf4SumStokes = NULL;
float4* g_pf4SumStokes_d = NULL;
int g_iIsPFBOn = DEF_PFB_ON;
int g_iNTaps = 1;                       /* 1 if no PFB, NUM_TAPS if PFB */
/* BUG: crash if file size is less than 32MB */
int g_iSizeRead = DEF_SIZE_READ;
int g_iNumSubBands = DEF_NUM_SUBBANDS;
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};
float *g_pfPFBCoeff = NULL;
float *g_pfPFBCoeff_d = NULL;
static int Init(hashpipe_thread_args_t * args)
//int Init()
{
    int iDevCount = 0;
    cudaDeviceProp stDevProp = {0};
    int iRet = EXIT_SUCCESS;
    cufftResult iCUFFTRet = CUFFT_SUCCESS;
    int iMaxThreadsPerBlock = 0;

    iRet = RegisterSignalHandlers();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Signal-handler registration failed!\n");
        return EXIT_FAILURE;
    }

    /* since CUDASafeCallWithCleanUp() calls cudaGetErrorString(),
       it should not be used here - will cause crash if no CUDA device is
       found */
    (void) cudaGetDeviceCount(&iDevCount);
    if (0 == iDevCount)
    {
        (void) fprintf(stderr, "ERROR: No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }

    /* just use the first device */
    CUDASafeCallWithCleanUp(cudaSetDevice(0));

    CUDASafeCallWithCleanUp(cudaGetDeviceProperties(&stDevProp, 0));
    iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;

    if (g_iIsPFBOn)
    {
        /* set number of taps to NUM_TAPS if PFB is on, else number of
           taps = 1 */
        g_iNTaps = NUM_TAPS;

        g_pfPFBCoeff = (float *) malloc(g_iNumSubBands
                                        * g_iNTaps
                                        * g_iNFFT
                                        * sizeof(float));
        if (NULL == g_pfPFBCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }

        /* allocate memory for the filter coefficient array on the device */
        CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pfPFBCoeff_d,
                                           g_iNumSubBands
                                           * g_iNTaps
                                           * g_iNFFT
                                           * sizeof(float)));

        /* read filter coefficients */
        /* build file name */
        (void) sprintf(g_acFileCoeff,
                       "%s_%s_%d_%d_%d%s",
                       FILE_COEFF_PREFIX,
                       FILE_COEFF_DATATYPE,
                       g_iNTaps,
                       g_iNFFT,
                       g_iNumSubBands,
                       FILE_COEFF_SUFFIX);
        g_iFileCoeff = open(g_acFileCoeff, O_RDONLY);
        if (g_iFileCoeff < EXIT_SUCCESS)
        {
            (void) fprintf(stderr,
                           "ERROR: Opening filter coefficients file %s "
                           "failed! %s.\n",
                           g_acFileCoeff,
                           strerror(errno));
            return EXIT_FAILURE;
        }

        iRet = read(g_iFileCoeff,
                    g_pfPFBCoeff,
                    g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float));
        if (iRet != (g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float)))
        {
            (void) fprintf(stderr,
                           "ERROR: Reading filter coefficients failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }
        (void) close(g_iFileCoeff);

        /* copy filter coefficients to the device */
        CUDASafeCallWithCleanUp(cudaMemcpy(g_pfPFBCoeff_d,
                   g_pfPFBCoeff,
                   g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float),
                   cudaMemcpyHostToDevice));
    }

    /* allocate memory for data array - 32MB is the block size for the VEGAS
       input buffer */
    //CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pc4DataRead_d, g_iSizeRead));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pc4Data_d, g_iSizeRead));
    g_pc4DataRead_d = g_pc4Data_d;

    /* load data from the first file into memory */
    /*iRet = LoadDataToMem();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! Loading to memory failed!\n");
        return EXIT_FAILURE;
    }*/

    /* calculate kernel parameters */
    if (g_iNFFT < iMaxThreadsPerBlock)
    {
        g_dimBPFB.x = g_iNFFT;
        g_dimBCopy.x = g_iNFFT;
        g_dimBAccum.x = g_iNFFT;
    }
    else
    {
        g_dimBPFB.x = iMaxThreadsPerBlock;
        g_dimBCopy.x = iMaxThreadsPerBlock;
        g_dimBAccum.x = iMaxThreadsPerBlock;
    }
    g_dimGPFB.x = (g_iNumSubBands * g_iNFFT) / iMaxThreadsPerBlock;
    g_dimGCopy.x = (g_iNumSubBands * g_iNFFT) / iMaxThreadsPerBlock;
    g_dimGAccum.x = (g_iNumSubBands * g_iNFFT) / iMaxThreadsPerBlock;

    /*iRet = ReadData();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Reading data failed!\n");
        return EXIT_FAILURE;
    }*/

    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTIn_d,
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTOut_d,
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4)));

    g_pf4SumStokes = (float4 *) malloc(g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4));
    if (NULL == g_pf4SumStokes)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4SumStokes_d,
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                       '\0',
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4)));

    /* create plan */
    iCUFFTRet = cufftPlanMany(&g_stPlan,
                              FFTPLAN_RANK,
                              &g_iNFFT,
                              &g_iNFFT,
                              FFTPLAN_ISTRIDE,
                              FFTPLAN_IDIST,
                              &g_iNFFT,
                              FFTPLAN_OSTRIDE,
                              FFTPLAN_ODIST,
                              CUFFT_C2C,
                              FFTPLAN_BATCH);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan creation failed!\n");
        return EXIT_FAILURE;
    }

    iRet = InitPlot();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Plotting initialisation failed!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/* function that frees resources */
void CleanUp()
{
    /* free resources */
    if (g_pc4InBuf != NULL)
    {
        free(g_pc4InBuf);
        g_pc4InBuf = NULL;
    }
    if (g_pc4Data_d != NULL)
    {
        (void) cudaFree(g_pc4Data_d);
        g_pc4Data_d = NULL;
    }
    if (g_pf4FFTIn_d != NULL)
    {
        (void) cudaFree(g_pf4FFTIn_d);
        g_pf4FFTIn_d = NULL;
    }
    if (g_pf4FFTOut_d != NULL)
    {
        (void) cudaFree(g_pf4FFTOut_d);
        g_pf4FFTOut_d = NULL;
    }
    if (g_pf4SumStokes != NULL)
    {
        free(g_pf4SumStokes);
        g_pf4SumStokes = NULL;
    }
    if (g_pf4SumStokes_d != NULL)
    {
        (void) cudaFree(g_pf4SumStokes_d);
        g_pf4SumStokes_d = NULL;
    }

    free(g_pfPFBCoeff);
    (void) cudaFree(g_pfPFBCoeff_d);

    /* destroy plan */
    /* TODO: check for plan */
    (void) cufftDestroy(g_stPlan);

    if (g_pfSumPowX != NULL)
    {
        free(g_pfSumPowX);
        g_pfSumPowX = NULL;
    }
    if (g_pfSumPowY != NULL)
    {
        free(g_pfSumPowY);
        g_pfSumPowY = NULL;
    }
    if (g_pfSumStokesRe != NULL)
    {
        free(g_pfSumStokesRe);
        g_pfSumStokesRe = NULL;
    }
    if (g_pfSumStokesIm != NULL)
    {
        free(g_pfSumStokesIm);
        g_pfSumStokesIm = NULL;
    }
    if (g_pfFreq != NULL)
    {
        free(g_pfFreq);
        g_pfFreq = NULL;
    }

    /* TODO: check if open */
    cpgclos();
    return;
}

/*
 * Registers handlers for SIGTERM and CTRL+C
 */
int RegisterSignalHandlers()
{
    struct sigaction stSigHandler = {{0}};
    int iRet = EXIT_SUCCESS;

    /* register the CTRL+C-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGINT, &stSigHandler, NULL);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGINT);
        return EXIT_FAILURE;
    }

    /* register the SIGTERM-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGTERM, &stSigHandler, NULL);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGTERM);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/*
 * Catches SIGTERM and CTRL+C and cleans up before exiting
 */
void HandleStopSignals(int iSigNo)
{
    /* clean up */
    CleanUp();

    /* exit */
    exit(EXIT_SUCCESS);

    /* never reached */
    return;
}

void __CUDASafeCallWithCleanUp(cudaError_t iRet,
                               const char* pcFile,
                               const int iLine,
                               void (*pCleanUp)(void))
{
    if (iRet != cudaSuccess)
    {
        (void) fprintf(stderr,
                       "ERROR: File <%s>, Line %d: %s\n",
                       pcFile,
                       iLine,
                       cudaGetErrorString(iRet));
        /* free resources */
        (*pCleanUp)();
        exit(EXIT_FAILURE);
    }

    return;
}
/*
 * Prints usage information
 */
void PrintUsage(const char *pcProgName)
{
    (void) printf("Usage: %s [options] <data-file>\n",
                  pcProgName);
    (void) printf("    -h  --help                           ");
    (void) printf("Display this usage information\n");
    (void) printf("    -n  --nfft <value>                   ");
    (void) printf("Number of points in FFT\n");
    (void) printf("    -p  --pfb                            ");
    (void) printf("Enable PFB\n");
    (void) printf("    -a  --nacc <value>                   ");
    (void) printf("Number of spectra to add\n");
    (void) printf("    -s  --fsamp <value>                  ");
    (void) printf("Sampling frequency\n");

    return;
}


static void *run(hashpipe_thread_args_t * args)
{
    // Local aliases to shorten access to args fields
    demo3_input_databuf_t *db_in = (demo3_input_databuf_t *)args->ibuf;
    demo3_output_databuf_t *db_out = (demo3_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    int rv;
    uint64_t mcnt=0;
    int curblock_in=0;
    int curblock_out=0;
    
    int nhits = 0;
	char *data_raw; // raw data will be feed to gpu thread
    data_raw = (char *)malloc(g_iSizeRead*sizeof(char));
    //float *full_stokes; // full stokes data returned from gpu thread
    //full_stokes = (float *)malloc(N_CHANS_PER_SPEC*N_IFS*sizeof(float));
	int n_frames; // number of frames has been processed

    int iRet = EXIT_SUCCESS;
    int iSpecCount = 0;
    int iNumAcc = DEF_ACC;
    int iProcData = 0;
    cudaError_t iCUDARet = cudaSuccess;
    struct timeval stStart = {0};
    struct timeval stStop = {0};
    const char *pcProgName = NULL;
    int iNextOpt = 0;
    /* valid short options */
    const char* const pcOptsShort = "hb:n:pa:s:";
    /* valid long options */
    const struct option stOptsLong[] = {
        { "help",           0, NULL, 'h' },
        { "nsub",           1, NULL, 'b' },
        { "nfft",           1, NULL, 'n' },
        { "pfb",            0, NULL, 'p' },
        { "nacc",           1, NULL, 'a' },
        { "fsamp",          1, NULL, 's' },
        { NULL,             0, NULL, 0   }
    };
	// initialize 
	/*iRet=Init();
    if (iRet != EXIT_SUCCESS)
    {   
        (void) fprintf(stderr, "ERROR! Init failed!\n");
        CleanUp();
    }*/
    while (run_threads()) {

        hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "GPUBLKIN", curblock_in);
        hputs(st.buf, status_key, "waiting");
        hputi4(st.buf, "GPUBKOUT", curblock_out);
	hputi8(st.buf,"GPUMCNT",mcnt);
        hashpipe_status_unlock_safe(&st);

        // Wait for new input block to be filled
        while ((rv=demo3_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked");
                hashpipe_status_unlock_safe(&st);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
                pthread_exit(NULL);
                break;
            }
        }

        // Got a new data block, update status and determine how to handle it
        /*hashpipe_status_lock_safe(&st);
        hputu8(st.buf, "GPUMCNT", db_in->block[curblock_in].header.mcnt);
        hashpipe_status_unlock_safe(&st);*/

        // Wait for new output block to be free
        while ((rv=demo3_output_databuf_wait_free(db_out, curblock_out)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked gpu out");
                hashpipe_status_unlock_safe(&st);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                pthread_exit(NULL);
                break;
            }
        }

        // Note processing status
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "processing gpu");
        hashpipe_status_unlock_safe(&st);
	
	//get data from input databuf to local
	memcpy(data_raw,db_in->block[curblock_in].data_block,g_iSizeRead*sizeof(char));
	for(n_frames=0;n_frames < SIZEOF_INPUT_DATA_BUF/g_iSizeRead;n_frames++){
	// write new data to the gpu buffer
    	CUDASafeCallWithCleanUp(cudaMemcpy(g_pc4Data_d,
                                       data_raw,
                                       g_iSizeRead*sizeof(char),
                                       cudaMemcpyHostToDevice));
	/* whenever there is a read, reset the read pointer to the beginning */
	g_pc4DataRead_d = g_pc4Data_d;
	printf("SIZEOF_INPUT_DATA_BUF/g_iSizeRead is: %d\n",SIZEOF_INPUT_DATA_BUF/g_iSizeRead);
        while(iSpecCount < iNumAcc){

	if (g_iIsPFBOn)
        {
            /* do pfb */
		printf("do pfb processing...,");
            DoPFB<<<g_dimGPFB, g_dimBPFB>>>(g_pc4DataRead_d,
                                            g_pf4FFTIn_d,
                                            g_pfPFBCoeff_d);
            CUDASafeCallWithCleanUp(cudaThreadSynchronize());
            iCUDARet = cudaGetLastError();
            if (iCUDARet != cudaSuccess)
            {
                (void) fprintf(stderr,
                               "ERROR: File <%s>, Line %d: %s\n",
                               __FILE__,
                               __LINE__,
                               cudaGetErrorString(iCUDARet));
                /* free resources */
                CleanUp();
                //return EXIT_FAILURE;
            }
               printf("done!\n");
            /* update the data read pointer */
            g_pc4DataRead_d += (g_iNumSubBands * g_iNFFT);
        }
        else
        {
            CopyDataForFFT<<<g_dimGCopy, g_dimBCopy>>>(g_pc4DataRead_d,
                                                       g_pf4FFTIn_d);
            CUDASafeCallWithCleanUp(cudaThreadSynchronize());
            iCUDARet = cudaGetLastError();
            if (iCUDARet != cudaSuccess)
            {
                (void) fprintf(stderr,
                               "ERROR: File <%s>, Line %d: %s\n",
                               __FILE__,
                               __LINE__,
                               cudaGetErrorString(iCUDARet));
                /* free resources */
                CleanUp();
                //return EXIT_FAILURE;
           }
            /* update the data read pointer */
            g_pc4DataRead_d += (g_iNumSubBands * g_iNFFT);
        }

        /* do fft */
                printf("do fft...,");
        iRet = DoFFT();
        if (iRet != EXIT_SUCCESS)
        {
            (void) fprintf(stderr, "ERROR! FFT failed!\n");
            CleanUp();
            //return EXIT_FAILURE;
        }
                printf("done!\n");

        /* accumulate power x, power y, stokes, if the blanking bit is
           not set */
                printf("do stokes calculation and accumulation...,");
        Accumulate<<<g_dimGAccum, g_dimBAccum>>>(g_pf4FFTOut_d,
                                                 g_pf4SumStokes_d);
        CUDASafeCallWithCleanUp(cudaThreadSynchronize());
        iCUDARet = cudaGetLastError();
        if (iCUDARet != cudaSuccess)
        {
            (void) fprintf(stderr,
                           "ERROR: File <%s>, Line %d: %s\n",
                           __FILE__,
                           __LINE__,
                           cudaGetErrorString(iCUDARet));
            /* free resources */
            CleanUp();
            //return EXIT_FAILURE;
        }
                printf("done!\n");
        ++iSpecCount;
	}
        if (iSpecCount == iNumAcc)
        {
            /* dump to buffer */
                printf("copy accumulation data from gpu to cpu memory...,");
            CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes,
                                               g_pf4SumStokes_d,
                                               (g_iNumSubBands
                                               * g_iNFFT
                                                * sizeof(float4)),
                                                cudaMemcpyDeviceToHost));

		memcpy(db_out->block[curblock_out].Stokes_Full+N_CHANS_PER_SPEC*N_IFS*n_frames,g_pf4SumStokes,N_CHANS_PER_SPEC*N_IFS*sizeof(float));
                printf("done!\n");
            /* NOTE: Plot() will modify data! */
                printf("number %d of plot.\n",n_frames);
            Plot();
            (void) usleep(5000);

            /* reset time */
            iSpecCount = 0;
            /* zero accumulators */
            CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                               '\0',
                                               (g_iNumSubBands
                                                * g_iNFFT
                                                * sizeof(float4))));

        	/* if time to read from input buffer */
            	iProcData = 0;
    		(void) gettimeofday(&stStop, NULL);
    		(void) printf("Time taken (barring Init()): %gs\n",
                  ((stStop.tv_sec + (stStop.tv_usec * USEC2SEC))
                   - (stStart.tv_sec + (stStart.tv_usec * USEC2SEC))));

    		//return EXIT_SUCCESS;
	
		//display number of frames in status
		hashpipe_status_lock_safe(&st);
		hputi4(st.buf,"NFRAMES",n_frames);
		hashpipe_status_unlock_safe(&st);
        	}
	}
        // Mark output block as full and advance
        demo3_output_databuf_set_filled(db_out, curblock_out);
        curblock_out = (curblock_out + 1) % db_out->header.n_block;

        // Mark input block as free and advance
        demo3_input_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;
	mcnt++;
        /* Check for cancel */
        pthread_testcancel();
    }
    	CleanUp();
}

static hashpipe_thread_desc_t demo3_gpu_thread = {
    name: "demo3_gpu_thread",
    skey: "GPUSTAT",
    init: Init,
    //init: NULL,
    run:  run,
    ibuf_desc: {demo3_input_databuf_create},
    obuf_desc: {demo3_output_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&demo3_gpu_thread);
}
#ifdef __cplusplus
}
#endif
