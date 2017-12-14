/**
 * @file demo3_gpu_proc.cu
 * Hashpipe Demo3
 *  Top-level header file
 *
 * @author Sparke Pei
 * @date 2017.05.11
 */

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
#include "cuda_runtime.h"
#include <cufft.h>
#include <time.h>
#include "demo3_plot.h"

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
int g_iNFFT = N_CHANS_PER_SPEC;
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
int g_iIsPFBOn = 1;
int g_iNTaps = 1;                       /* 1 if no PFB, NUM_TAPS if PFB */
/* BUG: crash if file size is less than 32MB */
int g_iSizeRead = DEF_SIZE_READ/4;
int g_iNumSubBands = 1;
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};
float *g_pfPFBCoeff = NULL;
float *g_pfPFBCoeff_d = NULL;

int Init()
{
    int iDevCount = 0;
    int iRet = EXIT_SUCCESS;
    int iMaxThreadsPerBlock = 0;
    cudaDeviceProp stDevProp = {0};
    cufftResult iCUFFTRet = CUFFT_SUCCESS;


    g_pc4InBufRead=(char4 *)malloc(g_iNFFT*sizeof(char4));
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Signal-handler registration failed!\n");
        return EXIT_FAILURE;
    }   

    /* since CUDASafeCallWithCleanUp() calls cudaGetErrorString(),
       it should not be used here - will cause crash if no CUDA device is
       found */
    /*(void) cudaGetDeviceCount(&iDevCount);
    if (0 == iDevCount)
    {
        (void) fprintf(stderr, "ERROR: No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }*/

    /* just use the first device */
    //CUDASafeCallWithCleanUp(cudaSetDevice(0));

    cudaGetDeviceProperties(&stDevProp, 0);
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
        cudaMalloc((void **) &g_pfPFBCoeff_d,
                                           g_iNumSubBands
                                           * g_iNTaps
                                           * g_iNFFT
                                           * sizeof(float));

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
        cudaMemcpy(g_pfPFBCoeff_d,
                   g_pfPFBCoeff,
                   g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float),
                   cudaMemcpyHostToDevice);

    }

    /* allocate memory for data array - 32MB is the block size for the VEGAS
       input buffer */
    cudaMalloc((void **) &g_pc4Data_d, g_iSizeRead*sizeof(char4));
    // g_pc4DataRead_d = g_pc4Data_d;
     g_pc4Data_d = g_pc4DataRead_d;
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
    cudaMalloc((void **) &g_pf4FFTIn_d,
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4));
    cudaMalloc((void **) &g_pf4FFTOut_d,
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4));

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
    cudaMalloc((void **) &g_pf4SumStokes_d,
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4));
    cudaMemset(g_pf4SumStokes_d,
                                       '\0',
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4));
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
   return EXIT_SUCCESS;
/*
    iRet = InitPlot();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Plotting initialisation failed!\n");
        return EXIT_FAILURE;
    }
*/

    return EXIT_SUCCESS;
}
/* function that frees resources */
void CleanUp()
{
    // free resources 
    /*if (g_pc4InBuf != NULL)
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
    }*/
	if (g_pc4DataRead_d != NULL)
	{
	(void) cudaFree(g_pc4DataRead_d);
	g_pc4DataRead_d = NULL;
	}
    /*if (g_pf4SumStokes != NULL)
    {
        free(g_pf4SumStokes);
        g_pf4SumStokes = NULL;
    }
    if (g_pf4SumStokes_d != NULL)
    {
        (void) cudaFree(g_pf4SumStokes_d);
        g_pf4SumStokes_d = NULL;
    }*/
    free(g_pfPFBCoeff);
    (void) cudaFree(g_pfPFBCoeff_d);

    /* destroy plan */
    /* TODO: check for plan */
    (void) cufftDestroy(g_stPlan);

    /*if (g_pfSumPowX != NULL)
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
    }*/
    if (g_pfFreq != NULL)
    {
        free(g_pfFreq);
        g_pfFreq = NULL;
    }

    // TODO: check if open
    cpgclos();

    return;
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

void CUDASafeCallWithCleanUp(cudaError_t iRet,
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

/* function that performs the PFB */
__global__ void DoPFB(char4 *pc4Data,
                      float4 *pf4FFTIn,
                      float *pfPFBCoeff)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iNFFT = (gridDim.x * blockDim.x);
    int j = 0;
    int iAbsIdx = 0;
    float4 f4PFBOut = make_float4(0.0, 0.0, 0.0, 0.0);
    char4 c4Data = make_char4(0, 0, 0, 0);
    for (j = 0; j < NUM_TAPS; ++j)
    {
        /* calculate the absolute index */
        iAbsIdx = (j * iNFFT) + i;
        /* get the address of the block */
        c4Data = pc4Data[iAbsIdx];
	//printf("%d ",iAbsIdx);
        f4PFBOut.x += (float) c4Data.x * pfPFBCoeff[iAbsIdx];
        f4PFBOut.y += (float) c4Data.y * pfPFBCoeff[iAbsIdx];
        f4PFBOut.z += (float) c4Data.z * pfPFBCoeff[iAbsIdx];
        f4PFBOut.w += (float) c4Data.w * pfPFBCoeff[iAbsIdx];
    }

    pf4FFTIn[i] = f4PFBOut;

    return;
}
__global__ void CopyDataForFFT(char4 *pc4Data,
                               float4 *pf4FFTIn)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    pf4FFTIn[i].x = (float) pc4Data[i].x;
    pf4FFTIn[i].y = (float) pc4Data[i].y;
    pf4FFTIn[i].z = (float) pc4Data[i].z;
    pf4FFTIn[i].w = (float) pc4Data[i].w;

    return;
}

/* function that performs the FFT - not a kernel, just a wrapper to an
   API call */
int DoFFT()
{
    cufftResult iCUFFTRet = CUFFT_SUCCESS;

    /* execute plan */
    iCUFFTRet = cufftExecC2C(g_stPlan,
                             (cufftComplex*) g_pf4FFTIn_d,
                             (cufftComplex*) g_pf4FFTOut_d,
                             CUFFT_FORWARD);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT for polarisation X failed!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    float4 f4FFTOut = pf4FFTOut[i];
    float4 f4SumStokes = pf4SumStokes[i];

    /* Re(X)^2 + Im(X)^2 */
    f4SumStokes.x += (f4FFTOut.x * f4FFTOut.x)
                         + (f4FFTOut.y * f4FFTOut.y);
    /* Re(Y)^2 + Im(Y)^2 */
    f4SumStokes.y += (f4FFTOut.z * f4FFTOut.z)
                         + (f4FFTOut.w * f4FFTOut.w);
    /* Re(XY*) */
    f4SumStokes.z += (f4FFTOut.x * f4FFTOut.z)
                         + (f4FFTOut.y * f4FFTOut.w);
    /* Im(XY*) */
    f4SumStokes.w += (f4FFTOut.y * f4FFTOut.z)
                         - (f4FFTOut.x * f4FFTOut.w);

    pf4SumStokes[i] = f4SumStokes;
}

int gpu_proc(int g_iFFT,char *g_data,float *full_stokes){
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

    /* get the filename of the program from the argument list */
    //pcProgName = argv[0];
    /* initialise */
    iRet = Init();

    if (iRet != EXIT_SUCCESS)
    {   
        (void) fprintf(stderr, "ERROR! Init failed!\n");
        CleanUp();
    }

    (void) gettimeofday(&stStart, NULL);
	
	cudaMalloc((void **) &g_pc4DataRead_d,g_iSizeRead*sizeof(char4));
        //copy data from cpu memory to gpu memory
	printf("copy raw data from host to device...,");
	memcpy(g_pc4InBufRead,g_data,g_iSizeRead*sizeof(char4));
        cudaMemcpy(g_pc4DataRead_d,g_pc4InBufRead,g_iSizeRead*sizeof(char4),cudaMemcpyHostToDevice);
	printf("done!\n");

	for (int k=0;k<g_iSizeRead/g_iNFFT;k++)
	{

        	/* do pfb */
		printf("do pfb processing...,");
        	DoPFB<<<g_dimGPFB, g_dimBPFB>>>(g_pc4DataRead_d,g_pf4FFTIn_d,g_pfPFBCoeff_d);
        	cudaThreadSynchronize();
        	iCUDARet = cudaGetLastError();
        	if (iCUDARet != cudaSuccess)
        	{
			printf("ERROR,iCUDARet is: %d\n",iCUDARet);
        	    (void) fprintf(stderr,
        	                       "ERROR: File <%s>, Line %d: %s\n",
        	                       __FILE__,
        	                       __LINE__,
        	                       cudaGetErrorString(iCUDARet));
        	    /* free resources */
        	    CleanUp();
        	}
		printf("done!\n");

        	/* do fft */
		printf("do fft...,");
        	iRet = DoFFT();
        	if (iRet != EXIT_SUCCESS)
        	{
                        printf("ERROR,iCUDARet is: %d\n",iCUDARet);
       			(void) fprintf(stderr, "ERROR! FFT failed!\n");
        	    CleanUp();
        	}
		printf("done!\n");

        	// do Stokes calculation and accumulation
		printf("do stokes calculation and accumulation...,");
        	Accumulate<<<g_dimGAccum, g_dimBAccum>>>(g_pf4FFTOut_d,g_pf4SumStokes_d);
		printf("done!\n");

        	//copy accumulation data from gpu to cpu memory
		printf("copy accumulation data from gpu to cpu memory...,");
		cudaMemcpy(g_pf4SumStokes,g_pf4SumStokes_d,(g_iNumSubBands* g_iNFFT* sizeof(float4)),cudaMemcpyDeviceToHost);
        	    /* dump to buffer 
        	    CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes,
        	                                       g_pf4SumStokes_d,
        	                                       (g_iNumSubBands
        	                                        * g_iNFFT
        	                                        * sizeof(float4)),
        	                                        cudaMemcpyDeviceToHost));*/
		printf("done!\n");

		/* NOTE: Plot() will modify data! */
		printf("number %d of plot.\n",k);
        	Plot();
        	(void) usleep(500000);

		//if (k<g_iSizeRead/g_iNFFT-1){g_pc4DataRead_d += (g_iNumSubBands * g_iNFFT);}
		g_pc4DataRead_d += g_iNFFT;

        	cudaThreadSynchronize();
        	iCUDARet = cudaGetLastError();
        	if (iCUDARet != cudaSuccess)
        		{
                        printf("ERROR,iCUDARet is: %d\n",iCUDARet);
        	    (void) fprintf(stderr,
        	                   "ERROR: File <%s>, Line %d: %s\n",
        	                   __FILE__,
        	                   __LINE__,
        	                   cudaGetErrorString(iCUDARet));
        	    // free resources 
        	    CleanUp();
			}
		}
	CleanUp();
	return 0;
}
