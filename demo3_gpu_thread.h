/**
 * @file demo3_gpu_proc.h
 * Hashpipe Demo3
 *  Top-level header file
 *
 * @author Sparke Pei
 * @date 2017.05.08
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string.h>     // for memset(), strncpy(), memcpy(), strerror() 
#include <sys/types.h>  // for open() 
#include <sys/stat.h>   // for open() 
#include <fcntl.h>      // for open() 
#include <unistd.h>     // for close() and usleep() 
#include <cpgplot.h>    // for cpg*() 
#include <float.h>      // for FLT_MAX 
#include <getopt.h>     // for option parsing 
#include <assert.h>     // for assert() 
#include <errno.h>      // for errno 
#include <signal.h>     // for signal-handling 
#include <math.h>       // for log10f() in Plot() 
#include <sys/time.h>   // for gettimeofday() 
//#include "demo3_databuf.h"

#define FALSE               0
#define TRUE                1

#define LEN_GENSTRING       256

#define DEF_PFB_ON          TRUE

#define NUM_BYTES_PER_SAMP  1
#define DEF_LEN_SPEC        4096        // default value for g_iNFFT 
//#define DEF_LEN_SPEC        N_CHANS_PER_SPEC        // default value for g_iNFFT 

#define DEF_ACC             16           // default number of spectra to
//#define DEF_ACC             ACC_LEN           // default number of spectra to accumulate 
//#define DEF_SIZE_READ       33554432    // 32 MB - block size in VEGAS input buffer
//#define DEF_SIZE_READ       N_CHANS_PER_SPEC*N_POLS*ACC_LEN    
#define DEF_SIZE_READ       DEF_LEN_SPEC*DEF_ACC*4    
#define LEN_DATA            (NUM_BYTES_PER_SAMP * g_iNFFT)

// for PFB 
#define NUM_TAPS            8       // number of multiples of g_iNFFT 
#define FILE_COEFF_PREFIX   "coeff"
#define FILE_COEFF_DATATYPE "float"
#define FILE_COEFF_SUFFIX   ".dat"

#define DEF_NUM_SUBBANDS    1

#define FFTPLAN_RANK        1
#define FFTPLAN_ISTRIDE     (2 * g_iNumSubBands)
#define FFTPLAN_OSTRIDE     (2 * g_iNumSubBands)
#define FFTPLAN_IDIST       1
#define FFTPLAN_ODIST       1
#define FFTPLAN_BATCH       (2 * g_iNumSubBands)

#define USEC2SEC            1e-6

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char BYTE;

/**
 * Initialises the program.
 */
//int Init(void);

/**
 * Reads all data from the input file and loads it into memory.
 */
//int LoadDataToMem(void);

/**
 * Reads one block (32MB) of data form memory.
 */
//int ReadData(void);

/*
 * Perform polyphase filtering.
 *
 * @param[in]   pc4Data     Input data (raw data read from memory)
 * @param[out]  pf4FFTIn    Output data (input to FFT)
 */
__global__ void DoPFB(char4* pc4Data,
                      float4* pf4FFTIn,
                      float* pfPFBCoeff);
__global__ void CopyDataForFFT(char4* pc4Data,
                               float4* pf4FFTIn);
int DoFFT(void);
__global__ void Accumulate(float4 *pf4FFTOut,
                           float4* pfSumStokes);
void CleanUp(void);

#define CUDASafeCallWithCleanUp(iRet)   __CUDASafeCallWithCleanUp(iRet,       \
                                                                  __FILE__,   \
                                                                  __LINE__,   \
                                                                  &CleanUp)

void __CUDASafeCallWithCleanUp(cudaError_t iRet,
                               const char* pcFile,
                               const int iLine,
                               void (*pCleanUp)(void));

#if BENCHMARKING
void PrintBenchmarks(float fAvgPFB,
                     int iCountPFB,
                     float fAvgCpInFFt,
                     int iCountCpInFFT,
                     float fAvgFFT,
                     int iCountFFT,
                     float fAvgAccum,
                     int iCountAccum,
                     float fAvgCpOut,
                     int iCountCpOut);
#endif

/* PGPLOT function declarations */
int InitPlot(void);
void Plot(void);
//int writedata(void);
int RegisterSignalHandlers();
void HandleStopSignals(int iSigNo);
void PrintUsage(const char* pcProgName);



#ifdef __cplusplus
}
#endif


