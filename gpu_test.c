#include <stdio.h>
#include <stdlib.h>
//#include "demo3_databuf.h"
#include "demo3_gpu_proc.h"
#include "demo3_plot.h"
//#define N_CHANS_PER_SPEC        4096    // number of FFT channels per spectrum
//#define N_POLS                  4       // number of polarizations
//#define N_IFS                   4       // number of IFs per Stokes

int main(){
	int nhits = 0;
	int iRet=EXIT_SUCCESS;
    	int iFileData = 0;
	int g_iSizeFile = 0;
        char data_raw[DEF_SIZE_READ]; // raw data will be feed to gpu thread
	//data_raw = (char *)malloc(N_CHANS_PER_SPEC*N_POLS*sizeof(char));
	float full_stokes[DEF_SIZE_READ]; // full stokes data returned from gpu thread
	//full_stokes = (float *)malloc(N_CHANS_PER_SPEC*N_IFS*sizeof(float));
    // initialise 
	/*printf("memory initialise...");
    iRet = Init();

    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! Init failed!\n");
        CleanUp();
    }*/
	printf("done!\n");
	printf("plot initialise...");
	iRet = InitPlot();
	if (iRet != EXIT_SUCCESS)
	{
	        (void) fprintf(stderr,
	                       "ERROR: Plotting initialisation failed!\n");
	        return EXIT_FAILURE;
	}
	printf("done!\n");
	

	iFileData = open("file0000", O_RDONLY);
	for(int j=0;j<10;j++){
		read(iFileData, data_raw, DEF_SIZE_READ);
		
		printf("number %d outer loop\n",j);
		//printf("\nraw time domain data is: \n");
		/*for (int i=0;i<DEF_SIZE_READ;i++){
        		//data_raw[i] = i%257;
			printf("%c ",data_raw[i]);
		}*/
		// call gpu processing function
		nhits = gpu_proc(DEF_SIZE_READ,data_raw,full_stokes);
		}
        /* NOTE: Plot() will modify data! */
        //Plot();

	// free memory
	//free(data_raw);
	//free(full_stokes);
   	(void) close(iFileData);
	CleanUp();
	return 0;



}

