/*
 * demo3_output_thread.c
 * Get the data from output databuffer and then write them to file.
 */


#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "hashpipe.h"
#include "demo3_databuf.h"
#include "demo3_output_thread.h"
//#include "filterbank.h"
//#include "cuda.h"
extern double MJD;

static void *run(hashpipe_thread_args_t * args)
{
	// Local aliases to shorten access to args fields
	// Our input buffer happens to be a demo3_ouput_databuf
	demo3_output_databuf_t *db = (demo3_output_databuf_t *)args->ibuf;
	hashpipe_status_t st = args->st;
	const char * status_key = args->thread_desc->skey;
	int c,rv;
	int block_idx = 0;
	uint64_t mcnt=0;
	int f_full_flag = 1;
	char f_fil[256];
	struct tm        *now;
	time_t           rawtime;
    	double  FILE_SIZE_MB = 1; // MB
	double  FILE_SIZE_NOW_MB = 0;

	FILE * demo3_file;
	while (run_threads()) {

		hashpipe_status_lock_safe(&st);
		hputi4(st.buf, "OUTBLKIN", block_idx);
		hputi8(st.buf, "OUTMCNT",mcnt);
		hputs(st.buf, status_key, "waiting");
		hashpipe_status_unlock_safe(&st);

		// get new data
		while ((rv=demo3_output_databuf_wait_filled(db, block_idx))
		!= HASHPIPE_OK) {
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

		hashpipe_status_lock_safe(&st);
		hputs(st.buf, status_key, "processing");
		hashpipe_status_unlock_safe(&st);

        if (f_full_flag ==1){
            char    f_fil[256];
            struct tm  *now;
            time_t rawtime;

            printf("\n\nopen new filterbank file...\n\n");
            time(&rawtime);
			now = localtime(&rawtime);
			strftime(f_fil,sizeof(f_fil), "./data_%Y-%m-%d_%H-%M-%S.fil",now);
			WriteHeader(f_fil,MJD);
			printf("file name is: %s\n",f_fil);
			printf("write header done!\n");
			demo3_file=fopen(f_fil,"a+");
			printf("starting write data...\n");	

        }

        FILE_SIZE_NOW_MB += SIZEOF_OUT_STOKES*sizeof(float)/1024/1024.0;
	//printf("FILE_SIZE_NOW_MB is %lf\n",FILE_SIZE_NOW_MB);
        if (FILE_SIZE_NOW_MB >= FILE_SIZE_MB){
			f_full_flag = 1;
			FILE_SIZE_NOW_MB = 0;
        }
        else{f_full_flag = 0;}

		fwrite(db->block[block_idx].Stokes_Full,sizeof(float),SIZEOF_OUT_STOKES,demo3_file);
		demo3_output_databuf_set_free(db,block_idx);
		block_idx = (block_idx + 1) % db->header.n_block;
		mcnt++;

		//Will exit if thread has been cancelled
		pthread_testcancel();

	}
	fclose(demo3_file);
	return THREAD_OK;
}

static hashpipe_thread_desc_t demo3_output_thread = {
    name: "demo3_output_thread",
    skey: "OUTSTAT",
    init: NULL, 
    run:  run,
    ibuf_desc: {demo3_output_databuf_create},
    obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&demo3_output_thread);
}

