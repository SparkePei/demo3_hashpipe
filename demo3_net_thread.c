/*
 * demo3_net_thread.c
 *
 * This allows you to receive pakets from local ethernet, and then write them into a shared memory buffer. 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include "hashpipe.h"
#include "demo3_databuf.h"
//#include "cuda.h"

double MJD;
//defining a struct of type hashpipe_udp_params as defined in hashpipe_udp.h
static struct hashpipe_udp_params params;

static int init(hashpipe_thread_args_t * args)
{
        hashpipe_status_t st = args->st;
        strcpy(params.bindhost,"127.0.0.2");
        //selecting a port to listen to
        params.bindport = 5009;
        params.packet_size = 0;
        hashpipe_udp_init(&params);
        hashpipe_status_lock_safe(&st);
        hputi8(st.buf, "NPACKETS", 0);
        hputi8(st.buf, "NBYTES", 0);
        hashpipe_status_unlock_safe(&st);
        return 0;

}

double UTC2JD(double year, double month, double day){
        double jd;
        double a;
        a = floor((14-month)/12);
        year = year+4800-a;
        month = month+12*a-3;
        jd = day + floor((153*month+2)/5)+365*year+floor(year/4)-floor(year/100)+floor(year/400)-32045;
        return jd;
}

static void *run(hashpipe_thread_args_t * args){

    	demo3_input_databuf_t *db  = (demo3_input_databuf_t *)args->obuf;
    	hashpipe_status_t st = args->st;
    	const char * status_key = args->thread_desc->skey;

    	/* Main loop */
    	int i, rv,input,n;
    	uint64_t mcnt = 0;
    	int block_idx = 0;
        unsigned long header; // 64 bit counter     
        unsigned char data_pkt[N_BYTES_PER_PKT]; // save received packet
	//unsigned char data_tmp[N_CHANS_PER_SPEC]; // for  temporary data storage
        unsigned long SEQ=0;
        unsigned long LAST_SEQ=0;
        unsigned long CHANNEL;
        unsigned long n_pkt_rcv; // number of packets has been received
        unsigned long pkt_loss; // number of packets has been lost
        int first_pkt=1;
        double pkt_loss_rate; // packets lost rate

	double Year, Month, Day;
	double jd;
	//double MJD;
	time_t timep;
	struct tm *p;
	struct timeval currenttime;
	time(&timep);
	p=gmtime(&timep);
	Year=p->tm_year+1900;
	Month=p->tm_mon+1;
	Day=p->tm_mday;
	jd = UTC2JD(Year, Month, Day); 
	MJD=jd+(double)((p->tm_hour-12)/24.0)
                               +(double)(p->tm_min/1440.0)
                               +(double)(p->tm_sec/86400.0)
                               +(double)(currenttime.tv_usec/86400.0/1000000.0)
				-(double)2400000.5;
	printf("MJD time of packets is %lf\n",MJD);

    	uint64_t npackets = 0; //number of received packets
    	uint64_t nbytes = 0;  //number of received bytes

    	while (run_threads()){

        	hashpipe_status_lock_safe(&st);
        	hputs(st.buf, status_key, "waiting");
        	hputi4(st.buf, "NETBKOUT", block_idx);
		hputi8(st.buf,"NETMCNT",mcnt);
        	hashpipe_status_unlock_safe(&st);
 
        	// Wait for data
        	/* Wait for new block to be free, then clear it
        	 * if necessary and fill its header with new values.
        	 */
        	while ((rv=demo3_input_databuf_wait_free(db, block_idx)) 
        	        != HASHPIPE_OK) {
        	    if (rv==HASHPIPE_TIMEOUT) {
        	        hashpipe_status_lock_safe(&st);
        	        hputs(st.buf, status_key, "blocked");
        	        hashpipe_status_unlock_safe(&st);
        	        continue;
        	    } else {
        	        hashpipe_error(__FUNCTION__, "error waiting for free databuf");
        	        pthread_exit(NULL);
        	        break;
        	    }
        	}

        	hashpipe_status_lock_safe(&st);
        	hputs(st.buf, status_key, "receiving");
        	hashpipe_status_unlock_safe(&st);
		for(int i=0;i<PAGE_SIZE*N_PKTS_PER_SPEC;i){
		        n = recvfrom(params.sock,data_pkt,N_BYTES_PER_PKT*sizeof(unsigned char),0,0,0);
			if(n>0){
				i++;
				npackets++;
				nbytes += n;
				printf("received %lu bytes of number %lu packets\n",nbytes,npackets);
				// neglect the packets loss if it is the first block
				if(mcnt == 0){
        	                        pkt_loss=0;
        	                        //LAST_SEQ = SEQ;
					}
				else{
        		                //pkt_loss += SEQ - (LAST_SEQ+1);
        	        	        //pkt_loss_rate = (double)pkt_loss/(double)npackets*100.0;
        	                	//LAST_SEQ = SEQ;
				}


				// copy data to input data buffer		              
				printf("db->block[block_idx]+%d\n",(i%(PAGE_SIZE*N_PKTS_PER_SPEC))*N_BYTES_PER_PKT); 
				memcpy(db->block[block_idx].data_block+(i%(PAGE_SIZE*N_PKTS_PER_SPEC))*N_BYTES_PER_PKT,data_pkt,N_BYTES_PER_PKT*sizeof(unsigned char));
	      			hashpipe_status_lock_safe(&st);
        			hputi8(st.buf, "NPACKETS", npackets);
			        hputi8(st.buf, "NBYTES", nbytes);
				hputi8(st.buf,"PKTLOSS",pkt_loss);
		      		hputr8(st.buf,"LOSSRATE",pkt_loss_rate);
			      	hashpipe_status_unlock_safe(&st);
			}

        		/* Will exit if thread has been cancelled */
	        	pthread_testcancel();
		}
	
		// Mark block as full
		if(demo3_input_databuf_set_filled(db, block_idx) != HASHPIPE_OK) {
			hashpipe_error(__FUNCTION__, "error waiting for databuf filled call");
		        pthread_exit(NULL);
		}
		db->block[block_idx].header.mcnt = mcnt;
        	block_idx = (block_idx + 1) % db->header.n_block;
		mcnt++;
        	/* Will exit if thread has been cancelled */
        	pthread_testcancel();
    	}
    	// Thread success!
	return THREAD_OK;
}

static hashpipe_thread_desc_t demo3_net_thread = {
    name: "demo3_net_thread",
    skey: "NETSTAT",
    init: init,
    run:  run,
    ibuf_desc: {NULL},
    obuf_desc: {demo3_input_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&demo3_net_thread);
}
