#include <stdint.h>
#include <stdio.h>
#include "hashpipe.h"
#include "hashpipe_databuf.h"
//#include <cuda.h>
//#include <cufft.h>

#define CACHE_ALIGNMENT         512	// cache alignment size	
#define N_INPUT_BLOCKS          3 	// number of input blocks
#define N_OUTPUT_BLOCKS         3	// number of output blocks
#define PAGE_SIZE	      	64	// number of spectra per memory, define memory size
#define N_CHANS_PER_SPEC	4096	// number of FFT channels per spectrum
#define N_BYTES_PER_SAMPLE	1	// number of bytes per sample
#define N_BEAMS			1	// number of beams
#define N_PKTS_PER_SPEC         4	// number packets per spectrum
#define N_POLS                  4       // number of polarizations
#define N_IFS			4	// number of IFs per Stokes
#define N_BYTES_HEAD		0	// number bytes of header in packets
#define N_BYTES_PER_PKT		4096	// number bytes per packets
//#define ACC_LEN			1024 // accumulation length
#define ACC_LEN			16 // accumulation length
#define SIZEOF_INPUT_DATA_BUF	N_BYTES_PER_PKT*N_BYTES_PER_SAMPLE*PAGE_SIZE*N_PKTS_PER_SPEC
#define SIZEOF_OUT_STOKES	PAGE_SIZE*N_CHANS_PER_SPEC/ACC_LEN*N_IFS
// Used to pad after hashpipe_databuf_t to maintain cache alignment
typedef uint8_t hashpipe_databuf_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(hashpipe_databuf_t)%CACHE_ALIGNMENT)
];

/* INPUT BUFFER STRUCTURES
  */
typedef struct demo3_input_block_header {
   uint64_t mcnt;                    // mcount of first packet
} demo3_input_block_header_t;

typedef uint8_t demo3_input_header_cache_alignment[
   CACHE_ALIGNMENT - (sizeof(demo3_input_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct demo3_input_block {
   demo3_input_block_header_t header;
   demo3_input_header_cache_alignment padding; // Maintain cache alignment
   char data_block[SIZEOF_INPUT_DATA_BUF*sizeof(char)]; // define input buffer
} demo3_input_block_t;

typedef struct demo3_input_databuf {
   hashpipe_databuf_t header;
   hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
   demo3_input_block_t block[N_INPUT_BLOCKS];
} demo3_input_databuf_t;


/*
  * OUTPUT BUFFER STRUCTURES
  */
typedef struct demo3_output_block_header {
   uint64_t mcnt;
} demo3_output_block_header_t;

typedef uint8_t demo3_output_header_cache_alignment[
   CACHE_ALIGNMENT - (sizeof(demo3_output_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct demo3_output_block {
   demo3_output_block_header_t header;
   demo3_output_header_cache_alignment padding; // Maintain cache alignment
// define full stokes
   float Stokes_Full[SIZEOF_OUT_STOKES*sizeof(float)];
/*	float Stokes_I[N_CHANS_PER_SPEC];
	float Stokes_Q[N_CHANS_PER_SPEC];
	float Stokes_U[N_CHANS_PER_SPEC];
	float Stokes_V[N_CHANS_PER_SPEC];
*/
} demo3_output_block_t;

typedef struct demo3_output_databuf {
   hashpipe_databuf_t header;
   hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
   demo3_output_block_t block[N_OUTPUT_BLOCKS];
} demo3_output_databuf_t;

/*
 * INPUT BUFFER FUNCTIONS
 */
hashpipe_databuf_t *demo3_input_databuf_create(int instance_id, int databuf_id);

static inline demo3_input_databuf_t *demo3_input_databuf_attach(int instance_id, int databuf_id)
{
    return (demo3_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int demo3_input_databuf_detach(demo3_input_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline void demo3_input_databuf_clear(demo3_input_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline int demo3_input_databuf_block_status(demo3_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_input_databuf_total_status(demo3_input_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int demo3_input_databuf_wait_free(demo3_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_input_databuf_busywait_free(demo3_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_input_databuf_wait_filled(demo3_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_input_databuf_busywait_filled(demo3_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_input_databuf_set_free(demo3_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_input_databuf_set_filled(demo3_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

/*
 * OUTPUT BUFFER FUNCTIONS
 */

hashpipe_databuf_t *demo3_output_databuf_create(int instance_id, int databuf_id);

static inline void demo3_output_databuf_clear(demo3_output_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline demo3_output_databuf_t *demo3_output_databuf_attach(int instance_id, int databuf_id)
{
    return (demo3_output_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int demo3_output_databuf_detach(demo3_output_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline int demo3_output_databuf_block_status(demo3_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_output_databuf_total_status(demo3_output_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int demo3_output_databuf_wait_free(demo3_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_output_databuf_busywait_free(demo3_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}
static inline int demo3_output_databuf_wait_filled(demo3_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_output_databuf_busywait_filled(demo3_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_output_databuf_set_free(demo3_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int demo3_output_databuf_set_filled(demo3_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}


