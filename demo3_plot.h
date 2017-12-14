/**
 * @file tut5_plot.h
 * CASPER Tutorial 5: Heterogeneous Instrumentation
 *  Header file for plotting
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

//#ifndef __TUT5_PLOT_H__
//#define __TUT5_PLOT_H__

#include <stdio.h>
#include <stdlib.h>
#include <cpgplot.h>    /* for cpg*() */
#include <math.h>       /* for log10f() in Plot() */
#include <errno.h>      /* for errno */
#include <float.h>      /* for FLT_MAX */

//#include "demo3_gpu_proc.h"
//#include "demo3_databuf.h"

#define PG_DEV              "1/XS"
#define PG_VP_ML            0.10    /* left margin */
#define PG_VP_MR            0.90    /* right margin */
#define PG_VP_MB            0.15    /* bottom margin */
#define PG_VP_MT            0.98    /* top margin */
#define PG_SYMBOL           2
#define PG_CI_DEF           1
#define PG_CI_PLOT          11
#define PG_SIZE_LABEL       4
#define PG_SIZE_DEF         3

#ifdef __cplusplus 
extern "C" {     
#endif

int InitPlot(void);
void Plot(void);

#ifdef __cplusplus 
} 
#endif
//#endif  /* __TUT5_PLOT_H__ */

