#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <iostream>
using namespace std;
/**\brief the class to store the time series. 
\param ptim the pointer to the data
\param Nsamples Number of samples to the time series
\param Var the variance of the data
\param Mean the mean of the data
\param plink the pointer to a array of the TimeSeires. This is used to store the reuslts of box-cart filter
 */
//FILE * f_fil;

#ifdef __cplusplus
extern "C"{
#endif
class TimeSeries
{
public:
	float * ptim;
	long int Nsamples;
	long int Nbox;
	float Var;
	float Mean;
	TimeSeries * plink;
	bool New(long int n);
	void SetStat(void);
	TimeSeries(void);
	~TimeSeries(void);
	void Free(void);
};
/**\brief Class to sore the subband data
\param pFreq the array to store the central frequequency
*/
class SubBandData
{
public:
	/** Frequency table for the subband */
	double * pFreq;
	/** Number of channel */
	int Nchans;
	/** Number of samples*/
	long int Nsamples;
	/** Pointer to the data */
	float * pData;
	double RefDM;
	/** Formed tim series */
	TimeSeries * pTim;
	double * pDM;
	long int NDM;
	SubBandData(void);
	~SubBandData(void);
	void Free(void);
};
class FilterBankData
{
public:

	/// parameters to descrive how much data to read in
	long double Startpenc;					//starting percentage
	long double Endpenc;						//ending percentage
	long int StartSample;							//where is the start of the smaller file
	long int EndSample;								//where is the end of the smaller file
	long int Headersize;					//size of the header
	bool UseFrequencyTable;
	string FileName;
	//parameters from the filterbank file
	float * pData;										//pointer to the start of data
	SubBandData Obj;									//Store the object data, such as Dedisperse and FormDedisperseChannelData
	TimeSeries Obj1D;									//Store the object data, such as Dedisperse and FormDedisperseChannelData
	float * pZeroDM;									//pointer to the zero dm time series
	float Sig_0DM;
	SubBandData * pSubband;								//pSubband[k] is the k-th subband
	long int Nsubband;						//number of subband files
	long int Nchsubband;					//number of channel for a subband file
	
	TimeSeries * pDedispersed;						//pDedispersed[k] is the k-th time series
	long int * vBin;
	int Nbox;
	float * pDM;										//DM series
	long int Ndm;									//number of DM
	
	long int Nsamples;
	int Machine_id;
	int Telescope_id;
	int Data_type;
	int Nchans;
	int Nbits;
	int Nifs;
	int Nifs_ori;
	int Scan_number;
	int Barycentric;
	int Pulsarcentric;
	int Nbins;
	double Tstart;
	double Mjdobs;
	double Tsamp;
	double Tsamp_ori;
	double Fch1;
	double Foff;
	double RefDM;
	double Az_start;
	double Za_start;
	double Src_raj;
	double Src_dej;
	double Gal_l;
	double Gal_b;
	double Header_tobs;
	double Raw_fch1;
	double Raw_foff;
	double Period;
	int Nbeams;
	int Ibeam;
	double Srcl,Srcb;
	double Ast0, Lst0;
	long Wapp_scan_number;
	char Project[8];
	char Culprits[24];
	double Analog_power[2];
	char Rawdatafile[80];
	char Source_name[80];

	/* added frequency table for use with non-contiguous data */
	double * frequency_table; //by default it initialized as 16320
	long int npuls; /* added for binary pulse profile format */
	FilterBankData(void);
	~FilterBankData();
	void CloseFile(void);
	bool ReadInHeader(const string fname);
	void PrintHeader(void);
	bool ReadInData(long double startp=0.0, long double endp=1.0);
	bool ReadInDatabySample(long int startbyte=0, long int endbyte=-1);
	void DumpTxt(const char * fname);
	void ZeroDM(const string method="dot");
	void CleanChannel(double threshold);
	bool WriteToFile(const char * fname);
	bool WriteHeaderToFile(const char * fname);
	bool DownSample(int nd);
	bool SubbandDeDispersion(double startdk, double ddm, double enddm);
	bool WriteSubbandData(const char * fname);
	bool WriteTimData(const char * fname);
	bool BoxCarFilter(long int nbox, double minw, float snrloss);
	bool ApplyBoxFilter(float * ori, float var, float mean, long int wi, long int n, float * obj);
	bool FormZeroDMSeriesandStatistics(void);
	bool Dedisperse(double dmv);
	bool FormDedisperseChannelData(double dmv);
	bool Strip2OnePol(const int * poli, int npol);
	bool Strip2OnePol_STD(const int * poli, int npol);
	bool RemoveBaseline(void);
	bool Equalize(void);
	void Free(void);
private:
	bool dmsub2tim(SubBandData * psub, double ldmv, double dmstep, double rdmv);
	bool dedmdata2sub(SubBandData * psub, double dmv);
	void get_string (FILE * inputfile,  string & strtmp);
	void put_string (FILE * outputfile, const string & strtmp);
	int sizeof_file(const char * name) ;
	int nsamples( const char *filename,int hs, int nb, int ni, int nch);
	bool readdata(long int ns);
	FILE * fp;
};
//extern double current_MJD;

/*void WriteHeader(const char * fname, double net_MJD);

void WriteData(const char * fname);
*/

#ifdef __cplusplus
}
#endif
