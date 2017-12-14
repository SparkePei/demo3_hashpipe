/**\brief
 *
 * =====================================================================================
 *
 *       Filename:  filterbank.cpp
 *
 *    Description:  This is the implementation of the FilterBankData class
 *
 *        Version:  1.0
 *        Created:  2014年12月08日 20时50分33秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *   Organization:
 *
 * =====================================================================================
 */
#include "filterbank.h"
#include "demo3_output_thread.h"
#include "time.h"
#include "sys/time.h"
#include <math.h>
#include <iostream>
using namespace std;
#define _CHAR_SWAP_SIZE 256

double UTCtoJD(double year, double month, double day){
        double jd;
        double a;
        a = floor((14-month)/12);
        year = year+4800-a;
        month = month+12*a-3;
        jd = day + floor((153*month+2)/5)+365*year+floor(year/4)-floor(year/100)+floor(year/400)-32045;
        return jd;
}


TimeSeries::TimeSeries(void)
{
	ptim=NULL;
	Nsamples=0;
	Nbox=0;
	plink=NULL;
}
TimeSeries::~TimeSeries(void)
{
	Free();
}
bool TimeSeries::New(long int n)
{
	Nsamples=n;
	ptim=new float[n];
	if (ptim==NULL ) return false;
	else return true;
}
void TimeSeries::Free(void)
{
	if (ptim!=NULL)
	{
		delete [] ptim;
		ptim=NULL;
	}
	if (plink!=NULL)
	{
		for (long int i=0; i<Nbox; i++)
			(plink+i)->Free();
		delete [] plink;
		plink=NULL;
	}
}
void TimeSeries::SetStat(void)
{
	float a,b;
	a=0;
	b=0;
	long int n=Nsamples;
	for (long int i=0; i<n; i++)
	{
		a+=ptim[i]*ptim[i];
		b+=ptim[i];
	}
	Var=(a-b*b/n)/n;
	Mean=b/n;
}

SubBandData::SubBandData(void)
{
	Nchans=0;
	Nsamples=0;
	pData=NULL;
	pFreq=NULL;
	pTim=NULL;
	pDM=NULL;
}
void SubBandData::Free(void)
{
	if (pFreq!=NULL)
	{
		delete [] pFreq;
		pFreq=NULL;

	}
	if (pData!=NULL)
	{
		delete [] pData;
		pData=NULL;

	}
	if (pDM!=NULL)
	{
		delete [] pDM;
		pDM=NULL;
	}
	if (pTim!=NULL)
	{
		for (long int j=0; j<NDM; j++)
			(pTim+j)->Free();
		delete [] pTim;
		pTim=NULL;
	}
}
SubBandData::~SubBandData(void)
{
	Free();
}
FilterBankData::FilterBankData(void)
{
	frequency_table=new double [16320];
	UseFrequencyTable=false;
	pData=NULL;
	pDM=NULL;
	pSubband=NULL;
	pZeroDM=NULL;
	pDedispersed=NULL;
	vBin=NULL;
	strcpy(Source_name, "J0000-0000");
	strcpy(Rawdatafile, "J0000-0000");
	strcpy(Project, "BEAR");
	strcpy(Culprits, "Husky");
}
FilterBankData::~FilterBankData()
{
	if (frequency_table!=NULL)
	delete [] frequency_table;
	Free();
}

/** \brief get a string from the header of *.fil file.
 *
 * This code is stealed from sigproc
 */
void FilterBankData::get_string (FILE * inputfile, string & strtmp)
{
	int nchar;
	strtmp = "ERROR";
	fread (&nchar, sizeof (int), 1, inputfile);
	if (feof (inputfile))
	{
		cerr << "Error in reading the header of File" << endl;
		exit (0);
	}
	if (nchar > 80 || nchar < 1)
		return;
	char chrtmp[_CHAR_SWAP_SIZE];
	fread (chrtmp, nchar, 1, inputfile);
	chrtmp[nchar] = '\0';
	strtmp = chrtmp;
}

void FilterBankData::put_string (FILE * outputfile, const string & strtmp)
{
	int nchar = strtmp.length ();
	fwrite (&nchar, sizeof (int), 1, outputfile);
	fwrite (strtmp.c_str (), nchar, 1, outputfile);
}


int FilterBankData::sizeof_file(const char name[]) /* includefile */
{
	struct stat stbuf;

	if(stat(name,&stbuf) == -1)
	{
		fprintf(stderr, "f_siz: can't access %s\n",name);
		exit(0);
	}
	return(stbuf.st_size);
}

int FilterBankData::nsamples(const char *filename,int headersize, int nbits, int nifs, int nchans)
{
	int datasize,numsamps;
	datasize=sizeof_file(filename)-headersize;
	numsamps=(int) ((long double) (datasize)/ (((long double) nbits) / 8.0)
	                /(long double) nifs/(long double) nchans);
	return(numsamps);
}

bool FilterBankData::ReadInHeader(const string fname)
{
	int expecting_rawdatafile = 0, expecting_source_name = 0;
	int nsamp;
	int expecting_frequency_table = 0, channel_index = 0;
	FILE *fpInput = fopen (fname.c_str (), "rb");
	FileName=fname;
	string strtmp;
	long int intTotalHeaderBytes = 0;
	get_string (fpInput, strtmp);
	if (strtmp != "HEADER_START")
	{
		cerr << "Non-Standard file format." << endl;
		return (false);
	}
	intTotalHeaderBytes += strtmp.length () + 4;
	while (1)
	{
		get_string (fpInput, strtmp);
		if (strtmp == "HEADER_END")
		{
			intTotalHeaderBytes += strtmp.length () + 4;
			break;
		}
		intTotalHeaderBytes += strtmp.length () + 4;
		if (strtmp == "rawdatafile")
		{
			expecting_rawdatafile = 1;
		}
		else if (strtmp == "source_name")
		{
			expecting_source_name = 1;
		}
		else if (strtmp == "FREQUENCY_START")
		{
			expecting_frequency_table = 1;
			UseFrequencyTable=true;
			channel_index = 0;
		}
		else if (strtmp == "FREQUENCY_END")
		{
			expecting_frequency_table = 0;
			Nchans=channel_index;
		}
		else if (strtmp == "az_start")
		{
			fread (&Az_start, sizeof (Az_start), 1, fpInput);
			intTotalHeaderBytes += sizeof (Az_start);
		}
		else if (strtmp == "za_start")
		{
			fread (&Za_start, sizeof (Za_start), 1, fpInput);
			intTotalHeaderBytes += sizeof (Za_start);
		}
		else if (strtmp == "src_raj")
		{
			fread (&Src_raj, sizeof (Src_raj), 1, fpInput);
			intTotalHeaderBytes += sizeof (Src_raj);
		}
		else if (strtmp == "src_dej")
		{
			fread (&Src_dej, sizeof (Src_dej), 1, fpInput);
			intTotalHeaderBytes += sizeof (Src_dej);
		}
		else if (strtmp == "tstart")
		{
			fread (&Tstart, sizeof (Tstart), 1, fpInput);
			intTotalHeaderBytes += sizeof (Tstart);
		}
		else if (strtmp == "tsamp")
		{
			fread (&Tsamp, sizeof (Tsamp), 1, fpInput);
			intTotalHeaderBytes += sizeof (Tsamp);
		}
		else if (strtmp == "period")
		{
			fread (&Period, sizeof (Period), 1, fpInput);
			intTotalHeaderBytes += sizeof (Period);
		}
		else if (strtmp == "fch1")
		{
			fread (&Fch1, sizeof (Fch1), 1, fpInput);
			intTotalHeaderBytes += sizeof (Fch1);
		}
		else if (strtmp == "fchannel")
		{
			fread (&frequency_table[channel_index], sizeof (double), 1, fpInput);
			intTotalHeaderBytes += sizeof (double);
			Fch1 = Foff = 0.0;				/* set to 0.0 to signify that a table is in use */
			UseFrequencyTable=true;
			channel_index++;
		}
		else if (strtmp == "foff")
		{
			fread (&Foff, sizeof (Foff), 1, fpInput);
			intTotalHeaderBytes += sizeof (Foff);
		}
		else if (strtmp == "nchans")
		{
			fread (&Nchans, sizeof (Nchans), 1, fpInput);
			intTotalHeaderBytes += sizeof (Nchans);
		}
		else if (strtmp == "telescope_id")
		{
			fread (&Telescope_id, sizeof (Telescope_id), 1, fpInput);
			intTotalHeaderBytes += sizeof (Telescope_id);
		}
		else if (strtmp == "machine_id")
		{
			fread (&Machine_id, sizeof (Machine_id), 1, fpInput);
			intTotalHeaderBytes += sizeof (Machine_id);
		}
		else if (strtmp == "data_type")
		{
			fread (&Data_type, sizeof (Data_type), 1, fpInput);
			intTotalHeaderBytes += sizeof (Data_type);
		}
		else if (strtmp == "ibeam")
		{
			fread (&Ibeam, sizeof (Ibeam), 1, fpInput);
			intTotalHeaderBytes += sizeof (Ibeam);
		}
		else if (strtmp == "nbeams")
		{
			fread (&Nbeams, sizeof (Nbeams), 1, fpInput);
			intTotalHeaderBytes += sizeof (Nbeams);
		}
		else if (strtmp == "nbits")
		{
			fread (&Nbits, sizeof (Nbits), 1, fpInput);
			intTotalHeaderBytes += sizeof (Nbits);
		}
		else if (strtmp == "barycentric")
		{
			fread (&Barycentric, sizeof (Barycentric), 1, fpInput);
			intTotalHeaderBytes += sizeof (Barycentric);
		}
		else if (strtmp == "pulsarcentric")
		{
			fread (&Pulsarcentric, sizeof (Pulsarcentric), 1, fpInput);
			intTotalHeaderBytes += sizeof (Pulsarcentric);
		}
		else if (strtmp == "nbins")
		{
			fread (&Nbins, sizeof (Nbins), 1, fpInput);
			intTotalHeaderBytes += sizeof (Nbins);
		}
		else if (strtmp == "nsamples")
		{
			/* read this one only for backwards compatibility */
			fread (&nsamp, sizeof (nsamp), 1, fpInput);
			intTotalHeaderBytes += sizeof (nsamp);
		}
		else if (strtmp == "nifs")
		{
			fread (&Nifs, sizeof (Nifs), 1, fpInput);
			intTotalHeaderBytes += sizeof (Nifs);
		}
		else if (strtmp == "npuls")
		{
			fread (&npuls, sizeof (npuls), 1, fpInput);
			intTotalHeaderBytes += sizeof (npuls);
		}
		else if (strtmp == "refdm")
		{
			fread (&RefDM, sizeof (RefDM), 1, fpInput);
			intTotalHeaderBytes += sizeof (RefDM);
		}
		else if (expecting_rawdatafile == 1)
		{
			strcpy (Rawdatafile, strtmp.c_str ());
			expecting_rawdatafile = 0;
		}
		else if (expecting_source_name == 1)
		{
			strcpy (Source_name, strtmp.c_str ());
			expecting_source_name = 0;
		}
		else
		{
			cerr << "read_header - unknown parameter : " << strtmp << endl;
			return (false);
		}
	}
	Headersize=intTotalHeaderBytes;
	Nsamples=nsamples(fname.c_str (), Headersize, Nbits, Nifs, Nchans);
	fp=fpInput;
	//Correct Frequency table any way
	if (!UseFrequencyTable)
	{
		for (long int i=0; i<Nchans; i++)
		{
			frequency_table[i]=Fch1+i*Foff;
		}
	}
	return (true);
}
void FilterBankData::PrintHeader(void)
{
	cout<< "Filename:"<<"\t"<<FileName<<endl;
	/// parameters to descrive how much data to read in
	cout<< "Startpenc:"<<"\t"<<Startpenc<<endl;					//starting percentage
	cout<<"Endpenc:"<<"\t"<<Endpenc<<endl;						//ending percentage
	cout<<"StartSample"<<"\t"<< StartSample<<endl;
	cout<<"EndSample"<<"\t"<< EndSample<<endl;
	cout<<"Headersize"<<"\t"<< Headersize<<endl;
	cout<<"Tau"<<"\t"<< Tsamp<<endl;
	cout<<"Nsamples"<<"\t"<< Nsamples<<endl;
	cout<<"Machine_id"<<"\t"<< Machine_id<<endl;
	cout<<"Telescope_id"<<"\t"<< Telescope_id<<endl;
	cout<<"Data_type"<<"\t"<< Data_type<<endl;
	cout<<"Nchans"<<"\t"<< Nchans<<endl;
	cout<<"Nbits"<<"\t"<< Nbits<<endl;
	cout<<"Nifs"<<"\t"<< Nifs<<endl;
	cout<<"Scan_number"<<"\t"<< Scan_number<<endl;
	cout<<"Barycentric"<<"\t"<< Barycentric<<endl;
	cout<<"Pulsarcentric"<<"\t"<< Pulsarcentric<<endl;
	cout<<"Nbins"<<"\t"<< Nbins<<endl;
	cout<<"Tstart"<<"\t"<< Tstart<<endl;
	cout<<"Mjdobs"<<"\t"<< Mjdobs<<endl;
	cout<<"Tsamp"<<"\t"<< Tsamp<<endl;
	cout<<"Fch1"<<"\t"<< Fch1<<endl;
	cout<<"Foff"<<"\t"<< Foff<<endl;
	cout<<"RefDM"<<"\t"<< RefDM<<endl;
	cout<<"Az_start"<<"\t"<< Az_start<<endl;
	cout<<"Za_start"<<"\t"<< Za_start<<endl;
	cout<<"Src_raj"<<"\t"<< Src_raj<<endl;
	cout<<"Src_dej"<<"\t"<< Src_dej<<endl;
	cout<<"Gal_l"<<"\t"<< Gal_l<<endl;
	cout<<"Gal_b"<<"\t"<< Gal_b<<endl;
	cout<<"Header_tobs"<<"\t"<< Header_tobs<<endl;
	cout<<"Raw_fch1"<<"\t"<< Raw_fch1<<endl;
	cout<<"Raw_foff"<<"\t"<< Raw_foff<<endl;
	cout<<"Nbeams"<<"\t"<< Nbeams<<endl;
	cout<<"Ibeam"<<"\t"<< Ibeam<<endl;
	cout<<"Srcl"<<"\t"<< Srcl<<endl;
	cout<<"Srcb"<<"\t"<< Srcb<<endl;
	cout<<"Ast0"<<"\t"<< Ast0<<endl;
	cout<<"Lst0"<<"\t"<< Lst0<<endl;
	cout<<"Wapp_scan_number"<<"\t"<< Wapp_scan_number<<endl;
	cout<<"Project"<<"\t"<< Project<<endl;
	cout<<"Culprits"<<"\t"<< Culprits<<endl;
	cout<<"Analog_power"<<"\t"<< Analog_power[0]<<"\t"<<Analog_power[1]<<endl;
	cout<<"Rawdatafile"<<"\t"<< Rawdatafile<<endl;
	cout<<"Source_name"<<"\t"<< Source_name<<endl;
}
bool FilterBankData::ReadInData(long double startp, long double endp)
{
	Startpenc=startp;
	Endpenc=endp;
	StartSample=startp*Nsamples;
	EndSample=endp*Nsamples;

	long int offset=(int) (long double)StartSample * (long double)Nchans * (long double)Nifs * ((long double)Nbits / 8.0);
	fseek (fp , offset+Headersize , SEEK_SET );
	return readdata(EndSample-StartSample);
}
bool FilterBankData::ReadInDatabySample(long int startsample, long int endsample)
{
	StartSample=startsample;
	if (endsample<0)
		EndSample=Nsamples;
	else
		EndSample=endsample;
	//cout<<Nchans<<"\t"<<Nifs<<"\t"<<Nbits<<endl;
	long int offset=(long int) (long double)StartSample * (long double)Nchans * (long double)Nifs * ((long double)Nbits / 8.0);
	fseek (fp , offset , SEEK_CUR );
	return readdata(EndSample-StartSample);
}
bool FilterBankData::readdata(long int ns)
{
	long int icnt=0;
	Free();
	switch (Nbits)
	{
	case 1:
	{
		long int nchr=ns*Nifs*Nchans/8;
		unsigned char * chb=new unsigned char [nchr];
		icnt=fread(chb,1,nchr,fp);
		if (icnt!=nchr)
			cerr<<"Data ends unexpected read to EOF"<<endl;
		pData=new float [icnt*8];
		long int k=0;
		for (long int i=0; i<icnt; i++)
		{
			for (long int j=0; j<8; j++)
			{
				pData[k]=(float)(chb[i]&1);
				k++;
				chb[i]>>=1;
			}
		}
		delete [] chb;
		Nsamples=icnt*8/Nifs/Nchans;
		break;
	}
	case 8:
	{
		long int nchr=ns*Nifs*Nchans;
#define EIGHTBIT char		
		EIGHTBIT * chb=new EIGHTBIT [nchr];
		icnt=fread(chb,1,nchr,fp);
		pData=new float [icnt];
		for (long int i=0; i<icnt; i++)
		{
			pData[i]=(float)(chb[i]);
		}
		delete [] chb;
		if (icnt!=nchr)
		{
			cerr<<"Data ends unexpected read to EOF"<<endl;
		}
		Nsamples=icnt/Nifs/Nchans;
		break;
	}
	case 32:
	{
		long int nchr=ns*Nifs*Nchans;
		pData=new float [nchr];
		icnt=fread(pData, 4, nchr, fp);
		if (icnt!=nchr)
		{
			cerr<<"Data ends unexpected read to EOF"<<endl;
			for (long int i=icnt; i<nchr; i++) pData[i]=0;
		}
		Nsamples=icnt/Nifs/Nchans;
		break;
	}
	}
	return true;
}
void FilterBankData::Free(void)
{
	if (pData!=NULL)
	{
		delete [] pData;
		pData=NULL;

	}
	if (pZeroDM!=NULL)
	{
		delete [] pZeroDM;
		pZeroDM=NULL;

	}
	if (pDM!=NULL)
	{
		delete [] pDM;
		pDM=NULL;

	}
	if (vBin!=NULL)
	{
		delete [] vBin;
		vBin=NULL;

	}
	if (pSubband!=NULL)
	{
		for (long int i=0; i<Nsubband; i++)
		{
			pSubband[i].Free();
		}
		delete [] pSubband;
		pSubband=NULL;
	}
	Obj.Free();
	Obj1D.Free();
}
/**\brief 滤除0DM信号
 * */
void FilterBankData::ZeroDM(const string method)
{
	double * pchunk;
	if (method=="dot")
	{
		long int ns=Nsamples*Nifs;
		pZeroDM=new float [ns];
		pchunk=new double [Nifs];
		long int l=0;
		long int nchk=Nifs*Nchans;
		double * avrsig=new double [nchk];
		double * avrzerodm=new double [Nifs];
		for (long int i=0; i<nchk; i++) avrsig[i]=0;
		for (long int k=0; k<Nifs; k++) avrzerodm[k]=0;
		long sft2=0;
		for(long int i=0; i<Nsamples; i++)
		{
			for (long int k=0; k<Nifs; k++) pchunk[k]=0;
			long int sft=0;
			for (long int j=0; j<Nchans; j++)
			{
				double * pt=avrsig+sft;
				for (long int k=0; k<Nifs; k++)
				{
					pchunk[k]+=pData[l];
					pt[k]+=pData[l];
					l++;
				}
				sft+=Nifs;
			}
			for (long int k=0; k<Nifs; k++)
			{
				double tmp;
				pZeroDM[sft2+k]=tmp=pchunk[k]/Nchans;
				avrzerodm[k]+=tmp;
			}
			sft2+=Nifs;
		}
		for (long int i=0; i<nchk; i++) avrsig[i]/=Nsamples;
		for (long int k=0; k<Nifs; k++) avrzerodm[k]/=Nsamples;

		double * ps2=new double [Nifs];

		long int dNchan=Nifs*Nchans;
		long int sft4=0;
		for(long int j=0; j<Nchans; j++)
		{
			for (long int k=0; k<Nifs; k++)
			{
				pchunk[k]=0;
				ps2[k]=0;
			}

			for (long int k=0; k<Nifs; k++)
			{
				long sft=0;
				long sft3=0;
				for (long int i=0; i<Nsamples; i++)
				{

					float a=pZeroDM[sft+k]-avrzerodm[k];
					pchunk[k]+=a*(pData[sft3+sft4+k]-avrsig[sft4+k]);
					ps2[k]+=a*a;
					sft+=Nifs;
					sft3+=dNchan;
				}
			}
			for (long int k=0; k<Nifs; k++)
			{
				double beta=pchunk[k]/ps2[k];
				long sft=0;
				long sft3=0;
				for (long int i=0; i<Nsamples; i++)
				{
					float a=pZeroDM[sft+k]-avrzerodm[k];

					pData[sft3+sft4+k]-=(float)a*beta;
					sft+=Nifs;
					sft3+=dNchan;
				}
			}
			sft4+=Nifs;
		}
		delete [] ps2;
		delete [] avrsig;
		delete [] avrzerodm;
	}
	else
	{
		long int ns=Nsamples*Nifs;
		pZeroDM=new float [ns];
		pchunk=new double [Nifs];
		double s2=0;
		double s0=0;
		long int l=0;
		long int sft=0;
		for(long int i=0; i<Nsamples; i++)
		{
			for (long int k=0; k<Nifs; k++) pchunk[k]=0;
			for (long int j=0; j<Nchans; j++)
			{
				for (long int k=0; k<Nifs; k++)
				{
					pchunk[k]+=pData[l];
					l++;
				}
			}
			for (long int k=0; k<Nifs; k++)
			{
				pZeroDM[sft+k]=pchunk[k]/Nchans;
			}
			sft+=Nifs;
		}
		l=0;
		long int delta=0;
		for(long int i=0; i<Nsamples; i++)
		{
			for (long int j=0; j<Nchans; j++)
			{
				for (long int k=0; k<Nifs; k++)
				{
					pData[l]-=pZeroDM[k+delta];
					l++;
				}
			}
			delta+=Nifs;
		}
	}
	delete [] pchunk;
}
/**\brief 写filterbank数据
 * */
bool FilterBankData::WriteHeaderToFile(const char* fname)
{
	FILE * f_fil;
	f_fil=fopen(fname, "w");
	//f_fil = fname
	if (f_fil==NULL) return false;

	put_string(f_fil, "HEADER_START");
	put_string(f_fil,"source_name");
	put_string(f_fil,Source_name);
	if (UseFrequencyTable || (Fch1==0.0 && Foff==0.0))
	{
		put_string(f_fil,"FREQUENCY_START");
		for (int channel_index=0; channel_index<Nchans; channel_index++)
		{
			put_string(f_fil, "fchannel");
			fwrite (&frequency_table[channel_index], sizeof (double), 1, f_fil);
		}
		put_string(f_fil, "FREQUENCY_END");
	}
	put_string(f_fil, "az_start");
	fwrite (&Az_start, sizeof (Az_start), 1, f_fil);
	put_string(f_fil, "za_start");
	fwrite (&Za_start, sizeof (Za_start), 1, f_fil);
	put_string(f_fil, "src_raj");
	fwrite (&Src_raj, sizeof (Src_raj), 1, f_fil);
	put_string(f_fil, "src_dej");
	fwrite (&Src_dej, sizeof (Src_dej), 1, f_fil);
	put_string(f_fil, "tstart");
	fwrite (&Tstart, sizeof (Tstart), 1, f_fil);
	put_string(f_fil, "tsamp");
	fwrite (&Tsamp, sizeof (Tsamp), 1, f_fil);
	//put_string(f_fil, "period");
	//fwrite (&Period, sizeof (Period), 1, f_fil);
	put_string(f_fil, "fch1");
	fwrite (&Fch1, sizeof (Fch1), 1, f_fil);
	put_string(f_fil, "foff");
	fwrite (&Foff, sizeof (Foff), 1, f_fil);
	put_string(f_fil, "nchans");
	fwrite (&Nchans, sizeof (Nchans), 1, f_fil);
	put_string(f_fil, "telescope_id");
	fwrite (&Telescope_id, sizeof (Telescope_id), 1, f_fil);
	put_string(f_fil, "machine_id");
	fwrite (&Machine_id, sizeof (Machine_id), 1, f_fil);
	put_string(f_fil, "data_type");
	fwrite (&Data_type, sizeof (Data_type), 1, f_fil);
	put_string(f_fil, "ibeam");
	fwrite (&Ibeam, sizeof (Ibeam), 1, f_fil);
	put_string(f_fil, "nbeams");
	fwrite (&Nbeams, sizeof (Nbeams), 1, f_fil);
	put_string(f_fil, "nbits");
	fwrite (&Nbits, sizeof (Nbits), 1, f_fil);
	put_string(f_fil, "barycentric");
	fwrite (&Barycentric, sizeof (Barycentric), 1, f_fil);
	put_string(f_fil, "pulsarcentric");
	fwrite (&Pulsarcentric, sizeof (Pulsarcentric), 1, f_fil);
	//put_string(f_fil, "nbins");
	//fwrite (&Nbins, sizeof (Nbins), 1, f_fil);
//	put_string(f_fil, "nsamples");
//			fwrite (&Nsamples, sizeof (Nsamples), 1, f_fil);
	put_string(f_fil, "nifs");
	fwrite (&Nifs, sizeof (Nifs), 1, f_fil);
	//put_string(f_fil, "npuls");
	//fwrite (&npuls, sizeof (npuls), 1, f_fil);
	//put_string(f_fil, "refdm");
	//fwrite (&RefDM, sizeof (RefDM), 1, f_fil);
	put_string(f_fil, "HEADER_END");
	fclose(f_fil);
	cout<<"Finish header write"<<endl;
	return 0;

}
void WriteHeader(const char * fname,double net_MJD)
//void WriteHeader(FILE * fname)
{

	double Fch1=1919.8828125; //1920-(960/4096)/2
	double Foff=-0.234375; // 960/4096
	int Nchans=4096;
	float Tsamp=213.3333e-6; // acc_len = 50
	int Nbits=8;
	int Nifs=2;
        int start=time((time_t*)NULL);
        double current_MJD;
        double Year;
        double  Month;
        double  Day;
        double jd;
        time_t timep;
        struct timeval currenttime;
        struct tm *p;
        time(&timep);
        p=gmtime(&timep);
        Year=p->tm_year+1900;
        Month=p->tm_mon+1;
        Day=p->tm_mday;

        //const char * fname= "filterbank-test";
        //Start to allocate the header information
        //long int header_size=sizeof(_MEMELEM)*Nseg;
        //current_MJD=5213.11111111111;
        jd = UTCtoJD(Year, Month, Day);
        current_MJD= jd+(double)((p->tm_hour-12)/24.0)
                               +(double)(p->tm_min/1440.0)
                               +(double)(p->tm_sec/86400.0)
                               +(double)(currenttime.tv_usec/86400.0/1000000.0)
                               -(double)2400000.5;
	cout.precision(10);
	cout <<"current MJD from header is:" << fixed << net_MJD << endl;
	//const char * fname= "filterbank-test";
	//Start to allocate the header information
	//long int header_size=sizeof(_MEMELEM)*Nseg;
	//current_MJD=50013.11111111111;
	FilterBankData fil;
	strcpy(fil.Source_name, "J0000-0000");
	fil.UseFrequencyTable=false;
	//fil.Telescope_id=1;
	//fil.Machine_id=1;
	fil.Data_type=1;
	fil.Az_start=0;
	fil.Za_start=0;
	fil.Src_dej=0;
	fil.Src_raj=0;
	fil.Tstart=net_MJD;
	fil.Tsamp=Tsamp;
	fil.Fch1=Fch1;
	fil.Foff=Foff;
	fil.Nchans=Nchans;
	fil.Nbits=Nbits;
	fil.Nbeams=1;
	fil.Ibeam=1;
	fil.Nifs=Nifs;
	fil.RefDM=0;
	//fil.Nsamples=Nsl/Nchans/Nifs;
	//fil.pData=(float *)pd_proc;
	//char outputname[4096];
	//sprintf(outputname, "%s.fil", fname);
	//fil.WriteHeaderToFile(outputname);
	fil.WriteHeaderToFile(fname);
	
	//usleep(100);
}
/*bool WriteDataToFile(const char * fname)
{
}
bool FilterBankData::WriteDataToFile(const char * fname)
{
	switch (Nbits)
	{
	case 1:
	{
		break;
	}
	case 8:
	{
		cout<<"use 8bit"<<endl;
		long int nchr=Nsamples*Nifs*Nchans;
		EIGHTBIT * chb=new EIGHTBIT [nchr];
		for (long int i=0; i<nchr; i++)
		{
			float tmp=pData[i]>255?255:pData[i];
			tmp=tmp<0?0:tmp;
			chb[i]=(char)tmp;
		}
		cout<<"start to write"<<endl;
		cout<<nchr<<endl;
		fwrite(chb,1,nchr,f_fil);
		fclose(f_fil);
		cout<<"Finish write"<<endl;
		delete [] chb;
		break;
	}
	case 32:
	{
		long int nchr=Nsamples*Nifs*Nchans;
		fwrite(pData, 4, nchr, f_fil);
		fclose(f_fil);
		break;
	}
	}
	return true;*/
/**!\brief 降低采样率
 * */
bool FilterBankData::DownSample(int nd)
{
	long int nnew=floor(Nsamples/nd);
	long int totals=Nsamples*Nchans*Nifs;
	long int nchk=Nchans*Nifs;
	float * pd2=new float [nnew*Nchans*Nifs];
	float * ptmp=new float [nchk];
	float * pori=pData;
	float * pobj=pd2;
	for (long int i=0; i<nnew; i++)
	{
		for (long int k=0; k<nchk; k++)
			ptmp[k]=0;

		for (long int j=0; j<nd; j++)
		{
			for (long int k=0; k<nchk; k++)
				ptmp[k]+=pori[k];
			pori+=nchk;
		}

		for (long int j=0; j<nchk; j++)
			pobj[j]=ptmp[j];
		pobj+=nchk;
	}
	delete [] ptmp;
	ptmp=pData;
	pData=pd2;
	delete [] ptmp;
	Nsamples=nnew;
	Tsamp_ori=Tsamp;
	Tsamp*=nd;
	return true;
}


bool FilterBankData::SubbandDeDispersion(double startdm, double ddm, double enddm)
{
	//构造子带数据
	Ndm=ceil((enddm-startdm)/ddm)+1;										//Number of total DM trials
	Nchsubband=(long int) sqrt(Nchans);									//number of channel of each subband file
	long int NMainDM=ceil( (float)Ndm / Nchsubband);		//主消色散数目
	Nsubband=NMainDM;																		//number of subband
	double MainDMStep=ddm*Ndm/NMainDM;								//主消色散步长
	pSubband=new SubBandData[Nsubband];
	pDM=new float[NMainDM];

	for (long int i=0; i<Nsubband; i++)
	{
		pDM[i]=startdm+(i)*MainDMStep;
		//if (!dedmdata2sub(pSubband+i, startdm+(i)*MainDMStep)) 	//形成子带数据
		//	return false;
		if (!dmsub2tim(pSubband+i, 0, ddm, MainDMStep))
			return false;
	}
	//每个子带的消色散
	Ndm=ceil(MainDMStep/ddm)*NMainDM;
	return true;
}

/**\brief write all the subband data  */
bool FilterBankData::WriteSubbandData(const char * fname)
{
	FILE * f_fil;
	for (int i=0; i<Nsubband; i++)
	{
		SubBandData * psub=pSubband+i;
		string fn=fname;
		char chrtmp[1024];
		sprintf(chrtmp, "%d_", i);
		fn=chrtmp+fn;
		f_fil=fopen(fn.c_str(), "wb");
		if (f_fil==NULL) return false;

		put_string(f_fil, "HEADER_START");
		put_string(f_fil,"source_name");
		put_string(f_fil,Source_name);
		put_string(f_fil,"FREQUENCY_START");
		for (int channel_index=0; channel_index<psub->Nchans; channel_index++)
		{
			put_string(f_fil, "fchannel");
			fwrite (&(psub->pFreq[channel_index]), sizeof (double), 1, f_fil);
		}
		put_string(f_fil, "FREQUENCY_END");
		put_string(f_fil, "az_start");
		fwrite (&Az_start, sizeof (Az_start), 1, f_fil);
		put_string(f_fil, "za_start");
		fwrite (&Za_start, sizeof (Za_start), 1, f_fil);
		put_string(f_fil, "src_raj");
		fwrite (&Src_raj, sizeof (Src_raj), 1, f_fil);
		put_string(f_fil, "src_dej");
		fwrite (&Src_dej, sizeof (Src_dej), 1, f_fil);
		put_string(f_fil, "tstart");
		fwrite (&Tstart, sizeof (Tstart), 1, f_fil);
		put_string(f_fil, "tsamp");
		fwrite (&Tsamp, sizeof (Tsamp), 1, f_fil);
		put_string(f_fil, "period");
		fwrite (&Period, sizeof (Period), 1, f_fil);
		put_string(f_fil, "fch1");
		fwrite (&Fch1, sizeof (Fch1), 1, f_fil);
		put_string(f_fil, "foff");
		fwrite (&Foff, sizeof (Foff), 1, f_fil);
		put_string(f_fil, "nchans");
		fwrite (&(psub->Nchans), sizeof (psub->Nchans), 1, f_fil);
		put_string(f_fil, "telescope_id");
		fwrite (&Telescope_id, sizeof (Telescope_id), 1, f_fil);
		put_string(f_fil, "machine_id");
		fwrite (&Machine_id, sizeof (Machine_id), 1, f_fil);
		put_string(f_fil, "data_type");
		fwrite (&Data_type, sizeof (Data_type), 1, f_fil);
		put_string(f_fil, "ibeam");
		fwrite (&Ibeam, sizeof (Ibeam), 1, f_fil);
		put_string(f_fil, "nbeams");
		fwrite (&Nbeams, sizeof (Nbeams), 1, f_fil);
		put_string(f_fil, "nbits");
		int nb=32;
		fwrite (&nb, sizeof (nb), 1, f_fil);
		put_string(f_fil, "barycentric");
		fwrite (&Barycentric, sizeof (Barycentric), 1, f_fil);
		put_string(f_fil, "pulsarcentric");
		fwrite (&Pulsarcentric, sizeof (Pulsarcentric), 1, f_fil);
		put_string(f_fil, "nbins");
		fwrite (&Nbins, sizeof (Nbins), 1, f_fil);
//	put_string(f_fil, "nsamples");
//			fwrite (&Nsamples, sizeof (Nsamples), 1, f_fil);
		put_string(f_fil, "nifs");
		fwrite (&Nifs, sizeof (Nifs), 1, f_fil);
		put_string(f_fil, "npuls");
		fwrite (&npuls, sizeof (npuls), 1, f_fil);
		put_string(f_fil, "refdm");
		fwrite (&psub->RefDM, sizeof (psub->RefDM), 1, f_fil);
		put_string(f_fil, "HEADER_END");
		switch (nb)
		{
		case 1:
		{
			break;
		}
		case 8:
		{
			cout<<"use 8bit"<<endl;
			long int nchr=(psub->Nsamples)*Nifs*(psub->Nchans);
			EIGHTBIT * chb=new EIGHTBIT [nchr];
			for (long int i=0; i<nchr; i++)
			{
				float tmp=psub->pData[i]>255?255:psub->pData[i];
				tmp=tmp<0?0:tmp;
				chb[i]=(char)tmp;
			}
			fwrite(chb,1,nchr,f_fil);
			delete [] chb;
			break;
		}
		case 32:
		{
			long int nchr=(psub->Nsamples)*Nifs*(psub->Nchans);
			fwrite(psub->pData, 4, nchr, f_fil);
			break;
		}
		}
		fclose(f_fil);
	}
	return true;
}
bool FilterBankData::dmsub2tim(SubBandData * psub, double ldmv, double dmstep, double rdmv)
{
	psub->NDM=ceil((rdmv-ldmv)/dmstep);
	psub->pTim=new TimeSeries [psub->NDM];
	//the DM grid of the subband file
	psub->pDM=new double [psub->NDM];
	if (psub->pTim==NULL) return false;
	if (psub->pDM==NULL) return false;

	TimeSeries * ptim=psub->pTim;
	//calculate the bin shifts
	double maxf=psub->pFreq[0];
	for(long int i=0; i<psub->Nchans; i++)
		maxf=maxf>psub->pFreq[i]?maxf:psub->pFreq[i];

	int * sfti=new int [psub->Nchans];
	if (sfti==NULL) return false;

	for (long int i=0; i<psub->NDM; i++)
	{
		double dmv=ldmv+dmstep*i;
		psub->pDM[i]=dmv+psub->RefDM;
		int maxdelay=0;
		for (long int j=0; j<psub->Nchans; j++)
		{
			//sfti[j]=round(DMDelay(dmv, psub->pFreq[j], maxf)/Tsamp);
			if (sfti[j]>maxdelay) maxdelay=sfti[j];
		}

		float * pt;
		(ptim+i)->Nsamples=(psub->Nsamples);
		(ptim+i)->ptim=pt=new float [(ptim+i)->Nsamples * Nifs];
		(ptim+i)->plink=NULL;
		if (pt==NULL) return false;
		for (long int j=0; j<(ptim+i)->Nsamples; j++)
		{
			for (long int l=0; l<Nifs; l++)
			{
				float tmp=0;
				for (long int k=0; k<psub->Nchans; k++)
				{
					long int isft=j+sfti[k];
					if (isft<Nsamples && isft>0)
						tmp+=psub->pData[(isft)*psub->Nchans*Nifs + k*Nifs+l];
				}
				pt[j*Nifs + l]=tmp;
			}
		}
	}
	delete [] sfti;
	return true;
}

bool FilterBankData::WriteTimData(const char * fname)
{
	FILE * f_fil;
	for (int i=0; i<Nsubband; i++)
	{
		SubBandData * psub=pSubband+i;
		for (long int j=0; j<psub->NDM; j++)
		{
			char chrtmp[1024];
			char ext[1024];
			//GetBaseName(fname, chrtmp,ext);
			string fn=chrtmp;
			sprintf(chrtmp, "_%f", psub->pDM[j]);
			fn=fn+chrtmp+"."+ext;
			f_fil=fopen(fn.c_str(), "wb");
			if (f_fil==NULL) return false;

			put_string(f_fil, "HEADER_START");
			put_string(f_fil,"source_name");
			put_string(f_fil,Source_name);
			put_string(f_fil, "az_start");
			fwrite (&Az_start, sizeof (Az_start), 1, f_fil);
			put_string(f_fil, "za_start");
			fwrite (&Za_start, sizeof (Za_start), 1, f_fil);
			put_string(f_fil, "src_raj");
			fwrite (&Src_raj, sizeof (Src_raj), 1, f_fil);
			put_string(f_fil, "src_dej");
			fwrite (&Src_dej, sizeof (Src_dej), 1, f_fil);
			put_string(f_fil, "tstart");
			fwrite (&Tstart, sizeof (Tstart), 1, f_fil);
			put_string(f_fil, "tsamp");
			fwrite (&Tsamp, sizeof (Tsamp), 1, f_fil);
			put_string(f_fil, "period");
			fwrite (&Period, sizeof (Period), 1, f_fil);
			put_string(f_fil, "fch1");
			fwrite (&Fch1, sizeof (Fch1), 1, f_fil);
			put_string(f_fil, "foff");
			double Fbw=Foff*Nchans;
			fwrite (&Fbw, sizeof (Fbw), 1, f_fil);
			put_string(f_fil, "nchans");
			int nc=1;
			fwrite (&(nc), sizeof (nc), 1, f_fil);
			put_string(f_fil, "telescope_id");
			fwrite (&Telescope_id, sizeof (Telescope_id), 1, f_fil);
			put_string(f_fil, "machine_id");
			fwrite (&Machine_id, sizeof (Machine_id), 1, f_fil);
			put_string(f_fil, "data_type");
			fwrite (&Data_type, sizeof (Data_type), 1, f_fil);
			put_string(f_fil, "ibeam");
			fwrite (&Ibeam, sizeof (Ibeam), 1, f_fil);
			put_string(f_fil, "nbeams");
			fwrite (&Nbeams, sizeof (Nbeams), 1, f_fil);
			put_string(f_fil, "nbits");
			int nb=32;
			fwrite (&nb, sizeof (nb), 1, f_fil);
			put_string(f_fil, "barycentric");
			fwrite (&Barycentric, sizeof (Barycentric), 1, f_fil);
			put_string(f_fil, "pulsarcentric");
			fwrite (&Pulsarcentric, sizeof (Pulsarcentric), 1, f_fil);
			put_string(f_fil, "nbins");
			fwrite (&Nbins, sizeof (Nbins), 1, f_fil);
			put_string(f_fil, "nifs");
			fwrite (&Nifs, sizeof (Nifs), 1, f_fil);
			put_string(f_fil, "npuls");
			fwrite (&npuls, sizeof (npuls), 1, f_fil);
			put_string(f_fil, "refdm");
			fwrite (psub->pDM+j, sizeof (psub->RefDM), 1, f_fil);
			put_string(f_fil, "HEADER_END");
			TimeSeries * pt=psub->pTim+j;
			switch (nb)
			{
			case 1:
			{
				break;
			}
			case 8:
			{
				cout<<"use 8bit"<<endl;
				long int nchr=(pt->Nsamples)*Nifs;
				EIGHTBIT * chb=new EIGHTBIT [nchr];

				for (long int l=0; l<nchr; l++)
				{
					float tmp=pt->ptim[l]>255?255:pt->ptim[l];
					tmp=tmp<0?0:tmp;
					chb[l]=(char)tmp;
				}
				fwrite(chb,1,nchr,f_fil);
				delete [] chb;
				break;
			}
			case 32:
			{
				long int nchr=(pt->Nsamples)*Nifs;
				fwrite(pt->ptim, 4, nchr, f_fil);
				break;
			}
			}
			fclose(f_fil);

		}
	}
	return true;
}

bool FilterBankData::BoxCarFilter(long int nbox, double minw, float snrloss)
{

	Nbox=nbox;
	vBin=new long int [Nbox];
	float step=pow(snrloss+1.0,4.0);
	float pre=minw;
	vBin[0]=pre/Tsamp;
	if (vBin[0]<1)
	{
		vBin[0]=1;
		pre=Tsamp;
	}

	//forming the step of boxcar filter
	for (long int i=1; i<Nbox; i++)
	{
		pre*=step;
		vBin[i]=round(pre/Tsamp);
	}
	for (long int j=0; j<Nsubband; j++)
	{
		SubBandData * psub=pSubband+j;
		for (long int k=0; k<psub->NDM; k++)
		{
			TimeSeries * pt=(psub->pTim)+k;
			pt->Nbox=Nbox;
			pt->plink=new TimeSeries [pt->Nbox];
			if (pt->plink==NULL) return false;
			pt->SetStat();
			for (long int ib=0; ib<pt->Nbox; ib++)
			{
				TimeSeries * pfted=pt->plink+ib;
				pfted->New(pt->Nsamples);
				pfted->plink=NULL;
				if (!ApplyBoxFilter(pt->ptim, pt->Var,pt->Mean, vBin[ib], pt->Nsamples,
				                    pfted->ptim))
					return false;
				/*
				if (!ApplyBoxFilter(pt->ptim, Sig_0DM,pt->Mean, vBin[ib], pt->Nsamples,
				                    pfted->ptim))
					return false;
				*/

			}
		}
	}
	return true;
}
/**\brief applying the boxcar filter to the data
 * \param ori is the original 1-D time series
 * \param var is the variance of original data
 * \param wi is the length of filter
 * \param n is the data length
 * \param obj is the place to store the data, it should be of the size n
 * */
bool FilterBankData::ApplyBoxFilter(float * ori, float var, float mean, long int wi, long int n, float * obj)
{
	long int k=0;
	long int sfti=wi>n?n:wi;
	double sigshif=sfti*var;
	float tmp=0;
	for (long int j=0; j<sfti/2; j++)
	{
		tmp+=ori[j]-mean;
	}
	obj[k]=tmp*tmp/sigshif;
	k++;
	for (long int j=sfti/2; j<sfti; j++)
	{
		tmp+=ori[j]-mean;
		obj[k]=tmp*tmp/(sigshif);
		k++;
		if (k>=n) return true;
	}
	for (long int j=sfti; j<n; j++)
	{
		tmp+=ori[j];
		tmp-=ori[j-sfti];
		obj[k]=tmp*tmp/sigshif;
		k++;
		if (k>=n) return true;
	}
	for (long int j=n; j<n+sfti/2; j++)
	{
		tmp-=(ori[j-sfti]-mean);
		obj[k]=tmp*tmp/sigshif;
		k++;
		if (k>=n) return true;
	}
	return true;
}
bool FilterBankData::FormZeroDMSeriesandStatistics(void)
{
	long int ns=Nsamples*Nifs;
	pZeroDM=new float [ns];
	double * pchunk=new double [Nifs];
	if (pZeroDM==NULL) return false;
	if (pchunk==NULL) return false;
	long int l=0;
	long int nchk=Nifs*Nchans;
	double * avrzerodm=new double [Nifs];
	double * varzeroDM=new double [Nifs];
	for (long int k=0; k<Nifs; k++) avrzerodm[k]=0;
	long sft2=0;
	for(long int i=0; i<Nsamples; i++)
	{
		for (long int k=0; k<Nifs; k++) pchunk[k]=0;
		long int sft=0;
		for (long int j=0; j<Nchans; j++)
		{
			for (long int k=0; k<Nifs; k++)
			{
				pchunk[k]+=pData[l];
				l++;
			}
			sft+=Nifs;
		}
		for (long int k=0; k<Nifs; k++)
		{
			double tmp;
			pZeroDM[sft2+k]=tmp=pchunk[k];
			avrzerodm[k]+=tmp;
			varzeroDM[k]+=tmp*tmp;
		}
		sft2+=Nifs;
	}
	for (long int k=0; k<Nifs; k++) avrzerodm[k]/=Nsamples;
	for (long int k=0; k<Nifs; k++) varzeroDM[k]=varzeroDM[k]/Nsamples-avrzerodm[k]* avrzerodm[k];
	Sig_0DM=varzeroDM[0];
	delete [] avrzerodm;
	delete [] varzeroDM;
	delete [] pchunk;
	return true;
}

bool FilterBankData::Dedisperse(double dmv)
{
	Obj1D.Free();
	Obj1D.Nbox=0;
	if (!Obj1D.New(Nsamples)) return false;

	int * sfti=new int [Nchans];
	if (sfti==NULL) return false;

	double maxf=frequency_table[0];
	for(long int i=0; i<Nchans; i++)
		maxf=maxf>frequency_table[i]?maxf:frequency_table[i];

	//for (long int j=0; j<Nchans; j++)
		//sfti[j]=round(DMDelay(dmv, frequency_table[j], maxf)/Tsamp);

	for (long int i=0; i<Nsamples; i++)
	{
		float tmp=0;
		for(long int k=0; k<Nifs; k++)
		{
			for(long int j=0; j<Nchans; j++)
			{
				long int isft=i+sfti[j];
				if (isft<Nsamples && isft>0)
					tmp+=pData[(isft)*Nchans*Nifs + j*Nifs+k];
			}
		}
		Obj1D.ptim[i]=tmp;
	}
	delete [] sfti;
	return true;
}
bool FilterBankData::FormDedisperseChannelData(double dmv)
{
	Obj1D.Free();
	Obj1D.Nbox=0;
	if (!Obj1D.New(Nsamples)) return false;

	Obj.pFreq=new double [Nchans];
	if (Obj.pFreq==NULL) return false;
	Obj.Nchans=Nchans;
	Obj.Nsamples=Nsamples;
	Obj.pData=new float [Nsamples*Nifs*Nchans];
	if (Obj.pData==NULL) return false;
	double RefDM=dmv;
	int * sfti=new int [Nchans];
	if (sfti==NULL) return false;

	double maxf=frequency_table[0];
	for(long int i=0; i<Nchans; i++)
	{
		Obj.pFreq[i]=frequency_table[i];
		maxf=maxf>frequency_table[i]?maxf:frequency_table[i];
	}
	//for (long int j=0; j<Nchans; j++)
		//sfti[j]=round(DMDelay(dmv, frequency_table[j], maxf)/Tsamp);

	for (long int i=0; i<Nsamples; i++)
	{
		for(long int k=0; k<Nifs; k++)
		{
			float tmp=0;
			for(long int j=0; j<Nchans; j++)
			{
				long int isft=i+sfti[j];
				if (isft<Nsamples && isft>0)
				{
					float tpv=pData[(isft)*Nchans*Nifs + j*Nifs+k];
					tmp+=tpv;
					Obj.pData[i*Nifs*Nchans+j*Nifs+k]=tpv;
				}
				else
					Obj.pData[i*Nifs*Nchans+j*Nifs+k]=0;
			}
			Obj1D.ptim[i]=tmp;
		}
	}
	delete [] sfti;
	return true;
}


/**\brief Strip the filterbank to single polarization
 * Data order Nsamples X Nchans X Nifs
 * \param poli the polarization channel index
 * \param npol the number of poli
 *
 * poli=[0,1,2...] e.g. if poli=[0,2], then the polarizaiton channel 0 and 2
 * will be added to form the single pol chan.
 * */
bool FilterBankData::Strip2OnePol(const int * poli, int npol)
{
	float * ptmp=new float [Nsamples*Nchans];
	if (ptmp==NULL) return false;
	for (long int i=0; i<Nsamples; i++)
	{
		for(long int j=0; j<Nchans; j++)
		{
			float * pt=ptmp+i*Nchans+j;
			pt[0]=0;
			float *pto=pData+i*Nchans*Nifs+j*Nifs;
			for (long int k=0; k<npol; k++)
			{
				pt[0]+=pto[ poli[k] ];
			}
		}
	}
	float * pt=pData;
	pData=ptmp;
	delete [] pt;
	Nifs_ori=Nifs;
	Nifs=1;
	return true;
}
/**\brief Strip the filterbank to single polarization, Here the data order is
 * Nsample X Nifs X Nchans
 * \param poli the polarization channel index
 * \param npol the number of poli
 *
 * poli=[0,1,2...] e.g. if poli=[0,2], then the polarizaiton channel 0 and 2
 * will be added to form the single pol chan.
 * */

bool FilterBankData::Strip2OnePol_STD(const int * poli, int npol)
{
	float * ptmp=new float [Nsamples*Nchans];
	if (ptmp==NULL) return false;
	float * pt=ptmp;
	for (long int i=0; i<Nsamples; i++)
	{
		for(long int j=0; j<Nchans; j++)
		{
				pt[j]=0;
		}
		for (long int k=0; k<npol; k++)
		{
			long int ich=poli[k];
			float *pto=pData+i*Nchans*Nifs+ich*Nchans;
			for(long int j=0; j<Nchans; j++)
			{
					pt[j]+=pto[j];
			}
		}
		pt+=Nchans;
	}
	pt=pData;
	pData=ptmp;
	delete [] pt;
	Nifs_ori=Nifs;
	Nifs=1;
	return true;
}

bool FilterBankData::Equalize(void)
{
	double * chnmean=new double [Nchans];
	double * chnRMS=new double [Nchans];
	if (chnmean==NULL)
		return false;
	for (long int j=0; j<Nchans; j++) 
	{
		chnmean[j]=0;
		chnRMS[j]=0;
	}
	long int k=0;
	for (long int i =0; i<Nsamples; i++)
	{
		for (long int j=0; j<Nchans; j++)
		{
			chnmean[j]+=pData[k];
			chnRMS[j]+=pData[k]*pData[k];
			k++;
		}
	}
	for (long int j=0; j<Nchans; j++) 
	{
		chnmean[j]/=Nsamples;
		chnRMS[j]/=Nsamples;
		chnRMS[j]-=chnmean[j]*chnmean[j];
		//chnRMS[j]=sqrt(chnRMS[j];
	}
	k=0;
	for (long int i =0; i<Nsamples; i++)
	{
		for (long int j=0; j<Nchans; j++)
		{
			pData[k]=(pData[k]-chnmean[j]);
			if (chnRMS[j]!=0) pData[k]/=chnRMS[j];
			k++;
		}
	}
	delete [] chnmean;
	return true;
}

bool FilterBankData::RemoveBaseline(void)
{
	double * chnmean=new double [Nchans];
	if (chnmean==NULL)
		return false;
	for (long int j=0; j<Nchans; j++) chnmean[j]=0;
	long int k=0;
	for (long int i =0; i<Nsamples; i++)
	{
		for (long int j=0; j<Nchans; j++)
		{
			chnmean[j]+=pData[k];
			k++;
		}
	}
	for (long int j=0; j<Nchans; j++) chnmean[j]/=Nsamples;
	k=0;
	for (long int i =0; i<Nsamples; i++)
	{
		for (long int j=0; j<Nchans; j++)
		{
			pData[k]-=chnmean[j];
			k++;
		}
	}
	delete [] chnmean;
	return true;
}
void FilterBankData::CloseFile(void)
{
	fclose(fp);
}

