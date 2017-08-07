/////////////////////////////////////////////////////////////////////////
//                                                                     //
// Written  by                                                         //
// Qingjie Liu                                                         //
//                                                                     //
// web   :                                                             //
// email : qingjie.liu@buaa.edu.cn                                     //
//                                                                     //
/////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <malloc.h>

inline void conv2_v(const float* srcMat,int h, int w, const float * gfilter1d, int filterSize, float* dstMat); //convolution in vertical direction;
inline void conv2_h(const float* srcMat,int h, int w, const float * gfilter1d, int filterSize, float* dstMat); //convolution in horizontal direction;
inline void conv2d(const float*  srcMat,int h, int w, const float * gfilter1d, int filterSize, float* dstMat);

//inline int min(int a, int b);
//inline int max(int a, int b);
template <class T>
inline T min(T x, T y);
template <class T>
inline T max(T x, T y);

inline int get_filter_size(float sigma);
inline void gaussian_filter_1d(float sigma, float mean, int filterSize, float* gfilter);

inline void conv2_v(const float* srcMat, int h, int w, const float* gfilter1d, int filterSize, float* dstMat)
{
	if(NULL == srcMat)
	{
		return;
	}
	if(NULL == dstMat)// Do not creat the matrix inside the function
	{
		printf("The output of function conv2_v() point to an invalid address!\n");
		return;
	}
	int c               = (int)(filterSize-1)/2;
	const float* pSrc   = 0;
	float*       pDst   = 0;
	
	if(filterSize >= h-1)//if input image is very small, for example a small patch of an image
	{
		for(int j = 0; j<w; j++)
		{
			pSrc = srcMat + j*h;
		    pDst = dstMat + j*h;
			for(int i = 0; i<h; i++)
		    {
			    pDst[i] = 0;
			    int ind = 0;
			    for(int s = 0; s<filterSize; s++)
			    {
				    ind = i-c+s;
				    if(ind < 0)   ind = -ind;
					if(ind > h-1) ind = min((h-1)-(ind-(h-1)),(h-1));
				    pDst[i] += pSrc[ind]*gfilter1d[s];
			    }
		    }
		}
    }
	else
	{
		for(int j = 0; j<w; j++)
		{
			pSrc = srcMat + j*h;
			pDst = dstMat + j*h;
			for(int i = c; i<h-c; i++)
			{
				pDst[i] = 0;
				for(int s = 0; s<filterSize; s++)
				{
					pDst[i] += pSrc[i-c+s]*gfilter1d[s];
				}
			}
		}
		for(int j = 0; j<w; j++)
		{
			pSrc = srcMat + j*h;
			pDst = dstMat + j*h;
			for(int i = 0; i<c; i++)
			{
				pDst[i] = 0;
				for(int s = 0; s<filterSize; s++)
				{
					pDst[i] += pSrc[abs(i+s-c)]*gfilter1d[s]; 
				}
			}
			for(int i = h-c; i<h; i++)
			{
				pDst[i] = 0;
				int ind = 0;
				for(int s = 0; s<filterSize; s++)
				{
					ind = i-c+s;
					if(ind > h-1) ind = (h-1)-(ind-(h-1));
					pDst[i] += pSrc[ind]*gfilter1d[s]; 
				}
			}
		}
	}	
	return;
}
inline void conv2_h(const float* srcMat, int h, int w, const float* gfilter1d, int filterSize, float* dstMat)
{
	if(NULL == srcMat)
	{
		return;
	}
	if(NULL == dstMat)// Do not creat the matrix inside the function
	{
		printf("The output of function conv2d() point to an invalid address!\n");
		return;
	}

	int c             = (int)(filterSize-1)/2;
	const float* pSrc = 0;
	float*       pDst = 0;
	
	if(filterSize >= w-1)//if input image is very small, for example a small patch of an image
	{
		for(int i = 0; i<h; i++)
		{
			pSrc = srcMat + i;
		    pDst = dstMat + i;
			for(int j = 0; j<w; j++)
			{
				pDst[j*h] = 0;
				int ind   = 0;
				for(int s = 0; s<filterSize; s++)
				{
					ind = j-c+s;
					if(ind < 0)   ind = -ind;
					if(ind > w-1) ind = min(w-1,w-1-(ind-(w-1)));
					pDst[j*h] += pSrc[ind*h]*gfilter1d[s];
				}
			}
		}
    }
	else
	{
		for(int i = 0; i<h; i++)
		{
			pSrc = srcMat + i;
		    pDst = dstMat + i;
			for(int j = c; j<w-c; j++)
			{
				pDst[j*h] = 0;
				for(int s = 0; s<filterSize; s++)
				{
					pDst[j*h] += pSrc[(j-c+s)*h]*gfilter1d[s];
				}
			}
		}
		for(int i = 0; i<h; i++)
		{
			pSrc = srcMat + i;
		    pDst = dstMat + i;
			for(int j = 0; j<c; j++)
			{
				pDst[j*h] = 0;
				for(int s = 0; s<filterSize; s++)
				{
					pDst[j*h] += pSrc[abs(j+s-c)*h]*gfilter1d[s]; 
				}
			}
			for(int j = w-c; j<w; j++)
			{
				pDst[j*h] = 0;
				int ind   = 0;
				for(int s = 0; s<filterSize; s++)
				{
					ind = j-c+s;
					if(ind > w-1) ind = (w-1)-(ind-(w-1));
					pDst[j*h] += pSrc[ind*h]*gfilter1d[s]; 
				}
			}
		}
	}	
	return;
}
inline void conv2d(const float* srcMat, int h, int w, const float* gfilter1d, int filterSize, float* dstMat)
{
	if(NULL == srcMat)
	{
		return;
	}
	if(NULL == dstMat)// Do not creat the matrix inside the function
	{
		printf("The output of function conv2d() point to an invalid address!\n");
		return;
	}

	float* pTmp = (float *)malloc(sizeof(float)*h*w);
	
	conv2_h(srcMat, h, w, gfilter1d, filterSize, pTmp);
	conv2_v(pTmp,   h, w, gfilter1d, filterSize, dstMat);
	
	free(pTmp); pTmp = 0;
	
	return;
}
inline int get_filter_size(float sigma)
{
	int fsz = int(5*sigma)+1;
	float reminder = fmodf((float)fsz,2.0);
	int i_reminder = (int)reminder;
	if(0 == i_reminder)
	{
		fsz += 1;
	}

	if(fsz < 3)
	{
		fsz = 3;
	}
	return fsz;
}
inline void gaussian_filter_1d(float sigma, float mean, int filterSize, float* gfilter)
{
	int r = (int)(filterSize-1)/2;
	float sum = 0.0;
	for(int i = 0; i < filterSize; i++)
	{
		gfilter[i] = expf(-(i-r-mean)*(i-r-mean)/(sigma*sigma*2));
		sum += gfilter[i];
	}
	if(sum != 0)
		for(int i = 0; i < filterSize; i++)
			gfilter[i] /= sum;
}
/* inline int min(int a,int b)
{
	return (a>b)?(b):(a);
}
inline int max(int a,int b)
{
	return (a>b)?(a):(b);
} */
template<class T>
inline T min(T a, T b)
{
	return(a<b)?(a):(b);
}
template<class T>
inline T max(T a, T b)
{
	return(a<b)?(b):(a);
}