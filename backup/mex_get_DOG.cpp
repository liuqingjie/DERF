
#include <mex.h>
#include <math.h>
#include <malloc.h>
#include <string.h> // Call to memset

inline int get_filter_size(float sigma);

inline void gaussian_filter_1d(float sigma, float mean, int filterSize, float* gfilter);

inline void smooth_layers(const float* L, int h, int w, int nChannel,float sigma,float* smoothedL);

inline void conv2d(const float* matrixA,int h, int w, const float * gfilter1d, int filterSize, float* matrixB);

inline void compute_DOG(const float* L, int h, int w, int nChannel, const float* sigma, int SQ, float* DOG);

inline void sub(const float* matrixA, const float* matrixB, int h, int w, float* matrixC);

inline void tensor_sub(const float* matrixA, const float* matrixB, int h, int w, int nChannel, float* matrixC);

inline void tensor_abs(float* matrixA, int h, int w, int nChannel, int nCube);

inline int min(int a, int b);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) //DOG = mex_get_DOG(L,SQ);
{
  if (nrhs != 2) {  printf("in nrhs != 2\n"); return; }
  if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) { printf("input 1 must be a single array\n"); return; } // L
  if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) { printf("input 2 must be a single array\n"); return; } // SQ

  if (nlhs != 1) { printf("out nlhs != 1\n"); return; }

  // Histograms
  int const num_dims1 = mxGetNumberOfDimensions(prhs[0]);
  size_t const *dims1    = mxGetDimensions(prhs[0]);
  float const *L     = (float *)mxGetData(prhs[0]);
  
  int h        = dims1[0];
  int w        = dims1[1];
  int nChannel = dims1[2];

  float* _SQ = (float *)mxGetData(prhs[1]);
  int SQ = (int)_SQ[0];
  
  #ifdef DEBUG
  printf("----------------------------------------------\n");
  printf("H         : %d\n",h );
  printf("W         : %d\n",w );
  printf("nChannel  : %d\n",nChannel );
  printf("SQ        : %d\n",SQ );
  printf("----------------------------------------------\n");
#endif
  
  float sigms[5] = {1, 1.5874, 2.916666, 4.70588, 7.47011};

  // output
  size_t odim[] = {h, w, nChannel, SQ-1};
  plhs[0] = mxCreateNumericArray(4, odim, mxSINGLE_CLASS, mxREAL);
  float *DOG_out  = (float *)mxGetData(plhs[0]);

  memset( DOG_out, 0, sizeof(float)*h*w*nChannel*(SQ-1) );
  compute_DOG(L, h, w, nChannel, sigms, SQ, DOG_out);
}

inline void compute_DOG(const float* L, int h, int w, int nChannel, const float* sigma, int SQ, float* DOG)
{
	if(NULL == DOG)
	{
		printf("The output of funtion compute_DOG() is invalid!!!\n");
	}
	
	float* pTmp = 0;
	float* pT   = 0;

	pTmp = (float *)malloc(sizeof(float)*h*w*nChannel*SQ);
	
	for(int r = 0; r < 3; r++)
	{
		pT = pTmp + h*w*nChannel*r;
		smooth_layers(L, h, w, nChannel, sigma[r], pT);
	}
	float* pTT = 0;
	for(int r = 3; r < SQ; r++)
	{
		float _sigma = sqrt(sigma[r]*sigma[r] - sigma[r-1]*sigma[r-1]);
		pT  = pTmp + h*w*nChannel*(r-1);
		pTT = pTmp + h*w*nChannel*r;
		smooth_layers(pT, h, w, nChannel, _sigma, pTT);
	}

	float* pT1  = 0;
	float* pT2  = 0;
	float* pTmp1 = (float *)malloc(sizeof(float)*h*w*nChannel*(SQ-1));
	float* _pTmp1 = 0;
	for(int r = 0; r<SQ-1; r++)
	{
		_pTmp1 = pTmp1 + h*w*nChannel*r;
		pT1    = pTmp  + h*w*nChannel*r;
		pT2    = pTmp  + h*w*nChannel*(r+1);
		tensor_sub(pT1,pT2,h,w,nChannel,_pTmp1);
	}
	
	tensor_abs(pTmp1,h,w,nChannel,SQ-1);
	
	/*
	int numofver = 1; //What Shit is this?
	
	float* pDOG = 0; _pTmp1 = 0;
	for(int j = 0; j<numofver; j++)
	{
		for(int i = 0; i<SQ-1; i++)
		{
			pDOG   = DOG   + h*w*nChannel*(i*numofver+j);
			_pTmp1 = pTmp1 + h*w*nChannel*(i*numofver+j);
			smooth_layers(_pTmp1,h,w,nChannel,1.2,pDOG);
		}
	}
	*/
	float* pDOG = 0; 
	for(int i = 0; i<SQ-1; i++)
	{

		_pTmp1 = pTmp1 + h*w*nChannel*i;
		pDOG   = DOG   + h*w*nChannel*i;		
		smooth_layers(_pTmp1,h,w,nChannel,1.2,pDOG);
	}
	
	free(pTmp1);  pTmp1 = 0;
	free(pTmp);   pTmp  = 0;
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

inline void smooth_layers(const float* L, int h, int w, int nChannel, float sigma, float* smoothedL)
{
	if(NULL == L)
	{
		printf("The input L of function smooth_layers() point to an invalid address!\n");
		return;
	}
	if(NULL == smoothedL)// Do not creat the matrix inside the function
	{
		printf("The output of function smooth_layers() point to an invalid address!\n");
		return;
	}
	
	int fsz = get_filter_size(sigma);
	float* gfilter1d = 0;
	gfilter1d = (float *)malloc(sizeof(float)*fsz);
	gaussian_filter_1d(sigma,0.0,fsz,gfilter1d);
	
	const float* pL = 0;
	float* psmoothedL = 0;
	for(int nch = 0; nch < nChannel; nch++)
	{
		pL         = L         + nch*h*w;
		psmoothedL = smoothedL + nch*h*w;
		conv2d(pL,h,w,gfilter1d,fsz,psmoothedL);
	}
	free(gfilter1d);gfilter1d = 0;
	return;
}

inline void conv2d(const float* matrixA, int h, int w, const float* gfilter1d, int filterSize, float* matrixB)
{
	if(NULL == matrixA)
	{
		return;
	}
	if(NULL == matrixB)// Do not creat the matrix inside the function
	{
		printf("The output of function conv2d() point to an invalid address!\n");
		return;
	}

	float* pTmp     = 0;
	pTmp            = (float *)malloc(sizeof(float)*h*w);
	int c           = (int)(filterSize-1)/2;
	const float* pA = 0;
	float*       pB = 0;
	float*       pT = 0;
	
	if(filterSize > h-1 || filterSize > w-1)//if input image is very small, for example a small patch of an image
	{
		for(int j = 0; j<w; j++)
		{
			pA = matrixA + j*h;
		    pT = pTmp    + j*h;
			for(int i = 0; i<h; i++)
		    {
			    pT[i] = 0;
			    int ind = 0;
			    for(int s = 0; s<filterSize; s++)
			    {
				    ind = i-c+s;
				    if(ind < 0)   ind = -ind;
					if(ind > h-1) ind = min((h-1)-(ind-(h-1)),(h-1));
				    pT[i] += pA[ind]*gfilter1d[s];
			    }
		    }
		}

		for(int i = 0; i<h; i++)
		{
			pB = matrixB + i;
		    pT = pTmp    + i;
			for(int j = 0; j<w; j++)
			{
				pB[j*h] = 0;
				int ind = 0;
				for(int s = 0; s<filterSize; s++)
				{
					ind = j-c+s;
					if(ind < 0)   ind = -ind;
					if(ind > w-1) ind = min(w-1,w-1-(ind-(w-1)));
					pB[j*h] += pB[ind*h]*gfilter1d[s];
				}
			}
		}
    }
	else
	{
		//vertical convolution
		for(int j = 0; j<w; j++)
		{
			pA = matrixA + j*h;
			pT = pTmp    + j*h;
			for(int i = c; i<h-c; i++)
			{
				pT[i] = 0;
				for(int s = 0; s<filterSize; s++)
				{
					pT[i] += pA[i-c+s]*gfilter1d[s];
				}
			}
		}
		//process the boundary
		//symmetric padding
		for(int j = 0; j<w; j++)
		{
			pA = matrixA + j*h;
			pT = pTmp    + j*h;
			for(int i = 0; i<c; i++)
			{
				pT[i] = 0;
				for(int s = 0; s<filterSize; s++)
				{
					pT[i] += pA[abs(i+s-c)]*gfilter1d[s]; 
				}
			}
			for(int i = h-c; i<h; i++)
			{
				pT[i] = 0;
				int ind = 0;
				for(int s = 0; s<filterSize; s++)
				{
					ind = i-c+s;
					if(ind > h-1) ind = (h-1)-(ind-(h-1));
					pT[i] += pA[ind]*gfilter1d[s]; 
				}
			}
		}
			
		//horizontal convolution
		for(int i = 0; i<h; i++)
		{
			pB = matrixB + i;
			pT = pTmp    + i;
			for(int j = c; j<w-c; j++)
			{
				pB[j*h] = 0;
				for(int s = 0; s<filterSize; s++)
				{
					pB[j*h] += pT[(j-c+s)*h]*gfilter1d[s];
				}
			}
		}
		//process the boundary
		//using symmetric padding
		for(int i = 0; i<h; i++)
		{
			pB = matrixB + i;
			pT = pTmp    + i;
			for(int j = 0; j<c; j++)
			{
				pB[j*h] = 0;
				for(int s = 0; s<filterSize; s++)
				{
					pB[j*h] += pT[abs(j+s-c)*h]*gfilter1d[s]; 
				}
			}
			for(int j = w-c; j<w; j++)
			{
				pB[j*h] = 0;
				int ind = 0;
				for(int s = 0; s<filterSize; s++)
				{
					ind = j-c+s;
					if(ind > w-1) ind = (w-1)-(ind-(w-1));
					pB[j*h] += pT[ind*h]*gfilter1d[s]; 
				}
			}
		}
	}

	free(pTmp); pTmp = 0;
	
	return;
}

inline void sub(const float* matrixA, const float* matrixB, int h, int w, float* matrixC)
{
	for(int i = 0; i<h*w; i++) matrixC[i] = matrixA[i] - matrixB[i];
}

inline void tensor_sub(const float* matrixA, const float* matrixB, int h, int w, int nChannel, float* matrixC)
{
	for(int i = 0; i < h*w*nChannel; i++)	matrixC[i] = matrixA[i] - matrixB[i];
}

inline void tensor_abs(float* matrixA,int h, int w, int nChannel, int nCube)
{
	for(int i = 0; i < h*w*nChannel*nCube; i++) matrixA[i] = abs(matrixA[i]);
}

inline int min(int a,int b)
{
	return a>b?b:a;
}