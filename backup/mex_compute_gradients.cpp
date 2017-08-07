
#include <mex.h>
#include <math.h>
#include <malloc.h>
#include <string.h> // Call to memset
#include "conv.h"

#define PI 3.14159265

inline void get_orientations(float* ori, int nOri);
inline void compute_gradients(const float* srcMat, int h, int w, const float* ori, int numOfOris, float* gradMat);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) //DOG = mex_get_DOG(L,SQ);
{
  if (nrhs != 2) {  printf("in nrhs != 2\n"); return; }
  if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) { printf("input 1 must be a single array\n"); return; } // L
  if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) { printf("input 2 must be a single array\n"); return; } // SQ

  if (nlhs != 1) { printf("out nlhs != 1\n"); return; }

  // Histograms
  int const num_dims = mxGetNumberOfDimensions(prhs[0]);
  size_t const *dims = mxGetDimensions(prhs[0]);
  float const *image = (float *)mxGetData(prhs[0]);
  
  int h        = dims[0];
  int w        = dims[1];

  float* _HQ = (float *)mxGetData(prhs[1]); //number of the orientations
  int HQ = (int)_HQ[0];
  
  #ifdef DEBUG
  printf("----------------------------------------------\n");
  printf("H         : %d\n",h );
  printf("W         : %d\n",w );
  printf("HQ        : %d\n",HQ );
  printf("----------------------------------------------\n");
  #endif
  
  float* ori = (float *)malloc(sizeof(float)*HQ);
  get_orientations(ori,HQ);
  // output
  size_t odim[] = {h, w, HQ};
  plhs[0] = mxCreateNumericArray(3, odim, mxSINGLE_CLASS, mxREAL);
  float *g_img  = (float *)mxGetData(plhs[0]);

  memset( g_img, 0, sizeof(float)*odim[0]*odim[1]*odim[2] );
  compute_gradients(image,h,w,ori,HQ,g_img);
  
  free(ori);
}
inline void compute_gradients(const float* srcMat, int h, int w, const float* ori, int nOri, float* gradMat)
{
	float* dx = (float *)malloc(sizeof(float)*h*w);
	float* dy = (float *)malloc(sizeof(float)*h*w);
	float* tp = (float *)malloc(sizeof(float)*h*w);
	
	float gfilter[5];
	gaussian_filter_1d(0.5f,0.0f,5,gfilter);
	conv2_h(srcMat,h,w,gfilter,5,dx);
	conv2_v(dx,h,w,gfilter,5,tp);
	
	float gf[] = {-0.5,0,0.5};	
	conv2_h(tp,h,w,gf,3,dx);
	conv2_v(tp,h,w,gf,3,dy);
	
	float* pGrad    = 0;
	float* _gradMat = 0;
	float* _dx      = 0;
	float* _dy      = 0;
    for(int n = 0; n<nOri; n++)
	{
		pGrad = gradMat + n*h*w;
		double dori = (double)ori[n];
		for(int j = 0; j<w; j++)
	    {
		    _gradMat = pGrad    + j*h;
		    _dx       = dx      + j*h;
		    _dy       = dy      + j*h;
		    for(int i = 0; i<h; i++)
		        {
			        _gradMat[i] = max(0.0f,(float)((double)_dx[i]*cos(dori) + (double)_dy[i]*sin(dori)));
		        }
     	}
	}
	
	free(dx); dx = 0;
	free(dy); dy = 0;
	free(tp); tp = 0;
}
inline void get_orientations(float* ori, int nOri)
{
	for(int i = 0; i<nOri; i++) ori[i] = (float)2*i*PI/nOri;
}