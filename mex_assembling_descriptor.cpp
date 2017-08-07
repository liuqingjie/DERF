/////////////////////////////////////////////////////////////////////////
//                                                                     //
// Written  by                                                         //
// Qingjie Liu                                                         //
//                                                                     //
// web   :                                                             //
// email : qingjie.liu@buaa.edu.cn                                     //
//                                                                     //
/////////////////////////////////////////////////////////////////////////

#include <mex.h>
#include <string.h> // Call to memset
#include "derf.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (nrhs != 4) {  printf("in nrhs != 5\n"); return; }
  if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) { printf("input 1 must be a single array\n"); return; } // H
  if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) { printf("input 2 must be a single array\n"); return; } // params
  if (mxGetClassID(prhs[2]) != mxSINGLE_CLASS) { printf("input 3 must be a single array\n"); return; } // grid 
  if (mxGetClassID(prhs[3]) != mxINT32_CLASS)  { printf("input 4 must be a Int array\n"); return; } // y x
  if (nlhs != 1) { printf("out nlhs != 1\n"); return; }

  // Histograms
  int const num_dims1    = mxGetNumberOfDimensions(prhs[0]);
  size_t const *dims1    = mxGetDimensions(prhs[0]);
  float const *H         = (float *)mxGetData(prhs[0]);
  int h                  = dims1[0];
  int w                  = dims1[1];
  int nChannl            = dims1[2];
  int nSize              = h*w;

  // params
  float const *params    = (float *)mxGetData(prhs[1]);
  int const nd_params    = mxGetNumberOfDimensions(prhs[1]);
  size_t const *nparams  = mxGetDimensions(prhs[1]);
  int DS                 = params[0];
#ifdef DEBUG
  printf("----------------------------------------------\n");
  printf("DS : %f\n",params[0] ); //dimension of derf 520
  printf("HN : %f\n",params[1] ); //number of sample points 33 = 1+8+8+8+8
  printf("H  : %f\n",params[2] ); //height of the image
  printf("W  : %f\n",params[3] ); //width of the image
  printf("SQ : %f\n",params[4] ); //number of rings 5
  printf("TQ : %f\n",params[5] ); //number of samples in a circle 8
  printf("HQ : %f\n",params[6] ); //number of the gradient orientation 8
  printf("RQ : %f\n",params[7] ); //number of bins 65
  printf("----------------------------------------------\n");
#endif

  // grid info
  float const *grid = (float *)mxGetData(prhs[2]);
  int const nd_grid = mxGetNumberOfDimensions(prhs[2]);
  size_t const *ngrid  = mxGetDimensions(prhs[2]);

  // coordinates
  int* pcoord = (int*)mxGetData(prhs[3]);
  int y = pcoord[0];
  int x = pcoord[1];

  // output
  size_t odim[] = {DS,1};
  plhs[0] = mxCreateNumericArray(2, odim, mxSINGLE_CLASS, mxREAL);
  float *desc_out  = (float *)mxGetData(plhs[0]);
  
  memset( desc_out, 0, sizeof(float)*odim[0]*odim[1] );
  
  assembling_descriptor(H, dims1, params, grid, y, x, desc_out);
  normalize_derf(desc_out, params[5], params[7], params[0]);
}