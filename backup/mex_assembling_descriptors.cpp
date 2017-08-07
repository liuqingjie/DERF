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
#include <math.h>
#include <string.h> // Call to memset

#define MAX_ITER 10

inline void assembling_descriptors( const float* H, size_t const *dims, const float* params, const float* grid, const float* coordinates, int num_coord, float* desc_out );
inline void assembling_descriptor( const float* H, size_t const *dims, const float* params, const float* grid, int y, int x, float* desc_out );

inline bool clip_vector( float* vec, int sz, float th );

inline void normalize_vector( float* vec, int hs );
inline void normalize_partial( float* desc, int gn, int hs );
inline void normalize_full( float* desc, int sz );
inline void normalize_derf(float* desc, int gn, int hs, int sz);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (nrhs != 4) {  printf("in nrhs != 5\n"); return; }
  if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) { printf("input 1 must be a single array\n"); return; } // H
  if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS) { printf("input 2 must be a single array\n"); return; } // params
  if (mxGetClassID(prhs[2]) != mxSINGLE_CLASS) { printf("input 3 must be a single array\n"); return; } // grid 
  if (mxGetClassID(prhs[3]) != mxSINGLE_CLASS) { printf("input 4 must be a single array\n"); return; } // y x .....
  if (nlhs != 1) { printf("out nlhs != 1\n"); return; }

  // Histograms
  int const num_dims1 = mxGetNumberOfDimensions(prhs[0]);
  size_t const *dims1    = mxGetDimensions(prhs[0]);
  float const *H     = (float *)mxGetData(prhs[0]);
  int h = dims1[1];
  int w = dims1[2];
  int nChannl = dims1[3];
  int nSize = h*w;

  // params
  float const *params = (float *)mxGetData(prhs[1]);
  int const nd_params = mxGetNumberOfDimensions(prhs[1]);
  size_t const *nparams  = mxGetDimensions(prhs[1]);
  int DS = params[0];
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
  float const *pcoord = (float*)mxGetData(prhs[3]);
  size_t const *dims_coord = mxGetDimensions(prhs[3]);
  int num_coord = dims_coord[0];

  // output
  size_t odim[] = {DS,num_coord};//each column is a descriptor
  plhs[0] = mxCreateNumericArray(2, odim, mxSINGLE_CLASS, mxREAL);
  float *desc_out  = (float *)mxGetData(plhs[0]);
  
  memset( desc_out, 0, sizeof(float)*DS*num_coord );
  
  assembling_descriptors(H, dims1, params, grid, pcoord, num_coord, desc_out);
}

inline void assembling_descriptors( const float* H, size_t const *dims, const float* params, const float* grid, const float* coordinates, int num_coord, float* desc_out)
{	
	int DS  = params[0];
	int h   = params[2];
    int w   = params[3];
	int y,x;
	float* _desc_out = 0;
	for(int i = 0; i < num_coord; i++)
	{
		y = (int)coordinates[i];
		x = (int)coordinates[i+num_coord];
		if(y>h-1 || x > w-1)
		{
			printf("point [%d,%d] outside the image, skip!!!\n",y,x);
			continue;
		}
		_desc_out = desc_out + i*DS;
		assembling_descriptor(H, dims, params, grid, y, x, _desc_out);
		normalize_derf(_desc_out, params[5], params[7], params[0]);
	}
}
inline void assembling_descriptor( const float* H, size_t const *dims, const float* params,  const float* grid, int y, int x, float* desc_out )
{
   int DS      = params[0];
   int nSample = params[1];
   //int h       = params[2];
   //int w       = params[3];
   int SQ      = params[4];
   int TQ      = params[5];
   int HQ      = params[6];
   int nBin    = params[7];
   
   int h       = dims[0];
   int w       = dims[1];
   int nSize   = dims[0]*dims[1];
   int nChannl = dims[2];
   int nLayer  = dims[3];

   int yy, xx;
 
   int current = 0;
   const float* cube=0;
   int nl = nSample;
   for(int nch=1;nch<=nChannl;nch++)
   {
	   for(int nS = 0; nS<nSample; nS ++)
	   {		   
		   int nsq = grid[nS];
		   float fyy = (float)(y-1) - grid[nS+1*nl];//coordinate in C/C++ start from 0, while in matlab start from 1;
		   float fxx = (float)(x-1) + grid[nS+2*nl];
		   if(1==nsq)
		   {
			   cube = H + (nch-1)*nSize;
			   yy = (int)fyy;
			   xx = (int)fxx;
			  			   
			   int nNbor = 0;
			   int neighbor[10] = {0,0,-1,-1,-1,1,1,1,1,-1};
			   for(int n=0;n<10;n+=2)
			   {
				   int ny = yy+neighbor[n];
				   int nx = xx+neighbor[n+1];
				   
				   if(ny >= 0 && ny < h && nx >= 0 && nx < w)
				   {
					   nNbor++;
					   desc_out[current] += cube[nx*h+ny];
				   }
			   }
			   desc_out[current] /= nNbor;			   
			   current ++;
		   }
		   else
		   {
			   yy = (int)fyy;
			   xx = (int)fxx;
			   float a = fyy - (float)yy;
			   float b = fxx - (float)xx;
			   //
			   int ny1 = yy;
			   int nx1 = xx+1;
			   int ny2 = yy+1;
			   int nx2 = xx;
			   int ny3 = yy+1;
			   int nx3 = xx+1;
			   
			   if(yy<0 || yy > h-1 || xx < 0 || xx > w-1
			   ||ny1<0 ||ny1 > h-1 || nx1 <0 || nx1 > w-1
			   ||ny2<0 ||ny2 > h-1 || nx2 <0 || nx2 > w-1
			   ||ny3<0 ||ny3 > h-1 || nx3 <0 || nx3 > w-1)
			   {
				   current++;
				   continue;
			   }

			   if(2 == nsq || SQ == nsq)
			   {
				   cube = H + (nch-1)*nSize + (nsq-2)*nSize*HQ;
				   
				   desc_out[current] 
				   = cube[xx*h+yy]*(1-a)*(1-b) 
				   + cube[nx1*h+ny1]*(1-a)*b 
			       + cube[nx2*h+ny2]*a*(1-b) 
			       + cube[nx3*h+ny3]*a*b;
				   
				   current ++;
			   }
			   else
		       {
			       for(int m = 0; m<3; m++)
			       {
				       cube = H + (nch-1)*nSize + (nsq-2+m-1)*nSize*HQ;
					   
					   desc_out[current] 
					   = cube[xx*h+yy]*(1-a)*(1-b) 
				       + cube[nx1*h+ny1]*(1-a)*b 
			           + cube[nx2*h+ny2]*a*(1-b) 
			           + cube[nx3*h+ny3]*a*b;
					   
				       current ++;					   
			       }
		       }
		   }
	   }
	}
}

inline void normalize_vector( float* hist, int hs )
{
   float s=0;
   for( int i=0; i<hs; i++ ) s+= hist[i]*hist[i];
   if( s!=0 ) {
      s = sqrt(s);
      for( int i=0; i<hs; i++ ) hist[i]/=s;
   }
}

inline void normalize_partial( float* desc, int gn, int hs )// gn = 8; hs = 65
{
    float* _desc = 0;
	float sum = 0;
    for(int h = 0; h<hs; h++)
	{
		float s = 0;
		_desc = desc + h;
		for(int g = 0; g<gn; g++)
		{
			s += _desc[g*hs]*_desc[g*hs];
		}
		sum += s;
	}
	if(sum!=0)
	{
		sum = sqrt(sum);
		for(int i = 0; i<gn*hs; i++) 
			desc[i] /= sum;
	}
}
inline void normalize_full( float* desc, int sz )
{
   normalize_vector(desc,sz);
}
inline bool clip_vector( float* vec, int sz, float th )
{
   bool retval=false;
   for( int i=0; i<sz; i++ )
      if( vec[i] > th ) {
         vec[i]=th;
         retval = true;
      }
   return retval;
}
inline void normalize_derf(float* desc, int gn, int hs, int sz)
{
	int iter = 0;
	bool change = true;
	float threshold = 1.6/sqrt((float)(gn*hs)); 
	while( iter<MAX_ITER && change )
    {
		normalize_partial(desc,gn,hs);
		change = clip_vector( desc, sz, threshold );
		iter++;
	}
}