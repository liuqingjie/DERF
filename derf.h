/////////////////////////////////////////////////////////////////////////
//                                                                     //
// Written  by                                                         //
// Qingjie Liu                                                         //
//                                                                     //
// web   :                                                             //
// email : qingjie.liu@buaa.edu.cn                                     //
//                                                                     //
/////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <string.h> // Call to memset

#define MAX_ITER 10

inline void assembling_descriptor( const float* H, size_t const *dims, const float* params, const float* grid, int y, int x, float* desc_out );

inline bool clip_vector( float* vec, int sz, float th );

inline void normalize_vector( float* vec, int hs );
inline void normalize_partial( float* desc, int gn, int hs );
inline void normalize_full( float* desc, int sz );
inline void normalize_derf(float* desc, int gn, int hs, int sz);

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
		clip_vector( desc, sz, threshold );
		iter++;
	}
}
