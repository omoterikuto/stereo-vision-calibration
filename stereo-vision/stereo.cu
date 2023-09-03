/**
	This file is part of z2zncc. 

	Copyright (c) 2021 Qiong Chang.

	z2zncc is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	z2zncc is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with z2zncc.  If not, see <http://www.gnu.org/licenses/>.

**/


#include "function.h"
#include "stereo.h"
#include "cuda_fp16.h"
#include <stdlib.h>


void exp_computing(void){
  for(int i = 0; i < 256; i++){
    exp_list_sub_cpu_1[i]= exp(-double(i)/SIGMA_A);
    exp_list_sub_cpu_2[i]= exp(-double(i)/SIGMA_B);
  }
  for(int t = 0; t < SIGMA_TH; t++){
    exp_list_sub_cpu_2[t] = 0.8f;
  }
}

void gpu_initial(int width, int height){


  exp_computing();
  cudaMemcpyToSymbol(exp_list_sub_1, exp_list_sub_cpu_1, sizeof(float) * 256);
  cudaMemcpyToSymbol(exp_list_sub_2, exp_list_sub_cpu_2, sizeof(float) * 256);
   
  cudaMalloc((void **)&dst_temp, sizeof(uint8)*width*height);
  cudaMalloc((void **)&costL2R, sizeof(uint8)*width*height*MAX_DISPARITY);
  cudaMalloc((void **)&costR2L, sizeof(uint16)*width*height*MAX_DISPARITY);
  cudaMalloc((void **)&costU2D, sizeof(uint16)*width*height*MAX_DISPARITY);
  cudaMalloc((void **)&avrimg1, sizeof(float)*width*height);
  cudaMalloc((void **)&avrimg2, sizeof(float)*width*height);
  cudaMalloc((void **)&sum2_L, sizeof(float)*width*height);
  cudaMalloc((void **)&sum2_R, sizeof(float)*width*height);

}

__device__ float ncc_cost(float sum, float avr_left, float avr_right, float avr2_left, float avr2_right){
  
  float sum_temp = sum-PIX_NUM*avr_left*avr_right;
  sum_temp = sum_temp*avr2_left*avr2_right;
  sum_temp = fminf(fabsf(1-sum_temp), 1.0f);
  return sum_temp;
}



__global__ void __avr_fast__(const uint8* src1, const uint8* src2, float* output_left, float* output_right,float* output2_left, float* output2_right, int M, int N){
  const int tidy = blockIdx.y*blockDim.y+threadIdx.y;
  const int tidx = blockIdx.x*blockDim.x+threadIdx.x;

  __shared__ uint8 img_left[AVR_WARP_V+NCC_W-1][AVR_WARP_H+NCC_W-1];
  __shared__ uint8 img_right[AVR_WARP_V+NCC_W-1][AVR_WARP_H+NCC_W-1];


  int y = tidy-NCC_R;
  int x = tidx-NCC_R;
  
  y=y<0?0:y;
  x=x<0?0:x;
  
  img_left[threadIdx.y][threadIdx.x] = src1[y*M+x];
  img_right[threadIdx.y][threadIdx.x] = src2[y*M+x];

  int x_temp = tidx-NCC_R+AVR_WARP_H>M-1?M-1:tidx-NCC_R+AVR_WARP_H;
  int y_temp = tidy-NCC_R+AVR_WARP_V>N-1?N-1:tidy-NCC_R+AVR_WARP_V;
  x_temp = x_temp<0?0:x_temp;
  y_temp = y_temp<0?0:y_temp;

  
  if(threadIdx.x<NCC_W-1){
   
    img_left[threadIdx.y][threadIdx.x+AVR_WARP_H] = src1[y*M+x_temp];
    img_right[threadIdx.y][threadIdx.x+AVR_WARP_H] = src2[y*M+x_temp];
  }

  if(threadIdx.y<NCC_W-1){
    img_left[threadIdx.y+AVR_WARP_V][threadIdx.x] = src1[y_temp*M+x];
    img_right[threadIdx.y+AVR_WARP_V][threadIdx.x] = src2[y_temp*M+x];
  }

  if(threadIdx.x<NCC_W-1&&threadIdx.y<NCC_W-1){
    img_left[threadIdx.y+AVR_WARP_V][threadIdx.x+AVR_WARP_H] = src1[y_temp*M+x_temp];
    img_right[threadIdx.y+AVR_WARP_V][threadIdx.x+AVR_WARP_H] = src2[y_temp*M+x_temp];
  }
  __syncthreads();

  
  float s2L= 0.0f, s2R = 0.0f;
  float avrL = 0.0f, avrR= 0.0f;

#pragma unroll
  for(int r = 0; r<NCC_W; r++){
    for(int c = 0; c < NCC_W;c++){
      float left_data = img_left[threadIdx.y+r][threadIdx.x+c];
      float right_data = img_right[threadIdx.y+r][threadIdx.x+c];
      avrL += left_data;
      s2L += (left_data*left_data);
      avrR += right_data;
      s2R += (right_data*right_data);
    }
  }
  if(tidx<M&&tidy<N){
    output_left[tidy*M+tidx] = (avrL*AVR_RECIP);
    output_right[tidy*M+tidx] = (avrR*AVR_RECIP);
    output2_left[tidy*M+tidx] = rsqrtf(s2L-avrL*avrL*AVR_RECIP);
    output2_right[tidy*M+tidx] = rsqrtf(s2R-avrR*avrR*AVR_RECIP);
  }

  
  
}



template<typename T>
__global__ void _StereoMatching_Kernel(const uint8* src1, const uint8* src2, const T* intimg_left_v, const T *intimg_right_v, const float* intimg_left_v2, const float* intimg_right_v2, uint8 * costL2R, const int width, const int height, uint8 *dst){
  const int y = blockIdx.x*MATCH_V_SEG;
  const int THRid = threadIdx.x;
 
  __shared__ uint8 SharedMatch[MATCH_W_V][MATCH_W_H+MAX_DISPARITY];
  __shared__ uint8 SharedBase[MATCH_W_V][MATCH_W_H];
  __shared__ float SharedAVR_left[MATCH_V_SEG][MATCH_H_SEG];
  __shared__ float SharedAVR2_left[MATCH_V_SEG][MATCH_H_SEG];
  __shared__ float SharedAVR_right[MATCH_V_SEG][MATCH_H_SEG+MAX_DISPARITY];
  __shared__ float SharedAVR2_right[MATCH_V_SEG][MATCH_H_SEG+MAX_DISPARITY];

      
  for(int row = 0; row < MATCH_W_V; row++){
    int y_temp = y+row-NCC_R<0?0:y+row-NCC_R;
    y_temp = y_temp>height-1?height-1:y_temp;
    
    SharedMatch[row][MATCH_W_H+THRid]=src2[(y_temp)*width];
    if(row<MATCH_V_SEG){
      SharedAVR_right[row][MATCH_H_SEG+THRid] = intimg_right_v[(y+row)*width];
      SharedAVR2_right[row][MATCH_H_SEG+THRid] = intimg_right_v2[(y+row)*width];
    }
  }
  __syncthreads();

  int n_iter = (width+MATCH_H_SEG-1)/(MATCH_H_SEG);
  uint8 img_temp = 0;
  float avr_temp = 0;
  float sum_v[MATCH_V_SEG];
  float cost_sum[MATCH_V_SEG];
  int sub = 0;
  float e1 = 0.0f;
  int index = 0;

  for(int i = 0; i <MATCH_V_SEG; i++ ){
    sum_v[i] = 0.0f;
    cost_sum[i] = 0.0f;
  }
  __syncthreads();
  for(int ix = 0; ix<n_iter; ix++){
    const int x = ix*MATCH_H_SEG;

    for(int row = 0; row < MATCH_W_V; row++){
      img_temp = SharedMatch[row][THRid+MATCH_W_H];
      __syncthreads();
	
      SharedMatch[row][THRid+NCC_W-1] = img_temp;
      if(row<MATCH_V_SEG){
	avr_temp = SharedAVR_right[row][THRid+MATCH_H_SEG];
	__syncthreads();
	SharedAVR_right[row][THRid] = avr_temp;
	avr_temp = SharedAVR2_right[row][THRid+MATCH_H_SEG];
	__syncthreads();
	SharedAVR2_right[row][THRid] = avr_temp;
      }
      int x_temp = x+THRid-NCC_R<0?0:x+THRid-NCC_R;
      int y_temp = y+row-NCC_R<0?0:y+row-NCC_R;
      x_temp = x_temp>width-1?width-1:x_temp; 
      y_temp = y_temp>height-1?height-1:y_temp;
      if(THRid<MATCH_W_H){
	SharedMatch[row][THRid+MAX_DISPARITY]=src2[(y_temp)*width+x_temp];
	SharedBase[row][THRid] = src1[(y_temp)*width+x_temp];
	if(row<MATCH_V_SEG&&THRid<MATCH_H_SEG){
	  SharedAVR_left[row][THRid] = intimg_left_v[(y+row)*width+x+THRid];
	  SharedAVR2_left[row][THRid] = intimg_left_v2[(y+row)*width+x+THRid];
	  SharedAVR_right[row][THRid+MAX_DISPARITY] = intimg_right_v[(y+row)*width+x+THRid];
	  SharedAVR2_right[row][THRid+MAX_DISPARITY] = intimg_right_v2[(y+row)*width+x+THRid];
 	}
      }
    }
    __syncthreads();

    float sum = 0.0f;
    float base, match;
    
    for(int row = 0; row < MATCH_W_V; row++){
      float sum_h = 0.0f;
      for(int i = 0; i < NCC_W; i++){
	base = SharedBase[row][i];
	match = SharedMatch[row][MAX_DISPARITY-THRid+i];
	sum_h = fmaf(base,match,sum_h);
      }
      sum += sum_h;
 
      if(row<MATCH_V_SEG)
	sum_v[row] = sum_h;
 
      __syncthreads();
      if(row>=NCC_W-1){
	
	int y_t= y+row-NCC_W+1<height?y+row-NCC_W+1:height-1;
	float t = ncc_cost(sum, SharedAVR_left[row-NCC_W+1][0],SharedAVR_right[row-NCC_W+1][MAX_DISPARITY-THRid],SharedAVR2_left[row-NCC_W+1][0], SharedAVR2_right[row-NCC_W+1][MAX_DISPARITY-THRid]);

	sub = abs(SharedBase[row-NCC_R][NCC_R]-SharedBase[row-NCC_R][NCC_R-1]);

	index = ((y_t)*width+x)*MAX_DISPARITY+THRid;
	
	e1=exp_list_sub_1[sub];

	cost_sum[row-NCC_W+1] = fmaf(cost_sum[row-NCC_W+1],e1, t);	
	if(x<width){
	costL2R[index] = (cost_sum[row-NCC_W+1]);
      }
	
	sum_h = sum;
	sum -=sum_v[row-NCC_W+1];
	sum_v[row-NCC_W+1] = sum_h;
      }
      
    }

    for(int i = NCC_W; i < MATCH_W_H; i++){
      float sum1= 0.0f, sum2=0.0f;
      for(int row = 0; row < MATCH_W_V; row++){
    	base = SharedBase[row][i];
    	match = SharedMatch[row][MAX_DISPARITY-THRid+i];
    	sum1 = fmaf(base,match, sum1);   
    	base = SharedBase[row][i-NCC_W];
    	match = SharedMatch[row][MAX_DISPARITY-THRid+i-NCC_W];
    	sum2 = fmaf(base,match,sum2);
	
    	int y_t= y+row-NCC_W+1<height?y+row-NCC_W+1:height-1;
	int x_t= x+i-NCC_W+1<width?x+i-NCC_W+1:width-1;
	index  =((y_t)*width+x_t)*MAX_DISPARITY+THRid;
    	  
    	if(row==NCC_W-1){
    	  sum_v[0] = sum_v[0]+sum1-sum2;
    	  float t = ncc_cost(sum_v[0], SharedAVR_left[0][i-NCC_W+1],SharedAVR_right[0][MAX_DISPARITY-THRid+i-NCC_W+1],SharedAVR2_left[0][i-NCC_W+1],SharedAVR2_right[0][MAX_DISPARITY-THRid+i-NCC_W+1]);
    	  cost_sum[0] = fmaf(cost_sum[0],e1, t);
	  costL2R[index] = cost_sum[0];
	}	

	if(row>NCC_W-1){
	  base = SharedBase[row-NCC_W][i];
    	  match = SharedMatch[row-NCC_W][MAX_DISPARITY-THRid+i];
    	  sum1 = fmaf(-base,match,sum1);   
	  
    	  base = SharedBase[row-NCC_W][i-NCC_W];
    	  match = SharedMatch[row-NCC_W][MAX_DISPARITY-THRid+i-NCC_W];
    	  sum2 = fmaf(-base,match,sum2);
    	  sum_v[row-NCC_W+1] +=(sum1-sum2);
 
    	  float t = ncc_cost(sum_v[row-NCC_W+1], SharedAVR_left[row-NCC_W+1][i-NCC_W+1],SharedAVR_right[row-NCC_W+1][MAX_DISPARITY-THRid+i-NCC_W+1], SharedAVR2_left[row-NCC_W+1][i-NCC_W+1], SharedAVR2_right[row-NCC_W+1][MAX_DISPARITY-THRid+i-NCC_W+1]);

	  sub = abs(SharedBase[row-NCC_R][i-NCC_R]-SharedBase[row-NCC_R][i-NCC_R-1]);
	  e1=exp_list_sub_1[sub];
	  cost_sum[row-NCC_W+1] = fmaf(cost_sum[row-NCC_W+1],e1, t);
    	  costL2R[index] = (cost_sum[row-NCC_W+1]);
    	}
      }
    }    
  }
}


template<typename T>
__global__ void _CostAggregation_Kernel_R2L(const uint8* src1, T* cost_input, uint16* cost_output, int width, int height){
  const int y  =  blockIdx.x*ROWS;  
  const int THRid = blockDim.x*threadIdx.y+threadIdx.x; 
  __shared__ uint8 img[ROWS][1281];
  
#pragma unroll
  for(int ix = 0; ix < (width+MAX_DISPARITY-1)/MAX_DISPARITY; ix++){
    const int x = ix*MAX_DISPARITY; 
    for(int m = 0; m < ROWS; m++){
      img[m][THRid+x+1] = src1[(y+m)*width+x+THRid];
    }   
  }
  __syncthreads();

  float cost_current[AGG_STEP+1][ROWS];
  float cost_sum[ROWS];
  
  for(int i = 0 ; i < AGG_STEP; i++){
    for(int j = 0; j < ROWS; j++){
      cost_sum[j] = 0;
      cost_current[i][j]=0;
    
    }
  }
  
  uint8 sub=0;
  float e1=0;

#pragma unroll 
  for(int i=width-1; i>0 ;i-=AGG_STEP){
#pragma unroll
    for(int count = 0; count < AGG_STEP; count++){
      for(int warpid = 0; warpid < MAX_DISPARITY/WARP_SIZE; warpid++){
	int index_temp = ((y+threadIdx.y)*width+(i-count))*MAX_DISPARITY+threadIdx.x+warpid*WARP_SIZE;
	int temp_local =  cost_input[index_temp];
	cost_current[count][warpid] = temp_local;
      }
    }
    __syncthreads();
#pragma unroll
    for(int count = 0; count < AGG_STEP; count++){
      int offset = (i-count);
      sub = abs(img[threadIdx.y][offset+1]-img[threadIdx.y][offset]);

      e1 = exp_list_sub_1[sub];
	
#pragma unroll
      for(int warpid = 0; warpid < MAX_DISPARITY/WARP_SIZE; warpid++){
	cost_sum[warpid] = fmaf(cost_sum[warpid],e1, cost_current[count][warpid]);
	cost_output[((y+threadIdx.y)*width+offset)*MAX_DISPARITY+threadIdx.x+warpid*WARP_SIZE] = uint16(cost_sum[warpid]);
      }      
    }
  }
}


template<typename T>
__global__ void _CostAggregation_Kernel_U2D(const uint8* src1 ,  T* cost_input, uint16* cost_output, int width, int height){

  const int x  =  blockIdx.x*ROWS; 
  const int THRid = blockDim.x*threadIdx.y+threadIdx.x;
  __shared__ uint8 img[ROWS][420];
  __shared__ uint16 sharedstorage;

#pragma unroll
  for(int ix = 0; ix < (height+MAX_DISPARITY-1)/MAX_DISPARITY; ix++){
    const int y = ix*MAX_DISPARITY; 
    for(int m = 0; m < ROWS; m++){
      img[m][THRid+y+1] = src1[(y+THRid)*width+(x+m)];
      
    }
  }

  __syncthreads();
  uint16 cost_current[AGG_STEP+1][ROWS];
  uint16 cost_sum[ROWS];

  
  for(int j = 0; j < ROWS; j++){
    cost_sum[j] = 0;
    for(int i = 0; i < AGG_STEP+1; i++)
      cost_current[i][j]=0;
  }

  uint8 sub=0;
  int index[AGG_STEP];
  float e2 = 0.0f;
#pragma unroll
  for(int i=0; i<height-1 ;i+=AGG_STEP){
    for(int warpid = 0; warpid < MAX_DISPARITY/WARP_SIZE; warpid++){
      cost_current[0][warpid]=cost_input[(i*width+x+threadIdx.y)*MAX_DISPARITY+threadIdx.x+(warpid)*WARP_SIZE];
    }

#pragma unroll
    for(int count = 0; count < AGG_STEP; count++){  
      int offset = i+count;
      index[count] = (offset*width+x+threadIdx.y)*MAX_DISPARITY+threadIdx.x;

      sub = abs(img[threadIdx.y][offset+1]-img[threadIdx.y][offset]);
      e2=exp_list_sub_2[sub];

      if(threadIdx.x==0)
	sharedstorage = cost_current[count][0];
      __syncthreads();

#pragma unroll
      for(int warpid = 0; warpid < MAX_DISPARITY/WARP_SIZE; warpid++){
	cost_current[count+1][warpid] = cost_input[((i+count+1)*width+x+threadIdx.y)*MAX_DISPARITY+threadIdx.x+(warpid)*WARP_SIZE];

	cost_sum[warpid] = (cost_sum[warpid]*e2)+(((cost_current[count][warpid])-sharedstorage)>>1);
	cost_output[index[count]+warpid*WARP_SIZE]=(cost_sum[warpid]);
      }
      
    }    
  }
}



template<typename T>
__global__ void _CostAggregation_Kernel_D2U(const uint8* src1 , T* cost_input, uint8* dst, int width, int height){

  const int x  =  blockIdx.x*ROWS; 
  const int THRid = blockDim.x*threadIdx.y+threadIdx.x; 
  __shared__ uint8 img[ROWS][420];
  
#pragma unroll
  for(int ix = 0; ix < (height+MAX_DISPARITY-1)/MAX_DISPARITY; ix++){
    const int y = ix*MAX_DISPARITY; 
    for(int m = 0; m < ROWS; m++){
      img[m][THRid+y+1] = src1[(y+THRid)*width+x+m];;
    }
  }

  float cost_current[AGG_STEP+1][ROWS];
  float cost_sum[ROWS];

  for(int j = 0; j < ROWS; j++){
    for(int i = 0; i < AGG_STEP+1; i++)
      cost_current[i][j]=0;
    cost_sum[j] = 0;
  }

  uint8 sub=0;
#pragma unroll
  for(int i=height-1; i>0 ;i-=AGG_STEP){
#pragma unroll
    for(int warpid = 0; warpid < MAX_DISPARITY/WARP_SIZE; warpid++){
      int index_temp = (i*width+x+threadIdx.y)*MAX_DISPARITY+threadIdx.x+warpid*WARP_SIZE;
      cost_current[0][warpid] = cost_input[index_temp];	
    }
#pragma unroll
    for(int count = 0; count < AGG_STEP; count++){
      uint32 cost_disp = 0x0u;
      uint32 cost_min  =  1000000000;
      
      
      int offset = i-count;
      for(int warpid = 0; warpid < MAX_DISPARITY/WARP_SIZE; warpid++){
      	int index_temp = ((offset-1)*width+x+threadIdx.y)*MAX_DISPARITY+threadIdx.x+warpid*WARP_SIZE;
	cost_current[count+1][warpid] = cost_input[index_temp];	
	
	sub = abs(img[threadIdx.y][offset+1]-img[threadIdx.y][offset]);
	float e2=exp_list_sub_2[sub];
	cost_sum[warpid] = fmaf(cost_sum[warpid],e2,cost_current[count][warpid]);

	cost_disp = (((uint32)cost_sum[warpid]<<8) & 0xffffff00) + (threadIdx.x+warpid*WARP_SIZE);
	for(int id = WARP_SIZE/2; id > 0 ; id = id/2 ){
	  uint32 t=  __shfl_xor_sync(0xffffffffu, cost_disp, id, WARP_SIZE);
	  cost_disp =  (cost_disp < t? cost_disp : t);
	}
	cost_min = (cost_disp <= cost_min? cost_disp: cost_min);
	
      }
      
      cost_min &= 0xff; 
      if(threadIdx.x==0&&x+threadIdx.y<width){
	dst[offset*width+x+threadIdx.y] = cost_min;
      } 
      
    }
  }

}

__global__ void crosscheck_kernel(const uint8_t* __restrict__ d_input, uint8_t* __restrict__ d_out, const uint32_t rows, const uint32_t cols) {

  __shared__ uint8_t img[1280];

  int n_iter = (cols+blockDim.x-1)/blockDim.x;
  for(int  ix = 0; ix<n_iter; ix++){
    int x=ix*blockDim.x+threadIdx.x;
    if(x<cols)
      img[x] = d_input[blockIdx.x*cols+x];
  }
  __syncthreads();

  for(int  ix = 0; ix<n_iter; ix++){
    int x=ix*blockDim.x+threadIdx.x;
    if(x<cols){
      if(x<20){
	d_out[blockIdx.x*cols+x]=0;
      }	else{
	  for(int disp = 1; disp<64; disp++){
	    if(x-disp<=0)
	      break;
	    else{
	      if(abs(x-img[x])==abs(x-disp-img[x-disp])){
		d_out[blockIdx.x*cols+x]=0;
		break;
	      }
	      else
		d_out[blockIdx.x*cols+x]=img[x];
	    }
	  }
	}
    }
  }
}


__global__ void fill_kernel(const uint8_t* __restrict__ d_input, uint8_t* __restrict__ d_out, const uint32_t rows, const uint32_t cols) {

  __shared__ uint8_t result[1280];
  
  int n_iter = (cols+blockDim.x-1)/blockDim.x;
  for(int  ix = 0; ix<n_iter; ix++){
    int x=ix*blockDim.x+threadIdx.x;
    if(x<cols)
    result[x] = d_input[blockIdx.x*cols+x];
  }
  __syncthreads();

  for(int ix = 0; ix < n_iter; ix++){
    int x= ix*blockDim.x+threadIdx.x;
    int i=1, j=1;
    if(x<cols){
      while(result[x]==0&&x-i>=0&&result[x-i]==0)
  	i++;
      while(result[x]==0&&x+j<cols&&result[x+j]==0)
  	j++;
      d_out[blockIdx.x*cols+x]=x<20?result[x]=result[x+j]:(result[x]==0?((result[x-i<0?0:x-i]<=result[x+j]&&result[x-i<0?0:x-i]!=0)?result[x-i<0?0:x-i]:result[x+j]):result[x]);
    }

      
  }
}


void StereoMatching(const uint8 *src1, const uint8* src2, uint8 *dst, const int width, const int height){

  dim3 threads_ori(AVR_WARP_H,AVR_WARP_V);
  dim3 blocks_ori((width+AVR_WARP_H-1)/AVR_WARP_H, (height+AVR_WARP_V-1)/AVR_WARP_V);
  __avr_fast__<<<blocks_ori, threads_ori>>>(src1, src2, avrimg1, avrimg2, sum2_L, sum2_R, width, height);

  dim3 threads(MAX_DISPARITY,1);
  dim3 blocks((height+MATCH_V_SEG-1)/MATCH_V_SEG, 1);
  _StereoMatching_Kernel<<<blocks, threads>>>(src1, src2, avrimg1, avrimg2, sum2_L, sum2_R, costL2R, width, height, dst);


  dim3 threads_R(WARP_SIZE,ROWS);
  dim3 blocks_R((height+ROWS-1)/threads_R.y, 1);

  _CostAggregation_Kernel_R2L<<<blocks_R, threads_R>>>(src1, costL2R, costR2L, width, height);

  
  dim3 threads_V(WARP_SIZE,ROWS);
  dim3 blocks_V((width+ROWS-1)/threads_V.y, 1);

  _CostAggregation_Kernel_U2D<<<blocks_V, threads_V>>>(src1,  costR2L, costU2D, width, height);


  dim3 threads_V2(WARP_SIZE,ROWS);
  dim3 blocks_V2((width+ROWS-1)/threads_V2.y, 1);
  
  _CostAggregation_Kernel_D2U<<<blocks_V2, threads_V2>>>(src1, costU2D, dst, width, height);

  dim3 threads_V3(MAX_DISPARITY,1);
  dim3 blocks_V3(height,1);
  
  crosscheck_kernel<<<blocks_V3, threads_V3>>>(dst, dst_temp, height, width);
  
  fill_kernel<<<blocks_V3, threads_V3>>>(dst_temp, dst, height, width);
}


void gpu_cudafree(void){
  cudaFree( dst_temp);
  cudaFree(avrimg1);
  cudaFree(avrimg2);
  cudaFree(sum2_L);
  cudaFree(sum2_R);
  cudaFree( costL2R);
  cudaFree( costR2L);
  cudaFree( costU2D);

}
