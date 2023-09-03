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

// CUDA functions

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include "function.h"
#include <dirent.h>
#include <png.h>
#include "png_image.h"



using namespace std;


int main(int argc, char *argv[]) {


    ImageData* imgleft = png_image_load(argv[1]);
    ImageData* imgright = png_image_load(argv[2]);
    int height = imgleft->height;
    int width = imgleft->width;
    int size = height*width;
   


    char* outputname = argv[3];  
  
    
    uint8* d_left = NULL;
    uint8* d_right= NULL;
    uint8* d_dst = NULL;
    uint8* h_dst = NULL;
    uint8* leftimage = NULL;
    uint8* rightimage = NULL;
    leftimage = new uint8[size];
    rightimage = new uint8[size];
    h_dst = new uint8[size];

    for(int j = 0; j < height; j++)
      for(int i = 0; i < width; i++){
	leftimage[j*width+i]= imgleft->rows[j][i];
	rightimage[j*width+i]= imgright->rows[j][i];
      }
      

  
    
    cudaMalloc((void **)&d_left, sizeof(uint8)*size);
    cudaMalloc((void **)&d_right, sizeof(uint8)*size);
    cudaMalloc((void **)&d_dst, sizeof(uint8)*size);
    cudaMemcpyAsync(d_left, leftimage, sizeof(uint8)*size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_right, rightimage, sizeof(uint8)*size, cudaMemcpyHostToDevice);
    
    gpu_initial(width, height);

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
    cudaEventRecord( start, 0 );

    for(int m = 0; m < 100;m++){
      StereoMatching(d_left, d_right, d_dst, width, height);
    }
   
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop);

    float elapsed_time_ms=0;
    cudaEventElapsedTime( &elapsed_time_ms, start, stop );

	
    printf( "Time: %8.2f ms\n", elapsed_time_ms/100);


    cudaMemcpy(h_dst, d_dst, sizeof(uint8)*size, cudaMemcpyDeviceToHost);

    gpu_cudafree();
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_dst);


    ImageData* output=png_image_load(argv[1]);
    
    for(int j = 0 ; j < height; j++){
      for(int i = 0; i < width; i++){
      	output->rows[j][i] = h_dst[j*width+i];
      }
    }

    png_image_save(outputname, output);

    png_image_destroy(imgleft);
    png_image_destroy(imgright);
    png_image_destroy(output);
    delete[] h_dst;
    return 0;   
}


