#include "type.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel_affine_transform(unsigned char* SrcImg, unsigned char* DstImg, float a0, float a1, float a2,float a3, float a4, float a5) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
	int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
	int tx = threadIdx.x;
    if(idx < 0 || idx > IMG_SIZEX || idy < 0 || idy > IMG_SIZEY){
        return;
    }

    int x = a0* idx + a1*idy + a2;
    int y = a3* idx + a4*idy + a5;

    if(x < 0 || x > IMG_SIZEX || y < 0 || y > IMG_SIZEY){
        DstImg[idy*IMG_SIZEX + idx] = 0;
    }else{
        DstImg[idy*IMG_SIZEX + idx] = SrcImg[y*IMG_SIZEX+x];
    }
}
