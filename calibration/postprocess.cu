#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include "type.h"

least_squares_t least_squares(sad_match_t *sad_match, int num_sad){
    float ave_y =0;
    float ave_x =0;

    int count=0;
    for(int i=0; i < num_sad; i++){
        if (sad_match[i].x2 == 0 && sad_match[i].y2 == 0){
            continue;
        }
        ave_y += sad_match[i].y2; 
        ave_x += sad_match[i].x2;
        count++;
    }
    ave_y = ave_y / count;
    ave_x = ave_x / count;

    float a = 0;
    float b = 0;
    float tmp_1=0;
    float tmp_2=0;


    for(int i=0; i < count; i++){
        tmp_1 += (sad_match[i].y2 - ave_y) * (sad_match[i].x2 - ave_x);
        tmp_2 += (sad_match[i].x2 - ave_x) * (sad_match[i].x2 - ave_x);
    }
    a = tmp_1 / tmp_2;
    b = ave_y - a * ave_x;

    least_squares_t result = {a,b};
    return result; 
}


float estimateRotationY(least_squares_t *lines, int num_lines){
    float v[num_lines];
    float sum=0;
    for(int i=0; i < num_lines-1; i++){
        // v[i] = atan(F*(lines[i].a-lines[i+1].a) / (lines[i].b-lines[i+1].b));
        v[i] = atan(F*(lines[i].a-lines[i+1].a) / (lines[i].b-lines[i+1].b));
        sum += v[i];
    }
    float beta = sum / (num_lines-1);
    for(int i=0; i < num_lines; i++){
        float a = lines[i].a;
        float b = lines[i].b;

        lines[i].a = a*cos(beta) - (b/F)*sin(beta);
        lines[i].b = b*cos(beta) + F*a*sin(beta);
    }
    return beta;
}

float estimateRotationX(least_squares_t *lines, int num_lines){
    float v[num_lines];
    float sum=0;
    for(int i=0; i < num_lines-2; i++){
        v[i] = atan(-F*(lines[i].b + lines[i+2].b - 2*lines[i+1].b)/(lines[i].b*lines[i+1].b + lines[i+1].b*lines[i+2].b - 2*lines[i].b*lines[i+2].b));
        sum += v[i];
    }
    float alpha = sum / (num_lines-2);
    for(int i=0; i < num_lines; i++){
        float a = lines[i].a;
        float b = lines[i].b;

        lines[i].a = F*(a)/(-b*sin(alpha) + F*cos(alpha));
        lines[i].b = F*(b*cos(alpha) +F*sin(alpha))/(-b*sin(alpha) + F*cos(alpha));
    }
    return alpha;
}

float estimateRotationZ(least_squares_t *lines, int num_lines){
    float v[num_lines];
    float sum=0;
    for(int i=0; i < num_lines; i++){
        v[i] = atan(lines[i].a);
        sum += v[i];
    }
    float gamma = sum /num_lines;
    for(int i=0; i < num_lines; i++){
        float a = lines[i].a;
        float b = lines[i].b;
        lines[i].a = (-sin(gamma) + a*cos(gamma))/(cos(gamma) + a*sin(gamma));
        lines[i].b = (b)/(cos(gamma) + a*sin(gamma));
    }
    return gamma;
}

float estimateTranslationY(least_squares_t *lines, int num_lines){
    // float v[num_lines];
    // float sum=0;
    // for(int i=0; i < num_lines; i++){
    //     v[i] = Y_STEP*(i-NUM_LINES/2) - lines[i].b;
    //     sum += v[i];
    // }
    // float dy = sum /num_lines;
    float dy = lines[NUM_LINES/2].b;
    for(int i=0; i < num_lines; i++){
        float a = lines[i].a;
        float b = lines[i].b;
        lines[i].b = b - dy;
    }
    return dy;
}
float estimateTranslationZ(least_squares_t *lines, int num_lines){
    float v[num_lines];
    float sum=0;
    for(int i=0; i < num_lines-1; i++){
        v[i] = Y_STEP / (lines[i+1].b - lines[i].b);
        sum += v[i];
    }
    float dz = sum /(num_lines-1);
    for(int i=0; i < num_lines; i++){
        float b = lines[i].b;
        lines[i].b = b*dz;
    }
    return dz;
}

__global__ void kernel_transform_ry(unsigned char* SrcImg, unsigned char* DstImg, float beta){
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
	int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
    if(idx < 0 || idx > IMG_SIZEX || idy < 0 || idy > IMG_SIZEY){
        return;
    }
    idx -= IMG_SIZEX/2;
    idy -= IMG_SIZEY/2;
    int x = F*(idx*__cosf(beta) + F*__sinf(beta))/(-idx*__sinf(beta) + F*__cosf(beta));
    int y = F*(idy)/(-idx*__sinf(beta) + F*__cosf(beta));
    x += IMG_SIZEX/2;
    y += IMG_SIZEY/2;
    idx += IMG_SIZEX/2;
    idy += IMG_SIZEY/2;
    if(x < 0 || x > IMG_SIZEX || y < 0 || y > IMG_SIZEY){
        DstImg[idy*IMG_SIZEX + idx] = 0;
    }else{
        DstImg[idy*IMG_SIZEX + idx] = SrcImg[y*IMG_SIZEX+x];
    }
}

__global__ void kernel_transform_rx(unsigned char* SrcImg, unsigned char* DstImg, float alpha){
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
	int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
    if(idx < 0 || idx > IMG_SIZEX || idy < 0 || idy > IMG_SIZEY){
        return;
    }
    idx -= IMG_SIZEX/2;
    idy -= IMG_SIZEY/2;
    int x = F*(idx)/(idy*__sinf(alpha) + F*__cosf(alpha));
    int y = F*(idy*__cosf(alpha) - F*__sinf(alpha))/(idy*__sinf(alpha) + F*__cosf(alpha));
    x += IMG_SIZEX/2;
    y += IMG_SIZEY/2;
    idx += IMG_SIZEX/2;
    idy += IMG_SIZEY/2;
    if(x < 0 || x > IMG_SIZEX || y < 0 || y > IMG_SIZEY){
        DstImg[idy*IMG_SIZEX + idx] = 0;
    }else{
        DstImg[idy*IMG_SIZEX + idx] = SrcImg[y*IMG_SIZEX+x];
    }
}
__global__ void kernel_transform_rz(unsigned char* SrcImg, unsigned char* DstImg, float gamma){
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
	int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
    if(idx < 0 || idx > IMG_SIZEX || idy < 0 || idy > IMG_SIZEY){
        return;
    }
    idx -= IMG_SIZEX/2;
    idy -= IMG_SIZEY/2;

    int x = idx*__cosf(gamma) - idy*__sinf(gamma);
    int y = idx*__sinf(gamma) + idy*__cosf(gamma);
    x += IMG_SIZEX/2;
    y += IMG_SIZEY/2;
    idx += IMG_SIZEX/2;
    idy += IMG_SIZEY/2;
    if(x < 0 || x > IMG_SIZEX || y < 0 || y > IMG_SIZEY){
        DstImg[idy*IMG_SIZEX + idx] = 0;
    }else{
        DstImg[idy*IMG_SIZEX + idx] = SrcImg[y*IMG_SIZEX+x];
    }
}

__global__ void kernel_transform_ty(unsigned char* SrcImg, unsigned char* DstImg, float dy){
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
	int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
    if(idx < 0 || idx > IMG_SIZEX || idy < 0 || idy > IMG_SIZEY){
        return;
    }
    int x = idx;
    int y = idy + dy;

    if(x < 0 || x > IMG_SIZEX || y < 0 || y > IMG_SIZEY){
        DstImg[idy*IMG_SIZEX + idx] = 0;
    }else{
        DstImg[idy*IMG_SIZEX + idx] = SrcImg[y*IMG_SIZEX+x];
    }
}

__global__ void kernel_transform_tz(unsigned char* SrcImg, unsigned char* DstImg, float dz){
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
	int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
    if(idx < 0 || idx > IMG_SIZEX || idy < 0 || idy > IMG_SIZEY){
        return;
    }
    idx -= IMG_SIZEX/2;
    idy -= IMG_SIZEY/2;
    int x = 1/dz * idx;
    int y = 1/dz * idy;
    x += IMG_SIZEX/2;
    y += IMG_SIZEY/2;
    idx += IMG_SIZEX/2;
    idy += IMG_SIZEY/2;
    if(x < 0 || x > IMG_SIZEX || y < 0 || y > IMG_SIZEY){
        DstImg[idy*IMG_SIZEX + idx] = 0;
    }else{
        DstImg[idy*IMG_SIZEX + idx] = SrcImg[y*IMG_SIZEX+x];
    }
}
