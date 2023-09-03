#include <stdio.h>
#include <stdlib.h>
#include "type.h"

__global__ void kernel_sad4(int C,unsigned char* SrcImg_L, unsigned char* SrcImg_R, int dc, int x_center,sad_match_t* matched, int* num_matched) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
	int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
	int tx = threadIdx.x;

    int x_L = X_MARGIN + idx * X_STEP;
    if (x_L < X_MARGIN || x_L > (IMG_SIZEX - X_MARGIN) ){
        return;
    }

    unsigned char sad_window_L[13][13];
    for(int i=-6; i < 7; i++){ //SAD kernel size
        for(int j=-6; j < 7; j++){
            sad_window_L[i+6][j+6] = SrcImg_L[(C+i)*IMG_SIZEX + (x_L+j)];
        }
    }

    int sad_min = INFINITY;
    int sad_min_index_x;
    int sad_min_index_y;

    for(int y= C-dc; y < C+dc; y++){ // y= C+-dcの範囲で探す
        for(int x_R=(x_L + x_center - SEARCH_AREA_X); x_R < (x_L + x_center + SEARCH_AREA_X); x_R++){ //x_L + x_centerから+-SEARCH_AREA_Xの範囲で探す x_centerはFASTでどれぐらい視差があったのかの値
            if (x_R < X_MARGIN || x_R > IMG_SIZEX - X_MARGIN){
                continue;
            }
            int sad=0;
            for(int i=-6; i < 7; i++){ //SAD kernel size
                for(int j=-6; j < 7; j++){
                    sad += abs(sad_window_L[i+6][j+6] - SrcImg_R[(y+i)*IMG_SIZEX + (x_R+j)] );
                }
            }
            if(sad < sad_min){
                sad_min = sad;
                sad_min_index_x = x_R;
                sad_min_index_y = y;

            }
        }
    }
    if(sad_min > 0 && sad_min < 1500){ //distanceが一定以上のものと0は除外する
        int index = atomicAdd(num_matched, 1);
        matched[index-1]={
            C - IMG_SIZEY/2,
            x_L - IMG_SIZEX/2,
            sad_min_index_y - IMG_SIZEY/2,
            sad_min_index_x - IMG_SIZEX/2,
            sad_min
        };
    }
}


void kernel_sad_CPU(int C,unsigned char* SrcImg_L, unsigned char* SrcImg_R, int dc, int x_center,sad_match_t* matched, int* num_matched){
    for(int n=0; n < NUM_SAD; n++){
        int x_L = X_MARGIN + X_STEP*n;
        if (x_L < X_MARGIN || x_L > (IMG_SIZEX - X_MARGIN) ){
            continue;
        }
        unsigned char sad_window_L[13][13];
        for(int i=-6; i < 7; i++){ //SAD kernel size
            for(int j=-6; j < 7; j++){
                sad_window_L[i+6][j+6] = SrcImg_L[(C+i)*IMG_SIZEX + (x_L+j)];
            }
        }

        int sad_min = INFINITY;
        int sad_min_index_x;
        int sad_min_index_y;
        for(int y= C-dc; y < C+dc; y++){ 
            for(int x_R=(x_L + x_center - SEARCH_AREA_X); x_R < (x_L + x_center + SEARCH_AREA_X); x_R++){ //x_L + x_centerから+-SEARCH_AREA_Xの範囲で探す x_centerはFASTでどれぐらい視差があったのかの値
                if (x_R < X_MARGIN || x_R > IMG_SIZEX - X_MARGIN){
                    continue;
                }
                int sad=0;
                for(int i=-6; i < 7; i++){ //SAD kernel size
                    for(int j=-6; j < 7; j++){
                        sad += abs(sad_window_L[i+6][j+6] - SrcImg_R[(y+i)*IMG_SIZEX + (x_R+j)] );
                    }
                }
                if(sad < sad_min){
                    sad_min = sad;
                    sad_min_index_x = x_R;
                    sad_min_index_y = y;
                }
            }
        }
        if(sad_min > 0 && sad_min < 1500){ //distanceが一定以上のものと0は除外する
            *num_matched += 1;
            int index = *num_matched;
            matched[index-1]={
                C - IMG_SIZEY/2,
                x_L - IMG_SIZEX/2,
                sad_min_index_y - IMG_SIZEY/2,
                sad_min_index_x - IMG_SIZEX/2,
                sad_min
            };
        }   
    }
}
