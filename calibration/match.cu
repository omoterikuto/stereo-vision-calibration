#include "type.h"
#include <stdio.h>

// __global__ void kernel_sad(unsigned char* SrcImg1, unsigned char* SrcImg2, keypoint_t* keypoint1, keypoint_t* keypoint2, int size1, int size2, match_t* matched, int* num_matched) {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
//     int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
//     int tx = threadIdx.x;
//     if (idx >= size1) return;

//     int min_score = 65535;
//     int min_index = -1;
//     int pos1_y = keypoint1[idx].y;
//     int pos1_x = keypoint1[idx].x;

//     if (pos1_y < 15 || pos1_y > IMG_SIZEY - 15 || pos1_x < 15 || pos1_x > IMG_SIZEX - 15) return;
    
//     unsigned char window1[31][31];
//     unsigned char window2[31][31];

//     for (int y = 0; y < 31; y++) {
//         for (int x = 0; x < 31; x++) {
//             window1[y][x] = SrcImg1[(pos1_y - 15 + y) * IMG_SIZEX + pos1_x - 15 + x];
//         }
//     }

//     for (int j = 0; j < size2; j++) {
//         int pos2_y = keypoint2[j].y;
//         int pos2_x = keypoint2[j].x;
//         int r = (pos1_x - pos2_x) * (pos1_x - pos2_x) + (pos1_y - pos2_y) * (pos1_y - pos2_y);
//         if (r > 15000) continue;
//         if (pos2_y < 15 || pos2_y > IMG_SIZEY - 15 || pos2_x < 15 || pos2_x > IMG_SIZEX - 15) continue;

//         int sum = 0;
//         for (int y = 0; y < 31; y++) {
//             for (int x = 0; x < 31; x++) {
//                 window2[y][x] = SrcImg2[(pos2_y - 15 + y) * IMG_SIZEX + pos2_x - 15 + x];
//                 sum += abs(window1[y][x] - window2[y][x]);
//             }
//         }

//         if (sum < min_score) {
//             min_score = sum;
//             min_index = j;
//         }
//     }
//     int index = atomicAdd(num_matched, 1);
//     matched[index] = { pos1_y, pos1_x, keypoint2[min_index].y, keypoint2[min_index].x, min_score};

// }

__global__ void kernel_brief_match(unsigned char* SrcImg1, unsigned char* SrcImg2, keypoint_t* keypoint1, keypoint_t* keypoint2, int size1, int size2, match_t* matched, int* num_matched) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
    int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
    int tx = threadIdx.x;
    if (idx >= size1) return;

    int min_score = 65535;
    int min_index = 0;
    int pos1_y = keypoint1[idx].y;
    int pos1_x = keypoint1[idx].x;
    unsigned long long int min_xord1;
    unsigned long long int min_xord2;
    if (pos1_y < 15 || pos1_y > IMG_SIZEY - 15 || pos1_x < 15 || pos1_x > IMG_SIZEX - 15) return;

    for (int j = 0; j < size2; j++) {
        int pos2_y = keypoint2[j].y;
        int pos2_x = keypoint2[j].x;
        int r = (pos1_x - pos2_x) * (pos1_x - pos2_x) + (pos1_y - pos2_y) * (pos1_y - pos2_y);
        // if (r > 100000) continue;//近傍では無かった場合パスする
        // if(abs(pos1_y - pos2_y) > 100) continue;
        if (pos2_y < 15 || pos2_y > IMG_SIZEY - 15 || pos2_x < 15 || pos2_x > IMG_SIZEX - 15) continue;

        unsigned long long int xord1 = keypoint1[idx].feature1 ^ keypoint2[j].feature1; //64bit intを2個使ってハミング距離をXORで計算
        unsigned long long int xord2 = keypoint1[idx].feature2 ^ keypoint2[j].feature2;
        // printf("feature1 %lu, %lu\n", keypoint1[idx].feature1, keypoint1[j].feature1);
        // printf("feature2 %lu, %lu\n", keypoint2[idx].feature2, keypoint2[j].feature2);

        int popcount1 = __popcll(xord1);
        int popcount2 = __popcll(xord2);
        int sum = popcount1 + popcount2;
        // printf("feature:%lu, %lu : %lu %lu : %d\n", keypoint1[idx].feature1, keypoint1[idx].feature2, keypoint2[j].feature1, keypoint2[j].feature2, sum);
        if (sum < min_score) {
            min_score = sum;
            min_index = j;
            min_xord1 = xord1;
            min_xord2 = xord2;
        }
    }
    int index = atomicAdd(num_matched, 1);
    matched[index] = { pos1_y, pos1_x, keypoint2[min_index].y, keypoint2[min_index].x, min_score, keypoint1[idx].feature1, keypoint1[idx].feature2, keypoint2[min_index].feature1, keypoint2[min_index].feature2};
    // printf("feature:%lu, %lu : %lu %lu : %d\n", keypoint1[idx].feature1, keypoint1[idx].feature2, keypoint2[min_index].feature1, keypoint2[min_index].feature2, min_score);
    // printf("xord:%lu, %lu : %d\n", min_xord1, min_xord2, min_score);



}
