#include "type.h"
#include <stdio.h>

void kernel_brief_match_cpu(unsigned char* SrcImg1, unsigned char* SrcImg2, keypoint_t* keypoint1, keypoint_t* keypoint2, int size1, int size2, match_t* matched, int* num_matched) {
    for (int idx = 0; idx < size1; idx++) {
        //int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
        //int idy = blockDim.y * blockIdx.y + threadIdx.y; //自分のスレッドyのindex
        //int tx = threadIdx.x;
        if (idx >= size1) continue;

        int min_score = 65535;
        int min_index = -1;
        int pos1_y = keypoint1[idx].y;
        int pos1_x = keypoint1[idx].x;

        if (pos1_y < 15 || pos1_y > IMG_SIZEY - 15 || pos1_x < 15 || pos1_x > IMG_SIZEX - 15) continue;

        for (int j = 0; j < size2; j++) {
            int pos2_y = keypoint2[j].y;
            int pos2_x = keypoint2[j].x;
            int r = (pos1_x - pos2_x) * (pos1_x - pos2_x) + (pos1_y - pos2_y) * (pos1_y - pos2_y);
            // if (r > 15000) continue;
            if (pos2_y < 15 || pos2_y > IMG_SIZEY - 15 || pos2_x < 15 || pos2_x > IMG_SIZEX - 15) continue;

            unsigned long long int xord1 = keypoint1[idx].feature1 ^ keypoint2[j].feature1;
            unsigned long long int xord2 = keypoint1[idx].feature2 ^ keypoint2[j].feature2;
            int popcount1 = __builtin_popcountll(xord1);
            int popcount2 = __builtin_popcountll(xord2);
            int sum = popcount1 + popcount2;
            if (sum < min_score) {
                min_score = sum;
                min_index = j;
            }
        }
        *num_matched = *num_matched + 1;
        int index = *num_matched;
        matched[index] = { pos1_y, pos1_x, keypoint2[min_index].y, keypoint2[min_index].x, min_score, keypoint1[idx].feature1, keypoint1[idx].feature2, keypoint2[min_index].feature1, keypoint2[min_index].feature2};

    }
}