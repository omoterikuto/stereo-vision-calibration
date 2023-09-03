#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <chrono>
#include "type.h"
#include "fast.h"
#include "match.h"
#include "sad.h"
#include "postprocess.h"
#include "postprocess-cpu.h"
#include "fast-cpu.h"
#include "match-cpu.h"
#include "utils.h"


void main_cpu(){
    std::chrono::system_clock::time_point  start1, start2, start3, start4, start5, start6, start7, start8, end1, end2, end3, end4, end5, end6, end7, end8;
    start8 = std::chrono::high_resolution_clock::now();
    cv::Mat	image_L = read_images_cpu("input/left.bmp");
    cv::Mat	image_R = read_images_cpu("input/rifht.bmp");
    // cv::Mat	image_L = read_images("data/16IMG_1777.bmp");
    // cv::Mat	image_R = read_images("data/17IMG_1778.bmp");
    end8 = std::chrono::high_resolution_clock::now();
    std::cout << "ReadFrame(CPU):"<< std::chrono::duration_cast<std::chrono::microseconds>(end8-start8).count() << " usec" << std::endl;

    int size = IMG_SIZEX * IMG_SIZEY * sizeof(unsigned char);
    int* pHostNum_keypoint1;
	int* pHostNum_keypoint2;
	keypoint_t* pHostKeypoint1;
	keypoint_t* pHostKeypoint2;
	int* pHostNum_Matched;
	match_t* pHostMatched;

    cudaMallocHost(&pHostNum_keypoint1, sizeof(int));
	cudaMallocHost(&pHostNum_keypoint2, sizeof(int));
	cudaMallocHost(&pHostKeypoint1, MAX_KEYPOINT * sizeof(keypoint_t));
	cudaMallocHost(&pHostKeypoint2, MAX_KEYPOINT * sizeof(keypoint_t));
	cudaMallocHost(&pHostNum_Matched, sizeof(int));
	cudaMallocHost(&pHostMatched, MAX_KEYPOINT * sizeof(match_t));

    least_squares_t result[20];
    start6 = std::chrono::high_resolution_clock::now();

    //----------FAST---------------
    start1 = std::chrono::high_resolution_clock::now();
    *pHostNum_keypoint1 = 0;
    *pHostNum_keypoint2 = 0;
    fast_cpu(image_L.data, pHostKeypoint1, IMG_SIZEX, IMG_SIZEY, pHostNum_keypoint1);
    fast_cpu(image_R.data, pHostKeypoint2, IMG_SIZEX, IMG_SIZEY, pHostNum_keypoint2);
    printf("Left:%d\n", *pHostNum_keypoint1);
    printf("Right:%d\n", *pHostNum_keypoint2);
    end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1-start1).count();
    std::cout << "FAST(cpu):"<< duration1 << " usec" << std::endl;


    //----------Matching---------------
    start2 = std::chrono::high_resolution_clock::now();
    kernel_brief_match_cpu(image_L.data, image_R.data, pHostKeypoint1, pHostKeypoint2, (*pHostNum_keypoint1), (*pHostNum_keypoint2), pHostMatched, pHostNum_Matched);
    printf("Match:%d\n", *pHostNum_Matched);
    std::sort(pHostMatched, pHostMatched + *pHostNum_Matched, cmp_match2);
    end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count();
    std::cout << "Match(cpu):"<< duration2 << " usec" << std::endl;
    //----------Correction---------------
    start3 = std::chrono::high_resolution_clock::now();
    std::vector<cv::Point2f> pt1;
	std::vector<cv::Point2f> pt2;
    int move_x = 0;
    for(int i=0; i < 15; i++){
        pt1.push_back(cv::Point2f(pHostMatched[i].x1, pHostMatched[i].y1));
		pt2.push_back(cv::Point2f(pHostMatched[i].x2, pHostMatched[i].y2));
        move_x += (pHostMatched[i].x2 - pHostMatched[i].x1);
    }
    move_x = move_x / 15;
    printf("move_x:%d\n", move_x);
    cv::Mat affine_matrix = cv::estimateAffinePartial2D(pt1, pt2);
    float rotation = atan(affine_matrix.at<double>(1,2) / affine_matrix.at<double>(0,2));
    float da = atan2(affine_matrix.at<double>(1,0), affine_matrix.at<double>(0,0));
    float scale = affine_matrix.at<double>(0,0) / cos(da);
    printf("rotation:%f, da:%f, scale:%f\n", rotation, da, scale);
    cv::Mat correction_mat_left = cv::getRotationMatrix2D(cv::Point2f(IMG_SIZEX/2, IMG_SIZEY/2), -rotation*57.2958, 1);
    cv::Mat correction_mat_right = cv::getRotationMatrix2D(cv::Point2f(IMG_SIZEX/2, IMG_SIZEY/2), -rotation*57.2958, 1/scale);
    cv::Mat	corrected_image_L;
    cv::Mat	corrected_image_R;
    cv::warpAffine(image_L, corrected_image_L, correction_mat_left, cv::Size(IMG_SIZEX, IMG_SIZEY));
    cv::warpAffine(image_R, corrected_image_R, correction_mat_right, cv::Size(IMG_SIZEX, IMG_SIZEY));
    // cv::imwrite("result/corrected_image_L.png", corrected_image_L);
    // cv::imwrite("result/corrected_image_R.png", corrected_image_R);
    std::cout << "correction_mat_left:"<< correction_mat_left << std::endl;
    end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3-start3).count();
    std::cout << "Correction(cpu):"<< duration3 << " usec" << std::endl;
    //----------SAD---------------
    start5 = std::chrono::high_resolution_clock::now();
    int* pHostNum_SAD_Matched;
    sad_match_t* pHostSad_match;
    // cudaMallocHost(&pHostNum_SAD_Matched, sizeof(int));
    // cudaMallocHost(&pHostSad_match,  MAX_RESULT*sizeof(sad_match_t));
    pHostNum_SAD_Matched = (int*)malloc(sizeof(int));
    pHostSad_match = (sad_match_t*)malloc(MAX_RESULT*sizeof(sad_match_t));
    for(int i=0; i < NUM_LINES; i++){
        start4 = std::chrono::high_resolution_clock::now();
        *pHostNum_SAD_Matched=0;
        kernel_sad_CPU(Y_STEP*i + IMG_SIZEY/2 - Y_STEP*(NUM_LINES/2), corrected_image_L.data, corrected_image_R.data, SAD_dc, move_x , pHostSad_match, pHostNum_SAD_Matched);
        // for(int j=0; j < (*pHostNum_SAD_Matched)-1; j++){
        //     printf("%d %d %d %d %d\n", pHostSad_match[j].x1, pHostSad_match[j].y1, pHostSad_match[j].x2, pHostSad_match[j].y2, pHostSad_match[j].distance);
        // }
        result[i] = least_squares(pHostSad_match, *pHostNum_SAD_Matched);
        end4 = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end4-start4).count();
        std::cout << "SAD(cpu):"<< *pHostNum_SAD_Matched << " points " << std::endl;
        std::cout << "SAD(cpu):"<< duration4 << " usec" << std::endl;
    }
    end5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end5-start5).count();
    std::cout << "SAD(cpu total):"<< duration5 << " usec" << std::endl;

    //----------transform---------------
    start7 = std::chrono::high_resolution_clock::now();
    for(int i=0; i < NUM_LINES ; i++){
        // result[i].b -= Y_STEP*3;
        printf("v=%fu + %f\n", result[i].a, result[i].b);
    }
    printf("-----y-----\n");
    float beta = estimateRotationY(result, NUM_LINES);
    std::cout << "beta:" << beta << std::endl;
    for(int i=0; i < NUM_LINES ; i++){
        printf("v=%fu + %f\n", result[i].a, result[i].b);
    }
    
    printf("----z------\n");
    float gamma = estimateRotationZ(result, NUM_LINES);
    std::cout << "gamma:" << gamma << std::endl;
    for(int i=0; i < NUM_LINES ; i++){
        printf("v=%fu + %f\n", result[i].a, result[i].b);
    }

    printf("-----x-----\n");
    float alpha = estimateRotationX(result, NUM_LINES);
    std::cout << "alpha:" << alpha << std::endl;
    for(int i=0; i < NUM_LINES ; i++){
        printf("v=%fu + %f\n", result[i].a, result[i].b);
    }

    printf("-----dy-----\n");
    float dy = estimateTranslationY(result, NUM_LINES);
    std::cout << "dy:" << dy << std::endl;
    for(int i=0; i < NUM_LINES ; i++){
        printf("v=%fu + %f\n", result[i].a, result[i].b);
    }
    printf("-----dz-----\n");
    float dz = estimateTranslationZ(result, NUM_LINES);
    std::cout << "dz:" << dz << std::endl;
    for(int i=0; i < NUM_LINES ; i++){
        printf("v=%fu + %f\n", result[i].a, result[i].b);
    }
    printf("----------\n");

    cv::Mat image_rx, image_ry, image_rz, image_dy, image_dz;
    image_rx = image_R.clone();
    image_ry = image_R.clone();
    image_rz = image_R.clone();
    image_dy = image_R.clone();
    image_dz = image_R.clone();

    kernel_transform_ry_cpu(corrected_image_R.data,image_ry.data, beta);
    kernel_transform_rz_cpu(image_ry.data,image_rz.data, gamma);
    kernel_transform_rx_cpu(image_rz.data,image_rx.data, alpha);
    kernel_transform_ty_cpu(image_rx.data,image_dy.data, dy);
    kernel_transform_tz_cpu(image_dy.data,image_dz.data, dz);

    // cv::imwrite("result/ry_corrected-cpu.png", image_ry);
    // cv::imwrite("result/rz_corrected-cpu.png", image_rz);
    // cv::imwrite("result/rx_corrected-cpu.png", image_rx);
    // cv::imwrite("result/dy_corrected-cpu.png", image_dy);
    // cv::imwrite("result/dz_corrected-cpu.png", image_dz);

    end7 = std::chrono::high_resolution_clock::now();
    std::cout << "Apply rx(cpu):"<< std::chrono::duration_cast<std::chrono::microseconds>(end7-start7).count() << " usec" << std::endl;

    end6 = std::chrono::high_resolution_clock::now();
    std::cout << "Total:"<< std::chrono::duration_cast<std::chrono::microseconds>(end6-start6).count() << " usec" << std::endl;

}
