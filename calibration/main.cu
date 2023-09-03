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
#include "affine.h"
#include "match.h"
#include "sad.h"
#include "postprocess.h"
#include "main-cpu.h"
#include "utils.h"


int main_fast(){
    std::chrono::system_clock::time_point  start1, start2, start3, start4, start5, start6, start7, start8, end1, end2, end3, end4, end5, end6, end7, end8;
    start8 = std::chrono::high_resolution_clock::now();
    cv::Mat	image_L = read_images_cpu("input/left.bmp");
    cv::Mat	image_R = read_images_cpu("input/right.bmp");
    // cv::Mat	image_L = read_images("data/16IMG_1777.bmp");
    // cv::Mat	image_R = read_images("data/17IMG_1778.bmp");
    end8 = std::chrono::high_resolution_clock::now();
    std::cout << "ReadFrame(GPU):"<< std::chrono::duration_cast<std::chrono::microseconds>(end8-start8).count() << " usec" << std::endl;

    int size = IMG_SIZEX * IMG_SIZEY * sizeof(unsigned char);

    int* pHostNum_keypoint1;
	int* pHostNum_keypoint2;
	keypoint_t* pHostKeypoint1;
	keypoint_t* pHostKeypoint2;
	int* pHostNum_Matched;
	match_t* pHostMatched;
    int* pHostNum_SAD_Matched;
	sad_match_t* pHostSad_match;

	cudaMallocHost(&pHostNum_keypoint1, sizeof(int));
	cudaMallocHost(&pHostNum_keypoint2, sizeof(int));
	cudaMallocHost(&pHostKeypoint1, MAX_KEYPOINT * sizeof(keypoint_t));
	cudaMallocHost(&pHostKeypoint2, MAX_KEYPOINT * sizeof(keypoint_t));
	cudaMallocHost(&pHostNum_Matched, sizeof(int));
	cudaMallocHost(&pHostMatched, MAX_KEYPOINT * sizeof(match_t));
    cudaMallocHost(&pHostSad_match,  MAX_RESULT*sizeof(sad_match_t));
    cudaMallocHost(&pHostNum_SAD_Matched,  sizeof(int));

	keypoint_t* pDevKeypoint1;
	keypoint_t* pDevKeypoint2;
	unsigned char* pDevSrc1;
	unsigned char* pDevSrc2;
    unsigned char* pDevSrc3;
	unsigned char* pDevSrc4;
    unsigned char* pDevSrc5;
    unsigned char* pDevSrc6;
    unsigned char* pDevSrc7;
    unsigned char* pDevSrc8;
    unsigned char* pDevSrc9;
	int* pDevNum_Matched;
	match_t* pDevMatched;
    int* pDevNum_SAD_Matched;
	sad_match_t* pDevSad_match;
    double *pDevAffineMatrix_L;
    double *pDevAffineMatrix_R;


    cudaMalloc(&pDevKeypoint1, MAX_KEYPOINT * sizeof(keypoint_t));
	cudaMalloc(&pDevKeypoint2, MAX_KEYPOINT * sizeof(keypoint_t));
	cudaMalloc(&pDevSrc1, size);
	cudaMalloc(&pDevSrc2, size);
    cudaMalloc(&pDevSrc3, size);
	cudaMalloc(&pDevSrc4, size);
    cudaMalloc(&pDevSrc5, size);
    cudaMalloc(&pDevSrc6, size);
    cudaMalloc(&pDevSrc7, size);
    cudaMalloc(&pDevSrc8, size);
    cudaMalloc(&pDevSrc9, size);
	cudaMalloc(&pDevNum_Matched, sizeof(int));
	cudaMalloc(&pDevMatched, MAX_KEYPOINT * sizeof(match_t));
    cudaMalloc(&pDevSad_match, MAX_RESULT*sizeof(sad_match_t));
    cudaMalloc(&pDevNum_SAD_Matched,  sizeof(int));
    cudaMalloc(&pDevAffineMatrix_L,  6*sizeof(double));
    cudaMalloc(&pDevAffineMatrix_R,  6*sizeof(double));


	cudaStream_t stream1;
	cudaStream_t stream2;
    
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
    
    cv::Mat	corrected_image_L;
    cv::Mat	corrected_image_R;

    least_squares_t result[20];

    dim3 block1(BLOCK_SHARED_X, 1);
	dim3 grid1((IMG_SIZEX + BLOCK_SHARED_X - 2 - 1) / (BLOCK_SHARED_X - 2), IMG_SIZEY);

    start6 = std::chrono::high_resolution_clock::now();
    //----------FAST---------------
    start1 = std::chrono::high_resolution_clock::now();
	*pHostNum_keypoint1 = 0;
	cudaMemcpyAsync(pDevSrc1, image_L.data, size, cudaMemcpyHostToDevice, stream1);
	kernel_fast << <grid1, block1, 0, stream1 >> > (pDevSrc1, pDevKeypoint1, IMG_SIZEX, IMG_SIZEY, pHostNum_keypoint1);

    *pHostNum_keypoint2 = 0;
	cudaMemcpyAsync(pDevSrc2, image_R.data, size, cudaMemcpyHostToDevice, stream2);
	kernel_fast << <grid1, block1, 0, stream2 >> > (pDevSrc2, pDevKeypoint2, IMG_SIZEX, IMG_SIZEY, pHostNum_keypoint2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    printf("Left:%d\n", *pHostNum_keypoint1);
    printf("Right:%d\n", *pHostNum_keypoint2);

    end1 = std::chrono::high_resolution_clock::now();
    std::cout << "FAST(GPU):"<< std::chrono::duration_cast<std::chrono::microseconds>(end1-start1).count() << " usec" << std::endl;

    //----------Matching---------------
    start2 = std::chrono::high_resolution_clock::now();
    *pHostNum_Matched = 0;
    cudaMemcpy(pDevNum_Matched, pHostNum_Matched, sizeof(int), cudaMemcpyHostToDevice);
    dim3 block2(BLOCK_SHARED_X, 1);
    dim3 grid2(((*pHostNum_keypoint1) / BLOCK_SHARED_X) + 1, 1);
    kernel_brief_match << <grid2, block2, 0>> > (pDevSrc1, pDevSrc2, pDevKeypoint1, pDevKeypoint2, (*pHostNum_keypoint1), (*pHostNum_keypoint2), pDevMatched, pDevNum_Matched);//マッチングカーネル事項
    cudaMemcpy(pHostNum_Matched, pDevNum_Matched, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pHostMatched, pDevMatched, *pHostNum_Matched * sizeof(match_t), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream1);
    printf("Match:%d\n", *pHostNum_Matched);

    std::sort(pHostMatched, pHostMatched + *pHostNum_Matched, cmp_match2); //hamming distanceでソートする
    end2 = std::chrono::high_resolution_clock::now();
    std::cout << "Match(GPU):"<< std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count() << " usec" << std::endl;

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
    rotation = 0.004269;
    scale = 1.007016;
    printf("rotation:%f, da:%f, scale:%f\n", rotation, da, scale);
    cv::Mat correction_mat_left = cv::getRotationMatrix2D(cv::Point2f(IMG_SIZEX/2, IMG_SIZEY/2), -rotation*57.2958, 1);
    cv::Mat correction_mat_right = cv::getRotationMatrix2D(cv::Point2f(IMG_SIZEX/2, IMG_SIZEY/2), -rotation*57.2958, 1/scale);
    cv::Mat correction_mat_left_inv, correction_mat_right_inv;
    cv:: invertAffineTransform(correction_mat_left, correction_mat_left_inv);
    cv:: invertAffineTransform(correction_mat_right, correction_mat_right_inv);

    auto end3_1 = std::chrono::high_resolution_clock::now();
    std::cout <<"Correction(before Kernel)" << std::chrono::duration_cast<std::chrono::microseconds>(end3_1-start3).count() << std::endl;

    dim3 block4(BLOCK_SHARED_X, 1);
    dim3 grid4((IMG_SIZEX/ BLOCK_SHARED_X) + 1, IMG_SIZEY);
    kernel_affine_transform << <grid4, block4, 0, stream1>> > (pDevSrc1, pDevSrc3, correction_mat_left_inv.at<double>(0,0), correction_mat_left_inv.at<double>(0,1), correction_mat_left_inv.at<double>(0,2), correction_mat_left_inv.at<double>(1,0), correction_mat_left_inv.at<double>(1,1), correction_mat_left_inv.at<double>(1,2));
    kernel_affine_transform << <grid4, block4, 0, stream2>> > (pDevSrc2, pDevSrc4, correction_mat_right_inv.at<double>(0,0), correction_mat_right_inv.at<double>(0,1), correction_mat_right_inv.at<double>(0,2), correction_mat_right_inv.at<double>(1,0), correction_mat_right_inv.at<double>(1,1), correction_mat_right_inv.at<double>(1,2));
    cudaDeviceSynchronize();
	cudaMemcpy(image_L.data, pDevSrc3, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(image_R.data, pDevSrc4, size, cudaMemcpyDeviceToHost);
    cv::imwrite("result/GPU_corrected_image_L.png", image_L);
    cv::imwrite("result/GPU_corrected_image_R.png", image_R);
    std::cout << "correction_mat_left_inv:"<< correction_mat_left_inv << std::endl;
    end3 = std::chrono::high_resolution_clock::now();
    std::cout << "Correction(GPU):"<< std::chrono::duration_cast<std::chrono::microseconds>(end3-start3).count() << " usec" << std::endl;

    //----------SAD---------------
    start5 = std::chrono::high_resolution_clock::now();
    dim3 block3(BLOCK_SHARED_X, 1);
	dim3 grid3( NUM_SAD / BLOCK_SHARED_X, 1);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    for(int i=0; i < NUM_LINES; i++){
        start4 = std::chrono::high_resolution_clock::now();
        *pHostNum_SAD_Matched = 0;
        cudaMemcpy(pDevNum_SAD_Matched, pHostNum_SAD_Matched, sizeof(int), cudaMemcpyHostToDevice);
        kernel_sad4  <<<grid3, block3, 0>>> (Y_STEP*i + IMG_SIZEY/2 - Y_STEP*(NUM_LINES/2), pDevSrc3, pDevSrc4, SAD_dc, move_x, pDevSad_match, pDevNum_SAD_Matched); 
        cudaMemcpy(pHostNum_SAD_Matched, pDevNum_SAD_Matched, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pHostSad_match, pDevSad_match, (*pHostNum_SAD_Matched) * sizeof(sad_match_t), cudaMemcpyDeviceToHost);
        printf("pHostNum_SAD_Matched:%d\n", *pHostNum_SAD_Matched);
        // for(int j=0; j < (*pHostNum_SAD_Matched)-1; j++){
        //     printf("%d %d %d %d %d\n", pHostSad_match[j].x1, pHostSad_match[j].y1, pHostSad_match[j].x2, pHostSad_match[j].y2, pHostSad_match[j].distance);
        // }
        result[i] = least_squares(pHostSad_match, (*pHostNum_SAD_Matched)-1);
        // draw_SAD_CUDA(Y_STEP*i + IMG_SIZEY/2 - Y_STEP*(NUM_LINES/2), image_L, image_R, pHostSad_match, (*pHostNum_SAD_Matched)-1);
        end4 = std::chrono::high_resolution_clock::now(); 
        std::cout << "sad(GPU):"<< std::chrono::duration_cast<std::chrono::microseconds>(end4-start4).count() << " usec" << std::endl;
    }
    end5 = std::chrono::high_resolution_clock::now();
    std::cout << "SAD(GPU total):"<< std::chrono::duration_cast<std::chrono::microseconds>(end5-start5).count() << " usec" << std::endl;

    //----------transform---------------
    for(int i=0; i < NUM_LINES ; i++){
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
    image_ry = image_R.clone();
    image_rz = image_R.clone();
    image_rx = image_R.clone();
    image_dy = image_R.clone();
    image_dz = image_R.clone();

    start7 = std::chrono::high_resolution_clock::now();
    kernel_transform_ry << <grid4, block4, 0, stream1>> > (pDevSrc4, pDevSrc5, beta);
    kernel_transform_rz << <grid4, block4, 0, stream1>> > (pDevSrc5, pDevSrc6, gamma);
    kernel_transform_rx << <grid4, block4, 0, stream1>> > (pDevSrc6, pDevSrc7, alpha);
    kernel_transform_ty << <grid4, block4, 0, stream1>> > (pDevSrc7, pDevSrc8, dy);
    kernel_transform_tz << <grid4, block4, 0, stream1>> > (pDevSrc8, pDevSrc9, 1/dz);
    // kernel_transform_ry << <grid4, block4, 0, stream1>> > (pDevSrc4, pDevSrc5, 0.000343);
    // kernel_transform_rz << <grid4, block4, 0, stream1>> > (pDevSrc5, pDevSrc6, 0.001983);
    // kernel_transform_rx << <grid4, block4, 0, stream1>> > (pDevSrc6, pDevSrc7, 0.000357);
    // kernel_transform_ty << <grid4, block4, 0, stream1>> > (pDevSrc7, pDevSrc8,  400 * 0.001901/0.566682);
    // kernel_transform_tz << <grid4, block4, 0, stream1>> > (pDevSrc8, pDevSrc9, 1 + 0.001163/0.566682);

	cudaMemcpyAsync(image_ry.data, pDevSrc5, size, cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(image_rz.data, pDevSrc6, size, cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(image_rx.data, pDevSrc7, size, cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(image_dy.data, pDevSrc8, size, cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(image_dz.data, pDevSrc9, size, cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream1);
    cv::imwrite("result/ry_corrected.png", image_ry);
    cv::imwrite("result/rz_corrected.png", image_rz);
    cv::imwrite("result/rx_corrected.png", image_rx);
    cv::imwrite("result/dy_corrected.png", image_dy);
    cv::imwrite("result/dz_corrected.png", image_dz);

    end7 = std::chrono::high_resolution_clock::now();
    std::cout << "Apply rx:"<< std::chrono::duration_cast<std::chrono::microseconds>(end7-start7).count() << " usec" << std::endl;

    end6 = std::chrono::high_resolution_clock::now();
    std::cout << "Total:"<< std::chrono::duration_cast<std::chrono::microseconds>(end6-start6).count() << " usec" << std::endl;
    return 0;
}


int main(){
    // main_opencv();
    main_fast();
    printf("---------------------------\n");
    // main_cpu();
}
