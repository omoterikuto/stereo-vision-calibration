
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "type.h"


cv::RNG rng(12345);
cv::Mat read_images(std::string path, cv::cuda::Stream *stream){
    cv::Mat	img;
    cv::Mat img_resized;
    cv::Mat img_gray;
    cv::cuda::GpuMat img_gpu, img_resized_gpu, img_gray_gpu;
    img = cv::imread(path);
    // cv::resize(img, img_resized, cv::Size(IMG_SIZEX, IMG_SIZEY));
    // cv::cvtColor(img_resized, img_gray, cv::COLOR_RGB2GRAY);
    // img = cv::imread(path);
    img_gpu.upload(img, *stream);
    cv::cuda::resize(img_gpu, img_resized_gpu, cv::Size(IMG_SIZEX, IMG_SIZEY));
    cv::cuda::cvtColor(img_resized_gpu, img_gray_gpu, cv::COLOR_RGB2GRAY);
    img_gray_gpu.download(img_gray, *stream);
    return img_gray;

}

cv::Mat read_images_cpu(std::string path){
    cv::Mat	img;
    cv::Mat img_resized;
    cv::Mat img_gray;
    cv::cuda::GpuMat img_gpu, img_resized_gpu, img_gray_gpu;
    img = cv::imread(path);
    cv::resize(img, img_resized, cv::Size(IMG_SIZEX, IMG_SIZEY));
    cv::cvtColor(img_resized, img_gray, cv::COLOR_RGB2GRAY);
    return img_gray;

}

void draw_SAD_CUDA(int C, cv::Mat img_L, cv::Mat img_R, sad_match_t *sad_match, int num_matched){
    cv::Mat img_vconcat;
    cv::vconcat(img_L, img_R, img_vconcat);
    for(int i=0; i < num_matched; i++){
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::circle(img_vconcat, cv::Point(sad_match[i].x1 + IMG_SIZEX/2, sad_match[i].y1 + IMG_SIZEY/2), 5, color, -1);
        cv::circle(img_vconcat, cv::Point(sad_match[i].x2 + IMG_SIZEX/2, IMG_SIZEY+sad_match[i].y2 +IMG_SIZEY/2), 5, color, -1);
        cv::line(img_vconcat, cv::Point(sad_match[i].x1 + IMG_SIZEX/2, sad_match[i].y1 + IMG_SIZEY/2), cv::Point(sad_match[i].x2 + IMG_SIZEX/2, IMG_SIZEY+sad_match[i].y2 + IMG_SIZEY/2), color, 2, 4);
    }
    cv::line(img_vconcat, cv::Point(0, C), cv::Point(IMG_SIZEX, C), cv::Scalar(255,0,0), 2, 4);
    cv::line(img_vconcat, cv::Point(0, IMG_SIZEY+C), cv::Point(IMG_SIZEX, IMG_SIZEY+C), cv::Scalar(255,0,0), 2, 4);
    char fname[64];
    sprintf(fname, "result/sad_cuda_%d.png", C);
    cv::imwrite(fname, img_vconcat);
}


bool cmp_match(cv::DMatch &p, cv::DMatch &q) { //マッチングペアsort用関数
	if (p.distance == 0) return false;
	return p.distance < q.distance;
}
bool cmp_match2(const struct match_t& p, const struct match_t& q) { //マッチングペアsort用関数
	if (p.distance == 0) return false;
	return p.distance < q.distance;
}