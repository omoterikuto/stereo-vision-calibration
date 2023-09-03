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
#include "type.h"

extern cv::RNG rng;

least_squares_t least_squares(std::vector<sad_match_t> *sad_match){
    float ave_y =0;
    float ave_x =0;
    printf("matching pair:%ld ", (*sad_match).size());

    for(int i=0; i < (*sad_match).size(); i++){
        ave_y += (*sad_match)[i].y2; 
        ave_x += (*sad_match)[i].x2;
    }
    ave_y = ave_y / (*sad_match).size();
    ave_x = ave_x / (*sad_match).size();

    float a = 0;
    float b = 0;
    float tmp_1=0;
    float tmp_2=0;


    for(int i=0; i < (*sad_match).size(); i++){
        tmp_1 += ((*sad_match)[i].y2 - ave_y) * ((*sad_match)[i].x2 - ave_x);
        tmp_2 += ((*sad_match)[i].x2 - ave_x) * ((*sad_match)[i].x2 - ave_x);
    }
    a = tmp_1 / tmp_2;
    b = ave_y - a * ave_x;
    printf("a:%f, b:%f\n", a, b);

    least_squares_t result = {a,b};
    return result; 
}

void calc_SAD(int C, cv::Mat img_L, cv::Mat img_R, std::vector<sad_match_t> *sad_match, least_squares_t *result){
        cv::Mat out_L;
        cv::Mat out_R;
        cv::cvtColor(img_L, out_L, cv::COLOR_GRAY2RGB);
        cv::cvtColor(img_R, out_R, cv::COLOR_GRAY2RGB);


        cv::Mat img_vconcat;
        cv::Mat img_hconcat;

        cv::vconcat(out_L, out_R, img_vconcat);
        cv::hconcat(out_L, out_R, img_hconcat);

        for(int x_L=X_MARGIN; x_L < IMG_SIZEX; x_L+=X_STEP){            
                int sad_min = INFINITY;
                int sad_min_index_x;
                int sad_min_index_y;
                for(int y= C-SAD_dc; y < C+SAD_dc; y++){ // y= C+-dcの範囲で探す
                    for(int x_R=X_MARGIN; x_R < IMG_SIZEX; x_R++){
                        int sad=0;
                        for(int i=-6; i < 7; i++){ //SAD kernel size
                            for(int j=-6; j < 7; j++){
                                sad += abs(img_L.at<unsigned char>(C+i, x_L+j) - img_R.at<unsigned char>(y+i, x_R+j));
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
                    (*sad_match).push_back({
                        C,
                        x_L,
                        sad_min_index_y,
                        sad_min_index_x,
                        sad_min
                    });
                    printf("%d %d %d %d %d\n", C, x_L, sad_min_index_y, sad_min_index_x, sad_min);
                }
            }
        printf("C=%d ", C);
        *result = least_squares(sad_match);
}

void draw_sad(int C, cv::Mat img_L, cv::Mat img_R, std::vector<sad_match_t> *sad_match, cv::Mat *img_vconcat, cv::Mat *img_hconcat, least_squares_t result){
    for(int i=0; i < (*sad_match).size(); i++){
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::circle(*img_vconcat, cv::Point((*sad_match)[i].x1, (*sad_match)[i].y1), 10, color, -1);
        cv::circle(*img_vconcat, cv::Point((*sad_match)[i].x2, IMG_SIZEY+(*sad_match)[i].y2), 10, color, -1);
        cv::line(*img_vconcat, cv::Point((*sad_match)[i].x1, (*sad_match)[i].y1), cv::Point((*sad_match)[i].x2, IMG_SIZEY+(*sad_match)[i].y2), color, 5, 4);

    }

    cv::line(*img_vconcat, cv::Point(0, IMG_SIZEY+result.b), cv::Point(IMG_SIZEX, IMG_SIZEY+IMG_SIZEX*result.a+result.b), cv::Scalar(0,255,0), 5, 4);
    cv::line(*img_vconcat, cv::Point(0, IMG_SIZEY+C), cv::Point(IMG_SIZEX, IMG_SIZEY+C), cv::Scalar(255,0,0), 5, 4);

    cv::line(*img_hconcat, cv::Point(IMG_SIZEX, result.b), cv::Point(IMG_SIZEX*2, IMG_SIZEX*result.a+result.b), cv::Scalar(0,255,0), 2, 4);
    cv::line(*img_hconcat, cv::Point(0, C), cv::Point(IMG_SIZEX*2, C), cv::Scalar(255,0,0), 2, 4);
}
