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

void kernel_transform_ry_cpu(unsigned char* SrcImg, unsigned char* DstImg, float beta){
    for(int idy=0; idy < IMG_SIZEY; idy++){
        for(int idx=0; idx < IMG_SIZEX; idx++){
            idx -= IMG_SIZEX/2;
            idy -= IMG_SIZEY/2;
            int x = F*(idx*cosf(beta) + F*sinf(beta))/(-idx*sinf(beta) + F*cosf(beta));
            int y = F*(idy)/(-idx*sinf(beta) + F*cosf(beta));
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
    }
}

void kernel_transform_rx_cpu(unsigned char* SrcImg, unsigned char* DstImg, float alpha){
    for(int idy=0; idy < IMG_SIZEY; idy++){
        for(int idx=0; idx < IMG_SIZEX; idx++){
            idx -= IMG_SIZEX/2;
            idy -= IMG_SIZEY/2;
            int x = F*(idx)/(idy*sinf(alpha) + F*cosf(alpha));
            int y = F*(idy*cosf(alpha) - F*sinf(alpha))/(idy*sinf(alpha) + F*cosf(alpha));
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
    }
}

void kernel_transform_rz_cpu(unsigned char* SrcImg, unsigned char* DstImg, float gamma){
    for(int idy=0; idy < IMG_SIZEY; idy++){
        for(int idx=0; idx < IMG_SIZEX; idx++){
            idx -= IMG_SIZEX/2;
            idy -= IMG_SIZEY/2;
            int x = idx*cosf(gamma) - idy*sinf(gamma);
            int y = idx*sinf(gamma) + idy*cosf(gamma);
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
    }
}

void kernel_transform_ty_cpu(unsigned char* SrcImg, unsigned char* DstImg, float dy){
    for(int idy=0; idy < IMG_SIZEY; idy++){
        for(int idx=0; idx < IMG_SIZEX; idx++){
            int x = idx;
            int y = idy + dy;

            if(x < 0 || x > IMG_SIZEX || y < 0 || y > IMG_SIZEY){
                DstImg[idy*IMG_SIZEX + idx] = 0;
            }else{
                DstImg[idy*IMG_SIZEX + idx] = SrcImg[y*IMG_SIZEX+x];
            }
        }
    }
}

void kernel_transform_tz_cpu(unsigned char* SrcImg, unsigned char* DstImg, float dz){
    for(int idy=0; idy < IMG_SIZEY; idy++){
        for(int idx=0; idx < IMG_SIZEX; idx++){
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
    }
}
