/**
	This file is part of z2zncc. 

	Copyright (c) 2021 Qiong Chang.

	z2zncc is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	z2zncc is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with z2zncc.  If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef DEFINE_H_
#define DEFINE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <device_launch_parameters.h>
#include <math.h> 
#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "cuda_runtime_api.h"

#define NCC_W 5

#define WARP 16
#define AVR_WARP_H 32
#define AVR_WARP_V 16
#define MAX_DISPARITY  96

#define MATCH_W_H  48
#define MATCH_H_SEG (MATCH_W_H-NCC_W+1)

#define MATCH_V_SEG  4
#define MATCH_W_V (MATCH_V_SEG+NCC_W-1)

#define SIGMA_A 52
#define SIGMA_B 52
#define SIGMA_TH 5
//#define UN 1

#ifdef UN
typedef  unsigned int uint32;
typedef  unsigned char uint8;
typedef  unsigned short int uint16;

#else
typedef  int uint32;
typedef  unsigned char uint8;
typedef  short int uint16;
#endif

static constexpr uint8 NCC_R = NCC_W/2;
static constexpr float PIX_NUM = 1.0f*(NCC_W*NCC_W);
static constexpr float AVR_RECIP = 1.0f/(PIX_NUM);


static constexpr uint8 WARP_SIZE = 32U;
static constexpr uint8 ROWS = (MAX_DISPARITY/WARP_SIZE);

static constexpr uint8 AGG_STEP = 2U;
static constexpr uint8 BITS = 4U;

static constexpr uint16 PIX_W = 1280;
//#define ACCURACY 1




#endif
 
