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

#ifndef STEREO_H
#define STEREO_H


static uint8 * gpu_cost_left;
static uint8 * gpu_cost_right;
static uint8 * dst_temp;
static float * avrimg1;
static float * avrimg2;
static float * sum2_L;
static float * sum2_R;                                                              
static uint8 * costL2R;
static uint16 * costR2L;
static uint16 * costU2D;

float exp_list_sub_cpu_1[256];
float exp_list_sub_cpu_2[256];

__constant__ float exp_list_sub_1[256];
__constant__ float exp_list_sub_2[256];


#endif
