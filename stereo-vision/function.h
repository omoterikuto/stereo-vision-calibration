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

#ifndef FUNCTION_H
#define FUNCTION_H

#include "define.h"




void gpu_initial(int width, int height);

void StereoMatching(const uint8* src1, const uint8* src2, uint8* dst, const int width, const int height);

void gpu_cudafree(void);





#endif
  
