#pragma once
#include "blockinfo.cuh"

__global__ void matchFormationKernel(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, int tb_size, int side_size, int version);