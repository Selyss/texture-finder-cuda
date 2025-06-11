#include <stdio.h>
#include "kernel.cuh"
#include "texture.cuh"

__global__ void matchFormationKernel(int x_min, int x_max, int y_min, int y_max, int z_min, int z_max, int tb_size, int side_size, int version)
{
    int x = x_min + blockIdx.x * blockDim.x + threadIdx.x;
    int y = y_min + blockIdx.y * blockDim.y + threadIdx.y;
    int z = z_min + blockIdx.z * blockDim.z + threadIdx.z;

    // FIXME: entering the same y coord to scan a single level does not work
    if (x > x_max || y > y_max || z > z_max)
        return;

    bool match = true;

    for (int i = 0; i < tb_size; i++)
    {
        int bx = x + d_topsAndBottoms[i].x;
        int by = y + d_topsAndBottoms[i].y;
        int bz = z + d_topsAndBottoms[i].z;
        int texture = (version == MODERN_VERSION) ? getTextureModern(bx, by, bz, MOD_TOP_BOTTOM) : getTextureLegacy(bx, by, bz, MOD_TOP_BOTTOM);

        // this is done instead of an if statement for performance reasons
        match &= (d_topsAndBottoms[i].rotation == texture);
        if (!match)
            break;
    }

    // NOTE: is this needed along with the other if?
    if (!match)
        return;

    for (int i = 0; i < side_size; i++)
    {
        int cx = x + d_sides[i].x;
        int cy = y + d_sides[i].y;
        int cz = z + d_sides[i].z;

        int texture = (version == MODERN_VERSION) ? getTextureModern(cx, cy, cz, MOD_SIDE) : getTextureLegacy(cx, cy, cz, MOD_SIDE);

        match &= (d_sides[i].rotation == texture);

        if (!match)
            break;
    }

    if (match)
    {
        printf("Match found at [%d, %d, %d]\n", x, y, z);
    }
}