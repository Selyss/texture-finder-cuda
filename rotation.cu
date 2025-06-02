#include "rotation.cuh"

__host__ __device__ void rotateFormation(BlockInfo &block)
{
    int temp = block.x;
    block.x = -block.z;
    block.z = temp;

    if (block.rotation == 3)
    {
        block.rotation = 0;
    }
    else
    {
        block.rotation++;
    }
}