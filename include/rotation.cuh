#pragma once
#include "blockinfo.cuh"

// Rotates one block of a formation by 90 degrees about the y axis:
// (x, z) -> (-z, x), i.e. east -> south viewed from above.
//
// The game rolls one of 4 model variants per position; rotating the whole
// formation by a quarter turn shifts the apparent variant of every block by
// one step. Top faces expose the full variant (0-3), so their expected value
// advances mod 4. Side faces only expose the variant's parity (0-1), so their
// expected value flips each quarter turn. Keeping a side's value in {0, 1} is
// essential: the kernel compares it against a mod-2 texture value, and a
// value of 2 or 3 could never match anything.
inline void rotateFormation(BlockInfo &block)
{
    int temp = block.x;
    block.x = -block.z;
    block.z = temp;

    if (block.isSide)
        block.rotation = (block.rotation + 1) & 1;
    else
        block.rotation = (block.rotation + 1) & 3;
}
