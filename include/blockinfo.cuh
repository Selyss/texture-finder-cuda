#pragma once

struct BlockInfo
{
    int x, y,z;
    int rotation;
    bool isSide;
};

constexpr int MOD_TOP_BOTTOM = 4;
constexpr int MOD_SIDE = 2;

// TODO: make enum
constexpr int MODERN_VERSION = 0;

extern __constant__ BlockInfo d_topsAndBottoms[1024];
extern __constant__ BlockInfo d_sides[1024];