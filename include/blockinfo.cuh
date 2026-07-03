#pragma once

struct BlockInfo
{
    int x, y, z;
    int rotation;
    bool isSide;
};

constexpr int MOD_TOP_BOTTOM = 4;
constexpr int MOD_SIDE = 2;

constexpr int MODERN_VERSION = 0; // 1.21.2+
constexpr int LEGACY_VERSION = 1; // 1.13 - 1.21.1

// Capacity of the __constant__ arrays the kernel reads the formation from.
constexpr int MAX_FORMATION_BLOCKS = 1024;
