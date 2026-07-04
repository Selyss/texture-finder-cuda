#pragma once

struct BlockInfo
{
    int x, y, z;
    int rotation;
    bool isSide;
};

constexpr int MOD_TOP_BOTTOM = 4;
constexpr int MOD_SIDE = 2;

constexpr int MODERN_VERSION = 0;   // 1.21.2+
constexpr int LEGACY_VERSION = 1;   // 1.13 - 1.21.1
constexpr int VANILLA12_VERSION = 2; // <= 1.12.2
constexpr int SODIUM_VERSION = 3;   // Sodium 1.0-4.1 (MC 1.16 - 1.18.2)
constexpr int SODIUM19_VERSION = 4; // Sodium 4.2-4.8 (MC 1.19 - 1.19.3)
constexpr int NUM_VERSIONS = 5;

// Capacity of the __constant__ arrays the kernel reads the formation from.
constexpr int MAX_FORMATION_BLOCKS = 1024;
