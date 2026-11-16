#pragma once

struct BlockInfo {
  int x, y, z;
  int rotation;
  bool is_side;
};

constexpr int MOD_TOP_BOTTOM = 4;
constexpr int MOD_SIDE = 2;

// Capacity of the __constant__ arrays the kernel reads the formation from.
constexpr int MAX_FORMATION_BLOCKS = 1024;
