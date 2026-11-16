#pragma once

// How the bitplane field is laid out, shared by the planner, every backend
// that searches with bitplanes, and the tests. The field stores the two rotation bits of
// every position on one z row in STRIDE, one 64-bit word pair per 64 x values.
//
// Like texture_rules.hpp this compiles as C++, CUDA and Metal Shading
// Language. The Metal shader is built by pasting the files together (see the
// Makefile), so it has nothing to include, and it names the address space of
// every constant and buffer pointer.
#ifdef __METAL_VERSION__
#define TF_CONSTANT constant
#define TF_BUFFER device
// The fields of search/backend.hpp's SearchBounds.
struct SearchBounds {
  int x_min, x_max;
  int y_min, y_max;
  int z_min, z_max;
};
#else
#include <cstddef>
#include <cstdint>

#include "search/backend.hpp"
#include "texture/texture_rules.hpp"
#define TF_CONSTANT constexpr
#define TF_BUFFER
#endif

namespace bitplanes {
TF_CONSTANT int STRIDE = 5;
TF_CONSTANT int WORD_BITS = 64;
TF_CONSTANT int THREADS = 256;
TF_CONSTANT int DEFAULT_SLAB_ROWS = 4096;
#ifdef __METAL_VERSION__
using Word = uint64_t; // MSL has no long long
#else
using Word = unsigned long long;
#endif

struct Planes {
  Word low, high;
};

struct Check {
  int x_offset, y_slot, z_offset, rotation;
  bool is_side;
};

struct PlaneCheck {
  size_t offset;
  int shift, rotation;
  bool is_side;
};

struct VerifyCheck {
  int64_t z_term;
  uint32_t x_term;
  int y_offset, rotation, mask;
};

struct Layout {
  SearchBounds bounds;
  int nx, ny, nz;
  int field_x, field_width, field_words, candidate_words;
  int y_slots, hashed_rows;
  int64_t first_hashed_z;
};

// The Metal host code hands these structs to the shader as they are.
static_assert(sizeof(Planes) == 16 && sizeof(PlaneCheck) == 24 && sizeof(VerifyCheck) == 24 &&
                  sizeof(Layout) == 72,
              "bitplane structs must have one layout in C++ and in the shader");

TF_HOST_DEVICE inline int residue(int64_t z) {
  const int r = static_cast<int>(z % STRIDE);
  return r < 0 ? r + STRIDE : r;
}

TF_HOST_DEVICE inline int ceil_div(int value, int divisor) {
  return value / divisor + (value % divisor != 0);
}

TF_HOST_DEVICE inline Word low_mask(int bits) {
  return bits == WORD_BITS ? ~0ull : (1ull << bits) - 1;
}

} // namespace bitplanes
