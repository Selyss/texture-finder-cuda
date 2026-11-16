#pragma once

// Compilable as C++, CUDA, and Metal Shading Language so the exact same
// code is used by every backend, the test suite, and the tools. Never copy
// these functions.
#ifdef __METAL_VERSION__
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t; // MSL long/ulong are 64-bit
typedef unsigned long uint64_t;
#define TF_HOST_DEVICE
#else
#include <cstdint>
#ifdef __CUDACC__
#define TF_HOST_DEVICE __host__ __device__
#else
#define TF_HOST_DEVICE
#endif
#endif

// Every function here is pure: discarding its result is always a bug, and
// none of them throw. Metal (MSL) may reject the C++ attributes, so they
// expand to nothing there.
#ifdef __METAL_VERSION__
#define TF_NODISCARD
#define TF_NOEXCEPT
#else
#define TF_NODISCARD [[nodiscard]]
#define TF_NOEXCEPT noexcept
#endif

// The game and renderer versions that roll rotations differently. The values
// are the CLI's version argument and are written into checkpoint signatures,
// so they are never renumbered.
enum class Version : int {
  modern = 0,    // 1.21.2+
  legacy = 1,    // 1.13 - 1.21.1
  vanilla12 = 2, // <= 1.12.2
  sodium = 3,    // Sodium 1.0-4.1 (MC 1.16 - 1.18.2)
  sodium19 = 4,  // Sodium 4.2-4.8 (MC 1.19 - 1.19.3)
};
#ifndef __METAL_VERSION__ // MSL has no program-scope constants
constexpr int NUM_VERSIONS = 5;
#endif

// Reference semantics: 19MisterX98/TextureRotations (texture/*.java) and the
// vanilla Mth.get_seed position hash. Java overflow wraps; C++ signed overflow
// is UB, so every wrapping step below is routed through unsigned arithmetic.

// Java: long l = (long)(x * 3129871) ^ (long)z * 116129781L ^ (long)y;
//       l = l * l * 42317861L + l * 11L;
//       return l >> 16;
// x * 3129871 is 32-bit Java int math (wraps); the rest is 64-bit.
//
// The three coordinate terms are exposed separately (coord_mix_from_parts)
// because both multiplications distribute over the formation offsets under
// wrapping: (x+bx)*3129871 mod 2^32 == x*3129871 + bx*3129871 mod 2^32, and
// the z product is exact in 64 bits. The kernel exploits this to hoist the
// per-candidate multiplies out of the per-block loop; the fused coord_random
// below stays the reference form.
//
// coord_mix is the full 64-bit mixed value BEFORE the >> 16: most versions
// consume mix >> 16 (reference get_coordinate_random), but <=1.12 truncates
// the full value to int first (reference get_coordinate_random_legacy).
TF_NODISCARD TF_HOST_DEVICE inline int64_t coord_mix_from_parts(uint32_t x_term, int64_t z_term,
                                                                int32_t y_term) TF_NOEXCEPT {
  int64_t l = (int64_t)(int32_t)x_term ^ z_term ^ (int64_t)y_term;
  uint64_t ul = (uint64_t)l;
  ul = ul * ul * 42317861ull + ul * 11ull;
  return (int64_t)ul;
}

TF_NODISCARD TF_HOST_DEVICE inline int64_t coord_random_from_parts(uint32_t x_term, int64_t z_term,
                                                                   int32_t y_term) TF_NOEXCEPT {
  return coord_mix_from_parts(x_term, z_term, y_term) >> 16;
}

TF_NODISCARD TF_HOST_DEVICE inline int64_t coord_mix(int32_t x, int32_t y, int32_t z) TF_NOEXCEPT {
  return coord_mix_from_parts((uint32_t)x * 3129871u, (int64_t)z * 116129781LL, y);
}

TF_NODISCARD TF_HOST_DEVICE inline int64_t coord_random(int32_t x, int32_t y, int32_t z) TF_NOEXCEPT {
  return coord_mix(x, y, z) >> 16;
}

// Java Math.abs(rand) % mod, computed in unsigned arithmetic: identical for
// mod 2 and 4 even at rand == INT_MIN (both yield 0), and never UB.
TF_NODISCARD TF_HOST_DEVICE inline int abs_mod(int32_t rand, int mod) TF_NOEXCEPT {
  uint32_t a = rand < 0 ? 0u - (uint32_t)rand : (uint32_t)rand;
  return (int)(a % (uint32_t)mod);
}

// Versions 1.13 - 1.21.1 (reference: Vanilla21_1Textures).
// Java: seed = (coord_random ^ 0x5DEECE66D) & ((1<<48)-1);
//       rand = (int)((seed * 0xBB20B4600A69 + 0x40942DE6BA) >>> 16);  // next_long, 2 LCG steps combined
//       return Math.abs(rand) % mod;
// The unsigned-negate trick equals Java Math.abs for mod 2 and 4 even at
// rand == INT_MIN (both yield 0), without the UB of abs(INT_MIN).
TF_NODISCARD TF_HOST_DEVICE inline int legacy_from_coord_random(int64_t coord_rand, int mod) TF_NOEXCEPT {
  constexpr int64_t MULTIPLIER = 0x5DEECE66DLL;
  constexpr int64_t MASK = (1LL << 48) - 1;
  int64_t seed = (coord_rand ^ MULTIPLIER) & MASK;
  uint64_t v = (uint64_t)seed * 0xBB20B4600A69ull + 0x40942DE6BAull;
  return abs_mod((int32_t)(uint32_t)(v >> 16), mod);
}

TF_NODISCARD TF_HOST_DEVICE inline int get_texture_legacy(int32_t x, int32_t y, int32_t z,
                                                          int mod) TF_NOEXCEPT {
  return legacy_from_coord_random(coord_random(x, y, z), mod);
}

// Versions <= 1.12.2 (reference: Vanilla12Textures).
// Java: rand = (int) coord_mix >> 16;   // truncate to int FIRST, then shift
//       return Math.abs(rand) % mod;
// No LCG scramble at all in this era. The int64 detour below performs the
// arithmetic >> 16 with fully defined semantics.
TF_NODISCARD TF_HOST_DEVICE inline int vanilla12_from_mix(int64_t mix, int mod) TF_NOEXCEPT {
  int32_t truncated = (int32_t)(uint32_t)(uint64_t)mix;
  int32_t rand = (int32_t)((int64_t)truncated >> 16);
  return abs_mod(rand, mod);
}

TF_NODISCARD TF_HOST_DEVICE inline int get_texture_vanilla12(int32_t x, int32_t y, int32_t z,
                                                             int mod) TF_NOEXCEPT {
  return vanilla12_from_mix(coord_mix(x, y, z), mod);
}

// Java: (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9; (z ^ (z >>> 27)) * 0x94D049BB133111EB; z ^ (z >>> 31)
TF_NODISCARD TF_HOST_DEVICE inline uint64_t stafford_mix13(uint64_t z) TF_NOEXCEPT {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Sodium 4.2 - 4.8 on MC 1.19 - 1.19.3 (reference: Sodium19Textures).
// Java: l = seed ^ 7640891576956012809L; m = l + -7046029254386353131L;
//       rand = (int)(Long.rotate_left(mix13(l) + mix13(m), 17) + mix13(l));
//       return Math.abs(rand) % mod;
// (-7046029254386353131 as u64 is 0x9E3779B97F4A7C15, the golden ratio.)
TF_NODISCARD TF_HOST_DEVICE inline int sodium19_from_coord_random(int64_t coord_rand, int mod) TF_NOEXCEPT {
  uint64_t l = (uint64_t)coord_rand ^ 0x6A09E667F3BCC909ull;
  uint64_t m = l + 0x9E3779B97F4A7C15ull;
  l = stafford_mix13(l);
  m = stafford_mix13(m);
  uint64_t sum = l + m;
  uint64_t rot = (sum << 17) | (sum >> 47); // Long.rotate_left(sum, 17)
  return abs_mod((int32_t)(uint32_t)(rot + l), mod);
}

TF_NODISCARD TF_HOST_DEVICE inline int get_texture_sodium19(int32_t x, int32_t y, int32_t z,
                                                            int mod) TF_NOEXCEPT {
  return sodium19_from_coord_random(coord_random(x, y, z), mod);
}

// Sodium 1.0 - 4.1 on MC 1.16 - 1.18.2 (reference: SodiumTextures).
// Java: murmur-style avalanche of the seed, then
//       rand1 = mix13(seed += PHI); rand2 = mix13(seed + PHI);
//       rand = (int)(rand1 + rand2); return Math.abs(rand) % mod;
TF_NODISCARD TF_HOST_DEVICE inline int sodium_from_coord_random(int64_t coord_rand, int mod) TF_NOEXCEPT {
  uint64_t s = (uint64_t)coord_rand;
  s ^= s >> 33;
  s *= 0xFF51AFD7ED558CCDull;
  s ^= s >> 33;
  s *= 0xC4CEB9FE1A85EC53ull;
  s ^= s >> 33;
  s += 0x9E3779B97F4A7C15ull; // seed += PHI
  uint64_t rand1 = stafford_mix13(s);
  uint64_t rand2 = stafford_mix13(s + 0x9E3779B97F4A7C15ull); // seed + PHI again
  return abs_mod((int32_t)(uint32_t)(rand1 + rand2), mod);
}

TF_NODISCARD TF_HOST_DEVICE inline int get_texture_sodium(int32_t x, int32_t y, int32_t z,
                                                          int mod) TF_NOEXCEPT {
  return sodium_from_coord_random(coord_random(x, y, z), mod);
}

// Versions 1.21.2+ (reference: VanillaTextures).
// Java: seed = coord_random ^ 0x5DEECE66D;
//       seed = seed * 0x5DEECE66D + 11 & ((1<<48)-1);
//       next = (int)(seed >> 48 - 31);                       // next_int(4) bound path
//       return (int)((4 * (long)next) >> 31) % mod;
// The game always rolls one of 4 model variants; a side face only exposes the
// parity of that roll, hence the hardcoded 4 with % mod applied after. Using
// next_int(2)-style (mod * next) >> 31 for sides is WRONG - it draws a number
// the game never draws.
TF_NODISCARD TF_HOST_DEVICE inline int modern_from_coord_random(int64_t coord_rand, int mod) TF_NOEXCEPT {
  constexpr int64_t MULTIPLIER = 0x5DEECE66DLL;
  constexpr int64_t MASK = (1LL << 48) - 1;
  int64_t seed = coord_rand ^ MULTIPLIER;
  seed = (int64_t)((uint64_t)seed * (uint64_t)MULTIPLIER + 11ull) & MASK;
  // seed is in [0, 2^48) after the mask, so `next` is non-negative and the
  // Java tail is computed in unsigned form (identical values for all
  // inputs; this lets the compiler fold the shift and constant mod instead
  // of emitting signed-division fixups it can't prove away).
  unsigned int next = (unsigned int)((uint64_t)seed >> (48 - 31));
  return (int)(((4ull * next) >> 31) % (unsigned int)mod);
}

TF_NODISCARD TF_HOST_DEVICE inline int get_texture_modern(int32_t x, int32_t y, int32_t z,
                                                          int mod) TF_NOEXCEPT {
  return modern_from_coord_random(coord_random(x, y, z), mod);
}

// The rotation at a position whose coord_mix is `mix`, for any version. The
// GPU kernels pass a compile-time version, so the switch folds away there.
TF_NODISCARD TF_HOST_DEVICE inline int texture_from_mix(Version version, int64_t mix, int mod) TF_NOEXCEPT {
  switch (version) {
  case Version::legacy:
    return legacy_from_coord_random(mix >> 16, mod);
  case Version::vanilla12:
    return vanilla12_from_mix(mix, mod);
  case Version::sodium:
    return sodium_from_coord_random(mix >> 16, mod);
  case Version::sodium19:
    return sodium19_from_coord_random(mix >> 16, mod);
  case Version::modern:
    break;
  }
  return modern_from_coord_random(mix >> 16, mod);
}

TF_NODISCARD TF_HOST_DEVICE inline int get_texture_for_version(Version version, int32_t x, int32_t y,
                                                               int32_t z, int mod) TF_NOEXCEPT {
  return texture_from_mix(version, coord_mix(x, y, z), mod);
}
