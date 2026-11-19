#pragma once

#include <cstdint>

#include "bitplanes/match.hpp"

// The field fill of the CPU backend. bitplanes::field_word shifts each
// rotation bit into two 32-bit registers, which suits a GPU. A CPU core is
// faster keeping the two bits of a rotation together: 32 rotations are packed
// into one word as bit pairs, with nothing carried between hashes but one OR,
// and the pairs are split into the two planes once per word.
namespace cpu_bitplanes {

using namespace bitplanes;

// Bits 0, 2, 4, ... of `value`, gathered into the low 32 bits.
[[nodiscard]] constexpr Word even_bits(Word value) noexcept {
  value &= 0x5555555555555555ull;
  value = (value | value >> 1) & 0x3333333333333333ull;
  value = (value | value >> 2) & 0x0F0F0F0F0F0F0F0Full;
  value = (value | value >> 4) & 0x00FF00FF00FF00FFull;
  value = (value | value >> 8) & 0x0000FFFF0000FFFFull;
  return (value | value >> 16) & 0x00000000FFFFFFFFull;
}

// The same word as bitplanes::field_word(VERSION, layout, y, z_index, word).
template <Version VERSION>
[[nodiscard]] inline Planes fill_word(const Layout& layout, std::int32_t y, int z_index, int word) noexcept {
  const std::int64_t z_term{(layout.first_hashed_z + std::int64_t{z_index} * STRIDE) * 116129781LL};
  const std::int64_t x_offset{std::int64_t{word} * WORD_BITS};
  const std::int64_t remaining{layout.field_width - x_offset};
  if (remaining <= 0)
    return {};

  auto x_term{static_cast<std::uint32_t>(layout.field_x + x_offset) * 3129871u};
  Word pairs[2];
  for (auto& half : pairs) {
    half = 0;
    for (int i{0}; i < 32; ++i, x_term += 3129871u)
      half |= static_cast<Word>(texture_from_mix(VERSION, coord_mix_from_parts(x_term, z_term, y), 4))
              << (2 * i);
  }
  const Word keep{low_mask(remaining < WORD_BITS ? static_cast<int>(remaining) : WORD_BITS)};
  return {(even_bits(pairs[0]) | even_bits(pairs[1]) << 32) & keep,
          (even_bits(pairs[0] >> 1) | even_bits(pairs[1] >> 1) << 32) & keep};
}

} // namespace cpu_bitplanes
