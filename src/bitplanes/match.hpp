#pragma once

#include "bitplanes/layout.hpp"
#include "texture/texture_rules.hpp"

// The three steps of a bitplane search, written once for every backend:
// hashing a field word, filtering a word of candidates against the field, and
// checking a survivor's remaining blocks. A side face shows the low bit of the
// rotation a top face shows, in every version, so one two-bit field serves
// both.

#ifdef __CUDACC__
#define TF_FORCE_INLINE __forceinline__
#else
#define TF_FORCE_INLINE inline
#endif

namespace bitplanes {

// The rotation bits of 64 consecutive x positions on one (y, z) row. `x_term`
// and `z_term` are the hash terms of the first position, and bits from `bits`
// up are cleared. The bits are gathered 32 at a time so the shifts stay in
// 32-bit registers on a GPU.
TF_NODISCARD TF_HOST_DEVICE TF_FORCE_INLINE Planes hash_word(Version version, uint32_t x_term,
                                                             int64_t z_term, int32_t y,
                                                             int bits) TF_NOEXCEPT {
  Planes pair{};
  if (bits <= 0)
    return pair;
  for (int half = 0; half < 2; ++half) {
    uint32_t low = 0, high = 0;
    for (int bit = 0; bit < 32; ++bit, x_term += 3129871u) {
      const uint32_t rotation =
          static_cast<uint32_t>(texture_from_mix(version, coord_mix_from_parts(x_term, z_term, y), 4));
      low = (low >> 1) | (rotation << 31);
      high = (high >> 1) | ((rotation >> 1) << 31);
    }
    pair.low |= static_cast<Word>(low) << (half * 32);
    pair.high |= static_cast<Word>(high) << (half * 32);
  }
  pair.low &= low_mask(bits);
  pair.high &= low_mask(bits);
  return pair;
}

// The candidates of one word that pass every plane check of their z coset.
// `y` and `z_row` count from the slab's first level and first row.
TF_NODISCARD TF_HOST_DEVICE TF_FORCE_INLINE Word filter_word(const Layout& layout, const PlaneCheck* checks,
                                                             const int* coset_offsets, const Planes* planes,
                                                             size_t y, size_t z_row, int word) TF_NOEXCEPT {
  const int remaining = layout.nx - word * WORD_BITS;
  Word mask = low_mask(remaining < WORD_BITS ? remaining : WORD_BITS);
  // The coset anchor is within the slab's first STRIDE rows.
  // This quotient is the number of strides since that anchor.
  const size_t base = (y * layout.hashed_rows + z_row / STRIDE) * layout.field_words + word;
  const int coset = residue(layout.bounds.z_min + static_cast<int>(z_row));
  for (int i = coset_offsets[coset]; i < coset_offsets[coset + 1] && mask; ++i) {
    const PlaneCheck check = checks[i];
    const size_t index = base + check.offset;
    Word low = planes[index].low >> check.shift;
    if (check.shift)
      low |= planes[index + 1].low << (WORD_BITS - check.shift);
    mask &= (check.rotation & 1) ? low : ~low;
    if (!check.is_side) {
      Word high = planes[index].high >> check.shift;
      if (check.shift)
        high |= planes[index + 1].high << (WORD_BITS - check.shift);
      mask &= (check.rotation & 2) ? high : ~high;
    }
  }
  return mask;
}

// True when the candidate also has every block the field does not cover:
// checks [first, last) of `verify`. `x_term` and `z_term` are the candidate
// origin's hash terms.
TF_NODISCARD TF_HOST_DEVICE TF_FORCE_INLINE bool survivor_matches(Version version, const VerifyCheck* verify,
                                                                  int first, int last, uint32_t x_term,
                                                                  int64_t z_term, int32_t y) TF_NOEXCEPT {
  for (int i = first; i < last; ++i) {
    const VerifyCheck check = verify[i];
    const int rotation = texture_from_mix(
        version, coord_mix_from_parts(x_term + check.x_term, z_term + check.z_term, y + check.y_offset), 4);
    if ((rotation & check.mask) != check.rotation)
      return false;
  }
  return true;
}

} // namespace bitplanes
