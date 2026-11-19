#pragma once

#include <cstddef>

#include "bitplanes/match.hpp"

namespace cpu_bitplanes {

// bitplanes::filter_word for every candidate word of one (y, z) row at once:
// masks[word] gets what filter_word returns for that word. The field is given
// as two arrays, the low and the high rotation bit of each field word, so each
// loop reads one contiguous run of memory. Each plane check is applied to the
// whole row with no branch and no early exit, so the compiler can vectorise
// it. `masks` has layout.candidate_words entries.
inline void filter_row(const bitplanes::Layout& layout, const bitplanes::PlaneCheck* checks,
                       const int* coset_offsets, const bitplanes::Word* low_plane,
                       const bitplanes::Word* high_plane, std::size_t y, std::size_t z_row,
                       bitplanes::Word* masks) noexcept {
  using namespace bitplanes;
  const auto words{static_cast<std::size_t>(layout.candidate_words)};
  for (std::size_t word{0}; word + 1 < words; ++word)
    masks[word] = ~Word{0};
  masks[words - 1] = low_mask(layout.nx - (layout.candidate_words - 1) * WORD_BITS);

  const auto row{(y * layout.hashed_rows + z_row / STRIDE) * layout.field_words};
  const int coset{residue(layout.bounds.z_min + static_cast<int>(z_row))};
  for (int i{coset_offsets[coset]}; i < coset_offsets[coset + 1]; ++i) {
    const auto& check{checks[i]};
    const Word* const low{low_plane + row + check.offset};
    const Word* const high{high_plane + row + check.offset};
    // A wanted bit of 0 flips the plane, so the AND keeps the positions that
    // have it. A side face ignores the high plane.
    const Word flip_low{(check.rotation & 1) ? Word{0} : ~Word{0}};
    const Word flip_high{(check.rotation & 2) ? Word{0} : ~Word{0}};
    const int down{check.shift}, up{WORD_BITS - check.shift};
    if (check.shift == 0) {
      for (std::size_t word{0}; word < words; ++word)
        masks[word] &= low[word] ^ flip_low;
      if (!check.is_side)
        for (std::size_t word{0}; word < words; ++word)
          masks[word] &= high[word] ^ flip_high;
    } else {
      for (std::size_t word{0}; word < words; ++word)
        masks[word] &= ((low[word] >> down) | (low[word + 1] << up)) ^ flip_low;
      if (!check.is_side)
        for (std::size_t word{0}; word < words; ++word)
          masks[word] &= ((high[word] >> down) | (high[word + 1] << up)) ^ flip_high;
    }
  }
}

} // namespace cpu_bitplanes
