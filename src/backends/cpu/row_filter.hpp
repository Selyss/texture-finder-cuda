#pragma once

#include <cstddef>

#include "bitplanes/match.hpp"

namespace cpu_bitplanes {

// bitplanes::filter_word for every candidate word of one (y, z) row at once:
// masks[word] gets what filter_word returns for that word. Each plane check is
// applied to the whole row in one loop with no branch and no early exit, so
// the compiler can vectorise it. `masks` has layout.candidate_words entries.
inline void filter_row(const bitplanes::Layout& layout, const bitplanes::PlaneCheck* checks,
                       const int* coset_offsets, const bitplanes::Planes* planes, std::size_t y,
                       std::size_t z_row, bitplanes::Word* masks) noexcept {
  using namespace bitplanes;
  const auto words{static_cast<std::size_t>(layout.candidate_words)};
  for (std::size_t word{0}; word + 1 < words; ++word)
    masks[word] = ~Word{0};
  masks[words - 1] = low_mask(layout.nx - (layout.candidate_words - 1) * WORD_BITS);

  const auto* row{planes + (y * layout.hashed_rows + z_row / STRIDE) * layout.field_words};
  const int coset{residue(layout.bounds.z_min + static_cast<int>(z_row))};
  for (int i{coset_offsets[coset]}; i < coset_offsets[coset + 1]; ++i) {
    const auto& check{checks[i]};
    const auto* field{row + check.offset};
    // A wanted bit of 0 flips the plane, so the AND keeps the positions that
    // have it. A side face ignores the high plane.
    const Word flip_low{(check.rotation & 1) ? Word{0} : ~Word{0}};
    const Word use_high{check.is_side ? Word{0} : ~Word{0}};
    const Word flip_high{!check.is_side && (check.rotation & 2) ? Word{0} : ~Word{0}};
    if (check.shift == 0) {
      for (std::size_t word{0}; word < words; ++word)
        masks[word] &= (field[word].low ^ flip_low) & ((field[word].high & use_high) ^ flip_high);
    } else {
      const int down{check.shift}, up{WORD_BITS - check.shift};
      for (std::size_t word{0}; word < words; ++word) {
        const Word low{(field[word].low >> down) | (field[word + 1].low << up)};
        const Word high{(field[word].high >> down) | (field[word + 1].high << up)};
        masks[word] &= (low ^ flip_low) & ((high & use_high) ^ flip_high);
      }
    }
  }
}

} // namespace cpu_bitplanes
