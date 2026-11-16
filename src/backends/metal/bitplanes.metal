// Metal kernels of the bitplane search. The Makefile pastes this file after
// texture_rules.hpp, match.metal, bitplanes/layout.hpp and
// bitplanes/match.hpp, so the field, the filter and the survivor check are
// the code the CPU and CUDA backends run. The host side is bitplane_search.mm.

using namespace bitplanes;

// One thread per field word: x is the word, y the hashed z row, z the level.
kernel void fill_planes(constant Layout& layout [[buffer(0)]], device const int* plane_y [[buffer(1)]],
                        device Planes* planes [[buffer(2)]], uint3 t [[thread_position_in_grid]]) {
  if (t.x >= (uint)layout.field_words || t.y >= (uint)layout.hashed_rows || t.z >= (uint)layout.y_slots)
    return;
  const size_t row = (size_t)t.z * layout.hashed_rows + t.y;
  planes[row * layout.field_words + t.x] =
      field_word(Version(TF_VERSION), layout, plane_y[t.z], (int)t.y, (int)t.x);
}

// One thread per candidate word: x is the word, y the y level, z the z row.
// Each surviving bit is one candidate, whose remaining blocks are hashed here.
kernel void match_words(constant Layout& layout [[buffer(0)]], device const PlaneCheck* checks [[buffer(1)]],
                        device const int* coset_offsets [[buffer(2)]],
                        device const Planes* planes [[buffer(3)]],
                        device const VerifyCheck* verify [[buffer(4)]],
                        device const int* verify_offsets [[buffer(5)]], device TfMatch* out [[buffer(6)]],
                        device atomic_uint* write_idx [[buffer(7)]],
                        device atomic_uint* pass_count [[buffer(8)]], uint3 t [[thread_position_in_grid]]) {
  if (t.x >= (uint)layout.candidate_words || t.y >= (uint)layout.ny || t.z >= (uint)layout.nz)
    return;
  const int word = (int)t.x;
  Word survivors = filter_word(layout, checks, coset_offsets, planes, t.y, t.z, word);
  if (!survivors)
    return;

  const int y = layout.bounds.y_min + (int)t.y;
  const int z = layout.bounds.z_min + (int)t.z;
  const int coset = residue(z);
  const int64_t z_term = (int64_t)z * 116129781L;
  for (; survivors; survivors &= survivors - 1) {
    const int x = layout.bounds.x_min + word * WORD_BITS + (int)ctz(survivors);
    if (survivor_matches(Version(TF_VERSION), verify, verify_offsets[coset], verify_offsets[coset + 1],
                         (uint32_t)x * 3129871u, z_term, y))
      record_match(out, write_idx, pass_count, x, y, z);
  }
}
