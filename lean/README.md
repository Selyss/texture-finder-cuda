# Texture RNG proofs

Run `lake exe cache get Mathlib.Data.BitVec Mathlib.Tactic.Ring` and then
`lake build` from this directory with Elan installed. The project pins Lean
4.19.0 and the corresponding Mathlib release. On macOS, if the cache binary
fails to load with a `__DATA_CONST` error, run `lake build mathlib/cache`, then
`lake env lean --run .lake/packages/mathlib/Cache/Main.lean get Mathlib.Data.BitVec Mathlib.Tactic.Ring`.

[Rng.lean](TextureFinder/Rng.lean) models the five modes in
[texture_rules.hpp](../src/texture/texture_rules.hpp) with explicit bit widths.
[Vectors.lean](TextureFinder/Vectors.lean) checks the 62 synthetic constant
vectors from [test.cpp](../test/test.cpp) and the signed minimum edge case.
The vector checks are part of `lake build`.

| Handoff theorem | Lean declarations | Reduction log |
| --- | --- | --- |
| T1. Aligned XOR | `xor_small_eq_add`, `xor_aligned_eq_add`, `xor_offset_in_range` | [Column sweep, aligned blocks](../docs/RNG_REDUCTION_LOG.md#2-r1--y-column-forward-difference-sweep) |
| T2. Offset permutation | `blockOffset_involutive` | [Column sweep, re-indexing](../docs/RNG_REDUCTION_LOG.md#2-r1--y-column-forward-difference-sweep) |
| T3. Negative y | `xor_not_not`, `complement_negative_y` | [Column sweep, negative y](../docs/RNG_REDUCTION_LOG.md#2-r1--y-column-forward-difference-sweep) |
| T4. Forward differences | `mix_first_diff`, `mix_second_diff`, `sweep_invariant` | [Column sweep, recurrence](../docs/RNG_REDUCTION_LOG.md#2-r1--y-column-forward-difference-sweep) |
| T5. Sweep correctness | `sweep_aligned`, `sweep_reindexed`, `sweep_negative_aligned` | [Column sweep, reference equivalence](../docs/RNG_REDUCTION_LOG.md#2-r1--y-column-forward-difference-sweep) |
| T6. Low-bit dependence | `positionMix_low`, `absMod_parity`, `neg_mod_four`, `legacy_parity_low33`, `legacy_even_low34`, `vanilla12_low32`, `vanilla12_parity_low17`, `vanilla12_even_low18` | [Low-bit structure](../docs/RNG_REDUCTION_LOG.md#5-low-bit-structure-per-version) |
| T7. Lattice soundness | `partition_match`, `lattice_coset` | [Sparse sublattice](../docs/RNG_REDUCTION_LOG.md#4-r3--only-hash-a-sparse-sublattice) |
| T8. Bitplane matching | `extractWord_bit`, `blockMask_bit`, `combinedMask_bit` | [Hash field as bitplanes](../docs/RNG_REDUCTION_LOG.md#3-r2--store-the-hash-field-as-bitplanes) |

`ring` closes the forward-difference identities directly on `BitVec 64`.
The alignment proof works for any block width up to the word width and
reduces addition to OR on disjoint bits. A separate `bv_decide` check
proved the 64-bit case with a 31-bit offset. The low-bit proofs use
truncation laws for the multiply and add steps. The even-value tag retains
0 and 2 and maps both odd outputs to 1.

The proofs cover the Lean model. The existing oracle and end-to-end tests
connect the model assumptions to the C++ and CUDA implementations.
