# RNG reduction log

Investigation of the texture-rotation generator (`getTexture*`) for structure
that lets the search do less work per candidate. Started 2026-11-12.

**Status:** three reductions found and verified bit-exact on CPU prototypes.
None is in the CUDA kernel yet and none has been measured on a GPU.

| # | Reduction | What it removes | CPU speedup (single thread) |
|---|---|---|---|
| R1 | y-column forward-difference sweep | the two 64-bit multiplies of the position mix | 1.2–1.5x |
| R2 | hash-field bitplanes | per-candidate evaluation and branching; one block check tests 64 candidates | 2.8–4.2x |
| R3 | sparse sublattice | most of the hash field itself (only 1/s of positions are ever hashed) | 7–12x with R2 |

All CPU numbers: Apple M4 Pro, one thread, `clang++ -O2`, modern (1.21.2+)
generator unless noted, formation = the 24-block `formation_a` fixture, boxes
are slices of the reference workload W from `docs/EXPERIMENTS.md` on the
`correctness-overhaul` branch. "ref" is a CPU loop doing the same
per-evaluation work as the current CUDA kernel (precomputed x/z terms, early
exit).

---

## 0. Scope and where the code lives

- The analysis was done against the `correctness-overhaul` branch: that is
  where the optimised kernel, the single-source `include/texture.cuh`, and the
  ledger of already-tried ideas live. The maths is identical on `main`.
- This file and the prototypes were written into the working tree on `main`
  (untracked, not committed).
- Prototypes live in `experiments/rng-reduction/`. On `rewrite`, pass
  `-I../../src` when building from that directory to use the shared
  `src/texture/texture_rules.hpp` implementation. The commands below
  reproduce the original setup with the headers from `correctness-overhaul`.

  ```bash
  cd experiments/rng-reduction
  git show correctness-overhaul:include/texture.cuh > texture.cuh
  git show correctness-overhaul:test/fixtures/formation_a.txt > formation_a.txt
  clang++ -O2 -std=c++17 sweep.cpp   -o sweep   && ./sweep
  clang++ -O2 -std=c++17 lattice.cpp -o lattice && ./lattice
  clang++ -O2 -std=c++17 flat.cpp    -o flat    && ./flat
  clang++ -O2 -std=c++17 lowbits.cpp -o lowbits && ./lowbits
  ```

  (`*.txt` is gitignored, so the fixture copy will not be picked up.)

## 1. The generator, as the hot path sees it

```
u    = sext32(x * 3129871)  ^  (z * 116129781)  ^  y        // 64-bit
mix  = 42317861 * u*u + 11 * u                  (mod 2^64)   // "position mix"
seed = mix >> 16
```

followed by a per-version output stage. For modern:

```
v      = ((seed ^ 0x5DEECE66D) * 0x5DEECE66D + 11) mod 2^48
result = top two bits of v                                   // 0..3; a side face reads result & 1
```

Current kernel cost (from the overhaul's log): ~1.33 evaluations per candidate
because of early exit (a top-face check passes 1 time in 4), each evaluation
being three 64-bit multiplies (two for the mix, one for the output stage),
about 40–44 issue slots. The x and z multiplies are already hoisted.

## 2. R1 — y-column forward-difference sweep

### The pattern

- For a fixed column (x, z), `u = K ^ y` with `K = sext32(x*3129871) ^ z*116129781`.
- y is small, so it only touches the low bits of K. Over an *aligned* block of
  2^k y-values, `K ^ y` takes every value `base, base+1, …, base+2^k−1`
  exactly once, in a permuted order: `u = base + t` with `t = (K & m) ^ j`,
  `m = 2^k − 1`, `j` = offset of y inside the block.
- `mix` is a quadratic polynomial in `u`, so walking t = 0, 1, 2, … needs only
  second-order forward differences:

  ```
  mix(0) = P(base)                       // 2 multiplies, once per segment
  d(0)   = 42317861 * (2*base + 1) + 11  // 1 multiply, once per segment
  mix(t+1) = mix(t) + d(t)
  d(t+1)   = d(t) + 2 * 42317861
  ```

- Negative y is the same case: `K ^ sext(y) == ~K ^ ~y`, and `~y ≥ 0`.
- An arbitrary y range is split into at most ~2·log2(n) aligned
  power-of-two segments.
- The output stage does **not** flatten: `(seed ^ M) * M` is not linear across
  the XOR, and consecutive seeds differ by a full-width value. One multiply
  per position remains.

### Multiply count per candidate

| | block 0 | later blocks | total |
|---|---|---|---|
| current kernel | 3 | 0.33 × 3 | ~4 |
| R1 (sweep block 0, normal path for survivors) | 1 + two adds | 0.33 × 3 | ~2 |

### Measured (`sweep.cpp`)

| Search | Version | Result | Speedup vs ref |
|---|---|---|---|
| y −64…−40 (25 levels) | modern | identical, finds the fixture's match | 1.20x |
| y −64…−40 (25 levels) | legacy | identical | 1.36x |
| y −64…319 (384 levels) | modern | identical, finds the fixture's match | 1.50x |
| single y level | modern | identical | **0.77x (slower)** |

- Equivalence: 2000 randomised trials (random formations with side faces,
  sign-straddling and out-of-world y ranges, world-border x/z, all five
  version modes) — all identical to the reference.
- Single-level searches regress because the per-segment setup (three
  multiplies) is paid for one position.
- On CPU the survivor path's branch mispredictions hide much of the gain.

## 3. R2 — store the hash field as bitplanes

### The idea

The hash value of a world position does not depend on which candidate asks
for it. Block *i* of candidate *c* reads position `c + d_i`; that is the same
position block *j* reads for candidate `c + d_i − d_j`. So instead of each
candidate evaluating its own blocks:

1. Hash each world position once and keep the 2-bit result in two bit arrays
   (bit 0 plane, bit 1 plane).
2. A block check for 64 candidates is then two loads, a shift, and a few
   logical ops: `M &= eq(plane0 >> shift, r&1) & eq(plane1 >> shift, r>>1)`.
   Side faces use plane 0 only.
3. Stop as soon as `M == 0`. With 64 candidates per word that takes 3–5 block
   checks, i.e. well under one operation per candidate.

Two layouts, depending on which axis the 64-bit word runs along:

- **y-axis words** (`lattice.cpp`): one pair of words per world column,
  filled with the R1 sweep. Best for tall searches. Candidate y range is
  processed in slabs of `64 − (formation height)` levels.
- **x-axis words** (`flat.cpp`): one pair of bit arrays per world row
  (z, y), filled with the ordinary full evaluation. Works for any number of
  levels, including one.

### Why the overhaul's rejection of "shared-memory tiling" does not apply

`docs/EXPERIMENTS.md` rejected tiling because early exit already brings the
naive scan to ~1.33 evaluations per position, and precomputing every position
cost more than that. Two things change the arithmetic:

- matching is bit-parallel, so the per-candidate matching cost is ~0, not one
  lookup per block;
- with R1 a precomputed position costs one multiply, not three — and with R3
  below most positions are never computed at all.

### Measured, R2 alone (stride s = 1)

| Search | Layout | Field evals / candidate | Speedup vs ref |
|---|---|---|---|
| W slice, 25 levels | y-axis | 1.085 | 3.1x |
| full height, 384 levels | y-axis | 1.051 | 4.2x |
| single level | y-axis | 3.0 | 0.4x (slower — wrong layout for this case) |
| single level | x-axis | 2.0 | 2.8x |

Field evals exceed 1.0 because the fixture has blocks at two heights (dy 0
and 2), plus a small halo at the box edges.

## 4. R3 — only hash a sparse sublattice

### The idea

A candidate is rejected as soon as *any* subset of its blocks mismatches, and
a 24-block formation has far more blocks than are needed to reject (three
top-face checks already reject 63 of 64 candidates). So:

- Pick a sublattice Λ of index s in the (x, z) plane, e.g.
  `{(px, pz) : a·px + b·pz ≡ 0 (mod s)}`.
- Hash **only** world columns on Λ — 1/s of the field.
- For a candidate in coset r, use only the blocks that land on Λ (bit-parallel
  lookups as in R2). Each coset gets its own block list, built once on the
  host.
- The few candidates that survive are verified against the remaining blocks
  with ordinary evaluations.

The host picks (s, a, b) by minimising a cost model:
`sweepCost / s  +  (1/s) · Σ_cosets P(survive the on-lattice blocks) · 1.33 · fullCost`.
Larger s hashes less but leaves fewer blocks per coset, so more survivors.

For the x-axis layout the lattice is just "every s-th z row"
(`pz ≡ 0 mod s`), which keeps the bit shifts along x trivial.

### Measured, R2 + R3

y-axis layout (`lattice.cpp`), fixture formation:

| Search | Version | s | Field evals / cand | Full evals / cand | Speedup | Throughput |
|---|---|---|---|---|---|---|
| W slice, 25 levels | modern | 7 | 0.155 | 0.014 | 7.2x | 1.50 G cand/s |
| W slice, 25 levels | legacy | 7 | 0.155 | 0.014 | 7.1x | 1.43 G cand/s |
| W slice, 25 levels | Sodium 1.19 | 7 | 0.155 | 0.014 | 8.0x | 1.25 G cand/s |
| full height, 384 levels | modern | 7 | 0.150 | 0.014 | 11.9x | 2.50 G cand/s |

Stride sweep on the W slice (modern, best lattice for each s):

| s | Field evals / cand | Full evals / cand | Speedup |
|---|---|---|---|
| 1 | 1.085 | 0 | 3.1x |
| 2 | 0.543 | 0.0000 | 4.8x |
| 3 | 0.362 | 0.0000 | 6.1x |
| 4 | 0.271 | 0.0003 | 6.5x |
| 5 | 0.217 | 0.0021 | 7.0x |
| 6 | 0.181 | 0.0072 | 7.3x |
| 8 | 0.136 | 0.0267 | 7.0x |
| 12 | 0.090 | 0.0990 | 5.0x |

The curve is flat from s = 5 to 8: by then the per-column matching loop, not
the hashing, dominates on CPU.

x-axis layout (`flat.cpp`), fixture formation, modern unless noted:

| Search | s (z stride) | Field evals / cand | Full evals / cand | Speedup | Throughput |
|---|---|---|---|---|---|
| single level | 1 | 2.002 | 0 | 2.8x | 0.55 G cand/s |
| single level | 2 | 1.001 | 0 | 5.0x | 0.99 G cand/s |
| single level | 3 | 0.667 | 0 | 6.8x | 1.35 G cand/s |
| single level | 6 | 0.334 | 0.019 | 9.2x | 1.82 G cand/s |
| single level, legacy | 6 | 0.334 | 0.019 | 9.2x | 1.75 G cand/s |
| 5 levels | 6 | 0.334 | 0.019 | 8.8x | 1.85 G cand/s |

Equivalence: 3000 randomised trials for each layout (random formations of
1–7 blocks with side faces, random boxes including world-border coordinates,
random lattices/strides 1–9, all five version modes) — all identical to the
reference. Every timed run above also reported identical match sets and found
the fixture's known match where the box contains it.

### Limits

- The achievable s depends on the formation: every coset needs roughly three
  or more top-face blocks landing on the lattice. A 24-block formation
  supports s ≈ 6–8; a 6-block one supports s ≈ 2. Side faces count half.
- Weak formations with many true matches spend their time in survivor
  verification regardless.
- Speedups above are CPU numbers and include the removal of branch
  mispredictions, which a GPU does not suffer from in the same way. See §6.

## 5. Low-bit structure per version

Verified on 20M random inputs with zero mismatches (`lowbits.cpp`).

| Version | Finding | Consequence |
|---|---|---|
| modern (1.21.2+) | result is the top two bits of the 48-bit state; changing only the top two input bits changes the result 75% of the time | no periodicity, no cheap pre-filter — every input bit matters |
| legacy (1.13–1.21.1) | parity depends only on `u mod 2^33`; "is it 0 or 2" only on `u mod 2^34` | a 32-bit-multiply pre-filter is possible; the full path is needed only for the sign when the target is 1 or 3 |
| ≤ 1.12 | the whole value depends only on `u mod 2^32`; parity on `u mod 2^17`; even values on `u mod 2^18` | pure 32-bit arithmetic; side-face parity is periodic with period 131072 in x and z |

These are independent of R1–R3 and would apply to the output stage of the
field fill.

## 6. What is not known yet (GPU)

- Nothing here has run on a GPU. The remote 3090 box was not used.
- Expected direction: on the GPU a 64-bit multiply is several instructions,
  so R1's multiply saving should be worth more than on CPU, while R2's
  branch-elimination benefit should be worth less.
- Rough operation count for W with R1+R2+R3 at s = 7: ~0.16 one-multiply
  evaluations plus ~0.014 full evaluations per candidate, against ~1.33 full
  evaluations today. That is an estimate from counts, not a measurement.
- Open design questions for a kernel:
  - where the bitplanes live (per-thread-block tile in shared memory with a
    formation-sized halo, vs. a global buffer filled by one kernel and
    matched by a second);
  - halo overhead for wide formations at small tile sizes;
  - how survivors are verified without reintroducing warp divergence
    (compact them into a list and run the existing per-candidate path);
  - choosing between y-axis and x-axis words from the search height.

## 7. Checked and ruled out

| Idea | Outcome |
|---|---|
| Flatten the output multiply along y | no: `(seed ^ M) * M` is not linear across the XOR and consecutive seeds differ by a full-width value |
| Forward differences along x or z | no: x and z enter through XOR of two products, so there is no additive structure; only y is "small" |
| Splitting `u` into 32-bit halves along x | works algebraically but costs about as many 32-bit multiplies as the plain 64-bit form |
| Periodicity / sieve for modern | no: result depends on all input bits (§5) |
| Inverting the hash instead of scanning | already a logged negative result on the overhaul branch (SMT experiment); not revisited |

## 8. Suggested order if this is taken further

1. x-axis bitplanes + z-stride lattice (R2 + R3, `flat.cpp`): simplest, works
   for every search height, no dependence on R1.
2. y-axis bitplanes + R1 sweep for tall searches (`lattice.cpp`).
3. Legacy / ≤1.12 narrow-arithmetic output stages (§5).

Each step can be validated against the existing reference scan the same way
the prototypes are: identical match sets on randomised boxes and formations,
then the fixture search.

## 9. Proposed A100 x-axis bitplane experiment (2026-11-12)

The [Narval baseline](BENCHMARKS.md#narval-a100-baseline-2026-11-12) has a
modern median of 0.681581 s. The estimates below use the fixture dimensions
and the loops in `flat.cpp`. They assume independent top-face matches with
probability 1/4. The two GPU stages still need timing.

| Design for W | Fill hashes per candidate | Survivor hashes per candidate | Total hashes |
|---|---:|---:|---:|
| Current early-exit kernel | 0 | 1.333 | 167 billion |
| Stride 6, rebuild each candidate y | 0.333 | 0.019 | 44.0 billion |
| Stride 6, index planes by world y | 0.180 | 0.019 | 24.9 billion |

The fixture assigns 2, 3, 4, 5, 5, and 5 top-face blocks to the six
cosets. The model predicts about 3.35 checks per full candidate word and
about 1.416% of candidates surviving the lattice checks. World-y indexing
reuses hashes across candidate levels, reducing 50 row fills to 27 for
each selected world z. A bulk GPU buffer holding every selected row would
write about 10.4 GB with per-candidate-y fills or 5.6 GB with world-y reuse.

The model gives 104.8 GB of logical two-plane read payload across W. With
the fixture's actual aligned and unaligned offsets, the word-load operands
total about 183.6 GB before cache reuse and memory-transaction effects. The
Narval A100 has 40 MB of L2. Reusing nearby rows is plausible, and the
reuse depends on how GPU blocks are scheduled. The predicted 0.2–0.4 s
end-to-end range assumes a fill rate derived from the existing kernel and
effective reuse in the match pass.
[NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#memory-system)

### First experiment

Use modern mode and fixed stride 6 on a slab of 4,096 candidate z rows,
full reference x width, and all 25 candidate y levels. That is about
10.24 billion candidates and a bulk plane buffer of about 856 MB with
per-candidate-y fills or 462 MB with world-y reuse.

1. Time the fill and match-count kernels separately with CUDA events.
   Start with one thread per word pair for fill and one thread per
   candidate word for matching. Reduce the survivor counts per block so a
   contended global counter does not dominate the match timing.
2. Check candidate masks against a direct CPU filter on a small case
   covering all six cosets, unaligned extracts, and the partial final
   word. Compare exact survivor counts per coset on the full slab against
   the CPU prototype. The model predicts about 145 million survivors on
   that slab. The exact CPU counts are the acceptance criterion.
3. Time the existing CUDA kernel on the identical slab and GPU. Record
   warm runs and the individual samples. Fable's initial targets are about
   14 ms for fill, about 7.5 ms with world-y reuse, and under 8 ms for
   matching. A miss measures a stage-level slowdown. Attributing it to
   DRAM requires more evidence, such as a cache-sized versus larger
   working-set comparison with comparable candidate work.

These stage timings can guide the port. A production comparison includes
survivor verification, result handling, and the full correctness suite.

### Running the prototype

Build `build/flat_cuda` with `make BACKEND=cuda ARCH=sm_80 build/flat_cuda`
on the A100. Pass the fixture path, then inclusive bounds in the order
`x_min x_max y_min y_max z_min z_max`. The optional arguments set the CPU
thread count and the maximum candidate z rows per GPU slab, in that order.
The default slab size is 4,096 rows. Use the bounds from `test/bench.sh`
for the full reference workload. For a one-slab comparison, set the upper
z bound to the lower z bound plus 4,095.

The executable checks individual GPU masks on a synthetic case and compares
exact survivor counts per coset against direct CPU evaluation on the slab.
It alternates the timing order of the prototype and the existing CUDA
backend after warming both. The baseline calls `runSearch` once on the slab.
The prototype reuses one GPU plane allocation for all slabs and fills the
halo needed for block lookups at each slab boundary. It verifies the
surviving candidates in the same kernel that filters them. It checks exact
final match sets and totals against the baseline and reports the sum of
CUDA event timings over all slabs and wall time per search.

In the filter-only Narval job `9195540`, the mask and survivor checks passed. The medians
were 8.868 ms for fill, 5.523 ms for match counting, and 56.051 ms for
the existing CUDA search on the same slab. See the
[experiment ledger](EXPERIMENTS.md#8-cuda-x-axis-bitplane-slab-experiment-2026-11-12)
for the complete samples and the scope of the timing comparison.

The complete prototype in Narval job `9196459` passed the known-match
and bounded-output checks and agreed with the existing search on the slab.
Median GPU time was 18.476 ms against 56.063 ms for the existing search.
[Complete slab measurements](EXPERIMENTS.md#complete-slab-search-job-9196459)

Narval job `9196908` reproduced the gain over the full reference workload.
The prototype returned the same single match as the existing search and
used 10.43 GB for the planes. Median GPU time was 224.970 ms against
681.448 ms for the existing search.
[Full reference measurements](EXPERIMENTS.md#full-reference-workload-job-9196908)

With 4,096-row slabs, job `9197641` reused an 856 MB plane buffer and
passed the full reference checks. Median search wall time was 218.752 ms
against 683.559 ms for the existing search on the same A100.
[Tiled reference measurements](EXPERIMENTS.md#full-reference-workload-with-a-reusable-plane-buffer-job-9197641)

## 10. Production integration review

Run the match-kernel diagnostics and measure compact world-y indexing on
the standalone prototype before choosing a production layout. The
[diagnostic guide](GUIDE_MATCH_DIAGNOSTIC_AND_WORLD_Y.md) gives the
measurement and correctness checks.

An opt-in modern path can enter inside `runSearch` after the formation
checks and before the existing CUDA upload. For a first integration,
evaluate the fixed stride of six on the rotated formation and use the
existing kernel for formations with empty cosets or insufficient expected
filtering. Estimate plane bytes and grid dimensions for each call using
checked arithmetic. Shrink z slabs to fit the budget and fall back if
one row with its halo exceeds it. This provides a bounded first release
with coverage for more shapes available through later y and x tiling.

Keep one result buffer and 64-bit count across internal slabs, sort at
the end of each call, and report the sum of fill and search kernel times.
Validate the existing end-to-end suite with the opt-in on and off. Add
cases that prove the bitplane path ran and that a deliberately small
budget selects the existing kernel. Include a checkpoint resumed with
the opposite setting. For truncated output, compare the true count and
check that the stored matches are valid and distinct.

### Running the match diagnostics

`build/flat_cuda` uses compact world-y planes by default. Set
`TF_FLAT_DUPLICATE_Y=1` to reproduce the original plane layout.
`TF_FLAT_DIAGNOSTICS=all` runs all five match variants after the
correctness checks. `TF_FLAT_DIAGNOSTICS=selected` runs the filter and
the synthetic-load variant. Each diagnostic prints nine CUDA event
samples with rotated run order.

The no-tail variant checks its survivor masks against the CPU on
synthetic cases. Its timed run skips the block reduction and count
output. The synthetic-load variant reports its survivor total so the
run can be checked for comparable selectivity. It replaces plane reads
with arithmetic, so its timing alone cannot establish DRAM throughput.

Narval job `9228736` builds the comparison on one A100. It runs all five
variants on the original 4,096-row slab, then compares the filter and
synthetic-load variants across both layouts and the full workload.
