# Guide: match-kernel diagnostic, then world-y plane indexing

Written 2026-11-12 for the assistant doing the next two pieces of work on the
CUDA bitplane prototype. It assumes you have read
`docs/HANDOFF_SERVER_AND_LEAN.md` (ground rules, repo map) and
`docs/EXPERIMENTS.md` §8 (the results so far).

All work happens in **one file**: `experiments/rng-reduction/flat_cuda.cu`
(built by `make build/flat_cuda`). Do not touch `src/`, `include/`, or the
production CLI in this task.

Line numbers below refer to the tiled revision of `flat_cuda.cu` measured in
job `9197641`. If the file has moved on, find the same code by name.

---

## 0. Where things stand

Reference workload W, one A100, modern mode, stride 6 (job `9197641`, medians):

| Stage | Time | Source |
|---|---|---|
| Existing kernel (`runSearch`) | 681.3 ms | measured |
| Prototype fill | 108.1 ms | measured |
| Prototype filter + verify | 108.9 ms | measured |
| — of which filter alone | ~67 ms | **estimate**: 5.52 ms per 4,096-row slab from the earlier count-only revision, × 12.2 slabs |
| — of which survivor verification | ~41 ms | **estimate**: the difference |

Two questions this task answers:

- **Part A (diagnostic):** why does the filter cost ~67 ms when it executes
  only a handful of bitwise operations per 64 candidates? Memory traffic, or
  per-thread overhead?
- **Part B (optimisation):** remove the duplicated hashing in the fill by
  indexing planes by world y. Predicted fill time: about 58 ms.

Do Part A first on the unmodified layout, then Part B, then repeat the two key
Part A measurements on the new layout.

## 1. Part A — match-kernel diagnostic

### 1.1 What to build

Add a compile-time or run-time **mode switch** to `matchWords`
(`flat_cuda.cu:207`) and time each mode on the same slab. Modes are
measurement-only; the default mode must stay bit-identical to today's
behaviour and keep passing every existing check.

| Mode | Change relative to the default | Question it answers |
|---|---|---|
| `full` | none (filter + inline verification) | reference |
| `filter` | skip the survivor loop (`:250`–`:274`); still count survivors | how much is verification? (replaces the estimate above with a measurement on the current code) |
| `filter-nomem` | as `filter`, but replace every `planes[...]` read with a pseudo-random 64-bit word computed from `(index, i)` | is the filter memory-bound? |
| `filter-div32` | as `filter`, but do the per-thread row/word/z/y decomposition (`:213`–`:217`) in 32-bit arithmetic | are the 64-bit divisions the cost? |
| `filter-notail` | as `filter`, but skip the per-block survivor tally (`:276`–`:288`, the `warpSum`/`__syncthreads`/`blockCounts` section) | is the diagnostic tail the cost? |

Notes on each:

- **`filter-nomem` must keep the loop statistics.** Do not substitute a
  constant: an all-ones or all-zeros word changes how many checks run before
  the mask empties. Each top-face check should clear about 3/4 of the
  remaining bits, as real data does. Any cheap mixing of the plane index and
  the check number is fine (multiply by an odd 64-bit constant, xor-shift);
  use two different words for the low and high planes. The survivor *count*
  will differ from the real one — that is expected; report it and confirm it
  is within a few percent of the real total so the workload is comparable.
- **`filter-div32` suspect.** Each thread currently computes
  `blockIdx.x / segments`, `row % layout.nz`, `row / layout.nz` with `size_t`
  operands. 64-bit division is a long software sequence on the GPU, and each
  thread otherwise does very little work. All of these quantities fit in 32
  bits for any grid CUDA accepts (the host already checks the grid size at
  `:437`). Results must be identical to `filter`.
- **`filter-notail`** changes nothing about the masks; it only removes the
  per-block count output, so the per-coset survivor check cannot run in this
  mode. Compare its match total against `filter` instead.

Optional, only if `filter-nomem` shows the filter is memory-bound:

| Mode | Change | Question |
|---|---|---|
| `filter-aligned` | read only `planes[index]`, never `planes[index + 1]` (wrong answers; timing only) | cost of the second, straddling load |
| `filter-soa` | store low and high planes in two separate arrays instead of the 16-byte `Planes` struct | does the array-of-structs layout hurt? |

### 1.2 How to measure

- Slab: the first 4,096 candidate z rows of W, full x width, all 25 levels —
  the same slab as jobs `9195540` and `9196459`. Then the two most informative
  modes on full W.
- Protocol from `docs/BENCHMARKS.md` ("Comparing future CUDA changes"): one
  warmup, then nine samples per mode, rotating the mode order between samples.
  Report median and min–max. Use CUDA event times around the match launch
  only; the fill is unchanged in Part A but record it anyway.
- Correctness first: before any timing, the default `full` mode must pass the
  built-in small synthetic checks and reproduce the recorded survivor counts
  (§3 below).

### 1.3 How to read the result

Let F = median time of `filter`.

| Observation | Conclusion | Next step (not in this task) |
|---|---|---|
| `filter-nomem` ≤ ~0.5 F | filter is memory-bound | layout work: `filter-soa`, smaller tiles, shared-memory tiles |
| `filter-div32` ≤ ~0.7 F | per-thread index arithmetic dominates | keep the 32-bit decomposition; consider more words per thread |
| `filter-notail` ≤ ~0.8 F | the diagnostic tail is distorting timings | production path drops it anyway; re-baseline without it |
| none of the above | launch/occupancy overhead or something unmodelled | try several words per thread (fewer, heavier threads) |
| `full` − `filter` | measured verification cost | feeds the later survivor-compaction work |

More than one row can be true; report all five numbers.

## 2. Part B — world-y plane indexing

### 2.1 What is wasteful today

Planes are indexed by (candidate y, dy slot):

- host: `fieldWordsPerZ_ = ny × ySlots × fieldWords` (`:433`);
- fill: `y = y_min + level / ySlots + yOffsets[level % ySlots]` (`:181`–`:182`);
- match: `fieldRow = (y × ySlots + check.ySlot) × hashedRows + zIndex` (`:228`).

For the fixture (`dy ∈ {0, 2}`, 25 candidate levels) that is 50 plane levels
per hashed row, but only 27 distinct world levels exist: candidate y's
`dy = 2` level is candidate (y+2)'s `dy = 0` level. 23 of 50 levels are hashed
twice.

### 2.2 Target layout

Index plane levels by **distinct world y**, compacted so the layout is never
worse than today's:

1. For each distinct dy value `d_k` the formation uses, the world levels it
   needs are the interval `[y_min + d_k, y_max + d_k]`.
2. Take the union of those intervals and number its members consecutively
   from 0 in ascending world y. Call the count `levels`. Because each interval
   has length `ny`, `levels ≤ ny × ySlots` always, and
   `levels = ny + (max dy − min dy)` when the intervals overlap into one run.
3. Host uploads `levelY[levels]` (the world y of each compact level). This
   replaces the `yOffsets` array.
4. Each on-lattice `Check` stores `levelBase` = the compact index of world
   level `y_min + block.y`, replacing `ySlot`. A candidate at y-index `j`
   (`0 ≤ j < ny`) reads compact level `levelBase + j`. This is valid because
   every interval is contiguous inside the union, so its members get
   consecutive compact indices.

Resulting changes, and nothing else:

| Place | Today | After |
|---|---|---|
| `Layout` | `ySlots` | `levels` |
| plane words per hashed z row | `ny × ySlots × fieldWords` | `levels × fieldWords` |
| fill: level → world y | arithmetic on `ySlots` and `yOffsets` | `levelY[level]` |
| match: plane row | `(y × ySlots + ySlot) × hashedRows + zIndex` | `(levelBase + y) × hashedRows + zIndex` |
| `Check` | `ySlot` | `levelBase` |

Keep the row ordering "level outermost, hashed z inside, x words innermost"
so the slab/halo logic in `DeviceExperiment::run` (`:464`–`:503`) is
untouched apart from the size term.

The survivor-verification path (`VerifyCheck`, `:250`–`:274`) hashes directly
and does not read planes; leave it alone.

### 2.3 Predictions to check

| Quantity | Today | Predicted after |
|---|---|---|
| Plane levels per hashed row (fixture, ny = 25) | 50 | 27 |
| Plane allocation, 4,096-row slab | 855,820,800 bytes | 462,143,232 bytes |
| Fill evaluations per candidate | 0.333 | 0.180 |
| Fill time, full W | 108.1 ms | ≈ 58 ms (108.1 × 27/50) |
| Filter + verify time | 108.9 ms | unknown; possibly slightly lower from the smaller working set |
| Survivor counts, final matches | — | **exactly unchanged** |

If the fill time does not scale close to 27/50, say so — it would mean the
fill is not purely evaluation-bound.

## 3. Correctness gates (both parts)

Run before recording any timing, and again after Part B:

1. The built-in `verifySmall()` checks print both `PASS` lines.
2. Slab survivor counts by coset equal the recorded values:
   106,710,019 / 6,666,155 / 1,668,734 / 1,668,151 / 1,665,536 / 26,637,239
   (total 145,015,834).
3. Full-W survivor counts by coset equal:
   1,302,225,705 / 81,392,044 / 20,347,922 / 20,345,216 / 20,341,899 /
   325,516,735 (total 1,770,169,521).
4. Full W returns the same single final match as `runSearch`, every sample.

Add these synthetic cases for Part B (extend `verifySmall`, same style: derive
rotations from a known origin, compare masks against `referenceCounts` and
final matches against `runSearch`):

| Case | Why |
|---|---|
| dy values far apart relative to ny (e.g. `{0, 50}` with 3 candidate levels) | the union is two disjoint intervals; `levels` must be 6, not 53 |
| dy values that make intervals exactly adjacent (gap of 0) and exactly overlapping by one level | off-by-one in interval merging |
| a single candidate level (`ny = 1`) with three distinct dy | `levels` equals the number of distinct dy |
| negative `y_min` with dy values of both signs | sign handling in `levelY` |
| all blocks at one dy | `levels == ny`; layout degenerates to today's with one slot |
| slab size 1 with any of the above | the halo/slab loop still indexes correctly |

## 4. What to report

Append to `docs/EXPERIMENTS.md` §8 (append only; never edit earlier rows),
with job id, node, driver, CUDA version, and source SHA-256 as the earlier
entries do:

- Part A: one table, modes × {slab median, slab min–max, full-W median where
  run}, plus the survivor total seen in `filter-nomem`.
- Part A: one sentence per row of §1.3 stating whether it held.
- Part B: predicted vs measured for every row of §2.3, and the paired speedup
  over `runSearch` on full W (same nine-sample protocol).
- Part A's `filter` and `filter-nomem` re-measured on the Part B layout.
- Anything that failed, was skipped, or could not be measured, stated as such.

Do not update `docs/BENCHMARKS.md` unless the full correctness gates passed on
the exact binary that produced the number.

## 5. Out of scope

Listed so they are not folded in by accident:

- survivor compaction / a separate verification pass;
- choosing the stride per formation (it stays 6 here);
- the y-sweep fill;
- integrating any of this into `src/kernel.cu` or the CLI;
- legacy or other version modes.
