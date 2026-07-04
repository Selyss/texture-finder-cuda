# Engineering log

Running log of the correctness/verification/optimization overhaul (started
2026-07-03). Newest entries at the bottom. Benchmark workload throughout:
the formation_a search volume — x [-175000, -75000], y [-64, -40],
z [-75000, -25000], ~1.25e11 positions, 24 top blocks, RTX 3090, CUDA 12.8,
`-O3 -arch=native`. "Kernel time" is CUDA-event time around the kernel;
"wall" includes ~0.2 s of CUDA context startup.

## Baseline (original code, unmodified)

Wall time: **4.63 s** mean of 3 (4.579 / 4.605 / 4.695). Build: sm_86, zero
warnings. Finds the fixture match. No kernel-only timing available (none was
instrumented).

Empirically confirmed correctness bugs in the baseline, all silent:

| Bug | Evidence |
|---|---|
| x/z max bound never searched at certain widths | grid used `(max-min+7)/8` blocks for an inclusive range: exact-multiple widths lose the last column. The fixture's real-world search bounds (widths 100001/50001) both hit it: x=-75000 and z=-25000 were never searched. |
| Narrow x/z ranges search nothing at all | single-cell probe at the known match returned no match: `(0+7)/8 = 0` blocks in that dim → invalid launch config → error silently discarded (no CUDA error checks) → "Search completed" as if clean. This was the old `FIXME: entering the same y coord... does not work` — the y formula `(diff+2)/2` was a partial fix of the same class of bug. |
| Side blocks + direction ≠ 0 can never match | parser reduces side rotations to {0,1} but `rotateFormation` increments all rotations mod 4 → sides reach 2-3, compared against mod-2 values → 0 matches. The likely source of the "side rotations don't work" reports. |
| Tests validated a *different* function than shipped | `test/test.cpp` had its own copy of `getTextureModern` using `(mod*next)>>31`; the device used `((4*next)>>31)%mod`. The device was right (see semantics below); the tests only exercised mod 4 where both coincide. |
| In-kernel printf as the only output channel | matches beyond the printf buffer would vanish; no host-side result capture. |
| UB wrapping arithmetic | `x * 3129871` (int32 overflow), `l*l*42317861` (int64 overflow), `abs(INT_MIN)` — all UB in C++, all happened to compile to Java-compatible wrapping on current nvcc; `staffordMix13(long)` would be 32-bit under MSVC (unused, removed). |
| `cudaFree` on `__constant__` symbols | invalid call, error ignored. |
| No validation anywhere | unreadable file → "0 blocks read" → every position matches; rotation=9 accepted → guaranteed 0 matches; version=7 silently treated as legacy; >1024 blocks would overflow constant memory. |

## Semantics decisions (from the Java reference, 19MisterX98/TextureRotations)

- The reference's modern generator is *literally* `((4 * next) >> 31) % mod`
  with a comment that the game always rolls 4 variants. So sides =
  parity of the 4-variant roll, same seed, y included. The "obvious fix"
  (`(mod*next)>>31`, i.e. nextInt(2) for sides) would have been wrong — the
  game never draws a 2-bound value. Full derivation in docs/SEMANTICS.md.
- Legacy `>>> 16` vs C++ `>> 16`: equivalent after the (int) cast (bits
  16-47 survive either way). Verified by oracle diff, not just argued.
- The reference has no direction feature; rotation-under-yaw is this repo's
  own semantics. Chosen convention: quarter turn = (x,z)→(-z,x), tops +1
  mod 4, sides flip parity. Since a user can still guess facing wrong, added
  `direction all` which tries all four and reports which matched — a wrong
  guess can no longer lose a find.

## Verification infrastructure built

- **Java oracle on the GPU box** (`/root/oracle/`): the reference's own
  texture classes compiled verbatim + a dump driver; regenerates ~29M ground
  truth values (grids near origin, world-border extremes, 1M random coords);
  regen gated on the fixture check (24/24 at the verified origin).
  Cross-checked: byte-identical dumps from JDK 17/Linux/x86_64 and JDK
  25/macOS/ARM.
- **`build/oracle_diff`**: compares all dumps against the C++ header
  implementations. Result: **29,178,112 values, 0 mismatches.**
- **Single-source RNG**: `include/texture.cuh` is host+device; kernel,
  tests, and tools compile the same functions. The old copy-divergence
  class of bug is now structurally impossible.
- **`build/cpu_search`**: independent CPU implementation of the whole
  pipeline (same CLI), used by e2e for GPU-vs-CPU differential tests,
  including dense cases (1-block formation → ~25% of all positions match)
  that prove exhaustive position coverage of the kernel.
- **`test/e2e.sh`** on the GPU: fixture (plus single-cell and box-corner
  regressions), synthetic round-trips (both versions × 4 directions ×
  single/all mode, ~40% sides), differentials, validation rejection tests.

## Performance history (kernel time on the reference workload, modern)

| Step | Kernel | Wall | Notes |
|---|---|---|---|
| Baseline (original) | — | 4.63 s | dim3(8,2,8), printf output, per-thread branch on version |
| Rewrite: grid-stride 3D, 256-thread blocks, result buffer, template dispatch | 1.26 s | 1.47 s | **3.15× vs baseline.** Correctness-driven rewrite; speed was a side effect. Geometry sweep (block 128/256/512, grid caps) all within ±1% → kernel is ALU-bound, not geometry-bound. |
| Warp-compacted kernel (persistent warps, per-lane candidate refill) | 1.17-1.20 s | ~1.40 s | Best config: chunk 1024/lane → 1.168 s (~7%). Far below the ~2× predicted from the divergence model (1.33 avg evals/position vs ~4.3 warp-paced): Ampere predication/scheduling already hides much of it, and two new costs appeared: divergent-address `__constant__` reads serialize (lanes now sit at different block indices), and the hash's serial dependency chain leaves the kernel latency-bound per lane. |

Compaction is kept: it's proven correct by the differentials, removes the
old geometry failure modes entirely (any position enumerated exactly once by
construction), and its 7% is real. Chunk default set to 1024/lane.

Next experiments queued: formation in dynamic shared memory (kills the
constant-serialization cost; sized to the formation so occupancy is
unaffected), register/occupancy audit via `-Xptxas -v`, and if
latency-bound: 2 interleaved candidates per lane for ILP.

## Optimization round 2 (2026-07-03, later)

| Experiment | Kernel (modern) | Verdict |
|---|---|---|
| Formation in dynamic shared memory instead of `__constant__` | 1.168 → 1.158 s | ~1%. Constant-serialization was NOT the story. Kept anyway (strictly better, costs nothing: dynamic size → no occupancy hit). |
| Register audit (`-Xptxas -v`) | 32 regs, full occupancy | Not latency/occupancy-starved. `ncu` profiling unavailable on the box (container blocks GPU perf counters, `ERR_NVGPUCTRPERM`), so bound analysis stayed model-based: ~44 issue slots/eval measured vs ~40 modeled → issue-rate bound. |
| Precompute per-block hash terms + ballot-free stripe loop | 1.158 → **0.855 s** | **~26%.** Two changes landed together: (a) the hash's x/z multiplies distribute over block offsets under wrapping, so `bx*3129871` and `bz*116129781` are precomputed per block on the host and the per-eval multiplies become adds (texture.cuh refactored into a two-stage API — coordRandomFromParts + {legacy,modern}FromCoordRandom — so kernel, tests, and tools still share one implementation); (b) the per-iteration `__all_sync` ballot/refill bookkeeping replaced by an outer uniform chunk-grab loop + inner per-lane stripe loop with zero warp coordination. |
| Chunk size re-sweep | 1024/lane: **0.823 s** vs 256: 0.856, 64: 0.944 | Default set to `TF_CHUNK_PER_LANE=1024`. Block size 128/256/512 within noise; 256 kept. |

**Final: kernel 0.826 s, wall 1.04 s on the reference workload — 4.45×
end-to-end vs the 4.63 s baseline** (legacy version essentially identical:
0.822 s). Estimated remaining headroom at this algorithm: ≤10% (the kernel
issues ~44 slots per evaluation vs ~35 of irreducible hash math + compare;
further cuts mean hand-scheduling PTX or truncated 48-bit multiplies —
complexity not worth it). Structural alternatives (shared-memory tiling of
texture values) were analyzed and rejected: early-exit makes the naive
evaluation count (~1.33/position) far cheaper than any precompute-everything
scheme (which needs ~18 evals/position equivalent for this formation shape).

All rounds re-validated: full e2e (incl. GPU-vs-CPU dense differentials) and
the 29M-value oracle diff pass on the final configuration.

## Adversarial review pass (2026-07-03, close-out)

An independent fresh-eyes review of the final tree (which fuzzed a host
replica of the kernel's enumeration over 415 shape configurations — clean,
exact-once visitation confirmed, matching the e2e differentials) surfaced
one major and a set of minor defects, all fixed and re-validated:

- **Major:** the device match counter was 32-bit; a dense weak-formation
  search (legal arguments, ~2 s runtime) can exceed 2^32 matches, wrapping
  the counter — silently wrong counts, a false "not truncated" state, and
  buffer overwrites. Now `unsigned long long` end-to-end; the CLI prints the
  true total and a "only the first N of M listed" warning; a new e2e case
  locks the behavior (GPU total vs CPU ground truth over a >1M-match
  volume: 1,088,572 == 1,088,572).
- Parser: trailing tokens on a line now rejected (previously a second block
  pasted on the same line was silently dropped); block offsets bounded
  (|x|,|z| ≤ 1e6, |y| ≤ 4096 — an absolute coordinate pasted as an offset
  used to be accepted and could overflow kernel math); contradictory
  duplicate entries rejected, including top-vs-side parity conflicts
  (side must equal top % 2), while consistent top+side pairs for the same
  block remain legal.
- runSearch API robustness: two-step u64 overflow guard for the volume
  check (ny*nz could overflow the guard itself), chunk-count math immune
  to u64 wrap at razor-edge volumes; weak-formation warning prints the
  double directly (the old cast was UB past 2^63).
- Compile-time asserts: chunk size > 0 (livelock guard), block threads a
  multiple of 32 (full-mask intrinsics), formation fits the 48 KB default
  dynamic-shared limit.
- oracle_diff now fails on truncated/malformed dump files instead of
  passing on partial coverage (eof + expected line counts enforced).
- bench.sh labels corrected after the chunk-default flip (the sweep now has
  a real c256 datapoint and no mislabeled duplicate); Makefile windows
  target creates build/ and documents NVCC=; doctest.h added to test deps;
  parser-test temp files cleaned up.
- Docs: oracle comparison count stated consistently as 29,178,112 values;
  corner-test claim softened to the two corners actually tested.

Post-fix validation: `make test` 11 cases / 99,975 assertions, full e2e
(29 checks incl. the new truncation case), oracle diff 0/29,178,112,
kernel time unchanged (0.815-0.825 s).

## Optimization round 3 + box migration (2026-07-04)

The original benchmark instance was recycled (fresh disk, new endpoint,
EPYC 7763 instead of 7H12, same RTX 3090 model and same 27.2-core quota).
Consequences handled:

- The Java oracle harness and benchmark sources previously lived only on
  the box. Now vendored: `test/oracle/Dump.java` + `test/oracle/regen.sh`
  rebuild the oracle on any machine with a JDK, and `test/bench-java/`
  holds the benchmark configuration of the reference tool. The locally
  regenerated dumps were verified **byte-identical (SHA256)** to the
  originals from the old box — the platform-independence claim held up
  exactly when it was needed.
- The previous best build was re-benchmarked on the new instance before
  anything else: 0.821-0.824 s kernel vs 0.822-0.826 s on the old box —
  within noise, so the benchmark table remains comparable across the swap.

Round-3 changes (predicted 8-12% from the instruction model, measured
**12.5%**: kernel 0.822 s → 0.719 s, wall ~0.94 s; legacy identical at
0.720 s):

| Change | Why it works |
|---|---|
| Modern tail in provably-unsigned form (`texture.cuh`) | `next` comes from a 48-bit-masked seed and is always non-negative, but the `(int)` cast hid that from the compiler, forcing signed `%`/shift fixup code on every evaluation. The unsigned form is value-identical for all inputs (re-proven: 29,178,112 oracle values, 0 mismatches) and folds to bare shifts/masks. |
| Per-chunk pull budget + incremental refill (`kernel.cu`) | The old refill did 64-bit `g = base + pull*32 + lane`, a 64-bit `g < total` compare, and two hash-term multiplies per candidate. Now: the number of in-range pulls is computed once per chunk (one ceil-div), the linear index advances implicitly, and the x hash term advances by the compile-time constant `32*3129871` (wrapping add) — the multiplies survive only on the rare z-carry path. Also shifts work from the saturated multiply pipe to the underused ALU pipe. |

Validated on the new box: all 29 e2e checks (including the dense
GPU-vs-CPU exhaustive differentials that pin the enumeration), oracle diff
clean, register count unchanged at full occupancy. BENCHMARKS.md row 7.

Remaining single-GPU headroom at this algorithm is now genuinely small
(~40 issue slots/eval vs ~35 irreducible); next meaningful steps are
multi-GPU (split the chunk cursor) or newer silicon.
