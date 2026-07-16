# Experiment ledger

Flat, objective index of every trial run during the correctness/verification/
optimization overhaul — kept *and* rejected — with the raw numbers and just
enough context to interpret them. This is the data-of-record for a later
writeup; the *narrative* (why each change works, decision trail, prose) lives
in [ENGINEERING_LOG.md](ENGINEERING_LOG.md) and [BENCHMARKS.md](BENCHMARKS.md).
Nothing here is editorialized; where a claim needs derivation it points at
[SEMANTICS.md](SEMANTICS.md).

Convention: append, don't rewrite. Each row records what was measured, not
what it means.

---

## 0. Fixed reference workload (W)

Every performance number below, unless stated otherwise, is this one search:

| Field | Value |
|---|---|
| x range | [-175000, -75000] (100,001 wide) |
| y range | [-64, -40] (25 tall) |
| z range | [-75000, -25000] (50,001 wide) |
| Candidate origins | 1.25 × 10¹¹ |
| Formation | `test/fixtures/formation_a.txt`, 24 top-face blocks |
| Version | modern (v0, MC 1.21.2+) unless noted |
| Correct result | exactly one match at (-108723, -54, -69736) |

"Kernel time" = CUDA-event time around the kernel launch. "Wall" = full
process including ~0.2 s CUDA context startup. Perf numbers are best-of-3
after one warmup.

## 0b. Machines

| ID | GPU | CPU | Notes |
|---|---|---|---|
| M1 | RTX 3090, driver 580.159.03 | 2× EPYC 7H12, cgroup quota 27.2 cores | original box; benchmark rows 1–6; recycled |
| M2 | RTX 3090, driver 580.126.20 | EPYC 7763, quota 27.2 cores | replacement box; benchmark row 7; row-6 build re-measured here first (within noise) |
| M3 | RTX 3090, CUDA 12.8, JDK 21 | RunPod instance, 256 hw threads | 2026-07-16 re-verification only (§5) |
| M4 | Apple M4 Pro, 20-core GPU | 14 CPU cores | Metal + portable-CPU backends (§4) |

---

## 1. Correctness defects found in the pre-overhaul baseline

All were silent (wrong/empty results with a clean "search completed"). Detail
and evidence in ENGINEERING_LOG.md §Baseline.

| # | Defect | Trigger | Observable failure |
|---|---|---|---|
| C1 | grid block count `(max-min+7)/8` drops the last column for exact-multiple widths | W's widths 100001/50001 | x=-75000, z=-25000 never searched |
| C2 | single-cell dim → `(0+7)/8 = 0` blocks → invalid launch, error swallowed | narrow ranges | searches nothing, reports success |
| C3 | `rotateFormation` advances side rotations mod 4 → reach 2–3 vs mod-2 values | side blocks + direction ≠ 0 | 0 matches ("side rotations don't work") |
| C4 | test file had its own `getTextureModern` = `(mod*next)>>31`; device used `((4*next)>>31)%mod` | any side face | tests validated a different function than shipped |
| C5 | in-kernel `printf` as sole output; bounded buffer | many matches | matches beyond buffer vanish |
| C6 | UB wrapping (`x*3129871` int32, `l*l*C` int64, `abs(INT_MIN)`) | all evaluations | happened to compile correctly on then-current nvcc; not guaranteed |
| C7 | no input validation | unreadable file / bad rotation / bad version / >1024 blocks | "0 blocks → everything matches", silent 0-match, silent wrong mode, constant-mem overflow |
| C8 (review) | device match counter 32-bit | dense weak formation (>2³² matches, ~2 s) | counter wraps: wrong count, false "not truncated", buffer overwrite |

All fixed; C8 found by a later adversarial-review pass. Fixes re-verified by
the suite in §3.

---

## 2. Performance experiments (chronological)

CUDA on M1 (E1–E9) then M2 (E10–E11). "Δ" is vs the immediately preceding
kept configuration.

| # | Change | Kernel | Wall | Δ | Verdict |
|---|---|---|---|---|---|
| E1 | Baseline (original: dim3(8,2,8), printf out, per-thread version branch) | — | 4.63 s | — | baseline |
| E2 | Rewrite: grid-stride 3D, 256-thread blocks, device result buffer, template version dispatch | 1.26 s | 1.47 s | 3.15× vs E1 | **kept** (correctness-driven; speed a side effect) |
| E3 | Launch-geometry sweep: block ∈ {128,256,512} × grid caps | ±1% | — | ~0 | **rejected as lever** — kernel is ALU-bound, not geometry-bound |
| E4 | Warp compaction: persistent warps, per-lane candidate refill, atomic chunk cursor | 1.17–1.20 s | ~1.40 s | ~7% | **kept** (also removes the C1/C2 bug class by construction) |
| E5 | Chunk-size sweep #1 | 1024/lane → 1.168 s | — | — | **kept**, default → 1024/lane |
| E6 | Formation in dynamic shared memory instead of `__constant__` | 1.168 → 1.158 s | — | ~1% | **kept** (free; no occupancy hit) — disproved the "constant-serialization" hypothesis |
| E7 | Register audit `-Xptxas -v` | 32 regs, full occupancy | — | — | diagnostic — not occupancy/latency starved; `ncu` unavailable (`ERR_NVGPUCTRPERM`), bound analysis stayed model-based |
| E8 | Precompute per-block hash terms (`bx·3129871`, `bz·116129781` host-side) + ballot-free stripe loop | 1.158 → 0.855 s | — | ~26% | **kept**; forced the `…FromParts` two-stage API in texture.cuh (single-source preserved) |
| E9 | Chunk-size sweep #2 | 64: 0.944 / 256: 0.856 / 1024: 0.823 | — | — | **kept**, default stays 1024/lane; block 128/256/512 within noise |
| E10 | Modern tail in provably-unsigned form (folds signed `%`/shift fixup away) | — | — | see E11 | **kept** |
| E11 | Per-chunk pull budget + incremental refill (kills per-pull 64-bit compare + 2 multiplies) — landed with E10 | 0.822 → 0.719 s | ~0.94 s | 12.5% (predicted 8–12%) | **kept** — current default |

Structural alternatives analyzed and **rejected without implementing**
(reasoning in ENGINEERING_LOG.md / BENCHMARKS.md):

| Idea | Why rejected |
|---|---|
| Shared-memory tiling of texture values | early-exit makes naive evaluation ~1.33 evals/position; precompute-everything needs ~18 equiv → net loss |
| Multi-pass filtering | same early-exit economics |
| ILP dual-candidates per lane | kernel already issues ~40 slots/eval vs ~35 irreducible; not worth the register/complexity cost |
| Hand-scheduled PTX / 48-bit truncated multiplies | ≤10% remaining headroom; complexity not justified |

Estimated remaining single-GPU headroom at this algorithm: small (~40 issue
slots/eval measured vs ~35 irreducible). Next real levers are multi-GPU
(split the chunk cursor) or newer silicon.

---

## 3. Algorithmic-alternative experiments

| # | Experiment | Result | Verdict |
|---|---|---|---|
| A1 | Hensel-lift the quadratic core `l = 42317861·u² + 11u mod 2⁶⁴` | exactly 2-to-1, invertible (derivative `2Cu+11` always odd; verified 2000/2000) | **holds** — full 48-bit state → coordinate falls out |
| A2 | Output stage `(cr ^ M)·M + 11 mod 2⁴⁸` treated as affine | affine, lattice-friendly *in isolation* | **holds** |
| A3 | Z3 4.16 QF_BV inversion of the full modern pipeline, 24 fixture constraints, coord pinned to answer | SAT instantly, forward-verifies | encoding **validated** (no false-negative risk) |
| A4 | Same, coord free + bounded to W (1.25e11 box the scanner solves in 0.72 s) | Z3 parallel mode, ~2 CPU-hours / >30 min wall on M4, **no solution**, killed | **negative** — off-the-shelf SMT does not beat forward scanning |

Barrier localized to the per-block coupling
`u_i = int32((x+dx_i)·A) XOR (z+dz_i)·B XOR (y+dy_i)` (XOR-of-arithmetic
defeats pure lattice methods). Untried, listed for a future attempt:
dedicated QF_BV solver (bitwuzla/kissat via SMT-LIB), fixing y (−9 unknown
bits), overdetermination via more observed blocks, custom search over the
Hensel inversion. Experiment script kept scratchpad-only (not in repo);
encoding facts preserved in ENGINEERING_LOG.md §SMT.

---

## 4. Cross-implementation & backend measurements

Cross-implementation benchmark on W (full table + per-row rationale in
BENCHMARKS.md):

| # | Implementation | Config | Wall | vs Java (as shipped) | Machine |
|---|---|---|---|---|---|
| 1 | Java reference (19MisterX98/TextureRotations) | as shipped, 10 threads | 192.5 s (1,562 CPU-core-s) | 1× | M1 |
| 2 | Java reference | 256 threads (quota-capped 27.2 cores) | 104.9 s (2,698 CPU-core-s) | 1.8× | M1 |
| 3 | CUDA original (pre-overhaul) | dim3(8,2,8), printf | 4.63 s | 41.6× | M1 — **correctness caveats C1–C3** |
| 4 | CUDA grid-stride + result buffer | 256-thread | 1.47 s (k 1.26) | 131× | M1 |
| 5 | CUDA warp compaction | chunk 256/lane | 1.40 s (k 1.17) | 138× | M1 |
| 6 | CUDA precomputed hash terms + stripes | chunk 1024/lane | 1.04 s (k 0.83) | 185× | M1 |
| 7 | CUDA unsigned tail + incremental refill | chunk 1024/lane | 0.94 s (k 0.72) | 205× | M2 |

Cross-device framing: at Java's best throughput (1,562 EPYC-core-s) vs the
0.72 s GPU kernel, matching one RTX 3090 needs ~2,200 EPYC cores scaling
perfectly; a full unrestricted 128-core dual-socket run projects to ~12 s —
the single GPU is still ~17× faster than the whole server at its theoretical
best.

Backend parity on W (identical `include/texture.cuh` compiled three ways):

| Backend | Hardware | Compute time |
|---|---|---|
| CUDA | RTX 3090 (M2) | 0.72 s |
| Metal | Apple M4 Pro, 20-core GPU (M4) | 7.49 s |
| CPU (threaded C++) | Apple M4 Pro, 14 cores (M4) | 74.6 s |

Per-version CUDA cost on W: modern/legacy ≈ 0.72 s; Sodium19 = 1.29 s (two
extra Stafford rounds/eval).

---

## 5. Verification artifacts (current pass state)

| Artifact | What it checks | Latest result |
|---|---|---|
| `make test` (doctest) | RNG vectors, parser, rotation, fixture | 12 cases / **249,765 assertions** pass |
| `test/e2e.sh` | fixture + corner + single-cell regressions, synthetic round-trips (5 versions × 4 directions × single/all), dense & weak GPU-vs-CPU differentials, truncation-total, checkpoint/resume, 100.0%-progress | **all checks pass** |
| `build/oracle_diff` | every C++ RNG path vs Java reference dumps (grids, world-border extremes, 1M random) | **72,945,280 values, 0 mismatches** |
| oracle dump determinism | byte-identical dumps across platforms | SHA256-identical: JDK17/Linux/x86_64, JDK25/macOS/ARM, JDK21/Linux |
| cross-impl unique match | Java, CUDA, CPU, Metal all solve W | all find (-108723, -54, -69736), count 1 |

**2026-07-16 re-verification (M3, fresh RunPod 3090, CUDA 12.8, JDK 21):**
clean checkout rsynced up, rebuilt from scratch. `make test` 249,765/249,765;
`test/e2e.sh` all pass; oracle regenerated on-box (JDK 21) and
`oracle_diff` = 72,945,280 / 0; kernel best-of-3 **0.71346 s** (runs 0.71346 /
0.71675 / 0.71790; wall 0.91–0.97 s) — reproduces benchmark row 7 (0.719 s)
within noise on a third independent instance.

---

## 6. Observability research (not yet implemented)

From decompiling a 1.21.11 client jar — future signal sources beyond the 24
top-face rotations already used. Context only; no code yet.

| Finding | Count | Bits/observation |
|---|---|---|
| Rotatable blocks (texture rotation via position hash) | 34 catalogued | 2 (top: mod 4; side: parity) |
| Flower/plant offset blocks, OffsetType.XZ | 32 | 8 (nibbles 0 & 2 of `getSeed(x,0,z)`) |
| Flower/plant offset blocks, OffsetType.XYZ | 5 | 8+ |

Flowers use the *same* position hash at `getSeed(x,0,z)`, so ~4× the bits per
observed block vs a rotation — relevant to the §3 overdetermination idea and
to shrinking search volumes.
