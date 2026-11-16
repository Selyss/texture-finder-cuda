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
| Correct result | exactly one match at the verified fixture position |

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
| M5 | 1× NVIDIA A100-SXM4-40GB (40,960 MiB, compute capability 8.0), driver 580.178.04 | AMD EPYC 7413, 4 Slurm CPUs allocated | Narval, CUDA 12.9.86 (§7) |

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
| cross-impl unique match | Java, CUDA, CPU, Metal all solve W | all find the verified fixture position, count 1 |

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

## 7. Narval A100 baseline and launch sweep (2026-11-12)

Slurm job `9193191` ran `correctness-overhaul` at `bbef439` on M5.
`make test` passed all 12 cases and 249,765 assertions, and the full
`test/e2e.sh` passed on the default CUDA build before `test/bench.sh`.
Slurm recorded a total job runtime of 1 min 44 s, including builds, tests,
the launch sweep, and the profiling attempt.

| Configuration | Modern kernel, best of 3 after warmup | Legacy kernel, single run |
|---|---:|---:|
| 256 threads, 1024 candidates per lane (default) | 0.681369 s | 0.671620 s |
| 128 threads, 1024 candidates per lane | 0.682844 s | 0.674180 s |
| 512 threads, 1024 candidates per lane | 0.678373 s | 0.667171 s |
| 256 threads, 256 candidates per lane | 0.713125 s | 0.703069 s |
| 256 threads, 4096 candidates per lane | 0.669153 s | 0.660097 s |

The 4096-candidate chunk was fastest in this sweep, with about 1.8% lower
kernel time than the default in both modes. The sweep builds used
`-arch=native`. The full end-to-end suite covered the default configuration,
and the alternate configurations have single-sweep measurements. The default
remains the baseline for future ports.

Nsight Compute attempted to profile one `matchFormationKernel` launch
with `--set full`. The driver reported an unavailable profiling resource.
The profiler produced no counter data. NVIDIA documents contention for the
performance monitor as a common cause, with DCGM among the possible
contenders. The log does not identify the owner of the resource, so the A100
kernel bottleneck remains unmeasured.
[NVIDIA Nsight Compute FAQ](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq)

### Repeat timings and CUDA timeline (job 9194685)

The verified default binary at `bbef439` ran on a second Narval node
(`ng31104`, after `ng30604` in job `9193191`) with the same A100 model
and driver. The job warmed each mode once, then ran nine samples per mode
with alternating mode order. All timing samples were taken outside the
profilers. Slurm recorded a total job runtime of 38 s.

| Sample | Modern kernel (s) | Legacy kernel (s) |
|---:|---:|---:|
| 1 | 0.681705 | 0.671485 |
| 2 | 0.681675 | 0.671968 |
| 3 | 0.681761 | 0.671987 |
| 4 | 0.681618 | 0.671831 |
| 5 | 0.681581 | 0.671745 |
| 6 | 0.681529 | 0.671681 |
| 7 | 0.681539 | 0.671712 |
| 8 | 0.681524 | 0.671701 |
| 9 | 0.681576 | 0.671746 |
| **Median** | **0.681581** | **0.671745** |
| **Min–max** | **0.681524–0.681761** | **0.671485–0.671987** |

Nsight Compute 2025.2.1 retried with `--set basic` and reported the same
unavailable driver resource on the second node. Hardware counters remain
unavailable, and the cause of the contention is unconfirmed.

Nsight Systems 2025.1.3 collected a CUDA timeline in
`nsys-cuda-9194685.nsys-rep` with CPU sampling and context-switch tracing
disabled. In launch order, the modern kernel took 1.606616 ms for the
initial slice and 730.540850 ms for the main slice, totaling 732.147466 ms
under tracing. The main slice accounts for 99.78% of the traced kernel time.
The five GPU copies transferred 1,180 bytes in 9.376 µs combined. The
first `cudaMemcpyToSymbol` call occupied 193.478 ms of host API time.
The CUDA timeline supports targeting the second kernel launch in a future
counter capture. It leaves the kernel's instruction and memory bottlenecks
unresolved.

## 8. CUDA x-axis bitplane slab experiment (2026-11-12)

Job `9195540` ran the modern, stride-6 prototype from
`experiments/rng-reduction/flat_cuda.cu` on M5 (`ng31104`, CUDA 12.9.86,
driver 580.178.04, `sm_80`). The source SHA-256 was
`9659f6a5fa49e008e9e197ad2084e5dcb8bc563ceb30edb97eac9518ae860c15`.
The slab used the first 4,096 candidate z rows of W, its full x width,
and all 25 candidate y levels, totaling 10,240,102,400 candidates.
The per-candidate-y plane layout allocated 855,820,800 bytes.

GPU and direct CPU masks matched on the small synthetic case. The full
slab matched exact CPU survivor counts in all nine GPU runs.

| Coset | Survivors |
|---:|---:|
| 0 | 106,710,019 |
| 1 | 6,666,155 |
| 2 | 1,668,734 |
| 3 | 1,668,151 |
| 4 | 1,665,536 |
| 5 | 26,637,239 |
| **Total** | **145,015,834** |

The baseline called the existing CUDA backend at `bbef439` once per
sample on the same slab. After warmup, the benchmark alternated the
baseline and prototype order and recorded CUDA event times. The prototype
timings include field construction and match counting. CPU validation,
survivor verification, and result handling are outside those timings.

| Sample | Baseline (ms) | Fill (ms) | Match count (ms) |
|---:|---:|---:|---:|
| 1 | 72.145439 | 11.409920 | 7.106208 |
| 2 | 63.197346 | 11.411328 | 7.107200 |
| 3 | 56.062977 | 8.870208 | 5.525024 |
| 4 | 56.061630 | 8.868032 | 5.525408 |
| 5 | 55.988159 | 8.776160 | 5.467552 |
| 6 | 56.013279 | 8.863264 | 5.520928 |
| 7 | 56.050751 | 8.869408 | 5.523008 |
| 8 | 56.013248 | 8.860384 | 5.520736 |
| 9 | 56.019135 | 8.864832 | 5.523264 |
| **Median** | **56.050751** | **8.868032** | **5.523264** |

The median per-sample sum of fill and match count was 14.392416 ms
(14.243712–18.518528 ms). The median paired ratio of the existing backend
to these two stages was 3.89×. Both stages met the initial timing targets.
The first two samples were slower across all three measurements and are
included in the summary. This experiment supports measuring survivor
verification and result handling on the same slab next.

### Complete slab search (job 9196459)

The same A100 node and toolchain ran a version that verifies each surviving
bit against the remaining formation blocks inside the match kernel. It
atomically counts all final matches, stores up to the existing result limit,
and sorts stored matches on the host. The baseline used the existing
`runSearch` on the same slab. The prototype source SHA-256 was
`48fcb14df186c1fd44670546792e24e90f69404c697bf299a4c83111327a5b4e`.

The synthetic test matched the CPU masks and the existing search's exact
match set, including a known positive. A dense case with a small result
buffer preserved the true count and returned valid, distinct matches.
The slab agreed with the direct CPU filter on all six cosets and
145,015,834 survivors. Both searches returned zero final matches on this
slab in every run.

Each sample used a fresh search allocation for both versions. Wall time
includes device allocations, kernel execution, result retrieval, sorting,
and device cleanup. The CPU correctness check ran outside the timing.

| Sample | Baseline GPU (ms) | Fill (ms) | Filter and verify (ms) | Baseline wall (ms) | Prototype wall (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 72.171356 | 11.410112 | 12.341088 | 74.652188 | 25.104964 |
| 2 | 56.730846 | 11.409760 | 12.349792 | 61.833510 | 25.288809 |
| 3 | 56.056416 | 8.870560 | 9.605568 | 60.747259 | 19.770330 |
| 4 | 56.077854 | 8.870016 | 9.609408 | 58.530945 | 19.760181 |
| 5 | 56.063969 | 8.869504 | 9.602080 | 57.572032 | 20.418789 |
| 6 | 56.062687 | 8.870112 | 9.607008 | 60.552944 | 19.947353 |
| 7 | 56.016384 | 8.782336 | 9.505760 | 61.588581 | 19.599629 |
| 8 | 56.042240 | 8.776672 | 9.558720 | 58.491640 | 19.720377 |
| 9 | 56.024513 | 8.864704 | 9.597440 | 57.759735 | 19.717571 |
| **Median** | **56.062687** | **8.870016** | **9.605568** | **60.552944** | **19.770330** |

The median per-sample GPU total was 18.476128 ms. The median paired
speedup was 3.03× on GPU and 2.97× in wall time. Slurm recorded 32 s for
the complete job, including compilation and validation. These
measurements cover modern mode and the 4,096-row slab. Full-volume
scaling and integration with the production CLI remain to be measured.

### Full reference workload (job 9196908)

The same standalone binary ran all 125,003,750,025 candidates in W on
`ng31003`, a Narval A100-SXM4-40GB node with driver 580.178.04 and
CUDA 12.9.86. The plane allocation was 10,428,752,000 bytes (10.43 GB).
The baseline used a single `runSearch` call on the full bounds.

Both GPU paths returned the same single verified match in every run. The
prototype matched the direct CPU survivor count in each coset.

| Coset | Survivors |
|---:|---:|
| 0 | 1,302,225,705 |
| 1 | 81,392,044 |
| 2 | 20,347,922 |
| 3 | 20,345,216 |
| 4 | 20,341,899 |
| 5 | 325,516,735 |
| **Total** | **1,770,169,521** |

| Sample | Baseline GPU (ms) | Fill (ms) | Filter and verify (ms) | Baseline wall (ms) | Prototype wall (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 681.441528 | 107.912033 | 117.045731 | 686.955540 | 230.794232 |
| 2 | 681.397156 | 107.918304 | 117.118912 | 689.939705 | 231.287174 |
| 3 | 681.447998 | 107.910591 | 117.045380 | 683.106842 | 230.744567 |
| 4 | 681.450012 | 107.909187 | 117.120033 | 690.051195 | 230.770096 |
| 5 | 681.490540 | 107.914627 | 117.047394 | 687.306103 | 233.498035 |
| 6 | 681.471558 | 107.910019 | 117.124191 | 689.798618 | 234.907941 |
| 7 | 681.452698 | 107.918526 | 117.051010 | 683.218413 | 230.815842 |
| 8 | 681.428284 | 107.906273 | 117.124512 | 696.390764 | 234.156260 |
| 9 | 681.442566 | 107.923843 | 117.045059 | 690.155803 | 234.307987 |
| **Median** | **681.447998** | **107.912033** | **117.051010** | **689.798618** | **231.287174** |

The median per-sample GPU total was 224.969536 ms. The median paired
speedup was 3.03× on GPU and 2.96× in search wall time. The wall timings
include device allocations and result retrieval, measured inside the
already-running harness. Slurm reported 2 min 14 s for the job, including
warmup, nine paired runs, and direct CPU verification.

### Full reference workload with a reusable plane buffer (job 9197641)

Narval node `ng10504` ran the tiled prototype on an A100-SXM4-40GB
with driver 580.178.04 and CUDA 12.9.86. The source SHA-256 was
`96865292fe380e50e273bd11a5cc619b3e51c837da068edadb54c6e634197291`.
The search covered all 125,003,750,025 candidates in 13 slabs with up to
4,096 candidate z rows per slab. One 855,820,800-byte plane allocation
served the whole search, a 12.19× reduction from the untiled prototype.
Each fill covered the halo required by that slab's formation offsets.

The synthetic case matched direct CPU masks and final matches across slab
boundaries. A dense synthetic case with one candidate z row per slab
preserved the total when the result buffer filled. On W, all six survivor
counts matched the direct CPU filter and job `9196908`, totaling
1,770,169,521 survivors. The existing and tiled searches returned the same
single final match in all nine pairs.

| Sample | Baseline GPU (ms) | Fill (ms) | Filter and verify (ms) | Baseline wall (ms) | Prototype wall (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 681.368164 | 108.150116 | 108.867867 | 688.890654 | 218.533983 |
| 2 | 681.333740 | 108.147156 | 108.867714 | 683.540564 | 218.752466 |
| 3 | 681.306885 | 108.147835 | 108.856026 | 682.790178 | 218.461877 |
| 4 | 681.308044 | 108.141914 | 108.865097 | 686.763449 | 218.767254 |
| 5 | 681.312744 | 108.152580 | 108.872231 | 685.374987 | 226.049863 |
| 6 | 681.364380 | 108.145767 | 108.867264 | 683.519565 | 218.754770 |
| 7 | 681.346252 | 108.152000 | 108.881821 | 685.480637 | 218.699476 |
| 8 | 681.318848 | 108.141403 | 108.865944 | 683.559450 | 218.763496 |
| 9 | 681.359619 | 108.144188 | 108.860733 | 682.852705 | 218.488959 |
| **Median** | **681.333740** | **108.147156** | **108.867264** | **683.559450** | **218.752466** |

The median per-sample GPU total was 217.013031 ms. The median paired
speedup over the existing search was 3.14× on GPU and 3.13× in search
wall time. The wall timings include allocations, transfers, and cleanup
inside the harness. Slurm recorded 2 min 26 s for compilation, warmup,
timings, and direct CPU validation. The untiled prototype ran in a
different allocation, so the paired speedup uses the existing search
from job `9197641`.

### Match kernel diagnostics (job 9228736)

Narval node `ng31104` ran the 4,096-row slab on an A100-SXM4-40GB with
driver 580.178.04 and CUDA 12.9.86. The source SHA-256 was
`b08466c96860a62b94f804aee563f93e880e6aec0ddf9b8ee35f4531d7420313`.
The original plane layout matched CPU survivor counts in all six cosets.
The synthetic case passed the mask and exact match checks, and the dense
case preserved the true count after the result buffer filled.

| Match mode | Slab median (ms) | Slab min–max (ms) |
|---|---:|---:|
| Filter and verify | 8.915264 | 8.822720–11.477056 |
| Filter only | 5.528928 | 5.471232–7.109376 |
| Synthetic plane words | 9.238208 | 9.236704–11.896736 |
| 32-bit index decomposition | 5.510752 | 5.510528–7.094720 |
| Filter without the block tally | 5.235744 | 5.234880–6.741088 |

Each mode had nine samples with rotated run order. The difference between
the full and filter medians was 3.39 ms. The 32-bit index decomposition
reduced filter time by 0.33%, and dropping the tally reduced it by 5.30%.
The synthetic plane words yielded 145,026,195 survivors versus 145,015,834
with real planes, a difference of about 0.007%. Synthetic words increased
filter time by 67%, so this substitution leaves the cost of real memory
reads unresolved.

### Compact world-y planes (job 9228736)

Both layouts ran on the same A100 allocation. Each matched the direct CPU
filter in every coset on the slab and on W, with totals of 145,015,834
and 1,770,169,521 survivors respectively. Both layouts found the same
single final match as `runSearch` on W in all nine timed pairs. The compact
layout also passed the synthetic mask checks for single levels, disjoint
world-y ranges, and ranges that touch or overlap.

| Full reference workload | Original planes | Compact world-y planes |
|---|---:|---:|
| Plane levels per hashed z row | 50 | 27 |
| Plane allocation (bytes) | 855,820,800 | 462,143,232 |
| Fill evaluations per candidate (approximate) | 0.333 | 0.180 |
| Fill median (ms) | 108.120483 | 56.464157 |
| Filter and verify median (ms) | 108.838760 | 111.697617 |
| Combined GPU median (ms) | 216.962730 | 168.163269 |
| Search wall median (ms) | 226.983666 | 169.593703 |

Compact indexing reduced the fill time by 47.8%, close to the 46% reduction
predicted from the plane level count. The paired comparison for the compact
layout used an existing-search median of 681.117554 ms on GPU and
683.235871 ms in wall time. The median paired speedups were 4.05× on GPU
and 4.03× in wall time. The two layout groups ran in sequence within the
allocation, with alternating baseline and prototype runs in each group.

| Layout and workload | Filter median (min–max), ms | Synthetic words median (min–max), ms | Synthetic survivors |
|---|---:|---:|---:|
| Original, slab | 5.529 (5.471–7.109) | 9.238 (9.237–11.897) | 145,026,195 |
| Original, full | 67.499 (67.386–67.503) | 112.776 (112.775–112.779) | 1,770,045,268 |
| Compact, slab | 6.939 (5.394–6.940) | 11.738 (9.119–11.741) | 145,022,162 |
| Compact, full | 65.809 (65.806–65.983) | 111.284 (111.119–111.287) | 1,770,007,798 |

The compact slab diagnostic had two timing clusters, with fill and match
times changing together. The full workload filter measurements were tightly
grouped. The gap between full-search filter-and-verify and filter-only
medians was about 41 ms for the original layout and 46 ms for the compact
layout. Synthetic loads were slower in all four comparisons, and no GPU
hardware counters were collected. The optional load layout diagnostics
were skipped. Slurm reported 4 min 53 s for the job.

### Opt-in CLI search (job 9229800)

The CUDA CLI now uses the compact world-y implementation when
`TF_CUDA_BITPLANES=1` and the search is eligible. It uses stride 6 and
reuses a bounded plane buffer across its internal slabs. The plane
budget is capped at 1 GiB and reduced if available device memory is
lower. A weak filter, insufficient plane space, or a legacy version
uses the standard kernel.

Narval job `9229800` completed in 119 s with exit code `0:0` on
`ng31401`, using one A100-SXM4-40GB, driver 580.178.04, CUDA 12.9,
and an `sm_80` build. Both modes passed the full CLI end-to-end suite
and the build passed all 12 unit tests (249,765 assertions). The
reference workload returned the same single match in both modes and
in every timed sample. Path tracing confirmed that every timed opt-in
run used bitplanes. A 1-byte plane budget forced fallback and returned
the same result. Stopping the bitplane CLI after one slice and
resuming the checkpoint with the standard kernel returned the same
match and total as the uninterrupted search and removed the state file.

Nine alternating pairs followed one untimed reference run per mode.
GPU time is the CLI's reported kernel time, including all GPU search
passes and slices. Whole-command wall time is measured from process
launch to exit, including CUDA startup. Times are seconds.

| Pair | Standard GPU | Bitplanes GPU | Standard wall | Bitplanes wall |
|---:|---:|---:|---:|---:|
| 1 | 0.681391 | 0.157367 | 1.257499 | 0.848831 |
| 2 | 0.681370 | 0.158027 | 1.220921 | 0.666937 |
| 3 | 0.681351 | 0.157330 | 1.322511 | 0.786920 |
| 4 | 0.681347 | 0.157420 | 1.333014 | 0.849043 |
| 5 | 0.681420 | 0.157400 | 1.335588 | 0.812208 |
| 6 | 0.681407 | 0.157445 | 1.213059 | 0.714509 |
| 7 | 0.681354 | 0.157418 | 1.244552 | 0.902906 |
| 8 | 0.681398 | 0.158262 | 1.390604 | 0.728267 |
| 9 | 0.681416 | 0.157396 | 1.378185 | 0.853347 |

The ratios of medians were 4.33× for GPU time and 1.63× for whole-command
wall time. The standalone timings above start inside an already-running
process. The CLI numbers include its normal slice scheduling, and this
run did not collect separate fill and match timings.

| Source | SHA-256 |
|---|---|
| `src/kernel.cu` | `a31cc12f34a31282e1f62cb7b5ae82ed58a3d3ff5bfc5dc2e5b5035ce2e7844e` |
| `include/cuda_bitplanes.cuh` | `73f6a9a1071eb0ce436f74030c2e832472700f41b8464d79ce31cee59cb128c5` |

### Proposal to verify survivors in a separate pass

Fable proposed storing the existing 64-bit survivor masks and having
a second CUDA kernel refill each lane from those masks as candidates
fail verification. The filter can use the existing mask indexing.
Each mask holds 64 candidates, so the allocation is bounded by the
slab dimensions even for a weak filter. On W, at 4,096 rows, one mask
buffer would use 1,280,409,600 bytes. Writing and rereading all masks
over W would move about 31.3 GB of mask payload. These are byte counts,
with no measured memory traffic or timings for this design yet.

The estimate of about 46 ms for inline verification comes from the
difference between the compact standalone filter-only and
filter-and-verify runs in job `9228736`. The CLI benchmark in job
`9229800` did not time those stages separately. For a future test,
compare the filter with inline verification against the filter with
mask writes plus the new verifier, at the same stride and slab size.
For the CLI decision, include any change in slab size needed to fit
the plane and mask allocations into the same device memory budget.
Use the proposal's 10 ms CLI improvement as its proposed threshold.

Check each stored mask against the CPU on small cases. Per-coset
survivor and word counts can catch coverage errors on W. On small
cases, instrument the word visits to check that each word is visited
exactly once, including partial warps and the final short slab. Check
exact final matches and true counts, including a dense case where the
result buffer fills.

### Compact world-y stride comparison (job 9230586)

Job `9230586` completed in 69 s with exit code `0:0` on Narval node
`ng31203`. The job used an A100-SXM4-40GB with driver 580.178.04, CUDA
12.9.86, and `sm_80` builds. The two builds differed only in the
compile-time stride. To reproduce the experiment, replace the
`STRIDE` initializer with `TF_EXPERIMENT_STRIDE` in the shared CUDA
header and build both executables with that definition set to 5 or 6.
The working-tree stride is 6.

Nine alternating CLI pairs on W followed an untimed run at each
stride. Every timed run used bitplanes and found exactly the match
reported by the standard kernel. GPU and whole-command times are
seconds.

| Pair | Stride 5 GPU | Stride 6 GPU | Stride 5 wall | Stride 6 wall |
|---:|---:|---:|---:|---:|
| 1 | 0.155619 | 0.157378 | 0.726339 | 0.808112 |
| 2 | 0.155629 | 0.158101 | 0.689679 | 0.807821 |
| 3 | 0.155599 | 0.157935 | 0.820569 | 0.655754 |
| 4 | 0.155600 | 0.157394 | 0.792766 | 0.754279 |
| 5 | 0.156141 | 0.157362 | 0.652947 | 0.755090 |
| 6 | 0.156309 | 0.157376 | 0.669642 | 0.810428 |
| 7 | 0.155608 | 0.157375 | 0.712593 | 0.993884 |
| 8 | 0.155611 | 0.157364 | 0.909123 | 0.696479 |
| 9 | 0.155613 | 0.157383 | 0.776928 | 0.773668 |
| Median | 0.155613 | 0.157378 | 0.726339 | 0.773668 |

Both standalone builds also passed the synthetic mask, exact match
and truncation checks. On the first 4,096 rows of W, the GPU survivor
counts matched the direct CPU count in every coset and the search
returned the standard kernel's zero matches. The table gives standalone
medians in milliseconds for the slab.

| Stride | Survivors | Plane bytes | Fill | Filter and verify | Combined |
|---:|---:|---:|---:|---:|---:|
| 5 | 38,491,946 | 554,707,008 | 5.558304 | 8.021120 | 13.578944 |
| 6 | 145,015,834 | 462,143,232 | 4.633280 | 9.155040 | 13.788896 |

Survivors by coset were 500,924 / 1,999,735 / 2,001,248 / 1,998,264 /
31,991,775 for stride 5 and 106,710,019 / 6,666,155 / 1,668,734 /
1,668,151 / 1,665,536 / 26,637,239 for stride 6.

| Slab sample | Stride 5 fill | Stride 5 filter and verify | Stride 6 fill | Stride 6 filter and verify |
|---:|---:|---:|---:|---:|
| 1 | 5.559392 | 8.021120 | 4.633888 | 9.155968 |
| 2 | 5.558304 | 8.023392 | 4.634048 | 9.158976 |
| 3 | 5.558496 | 8.020448 | 4.628864 | 9.149312 |
| 4 | 5.558656 | 8.022464 | 4.628672 | 9.150848 |
| 5 | 5.557760 | 8.019072 | 4.633344 | 9.154240 |
| 6 | 5.557280 | 8.021440 | 4.633280 | 9.155616 |
| 7 | 5.556992 | 8.018208 | 4.642880 | 9.155040 |
| 8 | 5.556832 | 8.020928 | 4.633056 | 9.156832 |
| 9 | 5.559744 | 8.022656 | 4.632448 | 9.154368 |

Stride 5 sent 3.77 times fewer candidates to verification. The slab
gained about 1.13 ms in filter and verify combined and spent about
0.93 ms more on fill. The full CLI gained 1.77 ms of GPU time, or
1.12%, on W. The wall time ranges overlap. The stride experiment did
not time verification separately. A follow-up to assess the proposed
mask verifier should measure filter-only and inline verification at
stride 5 to establish the headroom before adding a mask round trip.

| Source | SHA-256 |
|---|---|
| `src/kernel.cu` | `a31cc12f34a31282e1f62cb7b5ae82ed58a3d3ff5bfc5dc2e5b5035ce2e7844e` |
| `include/cuda_bitplanes.cuh` (experiment) | `e349eefda656623a0aa4a40ddb55dd476105fa53f4628bb4a62a677c9697b201` |
| `experiments/rng-reduction/flat_cuda.cu` | `4bd6c3220bb0569130c019ce6c2a800f64d33152a7b59f3de594abc1006d6bc1` |

### Stride 5 inline verification cost (job 9231142)

Narval job `9231142` completed in 4 min 30 s with exit code `0:0`
on `ng31104`, using one A100-SXM4-40GB, driver 580.178.04,
CUDA 12.9.86 and an `sm_80` build. The compact standalone harness
used 13 slabs of up to 4,096 z rows on W. It ran the production
`Search` kernel mode and the `NoTail` diagnostic with the same
plane construction and filter. The diagnostic keeps the computed
mask live with inline assembly. It omits verification, mask
writes, and the diagnostic survivor tally. The harness alternated
the order after warming up both modes and checked every inline
sample against the standard kernel's exact result.

The synthetic mask, exact match and bounded storage checks passed.
The GPU and the direct CPU count agreed on all 469,959,730 survivors
across the full workload. The five coset counts were 6,099,175 /
24,417,246 / 24,412,805 / 24,406,584 / 390,623,920.

| Pair | Inline fill (ms) | Filter and verify (ms) | Diagnostic fill (ms) | Filter only (ms) | Difference (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 67.712898 | 87.394073 | 67.714691 | 75.661247 | 11.732826 |
| 2 | 67.710396 | 87.396133 | 67.713547 | 75.596130 | 11.800003 |
| 3 | 67.716095 | 87.393120 | 67.713722 | 75.595901 | 11.797218 |
| 4 | 67.713661 | 87.394974 | 67.899590 | 75.596451 | 11.798523 |
| 5 | 67.715973 | 87.557373 | 67.711777 | 75.596161 | 11.961212 |
| 6 | 67.712578 | 87.392418 | 67.711205 | 75.597221 | 11.795197 |
| 7 | 67.714653 | 87.400551 | 67.788353 | 75.734177 | 11.666374 |
| 8 | 67.713570 | 87.398567 | 67.763069 | 75.599159 | 11.799408 |
| 9 | 67.740738 | 87.426399 | 67.734177 | 75.613503 | 11.812897 |
| Median | 67.713661 | 87.396133 | 67.714691 | 75.597221 | 11.798523 |

The paired difference ranged from 11.67 to 11.96 ms. It estimates
inline verification headroom and can include effects from compiling
the two modes separately. For the proposed 10 ms improvement, the
separate verifier and mask traffic would have about 1.8 ms combined
to preserve the gain, assuming comparable performance in the CLI.
The proposed full mask round trip would move about 31 GB of payload,
so the result favors pausing that design and evaluating fill
improvements. This diagnostic did not measure the separate verifier
or mask write cost.

| Source | SHA-256 |
|---|---|
| `src/kernel.cu` | `a31cc12f34a31282e1f62cb7b5ae82ed58a3d3ff5bfc5dc2e5b5035ce2e7844e` |
| `include/cuda_bitplanes.cuh` (experiment) | `e349eefda656623a0aa4a40ddb55dd476105fa53f4628bb4a62a677c9697b201` |
| `experiments/rng-reduction/flat_cuda.cu` (experiment) | `67e0e739d268f86f50530cfe8d6332979e3f40b07b1d92dec376c0a316cec593` |

### Fill arithmetic timing diagnostic (job 9231949)

Narval job `9231949` completed in 12 s with exit code `0:0` on
`ng31104`. Both stride builds ran on one A100-SXM4-40GB with driver
580.178.04 and CUDA 12.9.86, targeting `sm_80`. Each build used the
compact world-y layout and 13 slabs of up to 4,096 rows on W.

Both kernels used the same launch geometry, plane allocation,
output RNG, and bit packing. The diagnostic substituted the XOR of
the three coordinate terms for the normal coordinate RNG input.
Its output was used only for timing. CUDA events timed fill alone.
The real and diagnostic kernels each ran once before nine paired
samples with alternating order. The compiler reported 32 registers
per thread for the real fill and 28 for the diagnostic, with no
spills in either build.

| Stride | Pair | Real fill (ms) | XOR diagnostic (ms) | Difference (ms) |
|---:|---:|---:|---:|---:|
| 5 | 1 | 67.770432 | 49.628803 | 18.141628 |
| 5 | 2 | 67.380348 | 49.623074 | 17.757275 |
| 5 | 3 | 67.711014 | 49.622108 | 18.088905 |
| 5 | 4 | 67.711525 | 49.632736 | 18.078789 |
| 5 | 5 | 67.711044 | 49.622849 | 18.088196 |
| 5 | 6 | 67.708954 | 49.624481 | 18.084473 |
| 5 | 7 | 67.710167 | 49.623325 | 18.086842 |
| 5 | 8 | 67.709122 | 49.622913 | 18.086208 |
| 5 | 9 | 67.710526 | 49.622589 | 18.087936 |
| 6 | 1 | 56.460197 | 41.376225 | 15.083973 |
| 6 | 2 | 56.459103 | 41.380032 | 15.079071 |
| 6 | 3 | 56.460957 | 41.378399 | 15.082558 |
| 6 | 4 | 56.459518 | 41.376923 | 15.082596 |
| 6 | 5 | 56.459614 | 41.376610 | 15.083004 |
| 6 | 6 | 56.476128 | 41.381023 | 15.095104 |
| 6 | 7 | 56.460674 | 41.388802 | 15.071873 |
| 6 | 8 | 56.458687 | 41.381699 | 15.076988 |
| 6 | 9 | 56.458401 | 41.380283 | 15.078117 |

The median paired gap was 18.09 ms at stride 5 and 15.08 ms at
stride 6. Both exceed the proposed 10 ms threshold for trying the
grouped fill. The diagnostic omits the coordinate mix and uses
fewer registers, so the gap measures the effect of that
substitution. The grouped implementation still has to compute
the exact mix and keep several masks live. Its speedup and output
equality remain unmeasured.

| Source | SHA-256 |
|---|---|
| `include/cuda_bitplanes.cuh` (experiment) | `970bcb2a775d126aa3c96e6308b910b9a0ba3cf69c09272fc2d9a5d0a30f6230` |
| `experiments/rng-reduction/flat_cuda.cu` (experiment) | `08d26de501991932764d807871fc08d148820669f98196cb5d33afc4fb82533a` |

### Grouped fill decision and filter address calculation

Paused the grouped fill prototype after reviewing Fable's instruction
estimate. The XOR diagnostic saved 26.7% of fill time at both strides.
It measures the combined effect of changing the arithmetic and the
compiled kernel, which also used four fewer registers per thread.
It cannot isolate a per-multiply cost. Fable's estimate, based on
stand-in compiler output, predicts the grouped replacement would
give back the savings through narrower multiplies, setup and extra
instructions. The grouped kernel has no A100 timing.

The next filter experiment is to split the plane index into a base
computed once per thread and an offset computed on the host for each
check. For a given slab and coset, use the first candidate z in that
coset as an anchor. All later candidate rows in that coset advance by
exactly one hashed row for every stride of candidate z. The
per-check offset combines the anchor's hashed row for that check,
the check's compact y slot and its whole-word x offset. The thread
base combines the candidate's y, its hashed-row distance from the
anchor and its candidate word. Also store the intra-word shift in
the check metadata. The two terms add to the existing plane index.

A host arithmetic comparison matched the old and factored indices
in 53,035 cases across strides 5 and 6, including negative z, slabs
shorter than the stride, and large offsets. This exercised the index
formula only. The offset table depends on the slab's hashed-row
count and anchor alignment. A CUDA benchmark should time filter
events and also include table preparation and upload in whole-search
timing. Compare masks word for word on synthetic cases and check
exact matches on W. The previous 32-bit diagnostic changed the
initial row decomposition, so it did not measure removal of the
arithmetic inside the check loop.

### Precomputed filter plane offsets (job 9234752)

Narval job `9234752` completed in 39 s with exit code `0:0` on
`ng31205`, using one A100-SXM4-40GB, driver 580.178.04 and CUDA
12.9.86 with an `sm_80` target. The compact standalone harness
compared the existing filter with a variant that builds and uploads
a table of plane offsets before matching each slab. Each candidate
thread computes its base word index once and adds the stored offset
for each formation check. The table also stores the intra-word shift.

Both strides passed the synthetic word-for-word CPU mask checks,
including short slabs and partial words, and the bounded result
storage check. Both variants agreed on every per-coset survivor count
on W and returned the same single match as the standard kernel in
every timed search. The survivor totals were 469,959,730 at stride 5
and 1,770,169,521 at stride 6.

Nine alternating pairs followed warmups at each stride. CUDA events
measured filter-only kernels and kernels with inline verification.
Whole-search wall time includes device buffer allocation and release,
all fill and match kernels, per-slab table construction and upload,
and result retrieval. It starts inside the standalone process.

| Stride | Filter old → precomputed (ms) | Filter and verify old → precomputed (ms) | Search wall old → precomputed (ms) |
|---:|---:|---:|---:|
| 6 | 63.980389 → 45.002113 | 100.468231 → 87.214523 | 158.466027 → 145.193626 |
| 5 | 75.606720 → 50.802628 | 87.407173 → 70.095741 | 157.039299 → 139.982225 |

| Stride | Pair | Filter old (ms) | Filter precomputed (ms) | Filter and verify old (ms) | Filter and verify precomputed (ms) | Wall old (ms) | Wall precomputed (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1 | 63.978527 | 45.000675 | 100.627106 | 87.228386 | 158.580793 | 145.193626 |
| 6 | 2 | 63.979549 | 44.998051 | 100.456070 | 87.221107 | 158.365808 | 145.154392 |
| 6 | 3 | 63.983810 | 45.005634 | 100.661858 | 87.212036 | 158.904244 | 145.137239 |
| 6 | 4 | 63.980389 | 45.001446 | 100.483383 | 87.220772 | 158.402899 | 145.147549 |
| 6 | 5 | 63.980862 | 45.002750 | 100.463837 | 87.213211 | 158.386628 | 145.194427 |
| 6 | 6 | 63.979717 | 44.999390 | 100.468231 | 87.210945 | 161.627676 | 145.166665 |
| 6 | 7 | 63.981117 | 45.002113 | 100.456230 | 87.212669 | 158.378793 | 145.207752 |
| 6 | 8 | 63.990784 | 45.007393 | 100.591293 | 87.232414 | 159.928223 | 146.373670 |
| 6 | 9 | 63.980225 | 45.160542 | 100.466980 | 87.214523 | 158.466027 | 145.345472 |
| 5 | 1 | 75.605728 | 50.800259 | 87.715462 | 70.112541 | 161.268409 | 139.447898 |
| 5 | 2 | 75.763680 | 50.948994 | 87.410301 | 70.099197 | 159.851839 | 144.382837 |
| 5 | 3 | 75.606659 | 50.801983 | 87.407173 | 70.095741 | 156.687035 | 139.364250 |
| 5 | 4 | 75.606720 | 50.802784 | 87.406021 | 70.182587 | 166.496180 | 139.472574 |
| 5 | 5 | 75.608253 | 50.802788 | 87.407578 | 70.095009 | 156.657649 | 141.243261 |
| 5 | 6 | 75.610962 | 50.800034 | 87.422920 | 70.095390 | 157.039299 | 139.982225 |
| 5 | 7 | 75.605759 | 50.805824 | 87.404671 | 70.100098 | 156.675944 | 149.523856 |
| 5 | 8 | 75.605919 | 50.800999 | 87.402435 | 70.093925 | 156.670314 | 144.895784 |
| 5 | 9 | 75.608734 | 50.802628 | 87.404388 | 70.094856 | 160.023352 | 139.410066 |

Reproduce with `TF_FLAT_COMPARE_OFFSETS=1` and the compact standalone
benchmark on W. Build stride 5 by changing the header's stride
constant to 5 in an experiment copy. The prototype is selectable
through the standalone harness. The CLI continues to use its
existing index calculation pending integration and CLI measurements.

| Source | SHA-256 |
|---|---|
| `src/kernel.cu` | `a31cc12f34a31282e1f62cb7b5ae82ed58a3d3ff5bfc5dc2e5b5035ce2e7844e` |
| `include/cuda_bitplanes.cuh` | `5ebb16b1c88185707a93cc54453bcde6e0e866fcf6544883c396f49892612602` |
| `experiments/rng-reduction/flat_cuda.cu` | `75790c25b987a06cc8f86a92954df8a6efc1b3e07f28db0a100a8e1a39c09b70` |

### Precomputed plane offsets in the CUDA CLI (job 9235600)

The opt-in CUDA CLI uses the same offset preparation as the standalone
harness. It rebuilds and uploads one table per slab. The benchmark used
stride 6 for both binaries and compared the CLI immediately before and
after the offset change. Job `9235600` completed in 127 s with exit code
`0:0` on `ng31002`, an A100-SXM4-40GB with driver 580.178.04 and CUDA
12.9, targeting `sm_80`.

The 12 unit tests and both end-to-end suites passed. The 1-byte budget
forced fallback and returned the reference match. An 8 MiB budget kept
bitplanes active with shorter slabs and returned the same match. A
checkpoint created by the new bitplane search resumed with the standard
kernel, returned the same result, and removed its state file. Both
binaries took the bitplane path and returned the standard kernel's
exact match in all nine timed pairs.

Each mode ran once before the alternating pairs. GPU time is the
CLI's reported kernel time across all slices. Whole-command wall time
includes process startup, table preparation and upload, and process
exit. All times are seconds.

| Pair | Old GPU | New GPU | Old wall | New wall |
|---:|---:|---:|---:|---:|
| 1 | 0.157507 | 0.144197 | 0.859525 | 0.866059 |
| 2 | 0.157350 | 0.144282 | 0.857278 | 0.860062 |
| 3 | 0.157445 | 0.144388 | 0.864237 | 0.926646 |
| 4 | 0.157427 | 0.144326 | 0.850351 | 0.835544 |
| 5 | 0.157498 | 0.144154 | 0.996671 | 0.842369 |
| 6 | 0.157404 | 0.144198 | 0.906884 | 0.833566 |
| 7 | 0.157441 | 0.144214 | 0.863589 | 0.866560 |
| 8 | 0.157984 | 0.144279 | 0.839256 | 0.850069 |
| 9 | 0.157338 | 0.144293 | 0.842017 | 0.826126 |
| Median | 0.157441 | 0.144279 | 0.859525 | 0.850069 |

The GPU median fell by 13.16 ms (8.36%). Whole-command timing ranges
overlap, so these samples establish the GPU improvement more clearly
than the whole-command effect.

| Source | SHA-256 |
|---|---|
| `src/kernel.cu` before | `a31cc12f34a31282e1f62cb7b5ae82ed58a3d3ff5bfc5dc2e5b5035ce2e7844e` |
| `src/kernel.cu` after | `eb39bb28279be9fcc078772c46107b32b2d137880afd19d0586642f35b8e123d` |
| `include/cuda_bitplanes.cuh` | `3d0c89c44ba9f64a0d65b9e1ad89e0c55d1f15b9fb0b7f30ae1bab04c4ea3d35` |

### CUDA CLI stride coverage (job 9294135)

Narval job `9294135` completed in 122 s on `ng31104` with exit
code `0:0`. The job used one A100-SXM4-40GB, driver 580.178.04,
CUDA 12.9.86 and `sm_80`. Both CLI builds used precomputed offsets
and differed only in their stride constant. The standard path ran
with bitplanes disabled. The two stride builds ran with bitplanes
enabled and traced each search slice or direction.

Each mode warmed up before timed triples with rotated order. All
timed runs returned the exact standard-kernel match set and count.
The reference had one match in each direction setting. The synthetic
mixed formation had 12 top checks and 12 side checks and returned
zero matches at both sizes. The 10- and 8-top formations returned
18 and 11 matches. An independent local CPU search confirmed the
synthetic counts. Volumes per direction were 125,003,750,025 for
the reference, 725,409 and 1,088,340,017 for the mixed formation,
18,027,009 for 10 tops, and 725,409 for 8 tops.

Times are seconds. Fallback runs count commands with at least one
fallback. The last two columns count traced calls across all timed
commands. Each workload had five samples per mode, with nine for
the single-direction reference.

| Workload | Mode | GPU median | GPU min | GPU max | Wall median | Wall min | Wall max | Fallback runs | Bitplane calls | Fallback calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | standard | 0.68170800 | 0.68148600 | 0.68185100 | 1.28765916 | 1.27383232 | 1.32762935 | 0 | 0 | 0 |
| reference | 5 | 0.13834200 | 0.13833000 | 0.13839500 | 0.77656737 | 0.72510901 | 0.87943069 | 0 | 18 | 0 |
| reference | 6 | 0.14419000 | 0.14417200 | 0.14421000 | 0.79380990 | 0.72284460 | 0.80430322 | 0 | 18 | 0 |
| reference-all | standard | 2.72552000 | 2.72541000 | 2.72554000 | 3.39304038 | 3.36412884 | 3.45768703 | 0 | 0 | 0 |
| reference-all | 5 | 0.55351300 | 0.55330000 | 0.55375700 | 1.20997222 | 1.17176770 | 1.26911209 | 0 | 25 | 0 |
| reference-all | 6 | 1.65064000 | 1.65058000 | 1.65071000 | 2.30745837 | 2.24711479 | 2.35548870 | 5 | 15 | 10 |
| mixed-small | standard | 0.00051594 | 0.00051037 | 0.00054752 | 0.60317415 | 0.58952071 | 0.63595882 | 0 | 0 | 0 |
| mixed-small | 5 | 0.00047910 | 0.00047168 | 0.00055402 | 0.61362379 | 0.58271612 | 0.68399382 | 0 | 5 | 0 |
| mixed-small | 6 | 0.00048717 | 0.00047168 | 0.00051526 | 0.59433060 | 0.57817197 | 0.65018877 | 0 | 5 | 0 |
| mixed-large | standard | 0.00649542 | 0.00648944 | 0.00716979 | 0.60370573 | 0.59486708 | 0.64559276 | 0 | 0 | 0 |
| mixed-large | 5 | 0.00190490 | 0.00189888 | 0.00191146 | 0.64862327 | 0.60331926 | 0.65707691 | 0 | 10 | 0 |
| mixed-large | 6 | 0.00191917 | 0.00189690 | 0.00259750 | 0.60539617 | 0.59346786 | 0.64467900 | 0 | 10 | 0 |
| ten-tops | standard | 0.00048650 | 0.00047789 | 0.00051638 | 0.61842664 | 0.59547862 | 0.64122502 | 0 | 0 | 0 |
| ten-tops | 5 | 0.00053338 | 0.00051469 | 0.00054851 | 0.59774989 | 0.58818904 | 0.60028543 | 0 | 5 | 0 |
| ten-tops | 6 | 0.00049046 | 0.00047638 | 0.00052723 | 0.63450580 | 0.59116148 | 0.66257200 | 5 | 0 | 5 |
| eight-tops | standard | 0.00051190 | 0.00050378 | 0.00053354 | 0.60571259 | 0.58537601 | 0.61232511 | 0 | 0 | 0 |
| eight-tops | 5 | 0.00051344 | 0.00051066 | 0.00053235 | 0.61506549 | 0.59464766 | 0.63255576 | 5 | 0 | 5 |
| eight-tops | 6 | 0.00051165 | 0.00050893 | 0.00052064 | 0.60247498 | 0.58760106 | 0.64087200 | 5 | 0 | 5 |

Stride 5 saved 5.85 ms of GPU time on the single-direction reference.
In all-direction mode it stayed on bitplanes throughout, while
stride 6 fell back on 10 of 25 traced calls and in all five commands.
Stride 5 also qualified on the 10-top workload, where the standard
kernel had a lower median GPU time. Both builds fell back on the
8-top case. This benchmark used a temporary stride 5 build. The
workspace CLI still builds with stride 6.

Source SHA-256 values were
`eb39bb28279be9fcc078772c46107b32b2d137880afd19d0586642f35b8e123d`
for the kernel,
`3d0c89c44ba9f64a0d65b9e1ad89e0c55d1f15b9fb0b7f30ae1bab04c4ea3d35`
for the stride 6 header, and
`17e624929e3486a85c303484a540aa94972c47cc5436abf82acd6a570552c98a`
for the stride 5 header.

```csv
workload,sample,mode,gpu_s,wall_s,bitplane_calls,fallback_calls
reference,1,standard,0.68155500,1.32762935,0,0
reference,1,5,0.13833800,0.77656737,2,0
reference,1,6,0.14419500,0.80251028,2,0
reference,2,5,0.13836100,0.79031883,2,0
reference,2,6,0.14418400,0.80430322,2,0
reference,2,standard,0.68170800,1.28573951,0,0
reference,3,6,0.14417800,0.79380990,2,0
reference,3,standard,0.68180600,1.28011673,0,0
reference,3,5,0.13835100,0.87943069,2,0
reference,4,standard,0.68183600,1.27619927,0,0
reference,4,5,0.13834900,0.80334671,2,0
reference,4,6,0.14417600,0.72284460,2,0
reference,5,5,0.13833700,0.77436786,2,0
reference,5,6,0.14419900,0.79578432,2,0
reference,5,standard,0.68182600,1.29647572,0,0
reference,6,6,0.14419800,0.78538775,2,0
reference,6,standard,0.68185100,1.30131451,0,0
reference,6,5,0.13833900,0.75927870,2,0
reference,7,standard,0.68152800,1.27383232,0,0
reference,7,5,0.13839500,0.73635896,2,0
reference,7,6,0.14417200,0.78899131,2,0
reference,8,5,0.13833000,0.80138881,2,0
reference,8,6,0.14421000,0.79536739,2,0
reference,8,standard,0.68148600,1.28765916,0,0
reference,9,6,0.14419000,0.78549296,2,0
reference,9,standard,0.68154000,1.32004341,0,0
reference,9,5,0.13834200,0.72510901,2,0
reference-all,1,standard,2.72541000,3.39616109,0,0
reference-all,1,5,0.55330000,1.26911209,5,0
reference-all,1,6,1.65071000,2.30745837,3,2
reference-all,2,5,0.55333400,1.23829680,5,0
reference-all,2,6,1.65063000,2.32686991,3,2
reference-all,2,standard,2.72553000,3.45768703,0,0
reference-all,3,6,1.65064000,2.35548870,3,2
reference-all,3,standard,2.72551000,3.39304038,0,0
reference-all,3,5,0.55375700,1.19511271,5,0
reference-all,4,standard,2.72554000,3.36412884,0,0
reference-all,4,5,0.55356900,1.20997222,5,0
reference-all,4,6,1.65058000,2.24711479,3,2
reference-all,5,5,0.55351300,1.17176770,5,0
reference-all,5,6,1.65064000,2.28236442,3,2
reference-all,5,standard,2.72552000,3.39097970,0,0
mixed-small,1,standard,0.00051594,0.62414779,0,0
mixed-small,1,5,0.00047168,0.58271612,1,0
mixed-small,1,6,0.00047430,0.58085257,1,0
mixed-small,2,5,0.00047472,0.61840881,1,0
mixed-small,2,6,0.00047168,0.64242622,1,0
mixed-small,2,standard,0.00052230,0.63595882,0,0
mixed-small,3,6,0.00051526,0.57817197,1,0
mixed-small,3,standard,0.00051037,0.58952071,0,0
mixed-small,3,5,0.00047910,0.59762542,1,0
mixed-small,4,standard,0.00051357,0.59946430,0,0
mixed-small,4,5,0.00055402,0.68399382,1,0
mixed-small,4,6,0.00049066,0.59433060,1,0
mixed-small,5,5,0.00048138,0.61362379,1,0
mixed-small,5,6,0.00048717,0.65018877,1,0
mixed-small,5,standard,0.00054752,0.60317415,0,0
mixed-large,1,standard,0.00662010,0.59734029,0,0
mixed-large,1,5,0.00190182,0.60331926,2,0
mixed-large,1,6,0.00259750,0.64467900,2,0
mixed-large,2,5,0.00190490,0.64899446,2,0
mixed-large,2,6,0.00191917,0.59468962,2,0
mixed-large,2,standard,0.00649542,0.62926988,0,0
mixed-large,3,6,0.00190310,0.59346786,2,0
mixed-large,3,standard,0.00648944,0.59486708,0,0
mixed-large,3,5,0.00190765,0.62350576,2,0
mixed-large,4,standard,0.00649408,0.60370573,0,0
mixed-large,4,5,0.00191146,0.65707691,2,0
mixed-large,4,6,0.00195040,0.60539617,2,0
mixed-large,5,5,0.00189888,0.64862327,2,0
mixed-large,5,6,0.00189690,0.60693330,2,0
mixed-large,5,standard,0.00716979,0.64559276,0,0
ten-tops,1,standard,0.00047789,0.59650990,0,0
ten-tops,1,5,0.00051469,0.59774989,1,0
ten-tops,1,6,0.00048074,0.63450580,0,1
ten-tops,2,5,0.00053338,0.60028543,1,0
ten-tops,2,6,0.00050890,0.66257200,0,1
ten-tops,2,standard,0.00051638,0.61842664,0,0
ten-tops,3,6,0.00052723,0.65906460,0,1
ten-tops,3,standard,0.00048022,0.59547862,0,0
ten-tops,3,5,0.00053005,0.59777409,1,0
ten-tops,4,standard,0.00048650,0.62876297,0,0
ten-tops,4,5,0.00053926,0.58818904,1,0
ten-tops,4,6,0.00047638,0.61832389,0,1
ten-tops,5,5,0.00054851,0.58886063,1,0
ten-tops,5,6,0.00049046,0.59116148,0,1
ten-tops,5,standard,0.00049824,0.64122502,0,0
eight-tops,1,standard,0.00051654,0.60571259,0,0
eight-tops,1,5,0.00051066,0.63255576,0,1
eight-tops,1,6,0.00051094,0.58760106,0,1
eight-tops,2,5,0.00051142,0.59464766,0,1
eight-tops,2,6,0.00052064,0.63152280,0,1
eight-tops,2,standard,0.00050378,0.60535926,0,0
eight-tops,3,6,0.00051456,0.64087200,0,1
eight-tops,3,standard,0.00053354,0.60942195,0,0
eight-tops,3,5,0.00053235,0.62112124,0,1
eight-tops,4,standard,0.00050771,0.61232511,0,0
eight-tops,4,5,0.00051344,0.59645203,0,1
eight-tops,4,6,0.00051165,0.60247498,0,1
eight-tops,5,5,0.00052374,0.61506549,0,1
eight-tops,5,6,0.00050893,0.59781700,0,1
eight-tops,5,standard,0.00051190,0.58537601,0,0
```

### Stride 5 default for the opt-in CUDA path (job 9296853)

The opt-in CUDA bitplane path now builds with stride 5. Narval job
`9296853` completed in 93 s with exit code `0:0` on `ng10404`,
using one A100-SXM4-40GB, driver 580.178.04 and CUDA 12.9.
The promoted header SHA-256 matched the temporary stride 5 build
from job `9294135`.

All 12 unit tests and 249,765 assertions passed. Both CLI end-to-end
suites passed, with bitplanes enabled and disabled. The reference
search matched the standard kernel with an 8 MiB plane budget and
with a 1-byte budget that forced fallback. A checkpoint created
by the stride 5 path resumed with the standard kernel, returned
the exact match and removed the checkpoint on completion.

### Vulcan L40S validation (job 2323961)

Vulcan job `2323961` completed in 38 s
with exit code `0:0` on `rack04-06`. It used one NVIDIA L40S
(46,068 MiB reported), driver 595.112.07, and CUDA 12.9.86 targeting
`sm_89`. The kernel and stride 5 header hashes matched the sources
in Narval job `9294135`.

The same six workloads ran through the standard and opt-in bitplane
paths using one binary. Each mode warmed up before nine alternating
pairs for the single-direction reference and five for each other
workload. Every run returned the exact standard match set and
count. The eight-top workload fell back on all five runs. All other
opt-in runs used bitplanes throughout. GPU time includes all search
slices, and wall time covers process launch to exit. Times are seconds.

| Workload | Mode | GPU median | GPU min | GPU max | Wall median | Wall min | Wall max | Fallback runs | Bitplane calls | Fallback calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | standard | 0.31964600 | 0.31751400 | 0.32218200 | 0.53449614 | 0.53047664 | 0.57324319 | 0 | 0 | 0 |
| reference | 5 | 0.06679830 | 0.06478090 | 0.06754730 | 0.29612650 | 0.28909694 | 0.33605187 | 0 | 18 | 0 |
| reference-all | standard | 1.31435000 | 1.31169000 | 1.31647000 | 1.54172183 | 1.53325018 | 1.60018381 | 0 | 0 | 0 |
| reference-all | 5 | 0.27762100 | 0.27565500 | 0.27964500 | 0.51104732 | 0.50713080 | 0.58631680 | 0 | 25 | 0 |
| mixed-small | standard | 0.00028579 | 0.00028502 | 0.00028742 | 0.22334784 | 0.20522347 | 0.24813431 | 0 | 0 | 0 |
| mixed-small | 5 | 0.00038176 | 0.00036618 | 0.00066502 | 0.21119627 | 0.20761130 | 0.24669925 | 0 | 5 | 0 |
| mixed-large | standard | 0.00286608 | 0.00286269 | 0.00286675 | 0.21536158 | 0.20966114 | 0.25972052 | 0 | 0 | 0 |
| mixed-large | 5 | 0.00104931 | 0.00103392 | 0.00106550 | 0.21657629 | 0.20691285 | 0.25714432 | 0 | 10 | 0 |
| ten-tops | standard | 0.00026528 | 0.00026432 | 0.00026592 | 0.24418158 | 0.20695314 | 0.25783440 | 0 | 0 | 0 |
| ten-tops | 5 | 0.00039907 | 0.00038662 | 0.00096970 | 0.20659744 | 0.20503457 | 0.23945100 | 0 | 5 | 0 |
| eight-tops | standard | 0.00028605 | 0.00028394 | 0.00028630 | 0.21021092 | 0.20613054 | 0.27660613 | 0 | 0 | 0 |
| eight-tops | 5 | 0.00028544 | 0.00028422 | 0.00028621 | 0.21140552 | 0.20566695 | 0.28859084 | 5 | 0 | 5 |

```csv
workload,sample,mode,gpu_s,wall_s,bitplane_calls,fallback_calls
reference,1,standard,0.31751400,0.57324319,0,0
reference,1,5,0.06685930,0.29073464,2,0
reference,2,5,0.06718780,0.29612650,2,0
reference,2,standard,0.31877400,0.53982344,0,0
reference,3,standard,0.32052300,0.53814622,0,0
reference,3,5,0.06692630,0.29057657,2,0
reference,4,5,0.06643650,0.31022631,2,0
reference,4,standard,0.31964600,0.53424984,0,0
reference,5,standard,0.32035600,0.53661778,0,0
reference,5,5,0.06679830,0.29695876,2,0
reference,6,5,0.06478090,0.32181631,2,0
reference,6,standard,0.31890800,0.53047664,0,0
reference,7,standard,0.32132400,0.53449614,0,0
reference,7,5,0.06675760,0.29160856,2,0
reference,8,5,0.06666970,0.33605187,2,0
reference,8,standard,0.32218200,0.53292738,0,0
reference,9,standard,0.31947200,0.53207491,0,0
reference,9,5,0.06754730,0.28909694,2,0
reference-all,1,standard,1.31169000,1.60018381,0,0
reference-all,1,5,0.27964500,0.58631680,5,0
reference-all,2,5,0.27714200,0.51733772,5,0
reference-all,2,standard,1.31435000,1.53983340,0,0
reference-all,3,standard,1.31647000,1.54172183,0,0
reference-all,3,5,0.27762100,0.51104732,5,0
reference-all,4,5,0.27804700,0.50989420,5,0
reference-all,4,standard,1.31256000,1.53325018,0,0
reference-all,5,standard,1.31591000,1.54324452,0,0
reference-all,5,5,0.27565500,0.50713080,5,0
mixed-small,1,standard,0.00028502,0.22334784,0,0
mixed-small,1,5,0.00036832,0.20761130,1,0
mixed-small,2,5,0.00036618,0.24669925,1,0
mixed-small,2,standard,0.00028742,0.20522347,0,0
mixed-small,3,standard,0.00028614,0.24317239,0,0
mixed-small,3,5,0.00039469,0.20954883,1,0
mixed-small,4,5,0.00038176,0.21119627,1,0
mixed-small,4,standard,0.00028579,0.24813431,0,0
mixed-small,5,standard,0.00028547,0.20556476,0,0
mixed-small,5,5,0.00066502,0.24400237,1,0
mixed-large,1,standard,0.00286643,0.25972052,0,0
mixed-large,1,5,0.00104931,0.21650481,2,0
mixed-large,2,5,0.00105469,0.24774872,2,0
mixed-large,2,standard,0.00286608,0.21169669,0,0
mixed-large,3,standard,0.00286269,0.20966114,0,0
mixed-large,3,5,0.00106550,0.25714432,2,0
mixed-large,4,5,0.00103392,0.20691285,2,0
mixed-large,4,standard,0.00286269,0.24763527,0,0
mixed-large,5,standard,0.00286675,0.21536158,0,0
mixed-large,5,5,0.00104122,0.21657629,2,0
ten-tops,1,standard,0.00026432,0.24754195,0,0
ten-tops,1,5,0.00039907,0.20503457,1,0
ten-tops,2,5,0.00039738,0.20659744,1,0
ten-tops,2,standard,0.00026592,0.25783440,0,0
ten-tops,3,standard,0.00026451,0.20998292,0,0
ten-tops,3,5,0.00096970,0.23945100,1,0
ten-tops,4,5,0.00038662,0.21841454,1,0
ten-tops,4,standard,0.00026528,0.20695314,0,0
ten-tops,5,standard,0.00026534,0.24418158,0,0
ten-tops,5,5,0.00040506,0.20598210,1,0
eight-tops,1,standard,0.00028602,0.25055691,0,0
eight-tops,1,5,0.00028442,0.20566695,0,1
eight-tops,2,5,0.00028422,0.26133966,0,1
eight-tops,2,standard,0.00028630,0.21021092,0,0
eight-tops,3,standard,0.00028621,0.20613054,0,0
eight-tops,3,5,0.00028579,0.28859084,0,1
eight-tops,4,5,0.00028621,0.21140552,0,1
eight-tops,4,standard,0.00028605,0.20908337,0,0
eight-tops,5,standard,0.00028394,0.27660613,0,0
eight-tops,5,5,0.00028544,0.20798092,0,1
```

### Shared bitplane field across four directions (Vulcan job 2323980)

The standalone harness compared four independent stride 5 searches
against filling one union field per slab and running four match kernels
with independent output buffers. Both modes used precomputed offsets and
the same slab size. Job `2323980` completed in 21 s with exit code
`0:0` on one L40S, driver 595.112.07 and CUDA 12.9.86 targeting
`sm_89`. The harness SHA-256 was
`d2bae93c2cf29d2890f99a58c818fecc46389c0a442351789a71072951085874`.

The shared plane allocation was 555,382,656 bytes, compared with the
largest independent allocation of 554,707,008 bytes. Synthetic masks
agreed word for word with independent fields and the CPU oracle.
These cases covered one-row slabs, short final slabs, mixed faces,
negative offsets and offsets spanning multiple words. On W, all
five survivor counts agreed for each direction and every timed result
matched the standard kernel. The per-direction coset counts were:

| Direction | Matches | Coset 0 | Coset 1 | Coset 2 | Coset 3 | Coset 4 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 6099175 | 24417246 | 24412805 | 24406584 | 390623920 |
| 1 | 0 | 24417344 | 390621955 | 97666056 | 6102454 | 6107653 |
| 2 | 0 | 6100285 | 390635009 | 24406197 | 24409012 | 24418117 |
| 3 | 0 | 24420601 | 6105406 | 6101272 | 97660000 | 390612092 |

Nine alternating pairs followed warmups. Wall time covers the
standalone search, including allocation and copies, inside an
initialized process. Times are milliseconds.

| Pair | Independent fill | Shared fill | Independent match | Shared match | Independent wall | Shared wall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 138.683426 | 36.950466 | 145.879578 | 155.863770 | 310.035749 | 207.302530 |
| 2 | 137.487595 | 36.411812 | 144.902222 | 152.201416 | 315.964888 | 202.998691 |
| 3 | 139.449020 | 35.585243 | 146.428085 | 149.737930 | 311.338872 | 199.730242 |
| 4 | 140.711914 | 36.323971 | 147.985687 | 152.588272 | 320.270448 | 203.291507 |
| 5 | 138.518433 | 35.834621 | 145.715302 | 150.309616 | 309.673448 | 200.591464 |
| 6 | 138.353928 | 36.369343 | 145.545746 | 152.566833 | 309.393025 | 203.314310 |
| 7 | 140.243515 | 35.907742 | 147.288803 | 151.019714 | 312.932479 | 201.418603 |
| 8 | 139.756668 | 36.555073 | 146.744507 | 154.016739 | 316.800790 | 205.025279 |
| 9 | 140.257477 | 35.872738 | 147.380081 | 151.323486 | 313.044469 | 201.720531 |

Median total GPU time fell from 285.877 to 188.613 ms.
Median standalone wall time fell from 312.932 to 202.999 ms.
This experiment exercised shared field construction and matching.
CLI integration, checkpoint compatibility, mixed fallback and
independent truncation remain to be validated. Reproduce with
`TF_FLAT_SHARED_FIELD=1` on the standalone reference workload.

### Shared bitplanes in the CUDA CLI (Vulcan job 2327586)

The opt-in modern CUDA path builds a shared field for eligible directions
within each slice. Each direction has its own result buffer and count.
Ineligible directions use the standard kernel. If the shared field
cannot fit the budget, the CLI tries the individual paths. Existing
direction-first checkpoints resume in their original order. New
slice-first checkpoints use the `tfcp2` marker and can resume with
bitplanes disabled in builds that understand that marker. Streamed
output follows slice order and retains the facing on each line.

Vulcan job `2327586` completed in 47 s with exit code `0:0` on
`rack15-15`, using one L40S, driver 595.91.07, and CUDA 12.9.86
targeting `sm_89`. All 12 unit tests and the CUDA end-to-end suite
passed. Targeted checks covered the exact four-direction reference
match, a 1-byte fallback budget, short slabs with an 8 MiB budget,
a mixed formation with two shared and two fallback directions,
and a test build with a three-result cap per direction. A checkpoint
from the previous binary and new slice-first checkpoints resumed with
the expected results, including with bitplanes disabled.

Nine alternating pairs compared the previous and shared stride 5 CLI
on the full four-direction reference. Every run returned the exact
standard-kernel match. Times are seconds.

| Pair | Previous GPU | Shared GPU | Previous wall | Shared wall |
|---:|---:|---:|---:|---:|
| 1 | 0.27282700 | 0.17931000 | 0.49325531 | 0.46143940 |
| 2 | 0.27891600 | 0.17898700 | 0.53173985 | 0.38404240 |
| 3 | 0.27401600 | 0.17928000 | 0.55891135 | 0.42486030 |
| 4 | 0.27452900 | 0.17877700 | 0.50650856 | 0.46369556 |
| 5 | 0.27403100 | 0.17992300 | 0.49455434 | 0.38049524 |
| 6 | 0.27746800 | 0.18048700 | 0.57830190 | 0.37817532 |
| 7 | 0.27553000 | 0.17859700 | 0.61354442 | 0.45795283 |
| 8 | 0.27541100 | 0.18046500 | 0.53689500 | 0.39395867 |
| 9 | 0.27841700 | 0.17773000 | 0.49363532 | 0.37545879 |
| Median | 0.27541100 | 0.17928000 | 0.53173985 | 0.39395867 |

| Source | SHA-256 |
|---|---|
| `src/main.cpp` | `116e3b087bda9d2a348a5394be48cef531a414e3ae24edb0d583de562c5f6cae` |
| `src/kernel.cu` | `c886f516c822c130b48fd388759200b9f5f819a2658d31c78adf015cc3100a40` |
| `include/kernel.cuh` | `997ba03f476c3a161db977876deb378c3d483b7d543a741c56d2227c5a769dff` |

### Fused fill and filter prototype

The standalone stride 5 prototype keeps three levels in registers and
filters each candidate using one hashed z row. It uses overlapping
x words and verifies surviving positions inline. It has no global plane
allocation. The local CPU oracle passed 2,800 randomized comparisons
against brute force. GPU synthetic and reference slice masks matched the
CPU word for word. The slice had 270,324 survivors across the five cosets,
with counts of 46,920 / 11,776 / 11,649 / 11,527 / 188,452.

On the full reference, the CPU oracle measured 97,663,463 survivors
in coset 0. The GPU agreed and reproduced the previously measured
counts in the other four cosets, for 561,524,018 survivors in total.
Every timed verified run on Vulcan returned the exact standard-kernel
match. The 12-block line retained 1,250,025,194 of 5,000,150,001
candidates, approximately 25%, and returned the exact 288 matches.

#### Vulcan L40S (job 2327671)

Job `2327671` completed in 41 s on `rack04-16` with exit code `0:0`.
It used one NVIDIA L40S, driver 595.112.07, CUDA 12.9.86 and `sm_89`.
CUDA event times below are in milliseconds. The current path uses
stride 5 and precomputed offsets. Its full column includes fill,
filter and verification. The fused filter column includes hashing
and filtering, with the mask kept live and verification omitted.
The fused full column includes inline verification.

| Workload | Pair | Current fill | Current filter | Current full | Fused filter | Fused full |
|---|---:|---:|---:|---:|---:|---:|
| reference | 1 | 35.621246 | 27.975904 | 73.404640 | 49.498878 | 253.258408 |
| reference | 2 | 32.905281 | 28.436319 | 74.773628 | 42.781120 | 250.220551 |
| reference | 3 | 35.663872 | 28.003551 | 73.508545 | 49.983521 | 252.947388 |
| reference | 4 | 33.125309 | 28.401890 | 74.911263 | 43.239136 | 251.610397 |
| reference | 5 | 35.829376 | 28.005697 | 74.005692 | 49.420448 | 252.884293 |
| reference | 6 | 32.909214 | 28.471106 | 74.836105 | 43.390625 | 251.594620 |
| reference | 7 | 35.799107 | 28.049473 | 73.590393 | 49.691326 | 253.155579 |
| reference | 8 | 33.064640 | 28.430176 | 74.605347 | 43.165154 | 251.497894 |
| reference | 9 | 35.946236 | 28.097633 | 74.527519 | 50.037567 | 252.986145 |
| line | 1 | 1.207296 | 0.689664 | 3.831136 | 1.236064 | 9.275456 |
| line | 2 | 1.221856 | 0.710144 | 3.896768 | 1.248608 | 9.416416 |
| line | 3 | 1.211328 | 0.704544 | 3.903840 | 1.257536 | 9.452800 |
| line | 4 | 1.222080 | 0.702848 | 3.966464 | 1.270624 | 9.450976 |
| line | 5 | 1.232224 | 0.722816 | 3.990624 | 1.288128 | 9.689920 |
| line | 6 | 1.240064 | 0.723552 | 3.989024 | 1.293024 | 9.688384 |
| line | 7 | 1.236224 | 0.722848 | 3.990080 | 1.289120 | 9.688960 |
| line | 8 | 1.236384 | 0.721088 | 3.990688 | 1.291360 | 9.688768 |
| line | 9 | 1.239200 | 0.724192 | 3.992384 | 1.289344 | 9.689760 |

The reference median was 74.53 ms for the current full path, 49.42 ms
for fused hashing and filtering, and 252.88 ms for fused search with
verification. The line median was 3.99 ms for the current full path
and 9.69 ms for fused search. The compiler used 45 registers per
thread for the filter-only kernel and 48 with verification. Both had
zero stack bytes and zero spill loads or stores. The validation
variant used 80 registers and a 40-byte stack frame with zero spills.
The comparison shows a large cost when adding verification to the
fused kernel. It does not isolate the cause.

| Source | SHA-256 |
|---|---|
| `experiments/rng-reduction/flat_cuda.cu` | `635d66a6e98c89767af5385c2cef51c879b5a7c00d5a987245a4ae95fad2325f` |
| `experiments/rng-reduction/fused.cpp` | `850050a8598de7c7bb973d9da04c40397b1142660106101aaf8d8198c58f8b45` |
| `include/cuda_bitplanes.cuh` | `17e624929e3486a85c303484a540aa94972c47cc5436abf82acd6a570552c98a` |
| `src/kernel.cu` | `c886f516c822c130b48fd388759200b9f5f819a2658d31c78adf015cc3100a40` |

Reproduce in the standalone harness with `TF_FLAT_FUSED=1` and
`TF_CUDA_BITPLANES=0` on W.

#### Narval A100 (job 9316596)

Job `9316596` completed in 49 s on `ng31302` with exit code `0:0`.
It used one A100-SXM4-40GB, driver 580.178.04 and CUDA 12.9.86
targeting `sm_80`, with the source hashes listed above. It repeated
the same correctness checks and matched the standard kernel in every
timed sample on both workloads.

| Workload | Pair | Current fill | Current filter | Current full | Fused filter | Fused full |
|---|---:|---:|---:|---:|---:|---:|
| reference | 1 | 67.716606 | 50.785378 | 137.792130 | 100.379105 | 584.492554 |
| reference | 2 | 67.713089 | 50.785759 | 137.794037 | 100.378242 | 584.494324 |
| reference | 3 | 67.722107 | 50.785759 | 137.790527 | 100.375900 | 584.534058 |
| reference | 4 | 67.717407 | 50.792477 | 137.791809 | 100.381699 | 584.537354 |
| reference | 5 | 67.757339 | 50.837860 | 138.012344 | 100.378014 | 584.544434 |
| reference | 6 | 67.722267 | 50.788097 | 137.804184 | 100.381760 | 584.562012 |
| reference | 7 | 67.721565 | 50.788197 | 137.802704 | 100.379875 | 584.591370 |
| reference | 8 | 67.723480 | 50.789215 | 137.809280 | 100.379234 | 584.524719 |
| reference | 9 | 67.718880 | 50.789761 | 137.801880 | 100.377983 | 584.589844 |
| line | 1 | 2.614848 | 1.418592 | 8.522912 | 2.896448 | 21.655104 |
| line | 2 | 2.615904 | 1.416992 | 8.514208 | 2.897088 | 21.658688 |
| line | 3 | 2.615456 | 1.418368 | 8.523008 | 2.896192 | 21.655968 |
| line | 4 | 2.615680 | 1.417440 | 8.515424 | 2.897280 | 21.657152 |
| line | 5 | 2.615008 | 1.418656 | 8.517088 | 2.896512 | 21.658049 |
| line | 6 | 2.616864 | 1.418848 | 8.695841 | 2.896928 | 21.655169 |
| line | 7 | 2.616352 | 1.419136 | 8.514528 | 2.896288 | 21.653919 |
| line | 8 | 2.616992 | 1.418176 | 8.521248 | 2.896416 | 21.659391 |
| line | 9 | 2.615872 | 1.418688 | 8.519424 | 2.896352 | 21.656704 |

The reference medians were 137.80 ms for the current full path,
100.38 ms for fused hashing and filtering, and 584.54 ms for fused
search. Fused hashing and filtering was about 28.4 ms above the
estimated 72 ms hashing cost and passed the proposed filter-only
gate. Full fused search missed the proposed 120 ms target. The line
medians were 8.52 ms for the current path and 21.66 ms with fusion.
The timed kernels used 45 registers per thread without verification
and 56 with it. Both had zero stack bytes and zero spill loads or
stores. The validation build used 100 registers and a 40-byte stack
frame with zero spills.

The measured result rules out this implementation for the CLI.
The cost of adding inline verification was much larger than the
estimate on both GPUs. The experiment did not isolate divergence,
register pressure or other scheduling effects. These are standalone
CUDA event timings and do not include process startup.

### 32-bit fill packing on Vulcan L40S (job 2327788)

Vulcan job `2327788` completed in 15 s on `rack04-16` with exit
code `0:0`. It used one NVIDIA L40S, driver 595.112.07, CUDA 12.9.86,
and an `sm_89` build. The standalone variant packs each plane in
two 32-bit halves, then clears the bits past the end of a partial word.

The compiled original world-y fill has a four-position loop with
126 static SASS instructions, including 32 multiply or multiply-add
instructions, and a one-position cleanup loop with 37 instructions
including eight multiplies or multiply-adds. These counts include loop
control and exclude move and add forms from the multiply count. The
two variable 64-bit packing shifts use four funnel-shift instructions
per position, two for each plane. Both kernels use 40 registers per
thread and report zero spills.

Both halves of every plane pair agreed across eight synthetic
widths and all 34,669,188 plane pairs in the first reference slab.
The synthetic cases exercised negative coordinates, partial words,
and the zero padding word. The full reference fill then ran one
warmup per variant and nine alternating CUDA event pairs. Times
are milliseconds and cover plane construction.

| Pair | Original fill | 32-bit packing |
|---:|---:|---:|
| 1 | 34.625374 | 28.803265 |
| 2 | 35.592926 | 29.403233 |
| 3 | 34.360577 | 27.871138 |
| 4 | 37.098400 | 29.989088 |
| 5 | 35.331490 | 28.545570 |
| 6 | 36.486172 | 30.264736 |
| 7 | 34.590782 | 28.176735 |
| 8 | 36.642879 | 29.663012 |
| 9 | 34.815773 | 28.126850 |
| Median | 35.331490 | 28.803265 |

Median fill time fell by 6.53 ms (18.5%). This job measured fill
only on the L40S. Full-search timings, A100 timings, startup
instrumentation, and the diagonal lattice were outside this run.
The CUDA CLI uses the original packing. Reproduce with
`TF_FLAT_COMPARE_PACKING=1` in the standalone harness on W.

| Source | SHA-256 |
|---|---|
| `include/cuda_bitplanes.cuh` | `ff99d56b1b94d62f33efb17a64634c4a262e962a4d2429a3b823c270510246a7` |
| `experiments/rng-reduction/flat_cuda.cu` | `dbf07601f22e62b5bae9b9c469e229f30fa733d09109f20ce41df35f8a479498` |
| `src/kernel.cu` | `c886f516c822c130b48fd388759200b9f5f819a2658d31c78adf015cc3100a40` |

### Full search with 32-bit fill packing (Vulcan job 2327845)

Job `2327845` completed in 15 s with exit code `0:0` on
`rack03-13`, using one L40S, driver 595.112.07, CUDA 12.9.86 and
`sm_89`. The standalone stride 5 search used precomputed offsets
and changed only the fill packing. After one warmup per variant,
nine alternating pairs returned the exact standard-kernel match.
GPU search time sums fill and matching CUDA events. Search wall
time includes allocations, metadata uploads and result retrieval
inside an initialized process. Times are milliseconds.

| Pair | Original fill | Packed fill | Original GPU search | Packed GPU search | Original search wall | Packed search wall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33.424355 | 28.060062 | 68.626816 | 64.067291 | 76.019487 | 70.441094 |
| 2 | 34.529858 | 29.240482 | 70.731812 | 66.453918 | 77.064420 | 72.786184 |
| 3 | 33.327553 | 29.951200 | 68.481026 | 68.282112 | 74.947713 | 78.170709 |
| 4 | 35.791744 | 30.878654 | 73.124512 | 70.270111 | 79.463110 | 76.627486 |
| 5 | 33.707840 | 28.984451 | 69.211105 | 66.136612 | 75.602244 | 72.468774 |
| 6 | 34.811996 | 28.687614 | 71.229538 | 65.276894 | 77.567029 | 71.621016 |
| 7 | 33.707008 | 28.139103 | 69.215363 | 64.152481 | 81.531673 | 71.536104 |
| 8 | 34.540188 | 28.702335 | 70.804924 | 65.480164 | 77.135967 | 71.808616 |
| 9 | 33.711361 | 28.808895 | 69.241180 | 65.746628 | 75.596238 | 72.163045 |
| Median | 33.711361 | 28.808895 | 69.241180 | 65.746628 | 77.064420 | 72.163045 |

Reproduce with `TF_FLAT_COMPARE_PACKING_SEARCH=1` and
`TF_CUDA_BITPLANES=0` in the standalone harness on W.
The CLI still uses the original packing.

| Source | SHA-256 |
|---|---|
| `include/cuda_bitplanes.cuh` | `ff99d56b1b94d62f33efb17a64634c4a262e962a4d2429a3b823c270510246a7` |
| `experiments/rng-reduction/flat_cuda.cu` | `6488495a96a4c8fb163dd292d9eb68328aaf4411ae4b74f0ea402c9fbc8fc2bd` |
| `src/kernel.cu` | `c886f516c822c130b48fd388759200b9f5f819a2658d31c78adf015cc3100a40` |

### 32-bit packing in the opt-in CUDA CLI (Vulcan job 2327914)

Job `2327914` completed in 51 s with exit code `0:0` on
`rack04-02`, using one L40S, driver 595.112.07 and CUDA 12.9.86
targeting `sm_89`. Both builds used stride 5 and the same CLI
options. The new build uses 32-bit packing for both independent
and shared fields. Each variant warmed up before nine alternating
pairs per workload. Both variants used bitplanes and returned the
standard kernel's exact results in every run. All 12 unit tests
and the opt-in CUDA end-to-end suite passed.

The narrow synthetic search used 15 candidate x positions and
a 16-position field width. Times below are seconds.

| Workload | Pair | Original GPU | Packed GPU | Original command wall | Packed command wall |
|---|---:|---:|---:|---:|---:|
| one | 1 | 0.06760850 | 0.06341520 | 0.28815963 | 0.27825208 |
| one | 2 | 0.06736460 | 0.06255160 | 0.31122229 | 0.31157057 |
| one | 3 | 0.06523170 | 0.06408130 | 0.32025468 | 0.30750786 |
| one | 4 | 0.06523520 | 0.06150590 | 0.28600999 | 0.32392363 |
| one | 5 | 0.06744410 | 0.06394970 | 0.36833755 | 0.34457142 |
| one | 6 | 0.06542830 | 0.06408470 | 0.31600498 | 0.26614048 |
| one | 7 | 0.06775960 | 0.06455010 | 0.32461192 | 0.30046353 |
| one | 8 | 0.06774600 | 0.06356420 | 0.32440099 | 0.27172273 |
| one | 9 | 0.06528410 | 0.06156150 | 0.26963691 | 0.34921611 |
| all | 1 | 0.18393500 | 0.17782600 | 0.45473356 | 0.46636447 |
| all | 2 | 0.18268400 | 0.18061800 | 0.42442866 | 0.39473750 |
| all | 3 | 0.18315300 | 0.18128500 | 0.40514053 | 0.43140617 |
| all | 4 | 0.18488900 | 0.18139700 | 0.40308471 | 0.43681154 |
| all | 5 | 0.18160900 | 0.17929700 | 0.40642563 | 0.44280678 |
| all | 6 | 0.18408200 | 0.17962800 | 0.39770625 | 0.41264501 |
| all | 7 | 0.18298000 | 0.17933300 | 0.44953208 | 0.41704018 |
| all | 8 | 0.18107800 | 0.18348800 | 0.39048001 | 0.45351359 |
| all | 9 | 0.18243300 | 0.17721400 | 0.45136201 | 0.39066085 |
| narrow | 1 | 0.00050051 | 0.00050995 | 0.26203111 | 0.19510098 |
| narrow | 2 | 0.00051101 | 0.00106346 | 0.20629236 | 0.24340752 |
| narrow | 3 | 0.00051299 | 0.00051421 | 0.20666104 | 0.26360877 |
| narrow | 4 | 0.00049942 | 0.00050762 | 0.20551112 | 0.19650735 |
| narrow | 5 | 0.00049955 | 0.00050381 | 0.22571771 | 0.19607890 |
| narrow | 6 | 0.00048858 | 0.00051459 | 0.19697191 | 0.25735784 |
| narrow | 7 | 0.00052227 | 0.00051264 | 0.25877169 | 0.19842438 |
| narrow | 8 | 0.00050640 | 0.00051094 | 0.23476314 | 0.19952834 |
| narrow | 9 | 0.00049421 | 0.00134726 | 0.19616039 | 0.27076717 |

| Workload | Original GPU median | Packed GPU median | Original wall median | Packed wall median |
|---|---:|---:|---:|---:|
| One direction | 0.06736460 | 0.06356420 | 0.31600498 | 0.30750786 |
| All directions | 0.18298000 | 0.17962800 | 0.40642563 | 0.43140617 |
| Narrow | 0.00050051 | 0.00051264 | 0.20666104 | 0.19952834 |

GPU time fell by 3.80 ms for one direction and 3.35 ms for all
directions. The narrow GPU median increased by 12 microseconds.
Whole-command timing ranges overlapped in all three workloads.
The all-direction wall median increased in these samples.

| Source | SHA-256 |
|---|---|
| `src/kernel.cu` before | `c886f516c822c130b48fd388759200b9f5f819a2658d31c78adf015cc3100a40` |
| `src/kernel.cu` after | `8c17d2e644395825ae025a2fd8c0f3899195715b785591516fd0184a07907a88` |
| `include/cuda_bitplanes.cuh` | `ff99d56b1b94d62f33efb17a64634c4a262e962a4d2429a3b823c270510246a7` |

### CUDA command startup and teardown on Vulcan (job 2328111)

Vulcan job `2328111` completed in 21 s on `rack02-15` with exit
code `0:0`. It used one L40S, driver 595.112.07, CUDA 12.9.86
and an `sm_89` build. Temporary instrumentation logged monotonic
timestamps to stderr. The parent timed each command from just
before spawning through output collection and process exit.
After warmup, nine alternating traced and untraced pairs ran
for each workload. Both used the opt-in CUDA bitplane path and
returned the exact standard-kernel results. The reference used
one direction. The tiny workload had 855 candidates and a
synthetic ten-block top-face formation.

| Median interval (ms) | Reference | Tiny |
|---|---:|---:|
| Parent launch to main entry | 3.943 | 3.926 |
| Main entry to first CUDA call | 0.297 | 0.255 |
| First CUDA call, cudaMemGetInfo | 156.788 | 144.737 |
| First CUDA return to allocations ready | 0.315 | 0.240 |
| Allocations ready to first fill launch enqueued | 0.389 | 0.389 |
| First fill launch enqueued to last result copy | 64.412 | 0.064 |
| Last result copy to buffer release | 0.756 | 0.373 |
| Buffer release to end of main | 0.041 | 0.032 |
| End of main to parent return | 62.161 | 57.573 |
| Traced CUDA event time | 63.854 | 0.409 |
| Traced whole command | 325.164 | 207.683 |
| Untraced CUDA event time | 64.050 | 0.407 |
| Untraced whole command | 320.807 | 205.505 |

The launch marker follows the asynchronous launch call, so CUDA event
time can include work before that marker. The search interval includes
all remaining slices and the last result copy. The final interval
includes runtime shutdown, OS process exit and parent collection.
The run did not isolate those costs. Interval medians are calculated
independently and can differ from a sum of medians. Reusing one
process may amortize the first CUDA call and final exit. This run
did not measure repeated searches in one process.

Traced samples contain all columns. Untraced rows contain GPU and
wall time after the first three columns.

```csv
workload,sample,trace,launch_main,main_to_cuda,first_cuda_call,allocations,first_launch,search_to_last_copy,last_release_exit,exit_to_parent,gpu,wall,last_copy_to_release
reference,1,1,3.871489,0.250449,145.315898,0.315234,0.383818,64.402872,0.041350,68.821749,63.853200,284.153834,0.750975
reference,1,0,63.614000,325.967551
reference,2,0,63.754200,322.009119
reference,2,1,3.896918,0.278246,156.787579,0.301604,0.374513,65.164263,0.034403,61.470857,64.608100,289.051892,0.743509
reference,3,1,3.954023,0.312672,144.612764,0.308635,0.389268,64.412309,0.041900,119.042782,63.862800,338.090422,5.016069
reference,3,0,61.830400,272.055534
reference,4,0,65.038700,320.807209
reference,4,1,3.596981,0.303644,216.691483,0.456808,0.461024,64.186239,0.047874,68.958722,63.404200,355.476993,0.774218
reference,5,1,4.345033,0.302572,247.531003,0.341621,0.425442,67.989091,0.045075,62.160820,67.411700,383.897375,0.756718
reference,5,0,63.958800,341.179801
reference,6,0,64.050300,280.016452
reference,6,1,3.728812,0.297219,150.797904,0.307186,0.386650,64.396128,0.038937,109.096857,63.854400,332.190456,3.140763
reference,7,1,4.066872,0.479197,159.134997,0.311696,0.392640,64.642266,0.042580,61.850823,64.106400,291.664507,0.743436
reference,7,0,65.790300,295.567020
reference,8,0,64.312100,276.351836
reference,8,1,3.942500,0.256026,186.613814,3.621105,1.105453,67.533791,0.041442,61.319098,62.521800,325.163535,0.730306
reference,9,1,3.962295,0.289313,144.855669,0.317393,0.381457,63.738388,0.039537,61.355298,63.189400,275.695374,0.756024
reference,9,0,64.104600,357.935516
tiny,1,1,3.638687,0.239789,142.302156,0.241017,0.385421,0.063508,0.027732,58.370849,0.406720,205.642421,0.373262
tiny,1,0,0.402720,260.863425
tiny,2,0,0.403552,204.109571
tiny,2,1,3.926477,0.281994,165.506686,0.234224,1.746558,0.065984,0.033014,82.934072,1.767490,257.830706,3.101697
tiny,3,1,3.969979,0.236724,142.876121,0.233391,0.383698,0.061970,0.029246,57.327514,0.404480,205.499803,0.381160
tiny,3,0,0.401312,209.196781
tiny,4,0,0.402368,270.778988
tiny,4,1,4.053273,0.294348,144.671021,0.232836,0.389780,0.063387,0.032846,57.572657,0.408992,207.682885,0.372737
tiny,5,1,4.003274,0.262843,145.446129,3.784607,0.474093,0.063815,0.027871,114.077748,0.492736,270.571449,2.431069
tiny,5,0,0.410944,204.807463
tiny,6,0,0.414560,202.042666
tiny,6,1,3.837324,0.258235,215.144126,0.242753,0.396487,0.064750,0.032174,58.025615,0.415840,278.373680,0.372216
tiny,7,1,3.907487,0.255329,144.736706,0.244583,0.386817,0.063677,0.034266,57.487826,0.405824,207.491203,0.374512
tiny,7,0,0.413120,290.070359
tiny,8,0,0.407360,205.504709
tiny,8,1,3.942102,0.242530,142.302388,0.239966,0.385447,0.063014,0.029367,56.138978,0.407264,203.715221,0.371429
tiny,9,1,3.728034,0.243470,206.706043,0.232508,0.389240,0.063366,0.033043,56.770098,0.408736,268.537107,0.371305
tiny,9,0,0.421152,204.305594
```

| Temporary source | SHA-256 |
|---|---|
| `src/main.cpp` | `3c0f664773757979b87af873a30fd9d70a779429f527badcada2317058694429` |
| `src/kernel.cu` | `37f6ffaae852c3ea6081686f1af839ba3b8551285243e437db3b4802404167f7` |
| `src/startup-trace.h` | `d5822e2018afa5ddcb02b85cad8d80815d3950078671b56f4b4351692300fde6` |
| `measure.py` | `8aa04033bd6b7c2b5c8beb80a27f3da0e11730cbe926fe3a4381c91321696cfd` |

### Stand-in fixture for the diagonal lattice (Vulcan job 2328173)

Job `2328173` completed in 368 s with exit code `0:0` on
`rack01-16`, using one L40S, driver 595.112.07, CUDA 12.9.86 and
`sm_89`. The standalone harness used the current stride 5
z-row filter, precomputed offsets, and 32-bit fill packing for
both fixtures. A private copy of the fixture moved one top block
from the six-check class to the three-check class. This produced
the proposed diagonal lattice's multiset of check counts while
preserving the fill geometry and the reference match. The
diagonal read pattern was outside the scope of this test.

Synthetic cases and one full slab passed the exact CPU mask
comparison. The direct CPU count agreed with the GPU for every
class on the full workload. Both fixtures returned the standard
kernel's single exact match in all nine alternating timed pairs.

| Fixture | Survivors by class | Total survivors | CPU verify hashes |
|---|---|---:|---:|
| Original | 6,099,175 / 24,417,246 / 24,412,805 / 24,406,584 / 390,623,920 | 469,959,730 | 626,619,480 |
| Stand-in | 24,405,704 / 24,417,246 / 24,412,805 / 24,406,584 / 97,656,605 | 195,298,944 | 260,407,616 |

Times are CUDA event milliseconds. Verification is estimated by
subtracting the filter-only match kernel from the full match
kernel. Both fixtures used the same pair of compiled kernels.

| Pair | Original fill | Original filter | Original verify estimate | Original total | Stand-in fill | Stand-in filter | Stand-in verify estimate | Stand-in total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31.293631 | 29.145056 | 10.655107 | 71.093796 | 32.022434 | 29.306498 | 8.850882 | 70.179810 |
| 2 | 31.337921 | 29.232256 | 10.631517 | 71.201691 | 31.657024 | 29.303101 | 8.477093 | 69.437218 |
| 3 | 31.430464 | 29.126270 | 10.849823 | 71.406555 | 33.477215 | 29.440289 | 10.600483 | 73.517990 |
| 4 | 33.125084 | 31.550270 | 10.485825 | 75.161179 | 34.236897 | 31.809120 | 8.924572 | 74.970589 |
| 5 | 33.247555 | 29.848194 | 12.260250 | 75.356003 | 32.126400 | 29.554178 | 8.769819 | 70.450394 |
| 6 | 31.675806 | 29.574333 | 10.708103 | 71.958244 | 32.170082 | 29.734783 | 8.632706 | 70.537567 |
| 7 | 31.766466 | 29.316866 | 11.089727 | 72.173058 | 34.129726 | 30.497440 | 10.128254 | 74.755417 |
| 8 | 32.768990 | 30.475552 | 11.100704 | 74.345245 | 34.180195 | 31.307713 | 9.318336 | 74.806244 |
| 9 | 32.129025 | 30.158945 | 10.609087 | 72.897057 | 32.463551 | 29.828672 | 8.861090 | 71.153313 |
| Median | 31.766466 | 29.574333 | 10.708103 | 72.173058 | 32.463551 | 29.734783 | 8.861090 | 71.153313 |

The median paired total saving was 0.91 ms. Verification fell
from 10.71 to 8.86 ms despite a roughly 58% drop in survivors
and CPU verification hashes. Measured fill times also varied despite
identical geometry. The total saving fell below the proposed
2.5 ms L40S gate, so the diagonal implementation was skipped. The
result supports a weaker relationship between survivor count and
verification time on this workload. It does not validate the warp
step model or measure the diagonal layout itself.

| Source | SHA-256 |
|---|---|
| Temporary standalone harness | `bcb8298dcd422b8c1e818e6e7671c25743770a09bbf4a3fab92e54a0489e255d` |
| `src/kernel.cu` | `8c17d2e644395825ae025a2fd8c0f3899195715b785591516fd0184a07907a88` |
| `include/cuda_bitplanes.cuh` | `ff99d56b1b94d62f33efb17a64634c4a262e962a4d2429a3b823c270510246a7` |

### Reorganised source tree on Narval A100 (job 9435675)

First CUDA build and run of the reorganised tree (`src/backends/`,
`src/search/`, `src/texture/`). The source was commit `7bcdc69` plus the
then-untracked `test/validate_cuda.sh`, which ran the whole job. Narval job
`9435675` ran for 5 min 0 s on `ng10906` with one A100-SXM4-40GB, driver
580.178.04 and CUDA 12.9 (`cuda/12.9`, `gcc/12.3`) targeting `sm_80`.

The job exited `1:0` because one of 21 gates failed. That gate checked that
the binary reports the CUDA backend, and the failure was in the script: it
compared the first output line, which is the formation parser's message,
instead of the `Backend:` line. The script now selects the `Backend:` line.
That fix has been checked on a CPU binary only, and the job was not rerun.
The CLI was built by `nvcc` and printed the CUDA bitplane traces, so the
backend itself was the CUDA one.

The other 20 gates passed:

- The CLI, unit tests (12 cases, 249,765 assertions), tools and the
  standalone harness built and passed.
- `test/e2e.sh` passed with bitplanes off and with `TF_CUDA_BITPLANES=1`.
- On the reference workload, one direction used bitplanes with no
  fallback, all mode reported one shared field for four directions, and
  both returned the fixture's single match, as did the standard kernel.
- A 1-byte plane budget fell back and returned the match.
- A checkpoint written after one slice in all mode with bitplanes on
  resumed with bitplanes off, returned the match and removed the file.
- The harness default run passed its synthetic mask checks and the full
  GPU/CPU survivor counts: 6,099,175 / 24,417,246 / 24,412,805 /
  24,406,584 / 390,623,920.

CLI kernel time on the reference workload, in seconds, after one warmup
per mode, as five alternating pairs:

| Workload | Pair | Standard | Bitplanes |
|---|---:|---:|---:|
| one direction | 1 | 0.681705 | 0.126648 |
| one direction | 2 | 0.681832 | 0.126540 |
| one direction | 3 | 0.681802 | 0.126433 |
| one direction | 4 | 0.681695 | 0.126436 |
| one direction | 5 | 0.681673 | 0.126439 |
| all directions | 1 | 2.725250 | 0.339550 |
| all directions | 2 | 2.725420 | 0.338875 |
| all directions | 3 | 2.725390 | 0.338839 |
| all directions | 4 | 2.725270 | 0.338717 |
| all directions | 5 | 2.725240 | 0.338794 |

Medians were 0.126439 s for one direction and 0.338839 s for all
directions with bitplanes, against 0.681705 s and 2.725270 s for the
standard kernel. These are the first A100 CLI timings with 32-bit fill
packing and with the shared field. The earlier A100 CLI figures, before
either change, were 0.13834 s and 0.55351 s (job 9294135). The standard
kernel matches its earlier 0.6814 to 0.6817 s.

The harness default run reported nine samples with fill at 67.72 to
67.74 ms and match plus verification at 97.86 to 97.88 ms. The default
run uses the harness's original index calculation and 64-bit packing, and
has no earlier stride 5 record to compare against.

| Source | SHA-256 |
|---|---|
| `src/backends/cuda/search.cu` | `a535155bef8e3d15ddea86fa4018db453db11377304fc32d5f8a3408dbbefbcc` |
| `src/backends/cuda/bitplanes.cuh` | `a06dffab1a7087408c3a793d2b344fb1fdb6e75895289484947f3490461dc6b9` |
| `src/search/session.cpp` | `f983c5852385704a3ddaf32255f35bd5f8cd1f51d2eefb824dff466936527146` |
| `src/search/checkpoint.cpp` | `18aba30bd57dd91137809affcbfc1c3a3a9ef50ddc2945a8a6464d540c215b88` |
| `src/search/backend.hpp` | `84fdd3f0f28e032a4c069bd874b11afa562cd89287f1ebc01aafe044e6bf45e6` |
| `experiments/rng-reduction/flat_cuda.cu` | `b6b29d65b8cc57e85f5952f4467f9addb98d1bed546653e3fa7dc23e3cb74c24` |

### Restyled source on Narval A100 (job 9436591)

The same script ran against the restyled tree: commit `7bcdc69` plus an
uncommitted change that reformats the sources to 2-space style and renames
identifiers to snake_case, with no logic changes. Narval job `9436591`
completed in 4 min 54 s with exit code `0:0` on `ng31005`, using one
A100-SXM4-40GB, driver 580.178.04 and CUDA 12.9 targeting `sm_80`.

All 21 gates passed, including the backend check that failed in job
`9435675` through the script's own error. The harness reproduced the same
five survivor counts.

CLI kernel time on the reference workload, in seconds:

| Workload | Pair | Standard | Bitplanes |
|---|---:|---:|---:|
| one direction | 1 | 0.681568 | 0.126483 |
| one direction | 2 | 0.681578 | 0.126461 |
| one direction | 3 | 0.681579 | 0.126469 |
| one direction | 4 | 0.681517 | 0.126468 |
| one direction | 5 | 0.681572 | 0.126467 |
| all directions | 1 | 2.725620 | 0.338755 |
| all directions | 2 | 2.725620 | 0.338762 |
| all directions | 3 | 2.725720 | 0.338747 |
| all directions | 4 | 2.725540 | 0.338773 |
| all directions | 5 | 2.725870 | 0.338748 |

Bitplane medians were 0.126468 s and 0.338755 s, against 0.126439 s and
0.338839 s before the restyle.

| Source | SHA-256 |
|---|---|
| `src/backends/cuda/search.cu` | `35173d4dcabb2544cea0abc287d772e4a042841b5d30dc7ebde23f094e0b1d6b` |
| `src/backends/cuda/bitplanes.cuh` | `0a2af73983360f55691a1227ac93a9fe6fc3fefd2ae7656573360bc19eb44709` |
| `experiments/rng-reduction/flat_cuda.cu` | `7a2a3a75d25329d683cd8869c56d3fc0f9026ad448e8f4249c0a8b9499ed2945` |

### Host layer rewrite on Narval A100 (job 9446997)

The CLI, session, checkpoint, progress line and CPU backend were rewritten in
C++20, and the backend interface now takes one `SearchFormation` and reports
`kernel_seconds`. The CUDA sources changed only where they meet that
interface. `test/validate_cuda.sh` ran on the uncommitted tree: Slurm job
`9446997` completed in 5 min 0 s with exit code `0:0` on `ng30711`, with
`gcc/12.3`, CUDA 12.9 and `-std=c++20` targeting `sm_80`.

All 21 gates passed. The slice-first checkpoint written with bitplanes on
resumed with bitplanes off, which is the one session path that cannot run on
the CPU or Metal backends.

CLI kernel time on the reference workload, in seconds:

| Workload | Pair | Standard | Bitplanes |
|---|---:|---:|---:|
| one direction | 1 | 0.681478 | 0.126446 |
| one direction | 2 | 0.681467 | 0.126441 |
| one direction | 3 | 0.681479 | 0.126448 |
| one direction | 4 | 0.681456 | 0.126423 |
| one direction | 5 | 0.681465 | 0.126425 |
| all directions | 1 | 2.725430 | 0.338642 |
| all directions | 2 | 2.725630 | 0.338637 |
| all directions | 3 | 2.725540 | 0.338925 |
| all directions | 4 | 2.725340 | 0.338644 |
| all directions | 5 | 2.725430 | 0.338806 |

Bitplane medians were 0.126441 s and 0.338644 s, against 0.126468 s and
0.338755 s in job `9436591`.

| Source | SHA-256 |
|---|---|
| `src/backends/cuda/search.cu` | `bba95149d223aa68f157e6d69ffff25fe33c8b61811773abd0be5b08de891aea` |
| `src/backends/cuda/bitplanes.cuh` | `0a2af73983360f55691a1227ac93a9fe6fc3fefd2ae7656573360bc19eb44709` |
| `experiments/rng-reduction/flat_cuda.cu` | `c9d14beee37789cb16b885888cf196ddb6955e195facb2058c9ada28ba2d45a7` |

### CUDA backend split on Narval A100 (jobs 9453126 and 9453658)

The CUDA backend was split into scoped device owners (`device.hpp`,
`match_buffer.hpp`), a host-only planner with unit tests
(`bitplane_plan.hpp`), and the bitplane executor (`bitplane_search.cu`). The
kernels were not changed. Job `9453126` ran `test/validate_cuda.sh` on that
tree on `ng31006`. Job `9453658` ran it on `ng31302` after a second pass that
gives backends one block list per direction and makes a failed GPU call throw.
Both used one A100-SXM4-40GB, `gcc/12.3`, CUDA 12.9 and `-std=c++20`
targeting `sm_80`, and both passed all 21 gates.

Median CLI kernel time over five alternating pairs, in seconds:

| Job | Workload | Standard | Bitplanes |
|---|---|---:|---:|
| 9453126 | one direction | 0.682276 | 0.126245 |
| 9453126 | all directions | 2.727830 | 0.338771 |
| 9453658 | one direction | 0.681322 | 0.126140 |
| 9453658 | all directions | 2.724870 | 0.338569 |

Bitplane medians in job `9446997`, before the split, were 0.126441 s and
0.338644 s.

| Source in job 9453658 | SHA-256 |
|---|---|
| `src/backends/cuda/search.cu` | `04d996bbbe1873969af41260dfe9c74f205a3a9284867a024d2423fe57f461e2` |
| `src/backends/cuda/bitplanes.cuh` | `2415cf8c52d1ac0b8558106fe36536412be0fda9f994634b8a1f6db73b855fd1` |
| `experiments/rng-reduction/flat_cuda.cu` | `3072d8c6f6a4cddec48efda1932a7051bfc209f0732aa53df538798cd10c0e9b` |

### Version enum on Narval A100 (job 9454100)

The version became `enum class Version`, and the CPU path, the standard CUDA
kernel and the Metal shader now share one dispatch, `texture_from_mix`, in
place of three separate chains. The standard kernel's inner loop calls it with
its template constant. `test/validate_cuda.sh` ran on the uncommitted tree on
`ng31205` with one A100-SXM4-40GB, `gcc/12.3`, CUDA 12.9 and `-std=c++20`
targeting `sm_80`. All 21 gates passed.

Median CLI kernel time over five alternating pairs, in seconds:

| Workload | Standard | Bitplanes |
|---|---:|---:|
| one direction | 0.682195 | 0.126295 |
| all directions | 2.728590 | 0.339082 |

Standard one-direction medians in the two previous jobs, on other nodes, were
0.682276 s (`9453126`) and 0.681322 s (`9453658`), so the shared dispatch did
not measurably change the standard kernel.

| Source | SHA-256 |
|---|---|
| `src/backends/cuda/search.cu` | `3b2838e24a3023d223c5a420eeb8fafb99bc1c932e5423f18370922d703087b5` |
| `src/backends/cuda/bitplanes.cuh` | `2415cf8c52d1ac0b8558106fe36536412be0fda9f994634b8a1f6db73b855fd1` |
| `experiments/rng-reduction/flat_cuda.cu` | `74c524540d5bcf2f800b8f78689b28a180adeca18da948384cb8372b578f109c` |

### One bitplane implementation and bitplanes by default (jobs 9462804 and 9463175)

The diagnostic match modes and the unpacked fill moved out of
`src/backends/cuda/bitplanes.cuh` into
`experiments/rng-reduction/diagnostic_kernels.cuh`. The production header now
holds one fill kernel, the filter (`filter_word`), the survivor check
(`verify_survivors`) and one match kernel, without mode templates. The
precomputed diagnostic variants call the production filter and survivor
check. The harness builds its plan and slabs from `plan_search` and
`SearchPlan::slab`, uses the shared `cuda::DeviceArray`, `cuda::check` and
`cuda::Timer`, and compares the backend's own `cuda_bitplanes::search` with
the standard kernel in its default, small-case and shared-field runs.

In the same tree, version 0 searches on CUDA use bitplanes unless
`TF_CUDA_BITPLANES=0`. `rotate_formation` became `rotate_block`, with one
`rotated(formation, quarter_turns)` in `src/texture/rotation.hpp`. The CI
smoke run reads its bounds from the fixture's expected-match file.
`test/validate_cuda.sh` gained a gate that `TF_CUDA_BITPLANES=0` keeps the
standard kernel and an opt-in `HARNESS_MODES` switch that runs the harness
experiment modes.

Job `9462804` on `ng10803` ran the first half of this (kernel move and
planner) and failed one gate: the harness did not link, because the
production kernels had become non-template functions in a header that two
translation units compile. Its other 20 gates passed, with one-direction
medians of 0.681990 s standard and 0.125203 s bitplanes. The kernels are now
`static`.

Job `9463175` on `ng31302` ran the whole tree, uncommitted on top of
`1d2b9e6`, with one A100-SXM4-40GB, driver 580.178.04, `gcc/12.3`, CUDA 12.9
and `sm_80`. All 29 gates passed: the 21 build and CLI gates, the harness
default run, and the `DUPLICATE_Y`, `COMPARE_OFFSETS`, `COMPARE_PACKING`,
`COMPARE_PACKING_SEARCH`, `SHARED_FIELD`, `FUSED` and `DIAGNOSTICS` modes.

Median CLI kernel time over five alternating pairs, in seconds:

| Workload | Standard (`TF_CUDA_BITPLANES=0`) | Bitplanes (default) |
|---|---:|---:|
| one direction | 0.681605 | 0.125157 |
| all directions | 2.724700 | 0.334556 |

Bitplane medians in job `9454100`, before the kernels were slimmed, were
0.126295 s and 0.339082 s on `ng31205`. The two jobs ran on different nodes,
so the 1 to 5 ms difference is not attributed to the change.

The harness modes also give the current stage costs on W for one direction.
These are medians of nine CUDA event samples in milliseconds, from separate
modes of the same job:

| Stage | Median (ms) | Source |
|---|---:|---|
| Fill, one bit per iteration | 67.714 | `PACK_SEARCH` |
| Fill, 32-bit packing (production) | 55.828 | `PACK_SEARCH` |
| Filter alone, precomputed offsets | 50.796 | `OFFSET_PAIR` |
| Filter and survivor check (production) | 72.533 | `OFFSET_PAIR` |
| Four directions, matching only, shared field | 291.655 | `SHARED_PAIR` |

The filter-alone figure comes from the `NoTail` variant, which is compiled
separately from the production match, so the 21.7 ms difference estimates
the survivor check and is not a direct measurement of it. Survivors per
direction on W were 469,959,730 / 524,915,462 / 469,968,620 / 524,899,371.
In each direction one z coset holds 390.6 million of them.

| Source | SHA-256 |
|---|---|
| `src/backends/cuda/search.cu` | `095179f2cf3ac7253918bd83ba59ce54e212ef3562b442f7161ea3294a677755` |
| `src/backends/cuda/bitplanes.cuh` | `cf46089821ab39c5300fcc43e8a2ec46c4b4f0631b5bec5b5afd9ae69af5951d` |
| `experiments/rng-reduction/flat_cuda.cu` | `96a0b3c691e4a91d6ab050e354dc35e4e2e14467194cb0c57419ca831e65a9d5` |

### Long-lived match threads (job 9463518)

Tested whether the match is limited by thread launch overhead. The
production match gives each thread one 64-candidate word.
`match_word_columns` in the diagnostic header gives each thread one word
column through `rows_per_thread` consecutive (y, z) rows, calling the same
`filter_word` and `verify_survivors`. `TF_FLAT_WORD_COLUMNS=1` in the harness
checks it against the standard kernel on synthetic slabs with chunks that
straddle slab rows and y levels, then times it on W.

Narval job `9463518` ran on `ng31401` with one A100-SXM4-40GB, `gcc/12.3`,
CUDA 12.9 and `sm_80`, on the uncommitted tree later committed as `817dd44`
plus this experiment. All 23 gates passed, and every timed run returned the
standard kernel's exact match. Each variant ran once before nine samples in
rotated order with 32-bit fill packing, precomputed offsets and 13 slabs of
up to 4,096 rows. Times are CUDA event milliseconds for matching one
direction of W.

| Rows per thread | Median (ms) | Min (ms) | Max (ms) |
|---:|---:|---:|---:|
| production | 72.532 | 72.529 | 72.547 |
| 1 | 79.974 | 79.971 | 79.989 |
| 5 | 116.819 | 116.805 | 116.960 |
| 25 | 125.918 | 125.893 | 125.969 |
| 125 | 133.787 | 133.709 | 133.923 |
| 625 | 153.561 | 152.038 | 154.336 |
| 4096 | 292.245 | 292.178 | 292.333 |

The fill median was 55.83 ms in every variant. Longer-lived threads were
slower at every length, so launch overhead is not what limits the match and
this design is ruled out. The one-row column kernel does the same work as
production and was 7.4 ms slower; the experiment does not isolate why. The
cost grew with the number of z rows a block covers at once, which spreads
the field rows that concurrently running blocks read. That is consistent
with the match being sensitive to how local its field reads are. It was not
measured directly.

| Source | SHA-256 |
|---|---|
| `src/backends/cuda/bitplanes.cuh` | `cf46089821ab39c5300fcc43e8a2ec46c4b4f0631b5bec5b5afd9ae69af5951d` |
| `experiments/rng-reduction/diagnostic_kernels.cuh` | `a77594fd2f66b4d121b6596e2e0b4282de9620d3c4700bbd30383df40ff80125` |
| `experiments/rng-reduction/flat_cuda.cu` | `b483fa8adf82bd48f0b4960882786c79ca9e964a025559d6efd383135de77883` |

### Match launch order, pooled survivor check and field budget (job 9463975)

Three follow-ups to the long-lived thread result, all on one A100-SXM4-40GB
(`ng31304`, `gcc/12.3`, CUDA 12.9, `sm_80`), on `817dd44` plus the uncommitted
experiment code. All 39 gates passed, and every timed run returned the
standard kernel's exact match.

`match_words_variant<ZMajor, Pooled>` in the diagnostic header is the
production match with two independent changes. `ZMajor` numbers the launch by
z row first and y second, so the candidate y values that read one field level
run close together in time. `Pooled` has each block list its survivors in
shared memory and work through the list with all of its threads; survivors
beyond 1,024 per block are checked by the thread that found them.
`TF_FLAT_MATCH_VARIANTS=1` checks all four combinations on synthetic slabs,
including a dense case that overflows the pool, then times them on W. Each ran
once before nine samples in rotated order. Times are CUDA event milliseconds
for matching one direction, with a fill median of 55.83 ms throughout.

| Variant | Median (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|
| production (y-major, inline check) | 72.535 | 72.531 | 72.553 |
| z-major | 68.614 | 68.611 | 68.623 |
| pooled survivor check | 71.774 | 71.772 | 71.779 |
| z-major and pooled | 71.376 | 71.375 | 71.384 |

The z-major order was 3.92 ms faster than production, 5.4% of matching.
Pooling gained 0.76 ms alone and lost 2.76 ms on top of z-major, so it is
ruled out. Its small effect means the survivor check is not dominated by
idle lanes in the way the estimate assumed.

The field budget sweep ran the CLI with `TF_CUDA_BITPLANES_BYTES` from 1 GiB
down to 4 MiB, which shortens the slabs until the whole field is smaller than
the A100's 40 MB L2 cache. Each budget passed a gate that bitplanes ran and
found the match, then ran five times. Kernel seconds, median of five:

| Budget | Slab rows (approx.) | One direction | All directions |
|---:|---:|---:|---:|
| 1 GiB (default) | 4,096 | 0.125160 | 0.334267 |
| 256 MiB | 1,985 | 0.125530 | 0.335446 |
| 128 MiB | 990 | 0.126032 | 0.337053 |
| 64 MiB | 495 | 0.127322 | 0.340708 |
| 32 MiB | 245 | 0.129694 | 0.347810 |
| 16 MiB | 120 | 0.134402 | 0.362721 |
| 8 MiB | 60 | 0.145992 | 0.399663 |
| 4 MiB | 30 | 0.172343 | 0.499130 |

Slab rows are estimated from the budget and the 675,648 bytes one hashed z
row of W takes, before the halo is subtracted. Smaller fields were slower at
every step, so keeping the field in cache does not pay for the extra halo
rows and launches. Together with the 5% from launch order, this indicates
the match is not limited by field memory traffic. What does limit it was not
measured; an instruction count is the next thing to check.

| Source | SHA-256 |
|---|---|
| `src/backends/cuda/bitplanes.cuh` | `cf46089821ab39c5300fcc43e8a2ec46c4b4f0631b5bec5b5afd9ae69af5951d` |
| `experiments/rng-reduction/flat_cuda.cu` | `44a4cadddffad10f278ec99b4c8233a2cceb70a34b36c663c2396245c15504e2` |
| `experiments/rng-reduction/diagnostic_kernels.cuh` | `d3f33b6ea1d8bb85b91194b2d6195a901d334dad4bb0ac187b2017228defb6a5` |

### Bitplane search on the CPU backend (M4, local)

The planner and field layout moved from `src/backends/cuda/` to
`src/bitplanes/`, and `src/bitplanes/match.hpp` now holds the three steps
every backend shares: `hash_word` (the fill of one field word), `filter_word`
and `survivor_matches`. They take the version as a parameter. The CUDA
kernels call them with the modern version, and the unit suite runs them on
the host against brute force for all five versions.

The CPU backend gained `src/backends/cpu/bitplane_search.cpp`. Each thread
takes tiles of 4,096 x by 32 y by 64 z candidates from an atomic cursor, plans
the tile with the shared planner, fills a thread-local field and matches
every requested direction against it. It runs for every version when the
formation is eligible, and `TF_CPU_BITPLANES=0` forces the plain scan, which
is unchanged and stays the reference.

Measured on M4 (Apple M4 Pro, 14 threads) on W, kernel seconds:

| Path | One direction | All directions |
|---|---:|---:|
| Plain scan | 62.3485 (one run, at `1d2b9e6`) | not run |
| Bitplanes | 4.688 / 4.745 / 4.604 | 11.697 / 12.782 |

Bitplanes were 13.3 times faster than the plain scan on one direction, using
the median of the three runs. Metal on the same machine took 6.189 to
6.209 s for one direction at `1d2b9e6`.

Tile size sweep on W, one direction, three runs each in seconds:

| x | y | z | Runs |
|---:|---:|---:|---|
| 4096 | 32 | 64 | 4.665 / 4.669 / 4.665 |
| 1024 | 32 | 64 | 4.963 / 5.512 / 5.455 |
| 2048 | 32 | 64 | 4.796 / 4.847 / 4.861 |
| 8192 | 32 | 64 | 4.827 / 4.783 / 4.793 |
| 16384 | 32 | 64 | 4.838 / 4.883 / 4.902 |
| 4096 | 32 | 32 | 4.972 / 5.015 / 5.089 |
| 4096 | 32 | 128 | 4.813 / 4.834 / 4.829 |
| 4096 | 32 | 256 | 4.657 / 4.697 / 4.733 |
| 4096 | 8 | 64 | 5.293 / 5.356 / 5.402 |
| 4096 | 16 | 64 | 5.367 / 5.361 / 5.397 |

W is 25 levels tall, so a y tile of 32 is one tile. The default stays at
4096 by 32 by 64.

Correctness checks, all on the CPU backend:

- `make test`: 20 cases, 787,050 assertions.
- `test/e2e.sh` in full, and with `FAST=1` under `TF_CPU_BITPLANES=0`.
- A differential of 150 random formations (versions 0 to 4, 9 to 26 blocks,
  0 to 70% side faces, volumes spanning several tiles) searched in their own
  direction and with `all`, bitplanes against the plain scan: 300 runs, 107
  of them on the bitplane path, 128 with more than one match, no difference.
- 1,000 runs of the e2e synthetic round-trip with random origins. Two runs
  reported a second match besides the planted one, for the seed-16 formation
  (version 3, 15 side faces, 33 filter bits). The plain scan reported the
  same two matches, and so did the committed `817dd44`. The e2e check for a
  unique match can therefore fail on a real second match; one full e2e run in
  eight did during this work.

### Shared bitplane code on Narval A100 (job 9464541)

`test/validate_cuda.sh` ran with `HARNESS_MODES=all` on the tree described
in the previous entry, uncommitted on top of `817dd44`, on `ng10203` with one
A100-SXM4-40GB, `gcc/12.3`, CUDA 12.9 and `sm_80`. All 31 gates passed: the
21 build and CLI gates, the harness default run and its nine experiment modes.

Median CLI kernel time over five alternating pairs, in seconds:

| Workload | Standard (`TF_CUDA_BITPLANES=0`) | Bitplanes (default) |
|---|---:|---:|
| one direction | 0.681709 | 0.125219 |
| all directions | 2.725940 | 0.334658 |

Job `9463175` measured 0.125157 s and 0.334556 s for bitplanes before the
kernels called the shared `hash_word`, `filter_word` and `survivor_matches`,
so moving them into `src/bitplanes/match.hpp` did not measurably change the
CUDA path. The match variants repeated their earlier medians within 0.01 ms:
72.545 ms production, 68.622 ms z-major, 71.784 ms pooled, 71.389 ms both.

| Source | SHA-256 |
|---|---|
| `src/backends/cuda/search.cu` | `9d772485176522fb06fd259c0c9d68dcf064c7d6bfc2fdd75d6efa81e21f7cb2` |
| `src/backends/cuda/bitplanes.cuh` | `25db2273499b24e4d6980e0bf81a3cbb951cdd6a081bd2d901f3de7da588aed2` |
| `experiments/rng-reduction/flat_cuda.cu` | `129a170f7022cd72dff36a81f653cdb54166c4129480a3ba65ce08a8ece0d508` |
| `experiments/rng-reduction/diagnostic_kernels.cuh` | `d3f33b6ea1d8bb85b91194b2d6195a901d334dad4bb0ac187b2017228defb6a5` |

### Bitplane search on the Metal backend (M4, local)

Uncommitted on top of `6afb58d`, on an Apple M4 Pro with the 20-core GPU and
macOS 26.5.2.

The Metal shader is now `texture_rules.hpp`, `match.metal`,
`bitplanes/layout.hpp`, `bitplanes/match.hpp` and the new `bitplanes.metal`
pasted together, so the Metal kernels call the same `filter_word` and
`survivor_matches` as CUDA and the CPU. To compile as Metal Shading Language
the shared headers name the address space of constants and buffer pointers
through `TF_CONSTANT` and `TF_BUFFER`, take `Layout` by value, and no longer
use `half` as a variable name, which is a type there. The fill's index
arithmetic moved into a shared `field_word`, which the CUDA kernel, the CPU
fill and the unit test now call too. The duplicated `run_shared_search` body
of the backends moved to `bitplanes/shared_search.hpp`.

`src/backends/metal/bitplane_search.mm` runs one command buffer per slab: the
fill, then one match per direction. A kernel counts a pass's matches in 32
bits, so the planner takes a row limit and a Metal slab holds at most
2^32 - 1 candidates. On W that is 1,717 z rows per slab. The field is one
GPU-only buffer of at most 1 GiB, kept between calls. It runs for every
version when the formation is eligible, `TF_METAL_BITPLANES=0` forces the
plain kernel, and `TF_METAL_BITPLANES_TRACE=1` reports the path.

Measured on W, kernel seconds:

| Path | One direction | All directions |
|---|---:|---:|
| Plain kernel (`TF_METAL_BITPLANES=0`) | 6.12125 / 6.12105 | 24.498 (one run) |
| Bitplanes | 0.653785 / 0.641486 / 0.641278 / 0.641454 / 0.641418 | 1.62536 / 1.62566 / 1.62572 |

By the medians, bitplanes were 9.5 times faster on one direction and 15.1
times faster on all four.

Every version, one synthetic 24-block top-face formation each in a volume of
1.25e10 positions (100,001 by 25 by 5,001), one direction, three runs each:

| Version | Plain kernel | Bitplanes | Ratio |
|---:|---|---|---:|
| 0 | 0.608037 / 0.608078 / 0.608273 | 0.087396 / 0.069947 / 0.069854 | 8.7 |
| 1 | 0.628759 / 0.628691 / 0.628685 | 0.082165 / 0.082169 / 0.082162 | 7.7 |
| 2 | 0.439465 / 0.439141 / 0.439460 | 0.055628 / 0.055629 / 0.055629 | 7.9 |
| 3 | 1.45032 / 1.45001 / 1.45030 | 0.171771 / 0.171767 / 0.171759 | 8.4 |
| 4 | 1.20932 / 1.20915 / 1.20931 | 0.132913 / 0.132910 / 0.132903 | 9.1 |

Where the time goes on W, one direction, from temporary switches that are not
in the tree:

| Run | Kernel seconds |
|---|---|
| Fill only, no match | 0.342574 / 0.329735 / 0.329953 |
| Fill and filter, survivors counted but not checked, 32-bit field index | 0.553681 / 0.540396 / 0.540428 |
| Full search, 32-bit field index | 0.620247 / 0.613414 / 0.613501 |

So the fill is about half of the search, and by subtraction the filter is
about 0.21 s and the survivor check about 0.07 s.

Tried and ruled out on W, one direction, three runs each unless noted:

| Change | Kernel seconds |
|---|---|
| Match threadgroup 32 by 1 by 8 (kept) | 0.655399 / 0.641412 / 0.641455 |
| Match threadgroup 32 by 8 by 1 | 0.643479 / 0.643521 / 0.643712 |
| Match threadgroup 64 by 1 by 4 | 0.641609 / 0.641655 / 0.641870 |
| Match threadgroup 128 by 1 by 2 | 0.641837 / 0.642090 / 0.642109 |
| Match threadgroup 256 by 1 by 1 | 0.642869 / 0.642891 / 0.642870 |
| Match threadgroup 32 by 1 by 1 | 0.642450 / 0.643704 / 0.642490 |
| Match threadgroup 32 by 1 by 32 | 0.641627 / 0.641317 / 0.641348 |
| Match threadgroup 64 by 4 by 4 | 0.642411 / 0.642287 / 0.642584 |
| Match threadgroup 512 by 1 by 1 | 0.643725 / 0.643650 / 0.643885 |
| Match threadgroup 1024 by 1 by 1 | 0.643756 / 0.643855 / 0.643707 |
| Fill threadgroup, six shapes from 32 by 1 by 1 to 32 by 32 by 1, two runs each | 0.641185 to 0.641532 |
| Match grid with z rows before y levels | 0.660816 / 0.643367 / 0.643474 |
| 32-bit divide and residue in `filter_word` | 0.659690 / 0.641924 / 0.642188 |
| 32-bit field index in `filter_word` as well | 0.620247 / 0.613414 / 0.613501 |

Threadgroup shape and grid order do not matter on this GPU. The 32-bit field
index is 4.4% faster and exact while the field has fewer than 2^32 words,
which every backend's budget already guarantees, but it is not adopted: it
changes the shared filter and has not been measured on CUDA.

A defect in the committed plain Metal kernel turned up in the differential.
Each dispatch rounds its z range up to whole threadgroups of depth 8, and
the kernel only cut at the search's last z, so the rows past the end of one
dispatch were searched again by the next. A match in such a row was reported
twice: `6afb58d` prints one synthetic version 4 match twice in a 53,057 by 92
by 2,438 volume. The dispatch now passes its own last row. W has 429 rows per
dispatch, which leaves three such rows after each one.

Correctness checks:

- `make test`: 21 cases, 787,054 assertions. The brute-force test now builds
  its field with `field_word`, and a new case covers the planner's row limit.
- `test/e2e.sh` on Metal, by default and under `TF_METAL_BITPLANES=0`.
- A differential of 90 random formations (versions 0 to 4, 9 to 26 blocks, 0
  to 70% side faces, every third in a volume wide enough for several slabs)
  searched in their own direction and with `all`, bitplanes against the plain
  kernel after the fix above: 180 runs, 64 of them on the bitplane path, 93
  with more than one match, no difference in the sorted match lines. With
  bitplanes off an `all` search prints direction by direction, so the order
  of the lines differs.
- A checkpoint written after one slice of W with `all` resumed under
  `TF_METAL_BITPLANES=0`, found the match and removed the file.
- The CPU backend after the shared-code changes: 4.696 / 4.612 / 4.610 s on W
  for one direction and 11.537 s for all, a 60-formation differential against
  the plain scan with no difference, and `FAST=1 test/e2e.sh`.

### Stride between hashed rows on Metal and the CPU (M4, local)

Run in a scratch copy of the Metal-port tree with `STRIDE` edited, on W, one
direction, kernel seconds. The machine was also running other work, so stride
5 reads a little slower here than in the entry above.

| Stride | Metal | CPU | Path |
|---:|---|---|---|
| 3 | 0.872697 / 0.891036 / 0.850124 | 7.107 / 7.190 | bitplanes |
| 4 | 0.726400 / 0.701695 / 0.700176 | 5.975 / 5.403 | bitplanes |
| 5 (kept) | 0.665640 / 0.660881 / 0.651514 | 5.014 / 4.676 | bitplanes |
| 6 | 0.693320 / 0.675013 / 0.669906 | 5.318 / 5.195 | bitplanes |
| 7 | 6.14352 / 6.29855 / 6.42290 | 64.51 / 61.64 | plain |
| 8 | 6.13444 / 6.12175 / 6.12190 | 63.38 / 64.33 | plain |
| 10 | 6.14486 / 6.12808 / 6.13167 | 63.72 / 64.20 | plain |

Stride 5 is the fastest on both backends. From stride 7 the reference
formation has a z coset with fewer than four filter bits, so the search is
not eligible and takes the plain path. A longer stride is ruled out.

CPU stage split on W at stride 5, from temporary switches in the same copy,
three runs each:

| Run | Kernel seconds |
|---|---|
| Full search | 4.674 / 4.628 / 4.875 |
| Survivors counted but not checked | 5.134 / 4.998 / 4.587 |
| Fill only, no match | 2.557 / 2.556 / 2.534 |

The fill is about 55% of the CPU search. The filter is the rest, about
2.1 s, and the survivor check is below the run-to-run noise.

### Row-at-a-time filter on the CPU backend (M4, local)

Tried in the same scratch copy, not in the tree. The CPU match called the
shared `filter_word` once per candidate word. The variant keeps one mask per
candidate word of a (y, z) row and applies each plane check of the row's
coset to the whole row in one loop, with the shift-zero case split out and
the wanted bit folded into an XOR, so each pass has no branch and no early
exit. Survivors are then checked as before.

Measured on W, kernel seconds:

| CPU filter | One direction | All directions |
|---|---|---|
| Per word (in the tree) | 4.696 / 4.612 / 4.610 | 11.537 |
| Per row | 3.786 / 3.482 / 3.538 | 6.758 |

By the medians the row filter is 1.30 times faster on one direction, and 1.71
times faster on all four from one run each. With the fill at about 2.55 s,
the filter itself went from about 2.1 s to about 1.0 s per direction.

Checks in the copy: 150 random formations searched in their own direction and
with `all` against the plain scan (300 runs, 107 on the bitplane path, 128
with more than one match) with no difference, and `FAST=1 test/e2e.sh`.

The row filter then went into the tree as
`src/backends/cpu/row_filter.hpp`, called by the CPU match. The brute-force
unit test now requires `filter_row` to equal `filter_word` for every word in
every version (`make test`: 21 cases, 790,564 assertions). In the tree it
measured 3.493 / 3.513 / 3.462 s on W for one direction and 6.558 / 6.764 s
for all four. The 150-formation differential against the plain scan again
showed no difference in 300 runs, and `FAST=1 test/e2e.sh` passed by default
and under `TF_CPU_BITPLANES=0`. The shared headers are untouched by this
step.

### CPU tile sizes with the row filter (M4, local)

W, one direction, three runs each in kernel seconds, y tile 32 throughout.
The first and last rows are the same default configuration.

| x | z | Runs |
|---:|---:|---|
| 4096 | 64 | 3.512 / 3.555 / 3.472 |
| 4096 | 128 | 3.452 / 3.656 / 3.408 |
| 4096 | 256 | 3.502 / 3.341 / 3.418 |
| 4096 | 512 | 3.377 / 3.328 / 3.298 |
| 4096 | 1024 | 3.328 / 3.356 / 3.342 |
| 8192 | 64 | 3.422 / 3.488 / 3.841 |
| 8192 | 256 | 4.172 / 3.689 / 3.796 |
| 16384 | 256 | 3.330 / 3.325 / 3.339 |
| 2048 | 256 | 3.352 / 3.334 / 3.348 |
| 2048 | 512 | 3.394 / 3.329 / 3.355 |
| 2048 | 1024 | 3.428 / 3.393 / 3.363 |
| 1024 | 512 | 3.589 / 3.680 / 3.616 |
| 4096 | 64 | 3.472 / 3.981 / 3.581 |

The best configurations are about 5% faster than the default, which is close
to the spread of the default's own six runs (3.47 to 3.98 s). A z tile of 512
also leaves a volume under 512 rows deep and 4,096 wide as a single tile for
one thread. The default stays at 4096 by 32 by 64.

### Metal-port shared code on Narval A100 (job 9511085)

`test/validate_cuda.sh` ran with `HARNESS_MODES=all` on the Metal-port tree,
uncommitted on top of `6afb58d`, on `ng10607` with one A100-SXM4-40GB,
`gcc/12.3`, CUDA 12.9 and `sm_80`. All 31 gates passed. The tree it ran did
not yet have the CPU row filter, its unit-test assertion or the `test/e2e.sh`
change, none of which the CUDA build compiles.

Median CLI kernel time over five alternating pairs, in seconds:

| Workload | Standard (`TF_CUDA_BITPLANES=0`) | Bitplanes (default) |
|---|---:|---:|
| one direction | 0.681504 | 0.125269 |
| all directions | 2.725240 | 0.334495 |

Job `9464541` measured 0.125219 s and 0.334658 s, so taking `Layout` by
value, the shared `field_word` and the shared `run_shared_search` body did not
measurably change the CUDA path. The harness repeated 72.531 ms for the
production match and 68.608 ms for z-major.

| Source | SHA-256 |
|---|---|
| `src/backends/cuda/search.cu` | `4ee232e3e678e22f99dd465ea89920c03f5b899f50a150c56bc86993d6cf2689` |
| `src/backends/cuda/bitplanes.cuh` | `df2545e8753c1e1232ec00b91237f23759ae215ff66fbe93526ed9eaa61890c0` |
| `src/bitplanes/layout.hpp` | `f85c1d7e2455f3461f369c5a5f1af16c50652741e9593b1127ecc040d83e1df0` |
| `src/bitplanes/match.hpp` | `e23c7ec2155abc3443b8e71d34986b9994476c3e6828bc50e60c6cdcc0598bcd` |
| `src/bitplanes/plan.hpp` | `9625bc8141d62d53224785396b6726af031e4334e23bacb82dffe7f2f573f107` |
| `src/bitplanes/shared_search.hpp` | `2867bdad4dba6fc497ef74f80a2523d74503c1e42f61f5a422d0c49a5ec33a60` |
| `src/bitplanes/plan.cpp` | `ec00281071ede7767d550f85857a62cd9939577bb5bd4cfa300aa48ce804444b` |
| `experiments/rng-reduction/flat_cuda.cu` | `129a170f7022cd72dff36a81f653cdb54166c4129480a3ba65ce08a8ece0d508` |
| `experiments/rng-reduction/diagnostic_kernels.cuh` | `d3f33b6ea1d8bb85b91194b2d6195a901d334dad4bb0ac187b2017228defb6a5` |

The local files have the same hashes.

### e2e synthetic round-trips checked against the CPU reference (M4, local)

The synthetic round-trips in `test/e2e.sh` required exactly one match, which
failed whenever a random formation matched at a second place. They now
require the planted origin and a match list equal to `build/cpu_search`'s.
On the Metal build, 60 consecutive runs of the suite all passed. One of them
hit the case: `synthetic v3 dir3 (all mode)` reported two matches, the same
two as the CPU reference. A run of the suite took about 19 s, up from about
6 s, because the CPU tool now searches each of the 40 round-trip volumes.
