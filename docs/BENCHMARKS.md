# Benchmarks

Canonical performance log. Every entry is the **same workload** on the
**same machine**, so numbers are directly comparable across implementations
and iterations. Append a row (and an explanation below) for every future
iteration; never edit historical rows.

## Workload

Search a 100,001 × 25 × 50,001 volume — x ∈ [-175000, -75000],
y ∈ [-64, -40], z ∈ [-75000, -25000], **1.25 × 10¹¹ candidate origins** —
for a real 24-block top-face formation
([test/fixtures/formation_a.txt](../test/fixtures/formation_a.txt)),
Minecraft 1.21.2+ rotation rules. Every implementation must report exactly
one match, at the fixture's verified coordinate. Timing = wall clock, best
of 3 after one warmup (kernel-only time additionally reported for CUDA,
measured with CUDA events).

## Machine

- GPU: NVIDIA GeForce RTX 3090 (GA102, 82 SMs, 24 GB), CUDA 12.8, `-O3 -arch=native`
- CPU: 2 × AMD EPYC 7H12 (128 cores / 256 threads total), 1 TB RAM —
  **container cgroup CPU quota: 27.2 cores** (`cpu.max 2720000/100000`),
  which caps every CPU-side number below; CPU-seconds are reported so
  results can be projected onto unrestricted hardware
- OS: Ubuntu 22.04, OpenJDK 17 for the Java runs
- Rows 1-6: instance with driver 580.159.03. Row 7: a replacement instance,
  same GPU model (driver 580.126.20, CPU EPYC 7763, same 27.2-core quota) —
  row 6's build was re-measured there first and reproduced its numbers
  within noise (0.821-0.824 s vs 0.822-0.826 s kernel), so rows are
  comparable across the swap.

## Results

| # | Implementation | Config | Wall | vs Java (as shipped) | Notes |
|---|---|---|---|---|---|
| 1 | Java reference (19MisterX98/TextureRotations) | as shipped: 10 threads | 192.5 s (1,562 CPU-core-s) | 1× | search code unmodified; only formation/bounds/mode configured; correct unique match |
| 2 | Java reference | tuned: 256 threads (quota-capped at 27.2 cores) | 104.9 s (2,698 CPU-core-s) | 1.8× | same build, `-Dbench.threads=256`; correct unique match. Extra CPU-seconds vs #1 = contention overhead under the quota |
| 3 | CUDA original (this repo, pre-overhaul) | dim3(8,2,8), printf output | 4.63 s | 41.6× | **correctness caveats**: silently skipped the last x/z column at these exact bounds, searched nothing for single-cell ranges, sides broken for direction ≠ 0 |
| 4 | CUDA rewrite: grid-stride + result buffer | 256-thread blocks | 1.47 s (kernel 1.26 s) | 131× | first correct version; 3.15× vs #3 |
| 5 | CUDA warp compaction | chunk 256/lane | 1.40 s (kernel 1.17 s) | 138× | |
| 6 | CUDA precomputed hash terms + stripe loop | chunk 1024/lane | 1.04 s (kernel 0.83 s) | 185× | 4.45× vs original CUDA (#3) |
| 7 | CUDA unsigned tail folds + incremental refill | chunk 1024/lane | **0.94 s (kernel 0.72 s)** | **205×** | current default; 4.9× vs original CUDA (#3); legacy version identical (0.720 s kernel) |

Cross-device framing (since Java-on-CPU vs CUDA-on-GPU is the point of the
project): using the Java run's own best throughput (1,562 EPYC-core-seconds
for the workload) against the 0.72 s GPU kernel, it would take **~2,200
EPYC cores scaling perfectly** to match one RTX 3090 on this search.
Projected onto the full unrestricted 128-core machine (ideal scaling,
no quota), Java would finish in ~12 s — the single GPU is still ~17× faster
than the entire dual-socket server at its theoretical best.

## What each iteration did and why

**#1/#2 (Java reference).** The baseline everything is measured against —
the tool the community actually uses. Only its hardcoded formation, bounds,
and mode constants were edited to define the benchmark workload; the search
loops are untouched. It parallelizes by slicing the x range across threads.
Measured at the shipped thread count (10) and fully subscribed (256); the
container's 27.2-core cgroup quota caps the latter, so raw CPU-seconds are
recorded for quota-independent projection. Both runs find exactly the
verified match, which doubles as a cross-implementation correctness check.

**#3 → #4 (grid-stride rewrite, 3.15×).** The original launched one thread
per coordinate with a fixed grid whose size math was wrong at certain range
widths (positions silently skipped) and whose z-dimension would exceed the
65,535 grid-dimension limit on large searches. The rewrite made every thread
grid-stride over the volume in all three dimensions — coverage becomes
launch-geometry-independent (correctness first; the speed came along). It
also replaced in-kernel `printf` (bounded buffer, drops matches) with an
atomically-appended device result buffer, moved version dispatch from a
per-thread runtime branch to a template parameter, and switched to
256-thread blocks. A geometry sweep (block 128/256/512 × grid caps) then
showed ±1% variation → the kernel is ALU-bound, so further gains had to come
from doing less work per candidate, not from launch shapes.

**#4 → #5 (warp compaction, ~7%).** With a 1-in-4 pass rate per block check
and early exit, a warp advances at the pace of its slowest lane: ~4.3
evaluations per 32-candidate batch while the average candidate needs only
~1.33 → ~31% lane utilization. Rewrote the kernel as persistent warps: each
lane owns a candidate and pulls a fresh one the moment its candidate fails;
warps take work chunks from a global atomic cursor; within a chunk, lane L
owns linear indices `base + pull·32 + L`, so every position is enumerated
exactly once *by construction*. Measured gain (7%) was far below the naive
3× utilization model — Ampere's independent thread scheduling already hides
much of the divergence — but the structure eliminated the entire class of
launch-coverage bugs and set up iteration #6. Correctness of the enumeration
is proven by dense GPU-vs-CPU differential tests (a 1-block formation makes
~25% of all positions match; the GPU's match set must equal an independent
CPU implementation's exactly).

**#5 → #6 (precomputed hash terms + ballot-free stripes, ~29%).** Two
changes. (a) The position hash multiplies x by 3129871 (wrapping 32-bit) and
z by 116129781 (64-bit); both distribute over the formation offsets:
`(x+bx)·M ≡ x·M + bx·M`. So each block's `bx·M` / `bz·M` terms are
precomputed once on the host, and the per-evaluation multiplies become adds
(the hot loop runs one fused-multiply chain instead of three). To keep the
single-source-of-truth guarantee, `texture.cuh` was refactored into a
two-stage API (`coordRandomFromParts` + `modernFromCoordRandom` /
`legacyFromCoordRandom`) — the kernel, tests, CPU searcher, and oracle diff
still compile the identical functions, and the 29M-value Java-oracle
differential re-verified bit-exactness after the change. (b) The
per-iteration warp ballot (`__all_sync`) used to coordinate chunk refills
was restructured away: an outer uniform chunk-grab loop plus an inner
per-lane stripe loop with zero warp coordination. Register pressure stayed
at 32 (full occupancy). A chunk-size re-sweep moved the default to 1024
positions per lane (64: 0.94 s, 256: 0.86 s, 1024: 0.82 s). Remaining
headroom at this algorithm is estimated ≤10%: the kernel now issues ~44
instruction slots per evaluation against ~35 of irreducible hash math.

Full decision trail (including rejected ideas and why — shared-memory
value tiling, multi-pass filtering, ILP dual-candidates) in
[ENGINEERING_LOG.md](ENGINEERING_LOG.md).

## Reproducing

- CUDA: `test/bench.sh` (sweeps the tuning knobs on the current source).
- Java: sources vendored in `test/bench-java/` (reference code with only the
  fixture formation and benchmark bounds substituted; search loops
  untouched). Compile together with `RotationInfo.java` and the `texture/`
  classes from a reference clone (see `test/oracle/regen.sh`, which fetches
  it), then `java -Dbench.threads=N Main`.
- Oracle dumps: `test/oracle/regen.sh <workdir>` on any machine with a JDK.
- The verification suite (`make test`, `test/e2e.sh`, `build/oracle_diff`)
  must pass before any number lands in this table. CI (GitHub Actions) runs
  the unit suite, the CPU-backend e2e, and the full oracle differential on
  every push.

**#6 → #7 (unsigned tail folds + incremental refill, 12.5%).** Two
instruction-diet changes, both semantics-preserving and re-proven against
the oracle. (a) The modern generator's tail did signed `%` and shifts on a
value the compiler could not prove non-negative (the `(int)` cast hid the
48-bit mask's range), so it emitted signed-division fixup code on every
evaluation; rewriting the tail in provably-unsigned form (identical values
for all inputs) lets the compiler fold the constant mod and shifts away.
(b) The candidate-refill path did 64-bit index math, a 64-bit range compare,
and two hash-term multiplies per pull; replaced by a per-chunk pull budget
computed once (no per-pull 64-bit compare), an incremental linear index, and
an incremental x hash term (`+= 32·3129871`, wrapping add — the multiplies
now run only on the rare z-carry path). This also moves work off the
saturated multiply pipe onto the underused ALU pipe. Validated: 29,178,112
oracle values, 0 mismatches (the 2-version oracle of this round; the suite
now covers 72,945,280 values across all five versions); all 29 e2e checks including the dense
GPU-vs-CPU exhaustive differentials; predicted 8-12% from the instruction
model, measured 12.5%.

Also in this round: the oracle harness and Java benchmark sources were
vendored into the repo (`test/oracle/`, `test/bench-java/`) after the
original box was recycled — `test/oracle/regen.sh` rebuilds the dumps on
any machine with a JDK, and the regenerated dumps were verified
byte-identical (SHA256) to the originals.

## Portable backends (different machine — measured on an Apple M4 Pro laptop)

The same source builds three interchangeable backends behind one
`runSearch()` interface (`make BACKEND=cuda|metal|cpu`); the Metal shader
and the CPU backend compile the identical `include/texture.cuh` the CUDA
kernel uses (the Metal build wraps it into the shader source verbatim).
All three pass the full 29-check e2e suite including the exhaustive
GPU-vs-CPU differentials. Reference workload, best of 3 (single run for
CPU):

| Backend | Hardware | Compute time |
|---|---|---|
| CUDA | RTX 3090 (benchmark box) | 0.72 s |
| Metal | Apple M4 Pro, 20-core GPU | 7.49 s |
| CPU (threaded C++) | Apple M4 Pro, 14 CPU cores | 74.6 s |

For scale: the Metal backend on a fanless-class laptop chip is ~26× the
as-shipped Java reference running on the dual-EPYC server, and even the
portable CPU backend on the laptop beats it by ~2.6×. The 3090:M4-GPU gap
(~10×) is larger than raw specs suggest because the position hash is
64-bit-integer-multiply-heavy, which Apple GPUs emulate with 32-bit ops.

## Narval A100 baseline (2026-11-12)

Reference workload W on one NVIDIA A100-SXM4-40GB, using CUDA 12.9.86 and
driver 580.178.04. Slurm job `9193191` used revision `bbef439` and passed
`make test` and the full `test/e2e.sh` before the benchmark.

| Implementation | Config | Modern kernel, best of 3 after warmup | Legacy kernel, single run |
|---|---|---:|---:|
| CUDA warp-compacted scan | 256 threads, 1024 candidates per lane | 0.681369 s | 0.671620 s |

See the [experiment ledger](EXPERIMENTS.md#7-narval-a100-baseline-and-launch-sweep-2026-11-12)
for the full launch sweep and Nsight Compute outcome.

The repeat job `9194685` measured nine unprofiled samples per mode after
warmup on a second A100-SXM4-40GB node with the same driver and default
binary.

| Mode | Median kernel time | Min–max |
|---|---:|---:|
| Modern | 0.681581 s | 0.681524–0.681761 s |
| Legacy | 0.671745 s | 0.671485–0.671987 s |

The [repeat log summary](EXPERIMENTS.md#repeat-timings-and-cuda-timeline-job-9194685)
includes every sample and the separate Nsight Systems measurements.

### Comparing future CUDA changes

Build the baseline and candidate with the same toolchain and target GPU.
After the correctness suite passes, warm up both binaries and alternate
their run order on the same allocated GPU. Keep the individual timings and
report the median and range for each build. For a multi-pass search, include
field construction, filtering, and survivor verification in the total GPU
time. Record whole-program wall time separately.

Collect profiler data in a separate run so replay and tracing overhead stay
out of the benchmark. Record the profiler settings and any collection
failure alongside the timings.

### Opt-in CUDA bitplane CLI (Narval job 9229800)

The modern reference workload used the same CUDA binary and A100-SXM4-40GB
for nine alternating pairs. Both modes returned the same match. Whole-command
wall time includes CUDA startup and process exit.

| Metric | Standard median (min–max) | Bitplanes median (min–max) | Ratio of medians |
|---|---:|---:|---:|
| GPU search | 0.681391 s (0.681347–0.681420) | 0.157418 s (0.157330–0.158262) | 4.33× |
| Whole-command wall | 1.322511 s (1.213059–1.390604) | 0.812208 s (0.666937–0.902906) | 1.63× |

See the [individual samples and correctness checks](EXPERIMENTS.md#opt-in-cli-search-job-9229800).

### Bitplane stride comparison (Narval job 9230586)

Nine paired runs of the modern reference workload on one A100-SXM4-40GB
returned the same match as the standard kernel in both stride builds.

| Stride | GPU median (min–max) | Whole-command median (min–max) |
|---:|---:|---:|
| 5 | 0.155613 s (0.155599–0.156309) | 0.726339 s (0.652947–0.909123) |
| 6 | 0.157378 s (0.157362–0.158101) | 0.773668 s (0.655754–0.993884) |

See the [individual samples and slab breakdown](EXPERIMENTS.md#compact-world-y-stride-comparison-job-9230586).

### Standalone precomputed plane offsets (Narval job 9234752)

Nine paired runs per stride on one A100-SXM4-40GB measured the modern
reference workload. The new search included building and uploading
the offset table for each slab. Every timed search returned the same
single match as the standard kernel.

| Stride | Filter median, old → precomputed | Whole-search median, old → precomputed |
|---:|---:|---:|
| 5 | 75.61 → 50.80 ms | 157.04 → 139.98 ms |
| 6 | 63.98 → 45.00 ms | 158.47 → 145.19 ms |

These timings come from the standalone process with CUDA already
initialized. See the [experiment ledger](EXPERIMENTS.md#precomputed-filter-plane-offsets-job-9234752)
for paired samples and correctness checks.

### CUDA CLI with precomputed plane offsets (Narval job 9235600)

Nine alternating runs per build on one A100-SXM4-40GB compared the
stride 6 bitplane CLI before and after precomputing plane offsets.
Every run returned the standard kernel's exact match.

| Metric | Before median (min–max) | After median (min–max) |
|---|---:|---:|
| GPU search | 0.157441 s (0.157338–0.157984) | 0.144279 s (0.144154–0.144388) |
| Whole command | 0.859525 s (0.839256–0.996671) | 0.850069 s (0.826126–0.926646) |

See the [individual samples and correctness checks](EXPERIMENTS.md#precomputed-plane-offsets-in-the-cuda-cli-job-9235600).

### CUDA CLI stride comparison with precomputed offsets (Narval job 9294135)

Both stride builds returned the exact standard-kernel results
across six workloads on one A100-SXM4-40GB. Times are medians.

| Reference workload | Standard GPU | Stride 5 GPU | Stride 6 GPU | Standard wall | Stride 5 wall | Stride 6 wall |
|---|---:|---:|---:|---:|---:|---:|
| One direction | 0.681708 s | 0.138342 s | 0.144190 s | 1.287659 s | 0.776567 s | 0.793810 s |
| All four directions | 2.725520 s | 0.553513 s | 1.650640 s | 3.393040 s | 1.209972 s | 2.307458 s |

Stride 5 used bitplanes throughout the reference runs. Stride 6
fell back in all five four-direction runs. See the
[full workload results and individual samples](EXPERIMENTS.md#cuda-cli-stride-coverage-job-9294135).

Stride 5 was adopted for the opt-in CUDA path after the
[Narval regression run](EXPERIMENTS.md#stride-5-default-for-the-opt-in-cuda-path-job-9296853)
passed the end-to-end, fallback and checkpoint checks.

### CUDA stride 5 on Vulcan L40S (job 2323961)

The standard and opt-in paths returned the same exact match sets
across the six workloads used on Narval. The reference medians were:

| Reference workload | Standard GPU | Bitplanes GPU | GPU speedup | Standard wall | Bitplanes wall | Wall speedup |
|---|---:|---:|---:|---:|---:|---:|
| One direction | 0.319646 s | 0.066798 s | 4.79× | 0.534496 s | 0.296127 s | 1.80× |
| All four directions | 1.314350 s | 0.277621 s | 4.73× | 1.541722 s | 0.511047 s | 3.02× |

See the [full workload results and individual samples](EXPERIMENTS.md#vulcan-l40s-validation-job-2323961).

### Shared bitplane field prototype on Vulcan L40S (job 2323980)

The standalone four-direction search reduced median GPU time from
285.877 to 188.613 ms and median search wall time from 312.932
to 202.999 ms. The match set was exact in every direction. See the
[full samples and memory measurements](EXPERIMENTS.md#shared-bitplane-field-across-four-directions-vulcan-job-2323980).

### Shared bitplanes in the CUDA CLI on Vulcan L40S (job 2327586)

The four-direction reference search with opt-in stride 5 bitplanes
went from 275.4 to 179.3 ms of median GPU time (1.54×) and from
531.7 to 394.0 ms of median whole-command time (1.35×). All nine
paired runs matched the standard kernel exactly. See the
[individual samples and validation](EXPERIMENTS.md#shared-bitplanes-in-the-cuda-cli-vulcan-job-2327586).

## Bitplane search on every backend (2026-11-16)

The same workload, kernel seconds, with the bitplane search that is now the
default on all three backends. "Before" is the plain search of the same
build, forced with `TF_CUDA_BITPLANES=0`, `TF_METAL_BITPLANES=0` or
`TF_CPU_BITPLANES=0`. The CUDA rows are medians of five alternating pairs
from Narval job `9464541`. The Metal and CPU rows are medians of the runs
listed in the [experiment ledger](EXPERIMENTS.md).

| Backend | Hardware | Directions | Before | Bitplanes | Ratio |
|---|---|---|---:|---:|---:|
| CUDA | A100-SXM4-40GB | one | 0.681709 s | 0.125219 s | 5.4× |
| CUDA | A100-SXM4-40GB | all four | 2.725940 s | 0.334658 s | 8.1× |
| Metal | Apple M4 Pro, 20-core GPU | one | 6.121 s | 0.641 s | 9.5× |
| Metal | Apple M4 Pro, 20-core GPU | all four | 24.498 s | 1.626 s | 15.1× |
| CPU | Apple M4 Pro, 14 CPU cores | one | 62.35 s | 3.49 s | 17.9× |
| CPU | Apple M4 Pro, 14 CPU cores | all four | not run | 6.56 to 6.76 s | |

The CPU "before" is one run of the plain scan at `1d2b9e6`. The RTX 3090 of
the tables above has not run the bitplane search, so its 0.72 s stays the
latest figure for that card. Against the earlier rows of this file, the
Metal backend went from 7.49 s to 0.641 s and the CPU backend from 74.6 s to
3.49 s on the same laptop.

## CUDA bitplanes for the older versions (2026-11-16)

The CUDA backend now uses the bitplane search for every version, as the Metal
and CPU backends already did. Narval job `9549732`, A100-SXM4-40GB, kernel
seconds, the reference volume in one direction with one planted formation per
version, one run each. "Before" is the standard kernel of the same build.
This formation is not the reference one, so the modern row is not comparable
with the table above.

| Version | Before | Bitplanes | Ratio |
|---|---:|---:|---:|
| 0 modern | 0.681447 s | 0.155005 s | 4.4× |
| 1 legacy | 0.671894 s | 0.148785 s | 4.5× |
| 2 vanilla12 | 0.453555 s | 0.097538 s | 4.7× |
| 3 sodium | 1.295000 s | 0.300026 s | 4.3× |

On the reference workload the same job measured 0.123920 s for one direction
and 0.328846 s for all four (medians of five pairs), after the match kernel
changed to z-major order and a 32-bit field index.

## CPU fill packed as bit pairs (2026-11-18)

Apple M4 Pro, 14 CPU cores, mains power, the same reference workload, kernel
seconds, medians of seven alternating runs. "Before" is the build of the
previous commit, which already had the per-version fill.

| Backend | Directions | Before | After | Ratio |
|---|---|---:|---:|---:|
| CPU | one | 3.258 s | 2.658 s | 1.23× |
| CPU | all four | 6.528 s | 5.766 s | 1.13× |

Against the plain scan of the first table in the section above (62.35 s),
one direction is now 23× faster. The same session measured the Metal backend
at 0.613 s for one direction and 1.520 s for all four, unchanged in code
since the table above.

## CPU match with separate planes and a branch-free survivor check (2026-11-18)

Apple M4 Pro, 14 CPU cores, mains power, the same reference workload, kernel
seconds, medians of nine alternating runs. "Before" is the build of the
section above. Another application was using CPU during these runs, which is
why "before" is a little slower than there.

| Backend | Directions | Before | After | Ratio |
|---|---|---:|---:|---:|
| CPU | one | 2.739 s | 2.583 s | 1.06× |
| CPU | all four | 5.682 s | 4.769 s | 1.19× |
