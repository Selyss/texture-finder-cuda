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

- GPU: NVIDIA GeForce RTX 3090 (GA102, 82 SMs, 24 GB), driver 580.159.03, CUDA 12.8, `-O3 -arch=native`
- CPU: 2 × AMD EPYC 7H12 (128 cores / 256 threads total), 1 TB RAM —
  **container cgroup CPU quota: 27.2 cores** (`cpu.max 2720000/100000`),
  which caps every CPU-side number below; CPU-seconds are reported so
  results can be projected onto unrestricted hardware
- OS: Ubuntu 22.04, OpenJDK 17 for the Java runs

## Results

| # | Implementation | Config | Wall | vs Java (as shipped) | Notes |
|---|---|---|---|---|---|
| 1 | Java reference (19MisterX98/TextureRotations) | as shipped: 10 threads | 192.5 s (1,562 CPU-core-s) | 1× | search code unmodified; only formation/bounds/mode configured; correct unique match |
| 2 | Java reference | tuned: 256 threads (quota-capped at 27.2 cores) | 104.9 s (2,698 CPU-core-s) | 1.8× | same build, `-Dbench.threads=256`; correct unique match. Extra CPU-seconds vs #1 = contention overhead under the quota |
| 3 | CUDA original (this repo, pre-overhaul) | dim3(8,2,8), printf output | 4.63 s | 41.6× | **correctness caveats**: silently skipped the last x/z column at these exact bounds, searched nothing for single-cell ranges, sides broken for direction ≠ 0 |
| 4 | CUDA rewrite: grid-stride + result buffer | 256-thread blocks | 1.47 s (kernel 1.26 s) | 131× | first correct version; 3.15× vs #3 |
| 5 | CUDA warp compaction | chunk 256/lane | 1.40 s (kernel 1.17 s) | 138× | |
| 6 | CUDA precomputed hash terms + stripe loop | chunk 1024/lane | **1.04 s (kernel 0.83 s)** | **185×** | current default; 4.45× vs original CUDA (#3) |

Cross-device framing (since Java-on-CPU vs CUDA-on-GPU is the point of the
project): using the Java run's own best throughput (1,562 EPYC-core-seconds
for the workload) against the 0.83 s GPU kernel, it would take **~1,900
EPYC 7H12 cores scaling perfectly** to match one RTX 3090 on this search.
Projected onto the full unrestricted 128-core machine (ideal scaling,
no quota), Java would finish in ~12 s — the single GPU is still ~12× faster
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
- Java: `/root/javabench` on the benchmark box — reference sources with the
  fixture formation and benchmark bounds substituted, search loops
  untouched; `java -cp .:/root/oracle/harness/classes -Dbench.threads=N Main`.
- The verification suite (`make test`, `test/e2e.sh`, `build/oracle_diff`)
  must pass before any number lands in this table.
