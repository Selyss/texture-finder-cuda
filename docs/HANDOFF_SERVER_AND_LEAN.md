# Handoff: server setup and Lean formalisation

Written 2026-11-12 for whoever (human or AI assistant) sets up the A100
machine and starts the Lean work. It assumes no prior context on this project.

Read `docs/RNG_REDUCTION_LOG.md` first — it is the full record of what was
found and measured. This file is the "what to do next" companion.

**Things in this guide that have NOT been verified** are marked ⚠. In
particular: nothing here has been run on an A100, and none of the Lean
snippets has been compiled.

---

## 0. Ground rules for working in this repo

- **Privacy.** This tool locates real builds. Never put coordinates, base
  names, or anything location-identifying in commit messages, code comments,
  or docs. Coordinates live only inside `test/fixtures/` files, which use
  neutral names (`formation_a`). The existing `test/e2e.sh` is the one script
  that carries an expected coordinate; do not copy it elsewhere.
- **Commits.** Plain messages, no AI attribution / `Co-Authored-By` trailers.
- **Benchmarks.** `docs/BENCHMARKS.md` and `docs/EXPERIMENTS.md` (overhaul
  branch) are append-only: add rows, never edit historical ones, and only
  record a number after the verification suite passes on that build.
- **SSH.** Do not modify the owner's SSH keys or `known_hosts`. If a host-key
  prompt blocks a connection, hand the owner the command to run.
- **Single source of truth for the RNG.** `include/texture.cuh` (overhaul
  branch) is compiled by every backend and every test. Never copy its
  functions into another file; include it.

## 1. Repository map

Remote: `https://github.com/Selyss/texture-finder-cuda.git`

| Branch | What it is |
|---|---|
| `main` | old, naive kernel. Not the place to work. |
| `correctness-overhaul` | the real code: verified RNG for five version modes, optimised CUDA kernel, Metal and CPU backends, test pyramid, docs. **Work here.** |

On `correctness-overhaul`:

| Path | Purpose |
|---|---|
| `include/texture.cuh` | the RNG, host+device+Metal compatible |
| `include/kernel.cuh` | backend-neutral `runSearch()` interface |
| `src/kernel.cu` | CUDA backend (current kernel: warp-compacted per-candidate scan) |
| `src/backend_cpu.cpp`, `src/backend_metal.mm`, `src/match.metal` | other backends |
| `src/main.cpp`, `src/parser.cpp` | CLI driver (z-slicing, progress, checkpoint), formation parser |
| `test/test.cpp` | doctest unit suite (`make test`) |
| `test/e2e.sh` | end-to-end suite against the built binary |
| `test/bench.sh` | kernel tuning sweep on the reference workload |
| `test/oracle/` | Java ground-truth generator + `oracle_diff` |
| `docs/ENGINEERING_LOG.md`, `EXPERIMENTS.md`, `BENCHMARKS.md`, `SEMANTICS.md` | decision trail, ledger, benchmark table, RNG semantics |

**Not on the remote yet** (as of writing they exist only as untracked files on
the owner's laptop, created on `main`):

- `docs/RNG_REDUCTION_LOG.md`
- `docs/HANDOFF_SERVER_AND_LEAN.md` (this file)
- `experiments/rng-reduction/{sweep,lattice,flat,lowbits}.cpp`

These must be committed and pushed (or copied over) before a fresh clone on
the server can see them. Check for them after cloning; if they are missing,
ask the owner.

## 2. Server setup (A100 machine)

### 2.1 Inventory first

```bash
nvidia-smi
```

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
```

```bash
ls /usr/local/cuda/bin/nvcc; command -v nvcc; nvcc --version
```

```bash
c++ --version; java -version; nproc
```

Record GPU model/count, driver, CUDA version, and CPU in a new machine row in
`docs/EXPERIMENTS.md` §0b before recording any timing.

Notes:

- The Makefile finds `nvcc` on `PATH` or at `/usr/local/cuda/bin/nvcc`. If it
  is elsewhere, pass `NVCC=/path/to/nvcc`.
- `ARCH` defaults to `native` (needs CUDA 11.6+). ⚠ An A100 should be
  `sm_80` if you need to name it explicitly; confirm against `compute_cap`
  above (8.0 → `sm_80`).
- The code uses **one GPU** (the current device). With several A100s, pin runs
  with `CUDA_VISIBLE_DEVICES=0` so timings are comparable. Multi-GPU is not
  implemented.
- A JDK is only needed for regenerating the Java oracle (§2.4).

### 2.2 Clone and build

```bash
git clone https://github.com/Selyss/texture-finder-cuda.git
```

```bash
cd texture-finder-cuda && git checkout correctness-overhaul
```

```bash
make
```

```bash
make test
```

```bash
make tools
```

`make` prints nothing special on success; the binary is `build/main`. It
reports its backend at startup — confirm it says CUDA, not CPU (if `nvcc` was
not found the Makefile silently picks another backend). Run `make clean` when
switching backends.

### 2.3 Baseline verification — do this before changing anything

```bash
make test
```

```bash
test/e2e.sh
```

```bash
test/bench.sh
```

Expected:

- `make test`: all doctest cases pass (249,765 assertions at last count).
- `test/e2e.sh`: every line `PASS`. It builds what it needs. `FAST=1
  test/e2e.sh` uses a smaller fixture volume.
- `test/bench.sh`: prints kernel times for a few launch configurations on the
  reference workload W (1.25 × 10¹¹ candidates, 24-block fixture). On an
  RTX 3090 the default row (`t256_c1024`) was ~0.72 s for modern. ⚠ The A100
  number is unknown; whatever it prints **is the baseline** for this machine.
  Record it as a new row.

If any of these fail on a fresh clone, stop and report — do not start porting
on a red baseline.

### 2.4 Oracle (optional, needs a JDK)

```bash
test/oracle/regen.sh /tmp/tf-oracle
```

```bash
build/oracle_diff /tmp/tf-oracle/dumps
```

Regenerates ground truth from the unmodified Java reference and diffs every
C++ RNG path against it (~73M values, expect 0 mismatches). Only needed if
`include/texture.cuh` changes, which the planned work does not require.

### 2.5 Prototypes (CPU, no GPU needed)

Once `experiments/rng-reduction/` is present:

```bash
cd experiments/rng-reduction
```

```bash
git show correctness-overhaul:include/texture.cuh > texture.cuh
```

```bash
git show correctness-overhaul:test/fixtures/formation_a.txt > formation_a.txt
```

```bash
c++ -O2 -std=c++17 flat.cpp -o flat && ./flat
```

```bash
c++ -O2 -std=c++17 lattice.cpp -o lattice && ./lattice
```

```bash
c++ -O2 -std=c++17 sweep.cpp -o sweep && ./sweep
```

```bash
c++ -O2 -std=c++17 lowbits.cpp -o lowbits && ./lowbits
```

Each prints `IDENTICAL`/`all identical` lines and speedups versus a reference
scan. Any `MISMATCH` is a bug. (If already on `correctness-overhaul`, `cp
../../include/texture.cuh .` works too. Do not commit the copied
`texture.cuh`; `*.txt` is already gitignored.)

### 2.6 Profiling — the thing the old box could not do

On the rented 3090, Nsight Compute failed with `ERR_NVGPUCTRPERM`, so the
kernel's bottleneck analysis was model-based only. On an owned machine it may
work:

```bash
ncu --set full -o tf_baseline build/main -175000 -75000 -64 -40 -75000 -25000 0 test/fixtures/formation_a.txt 0
```

⚠ Flags from memory; check `ncu --help`. If it works, save the report and note
in `docs/EXPERIMENTS.md` whether the kernel is issue-bound, memory-bound, or
occupancy-bound. That answer decides which optimisation matters most.

## 3. What is being built next

Summary of `docs/RNG_REDUCTION_LOG.md`:

| # | Idea | Status |
|---|---|---|
| R1 | y-column forward differences: along a column the position mix is a quadratic in a counter, so it needs two adds instead of two multiplies | CPU-verified, not in kernel |
| R2 | bitplanes: hash each world position once, store the 2-bit result in two bit arrays, test 64 candidates per bitwise op | CPU-verified, not in kernel |
| R3 | sparse sublattice: only hash 1/s of the field; reject each candidate with the blocks that land on it; verify the rare survivors normally | CPU-verified, not in kernel |
| — | low-bit structure of legacy and ≤1.12 generators | CPU-verified, not used |

CPU single-thread result: 7–12× over the reference scan. GPU effect: unknown.

The planned first GPU port is the **x-axis bitplane version**
(`experiments/rng-reduction/flat.cpp`, function `scanFlat`). The owner intends
to write the kernel themselves, guided — so the server-side job is to provide
the test loop, not to write the kernel unprompted.

Acceptance for any new kernel, in order:

1. `make test` passes.
2. `test/e2e.sh` passes in full (it includes GPU-vs-CPU differentials on dense
   and weak formations — the strongest check that a restructured search still
   returns exactly the same match set and the same total count).
3. Only then `test/bench.sh`, compared with this machine's own baseline row.

Things a restructured kernel must preserve (from `include/kernel.cuh`):
`*totalMatches` is the true count even beyond `MAX_MATCHES`; `*truncated` is
set on overflow; results are sorted by x, y, z; `runSearch` is called once per
z-slice by `src/main.cpp` on large volumes, so everything is derived from the
bounds of that call.

## 4. Lean formalisation

### 4.1 Is it worth it?

Partly. Be selective.

- **Worth it:** the optimisations rest on a handful of *identities over 64-bit
  words*. Randomised tests cannot cover 2⁶⁴ inputs; a proof can. These are
  small, self-contained, and exactly the kind of statement where one missed
  corner (sign extension, block alignment, wraparound) would produce silent
  wrong answers.
- **Not worth it:** proving the C++/CUDA code itself, the kernel's
  concurrency, or performance. Lean would be reasoning about a model, not the
  shipped code. The link between model and code stays empirical:
  `oracle_diff` (C++ vs Java reference) and `e2e.sh` (new search vs old).

So the goal is: *a Lean model of the RNG, plus proofs that each shortcut is an
exact reformulation of it.* Keep it in its own directory (`lean/`) so it never
touches the build.

### 4.2 Setup

⚠ Commands from memory; check the current Lean/Mathlib install docs.

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

```bash
lake new lean math
```

```bash
cd lean && lake exe cache get && lake build
```

- `lake new <name> math` creates a project depending on Mathlib;
  `lake exe cache get` downloads prebuilt Mathlib (do not build it from
  source).
- Pin the toolchain (`lean-toolchain`) and commit `lake-manifest.json`.
- Tactics that should carry most of this: `ring` (identities in a commutative
  ring), `bv_decide` (SAT-based, for fixed-width bitwise facts), `omega` /
  `bv_omega` (linear arithmetic), `decide`/`native_decide` for small finite
  checks.
- ⚠ `bv_decide` bit-blasts. It is good at XOR/AND/shift/add facts at 64 bits
  but 64×64-bit **multiplication** identities will likely time out. Prove
  anything involving the multiplies algebraically (`ring` over `BitVec 64` or
  `ZMod (2^64)`), and use `bv_decide` only for the multiplication-free
  bit-twiddling lemmas. Mathlib provides a `CommRing (BitVec w)` instance —
  confirm it exists in the pinned version before relying on `ring`.

### 4.3 The model

Transcribe from `include/texture.cuh` and `docs/SEMANTICS.md`, nothing else.
The definitions, in maths:

```
A = 3129871, B = 116129781, C = 42317861, M = 0x5DEECE66D

u(x,y,z)   = sext₃₂→₆₄( (x·A) mod 2³² )  XOR  (sext(z)·B mod 2⁶⁴)  XOR  sext(y)
P(u)       = C·u² + 11·u                               (mod 2⁶⁴)
seed(u)    = P(u) arithmetic-shift-right 16

modern(s)  = let v = ((s XOR M)·M + 11) mod 2⁴⁸ in  v >> 46            -- 0..3
legacy(s)  = let t = (s XOR M) mod 2⁴⁸
             let r = low 32 bits of ((t·0xBB20B4600A69 + 0x40942DE6BA) >> 16), as signed int32
             in |r| mod 4            -- with |INT_MIN| treated as unsigned negate
vanilla12(mix) = let r = (low 32 bits of mix, as int32) arithmetic>> 16 in |r| mod 4
```

A side face reads `value AND 1`; a top face reads all two bits. (For modern,
the game always rolls one of four and the side face exposes the parity — see
the comment above `modernFromCoordRandom`.)

**Sanity-check the model before proving anything:** evaluate it in Lean
(`#eval`) on the test vectors in `test/test.cpp` and compare. A proof about a
mis-transcribed model is worthless.

### 4.4 Theorems, in priority order

Each is stated informally, then as a ⚠ *sketch* of a Lean statement (untested
syntax — treat as pseudocode).

**T1. XOR with a small value is addition inside an aligned block.** (R1)

For `K y : BitVec 64` and `k ≤ 63`, with `m = 2^k − 1` and `y ≤ m`:
`K ^^^ y = (K &&& ~~~m) + ((K &&& m) ^^^ y)`, and `(K &&& m) ^^^ y ≤ m`.

More generally with a block base `s` where `s &&& m = 0` and offset `j ≤ m`:
`K ^^^ (s + j) = ((K ^^^ s) &&& ~~~m) + ((K &&& m) ^^^ j)`.

```lean
theorem xor_small_eq_add (K y : BitVec 64) (k : Nat) (hk : k ≤ 63)
    (hy : y ≤ BitVec.ofNat 64 (2^k - 1)) :
    K ^^^ y = (K &&& ~~~(BitVec.ofNat 64 (2^k - 1))) + ((K &&& BitVec.ofNat 64 (2^k - 1)) ^^^ y) := by
  sorry
```

Approach: no multiplies, so `bv_decide` per fixed `k`, or a general proof via
"disjoint bits ⇒ OR = ADD".

**T2. The map `j ↦ (K &&& m) ^^^ j` is a bijection on `[0, m]`.** (R1 —
every level of the block is visited exactly once.) It is an involution.

**T3. Negative y.** `K ^^^ y = (~~~K) ^^^ (~~~y)`; and if `y` is the sign
extension of a negative 32-bit value then `~~~y` is small and non-negative.

```lean
theorem xor_not_not (a b : BitVec 64) : a ^^^ b = (~~~a) ^^^ (~~~b) := by bv_decide
```

**T4. Forward differences of the mix.** In any commutative ring:

```
P(u+1) − P(u)            = C·(2u+1) + 11
(P(u+2) − P(u+1)) − (P(u+1) − P(u)) = 2C
```

```lean
theorem mix_first_diff {R : Type} [CommRing R] (C u : R) :
    (C*(u+1)^2 + 11*(u+1)) - (C*u^2 + 11*u) = C*(2*u+1) + 11 := by ring
```

Corollary to state explicitly: the recurrence `l₀ = P(base)`,
`d₀ = C(2·base+1)+11`, `l_{t+1} = l_t + d_t`, `d_{t+1} = d_t + 2C` gives
`l_t = P(base + t)` for all `t` (induction + `ring`).

**T5. Sweep correctness (T1 + T2 + T3 + T4 combined).** For a column constant
`K` and an aligned block, the sequence produced by the recurrence, re-indexed
by `j = t ^^^ (K &&& m)`, equals `P(K ^^^ y)` for every y in the block. This is
the statement `sweep.cpp` tests empirically.

**T6. Low-bit dependence.** (§5 of the log.) If `u ≡ u' (mod 2ⁿ)` then
`P(u) ≡ P(u') (mod 2ⁿ)` (truncation is a ring homomorphism). Consequences to
prove:

- vanilla12 depends only on `u mod 2³²`; its parity only on `u mod 2¹⁷`.
- legacy parity depends only on `u mod 2³³`; "legacy ∈ {0,2}" is determined
  by `u mod 2³⁴`.
- |r| mod 4: negation preserves parity, and maps 0↦0, 2↦2, 1↔3 mod 4 (this is
  why the sign matters only for odd targets).

Approach: work in `ZMod (2^n)` or with `BitVec.truncate`/`setWidth`; the LCG
step `t·a + c` is also a ring map on low bits.

**T7. Lattice soundness.** (R3.) Pure logic, no arithmetic: if the block set
is partitioned into `on ∪ off` for a candidate, then "all `on` blocks match ∧
all `off` blocks match" ⇔ "all blocks match". So filtering by `on` then
verifying `off` returns exactly the reference match set, for any partition and
any stride. The only real content is that the `on`/`off` split used for a
candidate depends on its coset and is a partition.

**T8. Bitplane matching.** (R2.) Bit `j` of the combined mask is set iff
candidate `j` passes every block: a statement about `BitVec 64` AND/shift/NOT
and `getLsb`. Includes the unaligned 64-bit extract from two adjacent words.

**T9 (optional, already believed).** `P` is exactly 2-to-1 on `BitVec 64` and
its formal derivative `2Cu + 11` is always odd (Hensel lifting). Recorded as a
verified-by-experiment fact in `docs/EXPERIMENTS.md` §3; a proof would be a
nice-to-have, not needed for the optimisations.

Suggested order: T4 → T3 → T1 → T2 → T5 → T6 → T7 → T8. T4 and T3 should be
minutes; T1/T5 are where the alignment and sign-extension subtleties live and
are the ones most worth having.

### 4.5 Pitfalls specific to this RNG

- **Mixed widths.** `x·A` is a *32-bit* wrapping multiply, then sign-extended
  to 64. `z·B` is a 64-bit multiply of a sign-extended 32-bit value (it never
  wraps for in-range z, but model it as wrapping). `y` is sign-extended.
- **Arithmetic vs logical shift.** `mix >> 16` is arithmetic on a signed
  64-bit value. In the modern path the result is then masked to 48 bits, so
  the distinction disappears there — but not in general.
- **`abs(INT_MIN)`.** The code uses unsigned negate; for mod 2 and 4 the
  result is 0. Model it the same way.
- **Modern side faces.** Value is `(roll of 4) AND 1`, not a separate roll of
  2.
- **Do not "simplify" the model** to make proofs easier. If a proof needs a
  nicer form, prove the nicer form equal to the literal transcription.

### 4.6 What "done" looks like for Lean

- `lake build` succeeds with no `sorry` in T1–T5 (the R1 chain) and T6.
- The model's `#eval` output matches the vectors in `test/test.cpp`.
- A short `lean/README.md` mapping each theorem to the line of
  `docs/RNG_REDUCTION_LOG.md` it backs.
- No change to anything outside `lean/`.

## 5. Open questions to report back on

1. Baseline kernel time for W on one A100 (and the GPU's exact model/memory).
2. Does `ncu` work on this machine? If so, what bounds the current kernel?
3. Do the CPU prototypes reproduce (all `IDENTICAL`) on the server's compiler?
4. Lean: does `ring` close T4 over `BitVec 64` in the pinned Mathlib, and does
   `bv_decide` handle T1 at 64 bits in reasonable time?
