#!/usr/bin/env bash
# Regression gate for the CUDA backend after a source change: clean build,
# unit tests, the end-to-end suite with bitplanes off and on, forced fallback,
# the older versions on a planted formation, the standalone harness build, and
# a short reference timing.
#
# Usage, from a GPU node or under Slurm:
#   sbatch --account=<account> --gpus=1 --cpus-per-task=8 --mem=16G --time=00:20:00 test/validate_cuda.sh
#   test/validate_cuda.sh
# ARCH selects the nvcc target (default native). PAIRS sets timed pairs (default 5).
# HARNESS_MODES=all also runs every experiment mode of the harness (several
# minutes); a list such as HARNESS_MODES="FUSED WORD_COLUMNS" runs only those.
# PLANE_BYTES=a:b:c also times the bitplane path under those field budgets,
# which set the slab height.
# TF_MODULES lists environment modules to load first. A SOURCE_COMMIT file, if
# present, identifies a tree that was copied without its .git directory.
# Exit status is non-zero if any gate fails. Timings are printed, never judged.
#SBATCH --job-name=tf-validate
#SBATCH --output=tf-validate-%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Clusters that need environment modules: TF_MODULES="gcc/12.3 cuda/12.9".
if [ -n "${TF_MODULES:-}" ]; then
    # shellcheck disable=SC2086
    module load $TF_MODULES || { echo "module load failed: $TF_MODULES"; exit 1; }
fi
export BACKEND=cuda ARCH=${ARCH:-native}
PAIRS=${PAIRS:-5}
REF=(-175000 -75000 -64 -40 -75000 -25000 0 test/fixtures/formation_a.txt)
EXPECT=$(sed -n 's/.*Expected match: \*\*x=\(.*\) y=\(.*\) z=\(.*\)\*\*.*/Match found at [\1, \2, \3]/p' \
    test/fixtures/formation_a.expected.md)
fails=0

gate() { # <label> <command...>
    local label=$1
    shift
    if "$@"; then echo "GATE PASS: $label"; else echo "GATE FAIL: $label"; fails=$((fails + 1)); fi
}
sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

echo "== Environment"
date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "job ${SLURM_JOB_ID:-none} on $(hostname)"
if [ -f SOURCE_COMMIT ]; then cat SOURCE_COMMIT; else git rev-parse HEAD 2>/dev/null; git status --short 2>/dev/null | head -20; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvcc --version | tail -1
sha src/backends/cuda/search.cu src/backends/cuda/bitplanes.cuh src/bitplanes/*.hpp src/bitplanes/plan.cpp \
    src/search/session.cpp src/search/checkpoint.cpp src/search/backend.hpp \
    experiments/rng-reduction/flat_cuda.cu experiments/rng-reduction/diagnostic_kernels.cuh

echo "== Build"
make clean >/dev/null
gate "CUDA CLI builds" make BACKEND=cuda ARCH="$ARCH"
gate "unit tests" make test
gate "tools build" make tools
gate "standalone harness builds" make BACKEND=cuda ARCH="$ARCH" build/flat_cuda
[ -x build/main ] || { echo "No binary; stopping."; exit 1; }

echo "== Backend identity"
out=$(build/main "${REF[@]}" 0 2>&1 | grep -m1 "^Backend:")
gate "binary reports the CUDA backend" [ "$out" = "Backend: CUDA" ]

echo "== End-to-end, standard kernel"
gate "e2e with bitplanes off" env TF_CUDA_BITPLANES=0 test/e2e.sh
echo "== End-to-end, default path"
gate "e2e with bitplanes on" env -u TF_CUDA_BITPLANES test/e2e.sh

run() { # <mode: 0|1> <direction> -> stdout of the CLI; stderr kept for traces
    if [ "$1" = 1 ]; then
        env -u TF_CUDA_BITPLANES TF_CUDA_BITPLANES_TRACE=1 build/main "${REF[@]}" "$2" 2>&1
    else
        TF_CUDA_BITPLANES=0 TF_CUDA_BITPLANES_TRACE=1 build/main "${REF[@]}" "$2" 2>&1
    fi
}
matches() { grep "^Match found" <<<"$1" | sort; }
kernel() { sed -n 's/^Kernel time: \(.*\) seconds/\1/p' <<<"$1"; }

echo "== Path selection on the reference"
one=$(run 1 0)
all=$(run 1 all)
gate "one direction uses bitplanes" grep -q "CUDA search: bitplanes" <<<"$one"
gate "one direction never falls back" bash -c '! grep -q "baseline fallback" <<<"$1"' _ "$one"
gate "all directions share one field" grep -q "bitplanes shared (4 directions)" <<<"$all"
gate "reference match, one direction" [ "$(matches "$one")" = "$EXPECT" ]
gate "reference match, all directions" [ "$(matches "$all")" = "$EXPECT facing North" ]
off=$(run 0 0)
gate "TF_CUDA_BITPLANES=0 keeps the standard kernel" bash -c '! grep -q "CUDA search:" <<<"$1"' _ "$off"
gate "standard kernel agrees, one direction" [ "$(matches "$off")" = "$EXPECT" ]
gate "standard kernel agrees, all directions" [ "$(matches "$(run 0 all)")" = "$EXPECT facing North" ]

echo "== Forced fallback (1-byte plane budget)"
out=$(env -u TF_CUDA_BITPLANES TF_CUDA_BITPLANES_TRACE=1 TF_CUDA_BITPLANES_BYTES=1 build/main "${REF[@]}" 0 2>&1)
gate "fallback is taken" grep -q "baseline fallback" <<<"$out"
gate "fallback finds the match" [ "$(matches "$out")" = "$EXPECT" ]

echo "== Checkpoint written by bitplanes, resumed by the standard kernel"
ck=$(mktemp -u)
env -u TF_CUDA_BITPLANES TF_MAX_SLICES=1 build/main "${REF[@]}" all --checkpoint "$ck" >/dev/null 2>&1
gate "early stop leaves a checkpoint" [ -f "$ck" ]
out=$(TF_CUDA_BITPLANES=0 build/main "${REF[@]}" all --checkpoint "$ck" 2>/dev/null)
gate "resume reports the checkpoint" grep -q "Resuming from checkpoint" <<<"$out"
gate "resume finds the match" grep -qF "$EXPECT facing North" <<<"$out"
gate "resume removes the checkpoint" [ ! -f "$ck" ]
rm -f "$ck"

echo "== Reference timing (kernel seconds; one warmup, $PAIRS alternating pairs)"
echo "workload,pair,standard_s,bitplanes_s"
for dir in 0 all; do
    run 0 "$dir" >/dev/null
    run 1 "$dir" >/dev/null
    for pair in $(seq 1 "$PAIRS"); do
        if [ $((pair % 2)) = 1 ]; then s=$(kernel "$(run 0 "$dir")"); b=$(kernel "$(run 1 "$dir")")
        else b=$(kernel "$(run 1 "$dir")"); s=$(kernel "$(run 0 "$dir")"); fi
        echo "$dir,$pair,$s,$b"
    done
done

echo "== Older versions (a planted formation in the reference volume; kernel seconds)"
echo "version,standard_s,bitplanes_s"
form=$(mktemp)
# Not every generated formation filters well enough for bitplanes, so the
# seed is found here: the first whose formation does.
seed=
for try in $(seq 1 40); do
    build/gen_formation 0 0 -120000 -50 -50000 24 40 "$try" > "$form"
    if env -u TF_CUDA_BITPLANES TF_CUDA_BITPLANES_TRACE=1 build/main -120100 -119900 -60 -45 -50050 -49950 0 \
        "$form" 0 2>&1 | grep -q "CUDA search: bitplanes"; then
        seed=$try
        break
    fi
done
gate "a generated formation qualifies for bitplanes" [ -n "$seed" ]
for v in 0 1 2 3; do
    build/gen_formation "$v" 0 -120000 -50 -50000 24 40 "${seed:-1}" > "$form"
    old=("${REF[@]:0:6}" "$v" "$form" 0)
    on() { env -u TF_CUDA_BITPLANES TF_CUDA_BITPLANES_TRACE=1 build/main "${old[@]}" 2>&1; }
    off() { TF_CUDA_BITPLANES=0 build/main "${old[@]}" 2>&1; }
    out=$(on)
    std=$(off)
    gate "version $v uses bitplanes" grep -q "CUDA search: bitplanes" <<<"$out"
    gate "version $v finds the planted formation" grep -qF "Match found at [-120000, -50, -50000]" <<<"$out"
    gate "version $v agrees with the standard kernel" [ "$(matches "$out")" = "$(matches "$std")" ]
    echo "$v,$(kernel "$(off)"),$(kernel "$(on)")"
done
rm -f "$form"

if [ -n "${PLANE_BYTES:-}" ]; then
    echo "== Field budget sweep (kernel seconds, $PAIRS runs each)"
    echo "workload,plane_bytes,run,bitplanes_s"
    for dir in 0 all; do
        [ "$dir" = all ] && want="$EXPECT facing North" || want=$EXPECT
        for budget in ${PLANE_BYTES//:/ }; do
            out=$(export TF_CUDA_BITPLANES_BYTES=$budget; run 1 "$dir")
            gate "budget $budget, direction $dir: bitplanes find the match" \
                [ "$(matches "$out")" = "$want" -a -z "$(grep 'baseline fallback' <<<"$out")" ]
        done
        for pair in $(seq 1 "$PAIRS"); do
            for budget in ${PLANE_BYTES//:/ }; do
                echo "$dir,$budget,$pair,$(export TF_CUDA_BITPLANES_BYTES=$budget; kernel "$(run 1 "$dir")")"
            done
        done
    done
fi

echo "== Standalone harness on the reference"
if [ -x build/flat_cuda ]; then
    harness=(build/flat_cuda test/fixtures/formation_a.txt "${REF[@]:0:6}" "${SLURM_CPUS_PER_TASK:-4}")
    gate "harness default run" "${harness[@]}"
    modes=${HARNESS_MODES:-}
    if [ "$modes" = all ]; then
        modes="DUPLICATE_Y COMPARE_OFFSETS COMPARE_PACKING COMPARE_PACKING_SEARCH SHARED_FIELD FUSED WORD_COLUMNS MATCH_VARIANTS DIAGNOSTICS"
    fi
    # Each mode checks itself against the CPU reference or the standard kernel.
    for mode in $modes; do
        gate "harness $mode" env "TF_FLAT_$mode=all" "${harness[@]}"
    done
fi

echo "== Result: $fails gate(s) failed"
[ "$fails" = 0 ]
