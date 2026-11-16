#!/usr/bin/env bash
# Regression gate for the CUDA backend after a source change: clean build,
# unit tests, the end-to-end suite with bitplanes off and on, forced fallback,
# the standalone harness build, and a short reference timing.
#
# Usage, from a GPU node or under Slurm:
#   sbatch --account=<account> --gpus=1 --cpus-per-task=8 --mem=16G --time=00:20:00 test/validate_cuda.sh
#   test/validate_cuda.sh
# ARCH selects the nvcc target (default native). PAIRS sets timed pairs (default 5).
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
sha src/backends/cuda/search.cu src/backends/cuda/bitplanes.cuh src/search/session.cpp \
    src/search/checkpoint.cpp src/search/backend.hpp experiments/rng-reduction/flat_cuda.cu

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
gate "e2e with bitplanes off" env -u TF_CUDA_BITPLANES test/e2e.sh
echo "== End-to-end, opt-in bitplanes"
gate "e2e with bitplanes on" env TF_CUDA_BITPLANES=1 test/e2e.sh

run() { # <mode: 0|1> <direction> -> stdout of the CLI; stderr kept for traces
    if [ "$1" = 1 ]; then
        TF_CUDA_BITPLANES=1 TF_CUDA_BITPLANES_TRACE=1 build/main "${REF[@]}" "$2" 2>&1
    else
        env -u TF_CUDA_BITPLANES build/main "${REF[@]}" "$2" 2>&1
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
gate "standard kernel agrees, one direction" [ "$(matches "$(run 0 0)")" = "$EXPECT" ]
gate "standard kernel agrees, all directions" [ "$(matches "$(run 0 all)")" = "$EXPECT facing North" ]

echo "== Forced fallback (1-byte plane budget)"
out=$(TF_CUDA_BITPLANES=1 TF_CUDA_BITPLANES_TRACE=1 TF_CUDA_BITPLANES_BYTES=1 build/main "${REF[@]}" 0 2>&1)
gate "fallback is taken" grep -q "baseline fallback" <<<"$out"
gate "fallback finds the match" [ "$(matches "$out")" = "$EXPECT" ]

echo "== Checkpoint written by bitplanes, resumed by the standard kernel"
ck=$(mktemp -u)
TF_CUDA_BITPLANES=1 TF_MAX_SLICES=1 build/main "${REF[@]}" all --checkpoint "$ck" >/dev/null 2>&1
gate "early stop leaves a checkpoint" [ -f "$ck" ]
out=$(env -u TF_CUDA_BITPLANES build/main "${REF[@]}" all --checkpoint "$ck" 2>/dev/null)
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

echo "== Standalone harness on the reference"
if [ -x build/flat_cuda ]; then
    gate "harness default run" build/flat_cuda test/fixtures/formation_a.txt "${REF[@]:0:6}" "${SLURM_CPUS_PER_TASK:-4}"
fi

echo "== Result: $fails gate(s) failed"
[ "$fails" = 0 ]
