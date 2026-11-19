#!/usr/bin/env bash
# Regression gate for the CUDA backend after a source change: clean build,
# unit tests, the end-to-end suite with bitplanes off and on, forced fallback,
# the older versions on a planted formation, the multi-device code with several
# workers on one GPU, the standalone harness build, and a short reference
# timing. On a node with several GPUs it also times 1, 2, 4... devices.
#
# Usage, from a GPU node or under Slurm:
#   sbatch --account=<account> --gpus=1 --cpus-per-task=8 --mem=16G --time=00:20:00 test/validate_cuda.sh
#   test/validate_cuda.sh
# ARCH selects the nvcc target (default native). PAIRS sets timed pairs (default 5).
# HARNESS_MODES=all also runs every experiment mode of the harness (several
# minutes); a list such as HARNESS_MODES="FUSED DIAGNOSTICS" runs only those.
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
sha src/backends/cuda/search.cu src/backends/cuda/bitplane_search.cu src/backends/cuda/devices.hpp \
    src/backends/cuda/bitplanes.cuh src/bitplanes/*.hpp src/bitplanes/plan.cpp \
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

run() { # <mode: 0|1> <direction> -> output of the CLI, whose "Search:" line names the path
    if [ "$1" = 1 ]; then
        env -u TF_CUDA_BITPLANES build/main "${REF[@]}" "$2" 2>&1
    else
        TF_CUDA_BITPLANES=0 build/main "${REF[@]}" "$2" 2>&1
    fi
}
matches() { grep "^Match found" <<<"$1" | sort; }
kernel() { sed -n 's/^Kernel time: \(.*\) seconds/\1/p' <<<"$1"; }

echo "== Path selection on the reference"
one=$(run 1 0)
all=$(run 1 all)
gate "one direction uses bitplanes" grep -qx "Search: bitplanes" <<<"$one"
gate "one direction never falls back" bash -c '! grep -q "^Search: plain" <<<"$1"' _ "$one"
gate "all directions use bitplanes" grep -qx "Search: bitplanes" <<<"$all"
gate "reference match, one direction" [ "$(matches "$one")" = "$EXPECT" ]
gate "reference match, all directions" [ "$(matches "$all")" = "$EXPECT facing North" ]
off=$(run 0 0)
gate "TF_CUDA_BITPLANES=0 keeps the standard kernel" grep -qx "Search: plain" <<<"$off"
gate "standard kernel agrees, one direction" [ "$(matches "$off")" = "$EXPECT" ]
gate "standard kernel agrees, all directions" [ "$(matches "$(run 0 all)")" = "$EXPECT facing North" ]

echo "== Forced fallback (1-byte plane budget)"
out=$(env -u TF_CUDA_BITPLANES TF_CUDA_BITPLANES_BYTES=1 build/main "${REF[@]}" 0 2>&1)
gate "fallback is taken" grep -qx "Search: plain" <<<"$out"
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
    if env -u TF_CUDA_BITPLANES build/main -120100 -119900 -60 -45 -50050 -49950 0 "$form" 0 2>&1 |
        grep -qx "Search: bitplanes"; then
        seed=$try
        break
    fi
done
gate "a generated formation qualifies for bitplanes" [ -n "$seed" ]
for v in 0 1 2 3 4; do
    build/gen_formation "$v" 0 -120000 -50 -50000 24 40 "${seed:-1}" > "$form"
    old=("${REF[@]:0:6}" "$v" "$form" 0)
    on() { env -u TF_CUDA_BITPLANES build/main "${old[@]}" 2>&1; }
    off() { TF_CUDA_BITPLANES=0 build/main "${old[@]}" 2>&1; }
    out=$(on)
    std=$(off)
    gate "version $v uses bitplanes" grep -qx "Search: bitplanes" <<<"$out"
    gate "version $v finds the planted formation" grep -qF "Match found at [-120000, -50, -50000]" <<<"$out"
    gate "version $v agrees with the standard kernel" [ "$(matches "$out")" = "$(matches "$std")" ]
    echo "$v,$(kernel "$(off)"),$(kernel "$(on)")"
done
rm -f "$form"

echo "== Several workers on one device (the multi-device code on one GPU)"
# --devices 0,0,0 gives device 0 three workers, so the slab queue and the
# split of the standard kernel run here exactly as they do across real GPUs.
form=$(mktemp)
# Weak but well filtered: about 7,000 matches, with some on every z row, so a
# row that is skipped or searched twice at any seam changes the list.
DENSE=(-125000 124999 0 199 -300 299 0 "$form" 0)
build/gen_formation 0 0 0 0 0 11 0 19 > "$form"
dense() { # <bitplanes 0|1> <device list>
    if [ "$1" = 1 ]; then env -u TF_CUDA_BITPLANES build/main "${DENSE[@]}" --devices "$2" 2>&1
    else TF_CUDA_BITPLANES=0 build/main "${DENSE[@]}" --devices "$2" 2>&1; fi
}
ref=$(dense 0 0)
rows=$(matches "$ref" | sed 's/.*, \(-*[0-9]*\)\]$/\1/' | sort -u | wc -l)
gate "dense reference has matches on all 600 z rows" [ "$rows" -eq 600 ]
gate "dense reference, bitplanes on one device agree" [ "$(matches "$(dense 1 0)")" = "$(matches "$ref")" ]
for list in 0,0 0,0,0 0,0,0,0; do
    workers=$(tr -cd , <<<"$list" | wc -c); workers=$((workers + 1))
    for mode in 1 0; do
        [ "$mode" = 1 ] && path=bitplanes || path=plain
        out=$(dense "$mode" "$list")
        gate "$workers workers, $path: the single-device match list" [ "$(matches "$out")" = "$(matches "$ref")" ]
        gate "$workers workers, $path: same total" \
            [ "$(grep ' matches$' <<<"$out")" = "$(grep ' matches$' <<<"$ref")" ]
        gate "$workers workers, $path: path and device count reported" \
            [ "$(grep -c -e "^Search: $path\$" -e "^Devices: $workers x " <<<"$out")" = 2 ]
        gate "$workers workers, $path: work share printed" grep -q "^Work share: device 0 " <<<"$out"
    done
done
# A strong formation planted on and beside the rows where parts meet: an
# unsliced 300-row volume splits at rows 100 and 200 for three workers of the
# standard kernel, and into 64-row slabs for the queue.
for z in 63 64 99 100 127 128 199 200; do
    build/gen_formation 0 0 0 5 "$z" 24 40 "${seed:-1}" > "$form"
    for mode in 1 0; do
        out=$(if [ "$mode" = 1 ]; then env -u TF_CUDA_BITPLANES build/main -500 499 0 9 0 299 0 "$form" 0 --devices 0,0,0 2>&1
              else TF_CUDA_BITPLANES=0 build/main -500 499 0 9 0 299 0 "$form" 0 --devices 0,0,0 2>&1; fi)
        gate "origin on z row $z, bitplanes $mode, three workers" \
            [ "$(matches "$out")" = "Match found at [0, 5, $z]" ]
    done
done
out=$(env -u TF_CUDA_BITPLANES build/main "${REF[@]}" all --devices 0,0,0 2>&1)
gate "reference, all directions, three workers" [ "$(matches "$out")" = "$EXPECT facing North" ]
rm -f "$form"

gpus=$(nvidia-smi -L | wc -l)
if [ "$gpus" -gt 1 ]; then
    echo "== Scaling across $gpus devices (seconds)"
    # Fixed volumes that contain the reference match, searched on 1, 2, 4... devices.
    echo "workload,path,devices,run,kernel_s,whole_s,work_share"
    scale() { # <label> <bitplanes 0|1> <runs> <direction> <bounds...>
        local label=$1 mode=$2 runs=$3 dir=$4; shift 4
        local n list out path want=$EXPECT
        [ "$dir" = all ] && want="$EXPECT facing North"
        [ "$mode" = 1 ] && path=bitplanes || path=plain
        for run in $(seq 1 "$runs"); do
            # "auto" is the default: no --devices, the program decides.
            for n in 1 2 4 8 auto; do
                [ "$n" = auto ] || [ "$n" -le "$gpus" ] || continue
                [ "$n" = auto ] && list= || list=$(seq -s, 0 $((n - 1)))
                out=$(if [ "$mode" = 1 ]; then env -u TF_CUDA_BITPLANES build/main "$@" 0 test/fixtures/formation_a.txt "$dir" ${list:+--devices "$list"} 2>&1
                      else TF_CUDA_BITPLANES=0 build/main "$@" 0 test/fixtures/formation_a.txt "$dir" ${list:+--devices "$list"} 2>&1; fi)
                gate "scaling $label, $path, $n devices: finds the reference match" [ "$(matches "$out")" = "$want" ]
                echo "$label-$dir,$path,$n,$run,$(kernel "$out"),$(sed -n 's/^Search completed in \(.*\) seconds/\1/p' <<<"$out"),$(sed -n 's/^Work share: //p' <<<"$out" | tr -d ',')"
            done
        done
    }
    # 100k x 100k and 500k x 500k areas, 40 y levels, centred on the reference match.
    SMALL=(-158723 -58724 -64 -25 -119736 -19737)
    LARGE=(-358723 141276 -64 -25 -319736 180263)
    scale reference 1 3 0 "${REF[@]:0:6}"
    scale 100k 1 3 0 "${SMALL[@]}"
    scale 100k 1 3 all "${SMALL[@]}"
    scale 500k 1 3 0 "${LARGE[@]}"
    scale 500k 1 3 all "${LARGE[@]}"
    scale 100k 0 2 0 "${SMALL[@]}"
    scale 500k 0 1 0 "${LARGE[@]}"
fi

if [ -n "${PLANE_BYTES:-}" ]; then
    echo "== Field budget sweep (kernel seconds, $PAIRS runs each)"
    echo "workload,plane_bytes,run,bitplanes_s"
    for dir in 0 all; do
        [ "$dir" = all ] && want="$EXPECT facing North" || want=$EXPECT
        for budget in ${PLANE_BYTES//:/ }; do
            out=$(export TF_CUDA_BITPLANES_BYTES=$budget; run 1 "$dir")
            gate "budget $budget, direction $dir: bitplanes find the match" \
                [ "$(matches "$out")" = "$want" -a -z "$(grep '^Search: plain' <<<"$out")" ]
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
        modes="DUPLICATE_Y COMPARE_OFFSETS COMPARE_PACKING COMPARE_PACKING_SEARCH SHARED_FIELD FUSED DIAGNOSTICS"
    fi
    # Each mode checks itself against the CPU reference or the standard kernel.
    for mode in $modes; do
        gate "harness $mode" env "TF_FLAT_$mode=all" "${harness[@]}"
    done
fi

echo "== Result: $fails gate(s) failed"
[ "$fails" = 0 ]
