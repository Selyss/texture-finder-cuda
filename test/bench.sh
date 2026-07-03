#!/usr/bin/env bash
# Kernel tuning benchmark sweep. Builds one binary per configuration and
# times the reference workload (the formation_a search volume, ~1.25e11
# positions) plus a legacy-version run of the same volume.
# Usage: test/bench.sh   (run on a machine with the GPU)
set -euo pipefail
cd "$(dirname "$0")/.."

WORK="-175000 -75000 -64 -40 -75000 -25000"
FORM=test/fixtures/formation_a.txt
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
mkdir -p build

run_config() { # <label> <defines...>
    local label=$1; shift
    local bin="build/bench_$label"
    "$NVCC" -O3 -arch=native -std=c++17 -Iinclude "$@" \
        src/main.cu src/kernel.cu src/parser.cu -o "$bin"
    # warmup + 3 timed runs, take best kernel time
    "$bin" $WORK 0 "$FORM" 0 >/dev/null
    local best=""
    for _ in 1 2 3; do
        local t
        t=$("$bin" $WORK 0 "$FORM" 0 | awk '/Kernel time/{print $3}')
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best=$t; fi
    done
    local legacy
    legacy=$("$bin" $WORK 1 "$FORM" 0 | awk '/Kernel time/{print $3}')
    echo "$label modern=${best}s legacy=${legacy}s"
}

echo "config modern_best_of_3 legacy_single"
run_config "t256_c256"   # defaults
run_config "t128_c256"   -DTF_BLOCK_THREADS=128
run_config "t512_c256"   -DTF_BLOCK_THREADS=512
run_config "t256_c64"    -DTF_CHUNK_PER_LANE=64
run_config "t256_c1024"  -DTF_CHUNK_PER_LANE=1024
run_config "t128_c1024"  -DTF_BLOCK_THREADS=128 -DTF_CHUNK_PER_LANE=1024
rm -f build/bench_*
