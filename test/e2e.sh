#!/usr/bin/env bash
# End-to-end tests: runs the real GPU binary against the verified fixture and
# against synthetic formations with known origins, covering both versions,
# all directions, 'all' mode, and side faces. Requires a CUDA GPU.
# Usage: test/e2e.sh   (from the repo root; builds via make if needed)
set -euo pipefail
cd "$(dirname "$0")/.."

make -s build/main build/gen_formation build/cpu_search

BIN=build/main
GEN=build/gen_formation
DIRNAMES=(North West South East)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fails=0

check() { # <label> <expected-line> <output> [expected-match-count]
    local label=$1 expected=$2 out=$3 want_count=${4-1}
    local count
    count=$(grep -c "^Match found" <<<"$out" || true)
    if ! grep -qF "$expected" <<<"$out"; then
        echo "FAIL: $label — missing: $expected"
        echo "$out" | sed 's/^/    /'
        fails=$((fails + 1))
    elif [ "$count" -ne "$want_count" ]; then
        echo "FAIL: $label — expected $want_count match(es), got $count"
        echo "$out" | sed 's/^/    /'
        fails=$((fails + 1))
    else
        echo "PASS: $label"
    fi
}

# 1. Real-world fixture: must be found uniquely in the original search bounds.
out=$($BIN -175000 -75000 -64 -40 -75000 -25000 0 test/fixtures/formation_a.txt 0)
check "fixture formation_a" "Match found at [-108723, -54, -69736]" "$out"

# 2. Fixture at the exact single cell (regression: the old grid math searched
#    zero blocks for single-cell x/z ranges and missed max bounds).
out=$($BIN -108723 -108723 -54 -54 -69736 -69736 0 test/fixtures/formation_a.txt 0)
check "fixture single cell" "Match found at [-108723, -54, -69736]" "$out"

# 3. Fixture with the origin exactly at each corner of the search box.
out=$($BIN -108723 -108000 -54 -40 -69736 -69000 0 test/fixtures/formation_a.txt 0)
check "fixture at min corner" "Match found at [-108723, -54, -69736]" "$out"
out=$($BIN -109000 -108723 -64 -54 -70000 -69736 0 test/fixtures/formation_a.txt 0)
check "fixture at max corner" "Match found at [-108723, -54, -69736]" "$out"

# 4. Synthetic round-trips: random origins, both versions, every direction,
#    with ~40% side faces. The searcher must recover the origin uniquely.
seed=1
for version in 0 1; do
    for dir in 0 1 2 3; do
        ox=$(( (seed * 7919 + RANDOM) % 120001 - 60000 ))
        oy=$(( (RANDOM % 300) - 60 ))
        oz=$(( (seed * 104729 + RANDOM) % 120001 - 60000 ))
        $GEN "$version" "$dir" "$ox" "$oy" "$oz" 24 40 "$seed" > "$TMP/form.txt"

        out=$($BIN $((ox - 400)) $((ox + 400)) $((oy - 16)) $((oy + 16)) $((oz - 400)) $((oz + 400)) "$version" "$TMP/form.txt" "$dir")
        check "synthetic v$version dir$dir" "Match found at [$ox, $oy, $oz]" "$out"

        out=$($BIN $((ox - 400)) $((ox + 400)) $((oy - 16)) $((oy + 16)) $((oz - 400)) $((oz + 400)) "$version" "$TMP/form.txt" all)
        check "synthetic v$version dir$dir (all mode)" \
              "Match found at [$ox, $oy, $oz] facing ${DIRNAMES[$dir]}" "$out"

        seed=$((seed + 1))
    done
done

# 5. GPU vs CPU differential: on small volumes the GPU must produce exactly
#    the same match set as the independent CPU implementation — including
#    dense many-match cases that exercise every position (any enumeration bug
#    in the warp-compacted kernel shows up here deterministically).
CPU=build/cpu_search
diff_case() { # <label> <args...>
    local label=$1; shift
    local g c
    g=$("$BIN" "$@" 2>/dev/null | grep "^Match found" | sort) || true
    c=$("$CPU" "$@" 2>/dev/null | grep "^Match found" | sort) || true
    if [ "$g" == "$c" ] && [ -n "$c" ]; then
        echo "PASS: differential $label ($(wc -l <<<"$c" | tr -d ' ') matches)"
    else
        echo "FAIL: differential $label"
        diff <(echo "$g") <(echo "$c") | head -20 | sed 's/^/    /'
        fails=$((fails + 1))
    fi
}

echo "1 -1 2 2 0" > "$TMP/one.txt"           # single top block: ~25% of all positions match
diff_case "dense 1-block v0" 100 139 10 29 -220 -181 0 "$TMP/one.txt" 0
diff_case "dense 1-block v1" 100 139 10 29 -220 -181 1 "$TMP/one.txt" 0
printf '0 0 0 1 0\n2 0 1 1 1\n1 1 2 3 0\n' > "$TMP/three.txt"
diff_case "weak 3-block v0 all-mode" -3010 -2981 40 59 512 541 0 "$TMP/three.txt" all
diff_case "weak 3-block v1 dir2" -3010 -2981 40 59 512 541 1 "$TMP/three.txt" 2
$GEN 0 1 7777 30 -4444 24 40 99 > "$TMP/narrow.txt"
diff_case "narrow x window (nx=3)" 7776 7778 14 46 -4460 -4428 0 "$TMP/narrow.txt" 1
diff_case "single row (nx=1,nz=1)" 7777 7777 14 46 -4444 -4444 0 "$TMP/narrow.txt" 1

# 6. Input validation must fail loudly, not run garbage.
if $BIN 0 100 0 10 0 100 0 <(echo "1 2 3 9 0") 0 >"$TMP/bad.out" 2>&1; then
    echo "FAIL: invalid rotation accepted"; fails=$((fails + 1))
else
    echo "PASS: invalid rotation rejected"
fi
if $BIN 100 0 0 10 0 100 0 test/fixtures/formation_a.txt 0 >"$TMP/bad2.out" 2>&1; then
    echo "FAIL: min>max accepted"; fails=$((fails + 1))
else
    echo "PASS: min>max rejected"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "e2e: all tests passed"
else
    echo "e2e: $fails test(s) FAILED"
    exit 1
fi
