#!/usr/bin/env bash
# Rebuilds the Java ground-truth oracle from scratch on any machine with a
# JDK, and regenerates the dumps consumed by build/oracle_diff. The texture
# classes come UNMODIFIED from the reference repo; Dump.java refuses to
# generate anything if the built-in fixture gate fails.
#
# Usage: test/oracle/regen.sh <workdir> [path-to-TextureRotations-clone]
#   <workdir>/classes  compiled oracle
#   <workdir>/dumps    generated dumps + SHA256SUMS
set -euo pipefail
cd "$(dirname "$0")"

WORK=${1:?usage: regen.sh <workdir> [reference-clone]}
REF=${2:-$WORK/TextureRotations}

mkdir -p "$WORK"
if [ ! -d "$REF/src/main/java/texture" ]; then
    git clone --depth 1 https://github.com/19MisterX98/TextureRotations "$REF"
fi

mkdir -p "$WORK/src/texture"
cp "$REF"/src/main/java/texture/*.java "$WORK/src/texture/"
cp Dump.java "$WORK/src/"

mkdir -p "$WORK/classes"
javac -d "$WORK/classes" "$WORK"/src/texture/*.java "$WORK/src/Dump.java"

java -cp "$WORK/classes" Dump dumps "$WORK/dumps"

cd "$WORK/dumps"
(sha256sum * 2>/dev/null || shasum -a 256 *) | grep -v SHA256SUMS > SHA256SUMS
cat SHA256SUMS
