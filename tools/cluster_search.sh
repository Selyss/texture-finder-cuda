#!/usr/bin/env bash
# Runs one search across several Slurm jobs and prints the combined result.
#
# The search is cut into one piece per job (--part I/N). Each job searches its
# piece with every GPU it was given, and when all are done the pieces are
# merged and checked: the result is printed only if every piece is present
# exactly once. A piece whose job failed is submitted once more.
#
# Usage:
#   tools/cluster_search.sh [options] -- <search arguments>
#
#   tools/cluster_search.sh --jobs 16 --account def-someone_gpu --gpu-type a100 -- \
#       formation.txt --version 1.21.4 --y -64 100 --center 0 0 --radius 2000000
#
# Options:
#   --jobs N        pieces, and jobs running at once (default 4)
#   --gpus G        GPUs per job (default 1); a job shares its piece between them
#   --gpu-type T    GPU model to request, such as a100 (default: any)
#   --account A     Slurm account (default: $SLURM_ACCOUNT or $SBATCH_ACCOUNT)
#   --partition P   Slurm partition (default: the cluster's choice)
#   --time T        time limit per job (default 01:00:00)
#   --cpus C        CPU cores per job (default 2 per GPU, at least 4)
#   --mem M         memory per job (default 16G)
#   --modules "M"   environment modules to load in each job, such as "gcc/12.3 cuda/12.9"
#   --dir D         where the pieces' results go (default tf-search-<date>-<time>)
#   --bin B         the search binary (default build/main beside this script)
#   --local         no Slurm: run the pieces one after another on this machine
#
# The search arguments are those of the binary, without --part, --json or --quiet.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
jobs=4 gpus=1 gpu_type= account=${SLURM_ACCOUNT:-${SBATCH_ACCOUNT:-}} partition= time=01:00:00
cpus= mem=16G modules= dir= bin=$here/../build/main local_run=0

fail() { echo "Error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case $1 in
        --jobs) jobs=$2; shift 2 ;;
        --gpus) gpus=$2; shift 2 ;;
        --gpu-type) gpu_type=$2; shift 2 ;;
        --account) account=$2; shift 2 ;;
        --partition) partition=$2; shift 2 ;;
        --time) time=$2; shift 2 ;;
        --cpus) cpus=$2; shift 2 ;;
        --mem) mem=$2; shift 2 ;;
        --modules) modules=$2; shift 2 ;;
        --dir) dir=$2; shift 2 ;;
        --bin) bin=$2; shift 2 ;;
        --local) local_run=1; shift ;;
        --help|-h) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        --) shift; break ;;
        *) fail "unknown option $1 (search arguments go after --)" ;;
    esac
done
[ $# -gt 0 ] || fail "no search arguments; put them after --"
case $jobs in ''|*[!0-9]*|0) fail "--jobs must be a positive number" ;; esac
for word in "$@"; do
    case $word in --part|--json|--quiet) fail "$word is set by this script; leave it out" ;; esac
done
[ -x "$bin" ] || fail "no binary at $bin (build it first, or pass --bin)"
command -v python3 >/dev/null || fail "python3 is needed to merge the pieces"

dir=${dir:-tf-search-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$dir"
merge=("$here/merge_parts.py")

if [ "$local_run" = 1 ]; then
    for piece in $(seq 1 "$jobs"); do
        "$bin" "$@" --part "$piece/$jobs" --json > "$dir/part-$piece.json"
    done
    exec "${merge[@]}" "$dir"/part-*.json
fi

command -v sbatch >/dev/null || fail "sbatch not found; use --local to run without Slurm"
[ -n "$account" ] || fail "no Slurm account; pass --account"
cpus=${cpus:-$((gpus * 2 > 4 ? gpus * 2 : 4))}
gres=gpu:${gpu_type:+$gpu_type:}$gpus

# One job script for every piece; the array index is the piece number.
script=$dir/job.sh
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    [ -z "$modules" ] || printf 'module load %s\n' "$modules"
    printf 'exec %q' "$bin"
    printf ' %q' "$@"
    printf ' --part "$SLURM_ARRAY_TASK_ID/%s" --json > %q/part-"$SLURM_ARRAY_TASK_ID".json\n' "$jobs" "$dir"
} > "$script"

submit() { # <array spec>: submits and waits for every job of the array to end
    sbatch --wait --parsable --array="$1" --account="$account" ${partition:+--partition="$partition"} \
        --gres="$gres" --cpus-per-task="$cpus" --mem="$mem" --time="$time" \
        --job-name=tf-search --output="$dir/part-%a.log" "$script" || true
}

echo "Submitting $jobs jobs ($gres each); results in $dir" >&2
submit "1-$jobs" >&2
if missing=$("${merge[@]}" "$dir"/part-*.json 2>/dev/null); then
    echo "$missing"
    exit 0
fi
# Exit status 2 lists the pieces that did not finish. Anything else is an error.
missing=$("${merge[@]}" "$dir"/part-*.json 2>/dev/null | sed -n 's/^Missing: //p' | tr ' ' ',')
[ -n "$missing" ] || exec "${merge[@]}" "$dir"/part-*.json
echo "Pieces $missing did not finish; submitting them once more" >&2
submit "$missing" >&2
exec "${merge[@]}" "$dir"/part-*.json
