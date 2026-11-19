#!/usr/bin/env python3
"""Combines the --json results of a search that was run in pieces with --part.

Usage: merge_parts.py [--json] part-*.json

Refuses to print a result unless every piece of one search is present exactly
once and finished, so a missing or repeated piece cannot pass as an answer.
Exit status: 0 merged, 1 bad input, 2 pieces missing (their numbers are
printed one per line on stdout after "Missing:", for resubmitting).
"""
import json
import sys

FACINGS = ["North", "West", "South", "East"]


def fail(message):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def load(path):
    try:
        with open(path) as handle:
            text = handle.read().strip()
    except OSError as error:
        fail(f"{path}: {error.strerror}")
    if not text:
        return None  # a piece that was started but wrote nothing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fail(f"{path}: not a result written by --json")


def main(arguments):
    as_json = "--json" in arguments
    paths = [argument for argument in arguments if argument != "--json"]
    if not paths:
        fail("no result files given")

    pieces = {}
    first = None
    for path in paths:
        result = load(path)
        if result is None:
            continue
        if first is None:
            first = result
        for key in ("search_id", "version", "facing"):
            if result[key] != first[key]:
                fail(f"{path}: belongs to a different search ({key} differs)")
        index, count = result["part"]
        if count != first["part"][1]:
            fail(f"{path}: the search was cut into {count} pieces here and {first['part'][1]} elsewhere")
        if result["status"] != "finished":
            continue  # stopped early: counts as missing
        if index in pieces:
            fail(f"{path}: piece {index} is given twice")
        pieces[index] = result
    if first is None:
        fail("every result file is empty")

    count = first["part"][1]
    missing = [index for index in range(1, count + 1) if index not in pieces]
    if missing:
        print("Missing: " + " ".join(str(index) for index in missing))
        print(f"Error: {len(missing)} of {count} pieces are missing or unfinished", file=sys.stderr)
        sys.exit(2)

    ordered = [pieces[index] for index in range(1, count + 1)]
    # Pieces are consecutive runs of z rows, so together they must tile the range.
    for before, after in zip(ordered, ordered[1:]):
        if after["bounds"]["z"][0] != before["bounds"]["z"][1] + 1:
            fail(f"pieces {before['part'][0]} and {after['part'][0]} do not meet")

    matches = [match for piece in ordered for match in piece["matches"]]
    matches.sort(key=lambda match: (match["x"], match["y"], match["z"], FACINGS.index(match["facing"])))
    total = sum(piece["total_matches"] for piece in ordered)
    merged = {
        "status": "finished",
        "version": first["version"],
        "search_id": first["search_id"],
        "facing": first["facing"],
        "pieces": count,
        "bounds": {
            "x": first["bounds"]["x"],
            "y": first["bounds"]["y"],
            "z": [ordered[0]["bounds"]["z"][0], ordered[-1]["bounds"]["z"][1]],
        },
        "total_matches": total,
        "truncated": any(piece["truncated"] for piece in ordered),
        "matches": matches,
        "kernel_seconds": sum(piece["kernel_seconds"] for piece in ordered),
        "longest_piece_seconds": max(piece["seconds"] for piece in ordered),
    }
    if as_json:
        print(json.dumps(merged, separators=(",", ":")))
        return

    print(f"Pieces: {count}, all present")
    print(f"Version: {merged['version']}")
    for match in matches:
        line = f"Match found at [{match['x']}, {match['y']}, {match['z']}]"
        if merged["facing"] == "all":
            line += f" facing {match['facing']}"
        print(line)
    print("No matches found." if total == 0 else f"{total} {'match' if total == 1 else 'matches'}")
    if merged["truncated"]:
        print(f"Warning: only {len(matches)} of {total} matches are listed; shrink the search area.")
    print(f"Kernel time: {merged['kernel_seconds']:.6g} seconds over all pieces")
    print(f"Longest piece: {merged['longest_piece_seconds']:.6g} seconds")


if __name__ == "__main__":
    main(sys.argv[1:])
