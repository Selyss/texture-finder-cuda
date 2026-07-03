# Texture rotation semantics and verification

This documents exactly what the searcher computes, why, and how to prove it
is still right — including when a new Minecraft version changes the RNG.

Ground truth is the reference Java implementation
([19MisterX98/TextureRotations](https://github.com/19MisterX98/TextureRotations));
all formulas below are literal translations of it, kept in
[include/texture.cuh](../include/texture.cuh) as the single source shared by
the kernel, the tests, and the tools. Never copy those functions.

## Position hash (all versions)

```java
long l = (long)(x * 3129871) ^ (long)z * 116129781L ^ (long)y;
l = l * l * 42317861L + l * 11L;
seed48 = l >> 16;
```

`x * 3129871` is 32-bit Java int math and wraps; everything after is 64-bit
and wraps. In C++ both would be signed-overflow UB, so the implementation
routes every wrapping step through unsigned arithmetic (bit-identical
results, no UB).

## Legacy: 1.13 – 1.21.1 (`version = 1`, reference `Vanilla21_1Textures`)

```java
seed = (seed48 ^ 0x5DEECE66D) & ((1L << 48) - 1);
rand = (int)((seed * 0xBB20B4600A69L + 0x40942DE6BAL) >>> 16); // 2 LCG steps fused (nextLong upper path)
value = Math.abs(rand) % mod;
```

Notes:
- Java's `>>> 16` vs C++ `>> 16` on the signed 64-bit product makes no
  difference here: the `(int)` cast keeps bits 16–47 either way.
- `Math.abs(Integer.MIN_VALUE)` is negative in Java, but for `mod` 2 and 4
  the final value is 0 either way; the C++ uses an unsigned fold that is
  exactly equivalent for these mods and never UB.

## Modern: 1.21.2+ (`version = 0`, reference `VanillaTextures`)

```java
seed = seed48 ^ 0x5DEECE66D;
seed = seed * 0x5DEECE66D + 11L & ((1L << 48) - 1);
next = (int)(seed >> 48 - 31);
value = (int)((4 * (long)next) >> 31) % mod;   // the 4 is intentional!
```

The game always rolls one of **4** model variants (`nextInt(4)`); the `% mod`
afterwards only reduces what a given face can expose. Computing a side as a
direct 2-bound roll — `(2 * next) >> 31`, i.e. `nextInt(2)` — is **wrong**:
the game never draws that number. (An old copy of this function in the test
suite had exactly that bug; the tests passed because they only exercised
mod 4, where the two forms coincide.)

## Side faces

A side face exposes only the **parity** of the block's 4-variant roll:
`side = top_roll % 2`. Same coordinates, same seed, y included. This holds
for both versions (reference: `RotationInfo` stores `rotation % 2` for sides
and `TextureFinder` compares against `getTexture(x, y, z, 2)`).

## Formation direction

The reference tool has no direction feature; it is this project's addition.
Convention: `rotateFormation` turns the formation a quarter turn,
`(x, z) -> (-z, x)` (east -> south viewed from above), and advances the
expected values one variant step: tops `+1 mod 4`, sides flip parity
(`+1 mod 2`) — a side value must stay in {0, 1} or it could never match a
mod-2 texture value. Directions: 0 = North, 1 = West, 2 = South, 3 = East,
applied as that many quarter turns to the input formation.

If the recreation's orientation is uncertain, pass `all` as the direction:
the searcher tries all four orientations and reports which one matched.

## Verification pyramid

1. **Unit vectors** (`make test`, no GPU needed): hand-verified in-game
   values, the side/top parity invariant, rotation algebra, parser rejection
   cases — all against the real headers, never copies.
2. **Real-world fixture**: [test/fixtures/formation_a.txt](../test/fixtures/formation_a.txt)
   is an independently confirmed formation
   ([expected result](../test/fixtures/formation_a.expected.md)); the unit
   tests check it against the RNG directly, and the e2e suite requires the
   GPU search to find it — including with the origin on every corner of the
   search box, and as a single-cell search (regressions for old
   coverage-gap bugs).
3. **Oracle differential** (`make tools`, then
   `build/oracle_diff <dumps_dir>`): compares ~26M values (25.2M grid bytes
   near origin, 3k world-border extremes, 1M random coordinates) against
   dumps produced by the *actual Java reference code* running on a JVM
   (`/root/oracle/` on the GPU box; regenerate with `regen.sh`). Any
   divergence from Java semantics fails loudly.
4. **Synthetic end-to-end** (`test/e2e.sh`, needs the GPU): generates
   formations with known origins (both versions, all directions, ~40% side
   faces), and requires the search to recover each origin uniquely — plus
   'all'-mode direction identification and input-validation checks.

## When a new Minecraft version changes the RNG

1. Update the reference clone on the box and rerun `/root/oracle/regen.sh`
   (it refuses to produce dumps if the built-in fixture gate fails).
2. Run `build/oracle_diff` — it will show exactly where behavior changed.
3. Add/adjust the corresponding function in `include/texture.cuh` and a
   `version` mapping; the old versions keep their dumps and tests.
4. Re-run the whole pyramid; the real-world fixture must still be found by
   the version it was verified against.
