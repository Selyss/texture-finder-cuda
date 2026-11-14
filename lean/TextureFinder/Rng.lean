import Std

namespace TextureFinder

def positionInput (x y z : BitVec 32) : BitVec 64 :=
  (x * 3129871).signExtend 64 ^^^ (z.signExtend 64 * 116129781) ^^^ y.signExtend 64

def positionMix (u : BitVec 64) : BitVec 64 :=
  u * u * 42317861 + u * 11

def coordRandom (mix : BitVec 64) : BitVec 64 :=
  mix.sshiftRight 16

def absWord (r : BitVec 32) : BitVec 32 :=
  if r.msb then -r else r

def absMod (r : BitVec 32) (modulus : Nat) : Nat :=
  (absWord r).toNat % modulus

def modernFromSeed (seed : BitVec 64) (modulus : Nat) : Nat :=
  let state := ((seed ^^^ 0x5DEECE66D) * 0x5DEECE66D + 11).setWidth 48
  let next := (state >>> 17).zeroExtend 64
  (((4 * next) >>> 31).toNat) % modulus

def legacyState (seed : BitVec 64) : BitVec 64 :=
  ((seed ^^^ 0x5DEECE66D).setWidth 48).zeroExtend 64

def legacyStep (state : BitVec 64) : BitVec 64 :=
  state * 0xBB20B4600A69 + 0x40942DE6BA

def legacyRand (seed : BitVec 64) : BitVec 32 :=
  (legacyStep (legacyState seed) >>> 16).setWidth 32

def legacyFromSeed (seed : BitVec 64) (modulus : Nat) : Nat :=
  absMod (legacyRand seed) modulus

def vanilla12Rand (mix : BitVec 64) : BitVec 32 :=
  (mix.setWidth 32).sshiftRight 16

def vanilla12FromMix (mix : BitVec 64) (modulus : Nat) : Nat :=
  absMod (vanilla12Rand mix) modulus

def staffordMix13 (z : BitVec 64) : BitVec 64 :=
  let z := (z ^^^ (z >>> 30)) * 0xBF58476D1CE4E5B9
  let z := (z ^^^ (z >>> 27)) * 0x94D049BB133111EB
  z ^^^ (z >>> 31)

def sodiumFromSeed (seed : BitVec 64) (modulus : Nat) : Nat :=
  let s := (seed ^^^ (seed >>> 33)) * 0xFF51AFD7ED558CCD
  let s := (s ^^^ (s >>> 33)) * 0xC4CEB9FE1A85EC53
  let s := (s ^^^ (s >>> 33)) + 0x9E3779B97F4A7C15
  absMod ((staffordMix13 s + staffordMix13 (s + 0x9E3779B97F4A7C15)).setWidth 32) modulus

def sodium19FromSeed (seed : BitVec 64) (modulus : Nat) : Nat :=
  let l := seed ^^^ 0x6A09E667F3BCC909
  let mixed := staffordMix13 l
  let sum := mixed + staffordMix13 (l + 0x9E3779B97F4A7C15)
  let rot := (sum <<< 17) ||| (sum >>> 47)
  absMod ((rot + mixed).setWidth 32) modulus

def texture (version : Nat) (x y z : BitVec 32) (modulus : Nat) : Nat :=
  let mix := positionMix (positionInput x y z)
  let seed := coordRandom mix
  match version with
  | 1 => legacyFromSeed seed modulus
  | 2 => vanilla12FromMix mix modulus
  | 3 => sodiumFromSeed seed modulus
  | 4 => sodium19FromSeed seed modulus
  | _ => modernFromSeed seed modulus

end TextureFinder
