import Std.Tactic.BVDecide
import TextureFinder.Rng

namespace TextureFinder

/-- T6. Truncation commutes with the position polynomial. -/
theorem positionMix_low {u v : BitVec 64} (hn : n ≤ 64)
    (h : u.setWidth n = v.setWidth n) :
    (positionMix u).setWidth n = (positionMix v).setWidth n := by
  simp only [positionMix, BitVec.setWidth_add _ _ hn, BitVec.setWidth_mul _ _ hn, h]

private theorem sshiftRight_low {x y : BitVec w} (hn : n + s ≤ w)
    (h : x.setWidth (n + s) = y.setWidth (n + s)) :
    (x.sshiftRight s).setWidth n = (y.sshiftRight s).setWidth n := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hidx : s + i < n + s := by omega
  have hw : s + i < w := by omega
  have hiw : ¬ w ≤ i := by omega
  have hb := congrArg (fun v : BitVec (n + s) => v.getLsbD (s + i)) h
  simp only [BitVec.getLsbD_setWidth, hidx, decide_true, Bool.true_and] at hb
  simpa only [BitVec.getLsbD_setWidth, BitVec.getLsbD_sshiftRight, hi, hw, hiw,
    decide_true, decide_false, Bool.not_false, Bool.true_and, if_true] using hb

private theorem ushiftRight_low {x y : BitVec w}
    (h : x.setWidth (n + s) = y.setWidth (n + s)) :
    (x >>> s).setWidth n = (y >>> s).setWidth n := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have hidx : s + i < n + s := by omega
  have hb := congrArg (fun v : BitVec (n + s) => v.getLsbD (s + i)) h
  simpa only [BitVec.getLsbD_setWidth, BitVec.getLsbD_ushiftRight, hi, hidx,
    decide_true, Bool.true_and] using hb

private theorem legacyState_low {x y : BitVec 64} (hn : n ≤ 48)
    (h : x.setWidth n = y.setWidth n) :
    (legacyState x).setWidth n = (legacyState y).setWidth n := by
  simp only [legacyState, BitVec.zeroExtend_eq_setWidth,
    BitVec.setWidth_setWidth_of_le _ (Nat.le_trans hn (by decide : 48 ≤ 64)),
    BitVec.setWidth_setWidth_of_le _ hn, BitVec.setWidth_xor, h]

private theorem legacyStep_low {x y : BitVec 64} (hn : n ≤ 64)
    (h : x.setWidth n = y.setWidth n) :
    (legacyStep x).setWidth n = (legacyStep y).setWidth n := by
  simp only [legacyStep, BitVec.setWidth_add _ _ hn, BitVec.setWidth_mul _ _ hn, h]

private theorem legacyRand_low {u v : BitVec 64} (hn : n + 16 ≤ 48)
    (h : u.setWidth (n + 16 + 16) = v.setWidth (n + 16 + 16))
    (hw : n + 16 + 16 ≤ 64) :
    (legacyRand (coordRandom (positionMix u))).setWidth n =
      (legacyRand (coordRandom (positionMix v))).setWidth n := by
  have hs := sshiftRight_low hw (positionMix_low hw h)
  have hstep := legacyStep_low (Nat.le_trans hn (by decide : 48 ≤ 64))
    (legacyState_low hn hs)
  simpa only [legacyRand, coordRandom, BitVec.setWidth_setWidth_of_le _ (by omega : n ≤ 32)]
    using ushiftRight_low hstep

private theorem vanilla12Rand_low {u v : BitVec 64} (hn : n + 16 ≤ 32)
    (h : u.setWidth (n + 16) = v.setWidth (n + 16)) :
    (vanilla12Rand (positionMix u)).setWidth n = (vanilla12Rand (positionMix v)).setWidth n := by
  apply sshiftRight_low hn
  simpa only [BitVec.setWidth_setWidth_of_le _ hn] using
    positionMix_low (Nat.le_trans hn (by decide : 32 ≤ 64)) h

theorem absMod_parity (r : BitVec 32) : absMod r 2 = (r.setWidth 1).toNat := by
  change (absWord r).toNat % 2 = (r.setWidth 1).toNat
  rw [← BitVec.toNat_setWidth 1 (absWord r)]
  exact congrArg BitVec.toNat (by unfold absWord; bv_decide)

theorem neg_low_two_bits (r : BitVec 32) :
    (-r).setWidth 2 = -(r.setWidth 2) := by bv_decide

theorem neg_mod_four :
    -(0 : BitVec 2) = 0 ∧ -(1 : BitVec 2) = 3 ∧
    -(2 : BitVec 2) = 2 ∧ -(3 : BitVec 2) = 1 := by decide

/-- Preserve the two even values and collapse both odd values to the tag 1. -/
def evenTag (r : BitVec 2) : BitVec 2 :=
  if r.getLsbD 0 then 1 else r

theorem absWord_evenTag (r : BitVec 32) :
    evenTag ((absWord r).setWidth 2) = evenTag (r.setWidth 2) := by
  unfold absWord evenTag
  bv_decide

theorem legacy_parity_low33 {u v : BitVec 64} (h : u.setWidth 33 = v.setWidth 33) :
    legacyFromSeed (coordRandom (positionMix u)) 2 = legacyFromSeed (coordRandom (positionMix v)) 2 := by
  simpa only [legacyFromSeed, absMod_parity] using
    congrArg BitVec.toNat (legacyRand_low (n := 1) (by decide) h (by decide))

theorem legacy_even_low34 {u v : BitVec 64} (h : u.setWidth 34 = v.setWidth 34) :
    evenTag ((absWord (legacyRand (coordRandom (positionMix u)))).setWidth 2) =
      evenTag ((absWord (legacyRand (coordRandom (positionMix v)))).setWidth 2) := by
  simp only [absWord_evenTag, legacyRand_low (n := 2) (by decide) h (by decide)]

theorem vanilla12_low32 {u v : BitVec 64} (h : u.setWidth 32 = v.setWidth 32) (modulus : Nat) :
    vanilla12FromMix (positionMix u) modulus = vanilla12FromMix (positionMix v) modulus := by
  unfold vanilla12FromMix vanilla12Rand
  rw [positionMix_low (by decide : 32 ≤ 64) h]

theorem vanilla12_parity_low17 {u v : BitVec 64} (h : u.setWidth 17 = v.setWidth 17) :
    vanilla12FromMix (positionMix u) 2 = vanilla12FromMix (positionMix v) 2 := by
  simpa only [vanilla12FromMix, absMod_parity] using
    congrArg BitVec.toNat (vanilla12Rand_low (n := 1) (by decide) h)

theorem vanilla12_even_low18 {u v : BitVec 64} (h : u.setWidth 18 = v.setWidth 18) :
    evenTag ((absWord (vanilla12Rand (positionMix u))).setWidth 2) =
      evenTag ((absWord (vanilla12Rand (positionMix v))).setWidth 2) := by
  simp only [absWord_evenTag, vanilla12Rand_low (n := 2) (by decide) h]

end TextureFinder
