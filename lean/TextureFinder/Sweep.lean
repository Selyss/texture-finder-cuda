import Mathlib.Data.BitVec
import Mathlib.Tactic.Ring
import TextureFinder.Alignment
import TextureFinder.Rng

namespace TextureFinder

private def firstDiff (u : BitVec 64) : BitVec 64 :=
  42317861 * (2 * u + 1) + 11

/-- T4. First and second forward differences in the ring of 64-bit words. -/
theorem mix_first_diff (u : BitVec 64) :
    positionMix (u + 1) - positionMix u = firstDiff u := by
  unfold positionMix firstDiff
  ring

theorem mix_second_diff (u : BitVec 64) :
    (positionMix (u + 2) - positionMix (u + 1)) -
      (positionMix (u + 1) - positionMix u) = 2 * 42317861 := by
  unfold positionMix
  ring

def sweep (base : BitVec 64) : Nat → BitVec 64 × BitVec 64
  | 0 => (positionMix base, firstDiff base)
  | t + 1 =>
    let (mix, diff) := sweep base t
    (mix + diff, diff + 2 * 42317861)

theorem sweep_invariant (base : BitVec 64) (t : Nat) :
    sweep base t = (positionMix (base + BitVec.ofNat 64 t), firstDiff (base + BitVec.ofNat 64 t)) := by
  induction t with
  | zero => simp [sweep]
  | succ t ih =>
    rw [sweep, ih, BitVec.ofNat_add]
    change _ = (positionMix (base + (BitVec.ofNat 64 t + 1)),
      firstDiff (base + (BitVec.ofNat 64 t + 1)))
    apply Prod.ext
    · change positionMix _ + firstDiff _ = positionMix _
      unfold positionMix firstDiff
      ring
    · change firstDiff _ + 2 * 42317861 = firstDiff _
      unfold firstDiff
      ring

/-- T5. The re-indexed recurrence computes the reference mix on an aligned block. -/
theorem sweep_aligned (K s j : BitVec 64) (hkw : k ≤ 64)
    (hs : s &&& lowMask 64 k = 0) (hj : j.toNat < 2^k) :
    (sweep ((K ^^^ s) &&& ~~~(lowMask 64 k)) (((K &&& lowMask 64 k) ^^^ j).toNat)).1 =
      positionMix (K ^^^ (s + j)) := by
  rw [sweep_invariant, BitVec.ofNat_toNat, BitVec.setWidth_eq, xor_aligned_eq_add K s j hkw hs hj]

/-- In recurrence order, XORing the counter recovers the corresponding y offset. -/
theorem sweep_reindexed (K s : BitVec 64) (hkw : k ≤ 64) (hs : s &&& lowMask 64 k = 0)
    (t : {t : BitVec 64 // t.toNat < 2^k}) :
    (sweep ((K ^^^ s) &&& ~~~(lowMask 64 k)) t.val.toNat).1 =
      positionMix (K ^^^ (s + (blockOffset K hkw t).val)) := by
  let j := blockOffset K hkw t
  have h := congrArg Subtype.val (blockOffset_involutive K hkw t)
  change ((K &&& lowMask 64 k) ^^^ j.val) = t.val at h
  simpa only [h] using sweep_aligned K s j.val hkw hs j.property

/-- For negative y, the sweep runs over aligned blocks of the complemented y values. -/
theorem sweep_negative_aligned (K : BitVec 64) (y : BitVec 32) (s j : BitVec 64)
    (hkw : k ≤ 64) (hs : s &&& lowMask 64 k = 0) (hj : j.toNat < 2^k)
    (hy : ~~~(y.signExtend 64) = s + j) :
    (sweep (((~~~K) ^^^ s) &&& ~~~(lowMask 64 k))
      ((((~~~K) &&& lowMask 64 k) ^^^ j).toNat)).1 =
      positionMix (K ^^^ y.signExtend 64) := by
  rw [sweep_aligned (~~~K) s j hkw hs hj, ← hy, ← xor_not_not]

end TextureFinder
