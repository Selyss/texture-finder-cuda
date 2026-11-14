import Mathlib.Tactic.Ring
import Std

namespace TextureFinder

/-- T7. Filtering on one subset and verifying its complement is exact. -/
theorem partition_match (blocks : List α) (onLattice matchBlock : α → Bool) :
    ((blocks.filter onLattice).all matchBlock && (blocks.filter (fun b => !onLattice b)).all matchBlock) =
      blocks.all matchBlock := by
  apply Bool.eq_iff_iff.mpr
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_filter, Bool.not_eq_true']
  constructor
  · rintro ⟨hon, hoff⟩ b hb
    cases h : onLattice b
    · exact hoff b ⟨hb, h⟩
    · exact hon b ⟨hb, h⟩
  · intro h
    exact ⟨fun b hb => h b hb.1, fun b hb => h b hb.1⟩

/-- The candidate coset test in lattice.cpp selects exactly the on-lattice positions. -/
theorem lattice_coset (a b stride x z dx dz : Int) :
    (a * dx + b * dz + (a * x + b * z) % stride) % stride =
      (a * (x + dx) + b * (z + dz)) % stride := by
  have h : a * (x + dx) + b * (z + dz) = (a * dx + b * dz) + (a * x + b * z) := by ring
  rw [h, Int.add_emod (a * dx + b * dz) ((a * x + b * z) % stride), Int.emod_emod,
    ← Int.add_emod]

/-- T8. Extract a candidate word starting at a bit offset in the first row word. -/
def extractWord (lo hi : BitVec 64) (offset : Nat) : BitVec 64 :=
  if offset = 0 then lo else (lo >>> offset) ||| (hi <<< (64 - offset))

theorem extractWord_bit (lo hi : BitVec 64) (hoffset : offset < 64) (hj : j < 64) :
    (extractWord lo hi offset).getLsbD j =
      if offset + j < 64 then lo.getLsbD (offset + j) else hi.getLsbD (offset + j - 64) := by
  by_cases hz : offset = 0
  · simp [extractWord, hz, hj]
  · by_cases h : offset + j < 64
    · have hlow : j < 64 - offset := by omega
      simp [extractWord, hz, h, hj, hlow]
    · have hhigh : ¬ j < 64 - offset := by omega
      have hindex : j - (64 - offset) = offset + j - 64 := by omega
      have hlo := BitVec.getLsbD_ge lo (offset + j) (by omega : 64 ≤ offset + j)
      simp only [extractWord, hz, if_false, BitVec.getLsbD_or, BitVec.getLsbD_ushiftRight,
        BitVec.getLsbD_shiftLeft, hj, hhigh, decide_true, decide_false, Bool.not_false,
        Bool.true_and, h, hindex, hlo, Bool.false_or]

/-- The mask for one block reads one plane for a side and two planes for a top. -/
def blockMask (plane0 plane1 : BitVec 64) (want0 want1 top : Bool) : BitVec 64 :=
  let low := if want0 then plane0 else ~~~plane0
  let high := if want1 then plane1 else ~~~plane1
  if top then low &&& high else low

theorem blockMask_bit (plane0 plane1 : BitVec 64) (want0 want1 top : Bool) (hj : j < 64) :
    (blockMask plane0 plane1 want0 want1 top).getLsbD j =
      ((plane0.getLsbD j == want0) && (!top || (plane1.getLsbD j == want1))) := by
  cases top <;> cases want0 <;> cases want1 <;> simp [blockMask, hj]

/-- Each surviving bit is a valid candidate that passes every supplied block mask. -/
theorem combinedMask_bit (valid : BitVec 64) (masks : List (BitVec 64)) (j : Nat) :
    (masks.foldl (· &&& ·) valid).getLsbD j =
      (valid.getLsbD j && masks.all (fun mask => mask.getLsbD j)) := by
  induction masks generalizing valid with
  | nil => simp
  | cons mask masks ih => simp [ih, Bool.and_assoc]

end TextureFinder
