import Std.Tactic.BVDecide

namespace TextureFinder

private theorem split_add (m hi lo : BitVec w) (hhi : hi &&& m = 0)
    (hlo : lo &&& m = lo) : hi + lo = hi ||| lo := by
  apply BitVec.add_eq_or_of_and_eq_zero
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  have h₁ := congrArg (fun v : BitVec w => v.getLsbD i) hhi
  have h₂ := congrArg (fun v : BitVec w => v.getLsbD i) hlo
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero] at h₁ h₂ ⊢
  generalize hi.getLsbD i = a at *
  generalize lo.getLsbD i = b at *
  generalize m.getLsbD i = c at *
  cases a <;> cases b <;> cases c <;> simp_all

/-- Low-bit mask for an aligned block of size 2^k. -/
def lowMask (w k : Nat) : BitVec w := (BitVec.allOnes k).setWidth w

theorem lowMask_eq (w k : Nat) : lowMask w k = BitVec.ofNat w (2^k - 1) := by
  apply BitVec.eq_of_toNat_eq
  simp [lowMask]

private theorem and_lowMask_eq_self (hkw : k ≤ w) (x : BitVec w) :
    x &&& lowMask w k = x ↔ x.toNat < 2^k := by
  rw [lowMask_eq w k, BitVec.toNat_eq]
  have hmask : 2^k - 1 < 2^w := by
    have := Nat.pow_le_pow_right (by decide : 0 < 2) hkw
    have := Nat.two_pow_pos k
    omega
  rw [BitVec.toNat_and, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hmask,
      Nat.and_two_pow_sub_one_eq_mod]
  constructor
  · intro h
    have := Nat.mod_lt x.toNat (Nat.two_pow_pos k)
    omega
  · exact Nat.mod_eq_of_lt

private theorem xor_low_closed (K j m : BitVec w) (hj : j &&& m = j) :
    ((K &&& m) ^^^ j) &&& m = (K &&& m) ^^^ j := by
  apply BitVec.eq_of_getLsbD_eq
  intro i _
  have h := congrArg (fun v : BitVec w => v.getLsbD i) hj
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_xor] at h ⊢
  generalize K.getLsbD i = a at *
  generalize m.getLsbD i = b at *
  generalize j.getLsbD i = c at *
  cases a <;> cases b <;> cases c <;> simp_all

private theorem xor_block_masked (K s j m : BitVec w) (hs : s &&& m = 0)
    (hj : j &&& m = j) :
    K ^^^ (s + j) = ((K ^^^ s) &&& ~~~m) + ((K &&& m) ^^^ j) := by
  have hhi : ((K ^^^ s) &&& ~~~m) &&& m = 0 := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    simp [hi, Bool.and_assoc]
  rw [split_add m s j hs hj, split_add m _ _ hhi (xor_low_closed K j m hj)]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  have h₁ := congrArg (fun v : BitVec w => v.getLsbD i) hs
  have h₂ := congrArg (fun v : BitVec w => v.getLsbD i) hj
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero] at h₁ h₂
  simp only [BitVec.getLsbD_xor, BitVec.getLsbD_or, BitVec.getLsbD_and,
    BitVec.getLsbD_not, hi, decide_true, Bool.true_and]
  generalize K.getLsbD i = a at *
  generalize s.getLsbD i = b at *
  generalize j.getLsbD i = c at *
  generalize m.getLsbD i = d at *
  cases a <;> cases b <;> cases c <;> cases d <;> simp_all

/-- T1. XOR is an addition after splitting the aligned high bits and low offset. -/
theorem xor_aligned_eq_add (K s j : BitVec w) (hkw : k ≤ w)
    (hs : s &&& lowMask w k = 0) (hj : j.toNat < 2^k) :
    K ^^^ (s + j) = ((K ^^^ s) &&& ~~~(lowMask w k)) + ((K &&& lowMask w k) ^^^ j) :=
  xor_block_masked K s j _ hs ((and_lowMask_eq_self hkw j).mpr hj)

theorem xor_small_eq_add (K y : BitVec w) (hkw : k ≤ w) (hy : y.toNat < 2^k) :
    K ^^^ y = (K &&& ~~~(lowMask w k)) + ((K &&& lowMask w k) ^^^ y) := by
  simpa using xor_aligned_eq_add K 0 y hkw (by simp) hy

theorem xor_offset_in_range (K j : BitVec w) (hkw : k ≤ w) (hj : j.toNat < 2^k) :
    ((K &&& lowMask w k) ^^^ j).toNat < 2^k :=
  (and_lowMask_eq_self hkw _).mp (xor_low_closed K j _ ((and_lowMask_eq_self hkw j).mpr hj))

/-- T2. The low-bit XOR offset is its own inverse on the block. -/
def blockOffset (K : BitVec w) (hkw : k ≤ w) (j : {j : BitVec w // j.toNat < 2^k}) :
    {j : BitVec w // j.toNat < 2^k} :=
  ⟨(K &&& lowMask w k) ^^^ j.val, xor_offset_in_range K j.val hkw j.property⟩

theorem blockOffset_involutive (K : BitVec w) (hkw : k ≤ w)
    (j : {j : BitVec w // j.toNat < 2^k}) :
    blockOffset K hkw (blockOffset K hkw j) = j := by
  apply Subtype.ext
  simp [blockOffset, ← BitVec.xor_assoc]

/-- T3. Complementing both inputs preserves the XOR. -/
theorem xor_not_not (a b : BitVec 64) : a ^^^ b = (~~~a) ^^^ (~~~b) := by
  bv_decide

theorem complement_negative_y (y : BitVec 32) (hy : y.msb = true) :
    (~~~(y.signExtend 64)).toNat < 2^31 := by
  have h : (~~~(y.signExtend 64)) < (0x80000000 : BitVec 64) := by bv_decide
  exact h

end TextureFinder
