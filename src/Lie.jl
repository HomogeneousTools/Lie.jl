module Lie

using StaticArrays
using LinearAlgebra: dot as _dot, I as _I

# ─── Type-level Dynkin types ────────────────────────────────────────────────
include("DynkinTypes.jl")

# ─── Cartan matrices (compile-time specialized) ─────────────────────────────
include("CartanMatrix.jl")

# ─── Root systems ───────────────────────────────────────────────────────────
include("RootSystem.jl")

# ─── Weight lattice ─────────────────────────────────────────────────────────
include("WeightLattice.jl")

# ─── Weyl groups ────────────────────────────────────────────────────────────
include("WeylGroup.jl")

# ─── Characters and representation ring ─────────────────────────────────────
include("Characters.jl")

# ─── Precompilation ─────────────────────────────────────────────────────────

# ─── Precompilation ─────────────────────────────────────────────────────────
# Hint the compiler to precompile key methods for all simple Dynkin types
# below rank 10. This reduces first-call latency after module loading.

for _DT in (
  TypeA{1}, TypeA{2}, TypeA{3}, TypeA{4}, TypeA{5},
  TypeA{6}, TypeA{7}, TypeA{8}, TypeA{9},
  TypeB{2}, TypeB{3}, TypeB{4}, TypeB{5},
  TypeB{6}, TypeB{7}, TypeB{8}, TypeB{9},
  TypeC{2}, TypeC{3}, TypeC{4}, TypeC{5},
  TypeC{6}, TypeC{7}, TypeC{8}, TypeC{9},
  TypeD{4}, TypeD{5}, TypeD{6}, TypeD{7}, TypeD{8}, TypeD{9},
  TypeE{6}, TypeE{7}, TypeE{8},
  TypeF4, TypeG2,
)
  _R = rank(_DT)

  # CartanMatrix
  precompile(cartan_matrix, (Type{_DT},))
  precompile(cartan_symmetrizer, (Type{_DT},))
  precompile(cartan_bilinear_form, (Type{_DT},))
  precompile(cartan_matrix_inverse, (Type{_DT},))

  # RootSystem
  precompile(_make_root_system, (Type{_DT},))

  # WeylGroup
  precompile(_weyl_denominator, (Type{_DT},))
  precompile(_weyl_dim_scaled_roots, (Type{_DT},))
  precompile(degree, (Type{_DT}, WeightLatticeElem{_DT,_R}))
  precompile(conjugate_dominant_weight, (WeightLatticeElem{_DT,_R},))
  precompile(weyl_orbit, (Type{_DT}, WeightLatticeElem{_DT,_R}))

  # WeylGroup actions
  precompile(Base.:*, (RootSpaceElem{_DT,_R}, WeylGroupElem{_DT,_R}))
  precompile(Base.:*, (WeightLatticeElem{_DT,_R}, WeylGroupElem{_DT,_R}))

  # Characters
  precompile(freudenthal_formula, (WeightLatticeElem{_DT,_R},))
  precompile(dot_reduce, (WeightLatticeElem{_DT,_R},))
end

end # module
