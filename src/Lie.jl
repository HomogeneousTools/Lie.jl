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

# ─── Weylloop — systematic Weyl orbit traversal (LiE-style) ────────────────
include("Weylloop.jl")

# ─── Characters and representation ring ─────────────────────────────────────
include("Characters.jl")

# ─── Cache management ───────────────────────────────────────────────────────

"""
    clear_all_caches!()

Clear all internal caches used by Lie.jl.

This function empties the following caches:
- Root system cache (singleton RootSystem instances per Dynkin type)
- Longest Weyl element cache (cached per Dynkin type)
- Freudenthal formula cache (weight multiplicity computations)
- Tensor product cache (tensor product decompositions)
- Symmetric power cache (symmetric power decompositions)
- Exterior power cache (exterior power decompositions)

Caches are automatically populated on demand. Clearing them can be useful for:
- Benchmarking (to measure cold-start performance)
- Memory management (to free memory after large computations)
- Testing (to ensure reproducibility)

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> tensor_product(ω₁, ω₁);  # populates caches

julia> clear_all_caches!()      # clears all caches
```
"""
function clear_all_caches!()
  empty!(_root_system_cache)
  empty!(_longest_element_cache)
  empty!(_coset_reps_cache)
  empty!(_freudenthal_cache)
  empty!(_tensor_cache)
  empty!(_symmetric_power_cache)
  empty!(_exterior_power_cache)
  return nothing
end

export clear_all_caches!

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

# Littlewood–Richardson: precompile for TypeA
for _N in 1:9
  _DT = TypeA{_N}
  _R = _N
  precompile(lr_tensor_product, (WeightLatticeElem{_DT,_R}, WeightLatticeElem{_DT,_R}))
end

# ─── Startup banner ────────────────────────────────────────────────────────

function _print_banner()
  v = pkgversion(@__MODULE__)
  version_str = v === nothing ? "dev" : string(v)

  println()
  # Row 1: ▖ ▘     ▘▜
  printstyled("▖ ▘"; color=:white)
  print("     ")
  printstyled("▘"; color=:red)
  printstyled("▜ "; color=:magenta)
  println(" │  semisimple Lie algebras: root systems,")

  # Row 2: ▌ ▌█▌   ▌▐
  printstyled("▌ ▌█▌"; color=:white)
  print("   ")
  printstyled("▌"; color=:red)
  printstyled("▐ "; color=:magenta)
  println(" │  Weyl groups, and representations")

  # Row 3: ▙▖▌▙▖▗  ▌▐▖
  printstyled("▙▖▌▙▖"; color=:white)
  printstyled("▗"; color=:blue)
  print("  ")
  printstyled("▌"; color=:red)
  printstyled("▐▖"; color=:magenta)
  println(" │")

  # Row 4:        ▙▌
  print("       ")
  printstyled("▙▌"; color=:red)
  println("   │  Docs:    https://homogeneous.tools/lie.jl")

  print("            ")
  println("   │  Version: ", version_str)
end

function __init__()
  if displaysize(stdout)[2] >= 60
    _print_banner()
  end
  return nothing
end

end # module
