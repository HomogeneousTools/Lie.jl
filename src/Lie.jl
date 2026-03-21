# ═══════════════════════════════════════════════════════════════════════════════
#  Lie.jl — computations with semisimple Lie algebras
# ═══════════════════════════════════════════════════════════════════════════════

module Lie

using LRUCache
using PrecompileTools
using Preferences
using StaticArrays
using LinearAlgebra: dot as _dot, I as _I

# ─── Cache budget (computed before any cache is created) ─────────────────────

const _MIN_CACHE_BUDGET = 256 * 1024^2  # 256 MiB floor

_default_cache_budget() = max(div(Int(Sys.total_memory()), 4), _MIN_CACHE_BUDGET)

const _DEFAULT_DOMINANT_FRAC = 0.30
const _DEFAULT_TENSOR_FRAC = 0.40
const _DEFAULT_SYM_POWER_FRAC = 0.15
const _DEFAULT_EXT_POWER_FRAC = 0.15

_cache_maxsize(budget::Int, fraction::Float64) = max(1, round(Int, budget * fraction))

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

# ─── Cache configuration (Preferences + runtime API) ────────────────────────
include("CacheConfig.jl")

# ─── Cache management ───────────────────────────────────────────────────────

"""
    clear_all_caches!()

Clear all internal caches used by Lie.jl.  Alias for `clear_caches!`.
"""
clear_all_caches!() = clear_caches!()

export clear_all_caches!, clear_caches!, configure_caches!, cache_info

# ─── Precompilation ─────────────────────────────────────────────────────────
# @compile_workload executes real code during precompilation, so Julia
# transitively caches every callee (SMatrix constructors, getindex, etc.),
# not just the top-level method signatures that bare precompile() would cover.

@compile_workload begin
  # CartanMatrix, RootSystem, WeylGroup infrastructure, and Characters
  # for all simple Dynkin types below rank 10.
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
    # CartanMatrix
    cartan_matrix(_DT)
    cartan_symmetrizer(_DT)
    cartan_bilinear_form(_DT)
    cartan_matrix_inverse(_DT)

    # RootSystem
    _make_root_system(_DT)

    # WeylGroup internal helpers
    _weyl_denominator(_DT)
    _weyl_dim_scaled_roots(_DT)

    # WeightLattice + public WeylGroup API
    _ω₁ = fundamental_weight(_DT, 1)
    degree(_DT, _ω₁)
    conjugate_dominant_weight(_ω₁)
    _minus_ω₁ = -_ω₁
    degree(_DT, _ω₁)
    conjugate_dominant_weight(_ω₁)
    conjugate_dominant_weight(_minus_ω₁)
    conjugate_dominant_weight_with_length(_ω₁)
    conjugate_dominant_weight_with_length(_minus_ω₁)
    weyl_orbit(_DT, _ω₁)

    # WeylGroup actions on roots and weights
    simple_root(RootSystem(_DT), 1) * gen(weyl_group(_DT), 1)
    _ω₁ * gen(weyl_group(_DT), 1)

    # Characters
    freudenthal_formula(_ω₁)
    dot_reduce(_ω₁)
  end

  # Tensor products — skip high ranks to keep precompile time reasonable
  # (248⊗248 for E₈ etc. is expensive even for ω₁)
  for _DT in (
    TypeA{2}, TypeA{3}, TypeA{4}, TypeA{5},
    TypeB{2}, TypeB{3}, TypeB{4},
    TypeC{2}, TypeC{3}, TypeC{4},
    TypeD{4}, TypeD{5},
    TypeE{6}, TypeF4, TypeG2,
  )
    _ω₁ = fundamental_weight(_DT, 1)
    tensor_product(_ω₁, _ω₁)
  end

  # Littlewood–Richardson (TypeA only)
  for _N in 1:9
    _DT = TypeA{_N}
    _ω₁ = fundamental_weight(_DT, 1)
    lr_tensor_product(_ω₁, _ω₁)
  end
end

# ─── Startup banner ────────────────────────────────────────────────────────

function _print_banner()
  v = pkgversion(@__MODULE__)
  version_str = v === nothing ? "dev" : string(v)

  println()
  # Row 1: ▖ ▘     ▘▜
  print("▖ ▘     ▘▜ ")
  println(" │  semisimple Lie algebras: root systems,")

  # Row 2: ▌ ▌█▌   ▌▐
  print("▌ ▌█▌   ▌▐ ")
  println(" │  Weyl groups, and representations")

  # Row 3: ▙▖▌▙▖▗  ▌▐▖
  print("▙▖▌▙▖▗  ▌▐▖")
  println(" │")

  # Row 4:        ▙▌
  print("       ▙▌  ")
  println(" │  Docs:    https://homogeneous.tools/lie.jl")

  print("            ")
  println("│  Version: ", version_str)
end

function __init__()
  _apply_cache_preferences!()
  show_banner = @load_preference("show_banner", true)
  if show_banner && displaysize(stdout)[2] >= 60
    _print_banner()
  end
  return nothing
end

end # module
