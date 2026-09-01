# ═══════════════════════════════════════════════════════════════════════════════
#  Semisimple.jl — computations with semisimple Lie algebras
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Semisimple

Julia package for computations with semisimple Lie algebras over ℂ.

Provides root systems, Weyl groups, and weight-lattice arithmetic for all
classical and exceptional Dynkin types (A, B, C, D, E₆, E₇, E₈, F₄, G₂)
as well as their direct products.

See the [online documentation](https://homogeneous.tools/Semisimple.jl/) for
usage examples and mathematical background.
"""
module Semisimple

using LRUCache
using PrecompileTools
using Preferences
using PrettyTables: pretty_table, tf_unicode_rounded
using StaticArrays
using LinearAlgebra: I as _I

# ─── Cache budget (computed before any cache is created) ─────────────────────

const _MIN_CACHE_BUDGET = 256 * 1024^2  # 256 MiB floor

_default_cache_budget() = max(div(Int(Sys.total_memory()), 4), _MIN_CACHE_BUDGET)

const _DEFAULT_DOMINANT_FRAC = 0.30
const _DEFAULT_TENSOR_FRAC = 0.40
const _DEFAULT_SYM_POWER_FRAC = 0.15
const _DEFAULT_EXT_POWER_FRAC = 0.15

_cache_maxsize(budget::Int, fraction::Float64) = max(1, round(Int, budget * fraction))

# ─── Compact display toggle ──────────────────────────────────────────────────
# Governs the default show format for WeightLatticeElem and RootSpaceElem.
# When true, show always uses the coordinate form "DT[c₁,c₂,…]" regardless of
# the IOContext.  Per-call override is still possible via IOContext(:compact).
const _compact_display = Ref{Bool}(false)

# ─── Per-Dynkin-type Dict caches: despecialized access ───────────────────────
# Passing a Dynkin type as a `Dict{Type,Any}` key specializes `get`/`setindex!`
# on the singleton `Type{...}` argument, compiling fresh Base machinery for
# every Dynkin type.  These barriers keep that machinery compiled exactly once.
@inline function _typedict_get(d::Dict{Type,V}, @nospecialize(k::Type)) where {V}
  return get(d, k, nothing)
end

@inline function _typedict_set!(
  d::Dict{Type,V}, @nospecialize(k::Type), @nospecialize(v)
) where {V}
  d[k] = v
  return nothing
end

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

# ─── Bourbaki tables ─────────────────────────────────────────────────────────
include("BourbakiTable.jl")

# ─── Cache configuration (Preferences + runtime API) ────────────────────────
include("CacheConfig.jl")

# ─── Cache management ───────────────────────────────────────────────────────

"""
    clear_all_caches!()

Clear all internal caches used by Semisimple.jl.  Alias for `clear_caches!`.

# Examples
```jldoctest
julia> using Semisimple

julia> clear_all_caches!()
```
"""
clear_all_caches!() = clear_caches!()

export clear_all_caches!, clear_caches!, configure_caches!, cache_info

# ─── Compact display ─────────────────────────────────────────────────────────

"""
    compact_display!(val::Bool = true)

Set the global compact-display mode for [`WeightLatticeElem`](@ref) and
[`RootSpaceElem`](@ref).

When `true`, every call to `show` on these types prints the coordinate form
`DT[c₁,c₂,…]` (e.g. `A3[1,0,0]` for ω₁ in type A₃) regardless of the
`IOContext`.  When `false` (the default), the standard symbolic form is used
(`ω1`, `α1 + α2`, etc.).

The per-call `IOContext(:compact => true)` override always takes precedence.

# Examples
```jldoctest compact
julia> using Semisimple

julia> ω1 = fundamental_weight(TypeA{3}, 1);

julia> compact_display!(true)
true

julia> ω1
A3[1,0,0]

julia> fundamental_weights(TypeA{3})
3-element Vector{WeightLatticeElem{TypeA{3}, 3}}:
 A3[1,0,0]
 A3[0,1,0]
 A3[0,0,1]

julia> compact_display!(false)
false

julia> ω1
ω1
```
"""
function compact_display!(val::Bool=true)
  _compact_display[] = val
end

export compact_display!

# ─── Precompilation ─────────────────────────────────────────────────────────
# @compile_workload executes real code during precompilation, so Julia
# transitively caches every callee (SMatrix constructors, getindex, etc.),
# not just the top-level method signatures that bare precompile() would cover.
#
# The heavy numeric kernels (Freudenthal recursion, Weyl dimension formula,
# dominant-weight enumeration, dominant-chamber folds) are parametrized by the
# rank only, so covering e.g. A₇/B₇/C₇/D₇/E₇ compiles those kernels just once.
# The per-type marginal cost is limited to thin wrappers and the Weyl-orbit
# traversal, which keeps covering every type up to rank 10 affordable.
#
# Opt out (e.g. for development or CI) via
#   Preferences.set_preferences!(Semisimple, "precompile_workload" => false)

const _should_precompile_workload = @load_preference("precompile_workload", true)

if _should_precompile_workload
  @compile_workload begin
    # CartanMatrix, RootSystem, WeylGroup infrastructure, Characters, and
    # tensor products for all simple Dynkin types up to rank 10.
    for _DT in (
      TypeA{1}, TypeA{2}, TypeA{3}, TypeA{4}, TypeA{5},
      TypeA{6}, TypeA{7}, TypeA{8}, TypeA{9}, TypeA{10},
      TypeB{2}, TypeB{3}, TypeB{4}, TypeB{5},
      TypeB{6}, TypeB{7}, TypeB{8}, TypeB{9}, TypeB{10},
      TypeC{2}, TypeC{3}, TypeC{4}, TypeC{5},
      TypeC{6}, TypeC{7}, TypeC{8}, TypeC{9}, TypeC{10},
      TypeD{3}, TypeD{4}, TypeD{5}, TypeD{6}, TypeD{7},
      TypeD{8}, TypeD{9}, TypeD{10},
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

      # Tensor product (Brauer–Klimyk; Littlewood–Richardson for type A).
      # Execution is cheap for ω₁ ⊗ ω₁ — what is being cached here is the
      # compiled Brauer–Klimyk orbit traversal for this Dynkin type.
      tensor_product(_ω₁, _ω₁)
    end

    # Keep runtime cache state out of the shipped image; everything is
    # rebuilt lazily (and cheaply) on first use.
    clear_caches!()
  end
end

# ─── Startup banner ────────────────────────────────────────────────────────

# COV_EXCL_START
function _print_banner()
  v = pkgversion(@__MODULE__)
  version_str = v === nothing ? "dev" : string(v)

  println()
  println("▄▖     ▘  ▘     ▜      ▘▜ │ semisimple Lie algebras: root systems,")
  println("▚ █▌▛▛▌▌▛▘▌▛▛▌▛▌▐ █▌   ▌▐ │ Weyl groups, weight lattices")
  println("▄▌▙▖▌▌▌▌▄▌▌▌▌▌▙▌▐▖▙▖▗  ▌▐▖│ Docs:    https://homogeneous.tools/Semisimple.jl")
  println("              ▌       ▙▌  │ Version: ", version_str)
  println()
end
# COV_EXCL_STOP

function __init__()
  _apply_cache_preferences!()
  show_banner = @load_preference("show_banner", true)
  if show_banner && isinteractive() && !haskey(ENV, "CI") && displaysize(stdout)[2] >= 60
    # COV_EXCL_START
    _print_banner()
    # COV_EXCL_STOP
  end
  return nothing
end

end # module
