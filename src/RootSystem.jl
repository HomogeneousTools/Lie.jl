# ═══════════════════════════════════════════════════════════════════════════════
#  Root systems — root enumeration and root space elements
#
#  Roots are stored as SVector{R,Int} in the basis of simple roots.
#  Root data is computed once per Dynkin type at runtime (and cached) by a
#  value-level builder that is compiled only once, instead of being emitted
#  as @generated literals per type — keeping precompilation cheap.
# ═══════════════════════════════════════════════════════════════════════════════

export RootSystem, RootSpaceElem
export simple_roots, simple_root, positive_roots, positive_root
export negative_roots, negative_root, roots, root
export n_roots, n_simple_roots, highest_root, highest_short_root, highest_coroot
export simple_coroots, positive_coroots
export is_root, is_positive_root, height
export dot, coefficients, coxeter_coefficients, dual_coxeter_coefficients, coxeter_number,
  dual_coxeter_number, degrees_fundamental_invariants

import LinearAlgebra: dot

# ═══════════════════════════════════════════════════════════════════════════════
#  RootSpaceElem — a vector in the root space (linear combination of simple roots)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RootSpaceElem{DT,R}

An element of the root space for Dynkin type `DT` of rank `R`,
stored as an `SVector{R,Int}` of coordinates in the simple root basis.

# Examples
```jldoctest
julia> using Semisimple

julia> RootSpaceElem(TypeA{2}, [1, 1])
α1 + α2
```
"""
struct RootSpaceElem{DT<:DynkinType,R}
  vec::SVector{R,Int}
end

function RootSpaceElem(::Type{DT}, v::SVector{R,Int}) where {DT<:DynkinType,R}
  R == rank(DT) || throw(ArgumentError("Vector length $R does not match rank $(rank(DT))"))
  return RootSpaceElem{DT,R}(v)
end

function RootSpaceElem(::Type{DT}, v::NTuple{R,Int}) where {DT<:DynkinType,R}
  return RootSpaceElem(DT, SVector{R,Int}(v))
end

function RootSpaceElem(::Type{DT}, v::AbstractVector{<:Integer}) where {DT<:DynkinType}
  R = rank(DT)
  return RootSpaceElem(DT, SVector{R,Int}(v...))
end

"""
    coefficients(r::RootSpaceElem) -> SVector
    coefficients(w::WeightLatticeElem) -> SVector

Return the coordinate vector of a root space element (in the simple root basis)
or of a weight lattice element (in the fundamental weight basis).

# Examples
```jldoctest
julia> using Semisimple

julia> coefficients(RootSpaceElem(TypeA{2}, [1, 1])) == [1, 1]
true
```
"""
coefficients(r::RootSpaceElem) = r.vec

Base.:+(a::RootSpaceElem{DT,R}, b::RootSpaceElem{DT,R}) where {DT,R} = RootSpaceElem{DT,R}(
  a.vec + b.vec
)
Base.:-(a::RootSpaceElem{DT,R}, b::RootSpaceElem{DT,R}) where {DT,R} = RootSpaceElem{DT,R}(
  a.vec - b.vec
)
Base.:-(a::RootSpaceElem{DT,R}) where {DT,R} = RootSpaceElem{DT,R}(-a.vec)
Base.:*(n::Integer, a::RootSpaceElem{DT,R}) where {DT,R} = RootSpaceElem{DT,R}(n * a.vec)
Base.:*(a::RootSpaceElem, n::Integer) = n * a
Base.:(==)(a::RootSpaceElem{DT,R}, b::RootSpaceElem{DT,R}) where {DT,R} =
  a.vec == b.vec
Base.hash(a::RootSpaceElem, h::UInt) = hash(a.vec, h)
Base.zero(::Type{RootSpaceElem{DT,R}}) where {DT,R} = RootSpaceElem{DT,R}(
  zero(SVector{R,Int})
)
Base.iszero(a::RootSpaceElem) = iszero(a.vec)

"""
    height(r::RootSpaceElem) -> Int

Sum of coefficients in the simple root expansion.

# Examples
```jldoctest
julia> using Semisimple

julia> height(RootSpaceElem(TypeA{2}, [1, 1]))
2
```
"""
height(r::RootSpaceElem) = sum(r.vec)

function Base.show(io::IO, r::RootSpaceElem{DT,R}) where {DT,R}
  if get(io, :compact, _compact_display[])
    print(io, _type_name(DT), "[", join(r.vec, ","), "]")
    return nothing
  end
  terms = String[]
  for i in 1:R
    c = r.vec[i]
    c == 0 && continue
    if c == 1
      push!(terms, "α$i")
    elseif c == -1
      push!(terms, "-α$i")
    else
      push!(terms, "$(c)α$i")
    end
  end
  if isempty(terms)
    print(io, "0")
  else
    s = terms[1]
    for t in terms[2:end]
      if startswith(t, "-")
        s *= " - " * t[2:end]
      else
        s *= " + " * t
      end
    end
    print(io, s)
  end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  RootSystem — container holding all precomputed root data for a Dynkin type
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RootSystem{DT,R}

A root system for the Dynkin type `DT` of rank `R`. The data is built once
at runtime and cached per Dynkin type.

Fields:
- `positive_roots_list`: `Vector{SVector{R,Int}}` of positive roots,
  ordered by non-decreasing height (`pos_roots[end]` is the highest root).
- `positive_coroots_list`: `Vector{SVector{R,Int}}` of positive coroots,
  in the same order as the roots.
- `refl`: `Matrix{UInt}` reflection table of size `R × N` — `refl[s, i]` =
  index of `s_s(α_i)` among positive roots, or 0 if the result is negative.
- `highest_coroot_idx`: the index of the positive coroot with greatest height
  (= index of the highest short root in `positive_roots_list`).

# Examples
```jldoctest
julia> using Semisimple

julia> RootSystem(TypeA{2})
Root system of type A2, rank 2 with 3 positive roots
```
"""
struct RootSystem{DT<:DynkinType,R}
  positive_roots_list::Vector{SVector{R,Int}}
  positive_coroots_list::Vector{SVector{R,Int}}
  refl::Matrix{UInt}
  highest_coroot_idx::Int
end

function Base.:(==)(a::RootSystem{DT,R}, b::RootSystem{DT,R}) where {DT,R}
  return a.positive_roots_list == b.positive_roots_list &&
         a.positive_coroots_list == b.positive_coroots_list &&
         a.refl == b.refl &&
         a.highest_coroot_idx == b.highest_coroot_idx
end

"""
    RootSystem(::Type{DT}) -> RootSystem{DT,R}

Return the root system for Dynkin type `DT`. A single instance is cached per
Dynkin type; the data is computed by a compact value-level builder shared by
all types.

# Examples
```jldoctest
julia> using Semisimple

julia> RootSystem(TypeA{2}) === RootSystem(TypeA{2})
true
```
"""
const _root_system_cache = Dict{Type,Any}()
const _root_system_lock = ReentrantLock()

function RootSystem(::Type{DT}) where {DT<:DynkinType}
  check_dynkin_type(DT)
  R = rank(DT)
  # Explicit lock/try instead of do-block closures: closures specialize per
  # Dynkin type and add measurable precompilation cost for zero benefit here.
  lock(_root_system_lock)
  try
    cached = _typedict_get(_root_system_cache, DT)
    cached === nothing || return cached::RootSystem{DT,R}
    rs = _make_root_system(DT)
    _typedict_set!(
      _positive_roots_set_cache, DT, Set{SVector{R,Int}}(rs.positive_roots_list)
    )
    _typedict_set!(_root_system_cache, DT, rs)
    return rs::RootSystem{DT,R}
  finally
    unlock(_root_system_lock)
  end
end

# Rank-level conversion of value-level root data into `SVector` storage.
# Compiled once per rank R, shared by all Dynkin families of that rank.
Base.@constprop :none _as_svectors(::Val{R}, vs::Vector{Vector{Int}}) where {R} =
  [SVector{R,Int}(v...) for v in vs]

"""
    _make_root_system_runtime(::Type{DT}) -> RootSystem{DT,R}

Compact runtime builder for root systems: value-level root enumeration
(compiled once) plus a small per-rank conversion into `SVector` storage.
"""
function _make_root_system_runtime(::Type{DT}) where {DT<:DynkinType}
  check_dynkin_type(DT)
  R = rank(DT)
  pos_roots, pos_coroots, refl = _compute_positive_roots_and_reflections_runtime(
    _cartan_matrix_data(DT), R
  )
  hcr_idx = _highest_coroot_index(pos_coroots)
  return RootSystem{DT,R}(
    _as_svectors(Val(R), pos_roots), _as_svectors(Val(R), pos_coroots), refl, hcr_idx
  )
end

# ─── Core computation of positive roots ─────────────────────────────────────

function _root_height(v)
  total = zero(Int)
  @inbounds for x in v
    total += x
  end
  return total
end

function _highest_coroot_index(pos_coroots)
  best_idx = 1
  best_height = _root_height(pos_coroots[1])
  for i in 2:length(pos_coroots)
    h = _root_height(pos_coroots[i])
    if h > best_height
      best_idx = i
      best_height = h
    end
  end
  return best_idx
end

function _is_nonnegative(v)
  @inbounds for x in v
    x < 0 && return false
  end
  return true
end

function _sort_positive_root_data(pos_roots, pos_coroots, refl_data, rk)
  n_pos = length(pos_roots)
  heights = [_root_height(r) for r in pos_roots]
  perm = sortperm(heights; alg=MergeSort)
  inv_perm = Vector{Int}(undef, n_pos)
  for (new_i, old_i) in enumerate(perm)
    inv_perm[old_i] = new_i
  end

  sorted_roots = pos_roots[perm]
  sorted_coroots = pos_coroots[perm]
  refl = zeros(UInt, rk, n_pos)
  for ((s, old_i), old_v) in refl_data
    if 1 <= s <= rk && 1 <= old_i <= n_pos
      new_i = inv_perm[old_i]
      new_v = old_v == 0 ? UInt(0) : UInt(inv_perm[old_v])
      refl[s, new_i] = new_v
    end
  end

  return sorted_roots, sorted_coroots, refl
end

function _compute_positive_roots_and_reflections_runtime(C::AbstractMatrix{Int}, rk::Int)
  pos_roots = [zeros(Int, rk) for _ in 1:rk]
  pos_coroots = [zeros(Int, rk) for _ in 1:rk]
  for i in 1:rk
    pos_roots[i][i] = 1
    pos_coroots[i][i] = 1
  end

  root_index = Dict{Vector{Int},Int}()
  for i in 1:rk
    root_index[pos_roots[i]] = i
  end

  refl_data = Dict{Tuple{Int,Int},UInt}()
  for s in 1:rk
    refl_data[(s, s)] = 0
  end

  i = 1
  while i <= length(pos_roots)
    for s in 1:rk
      haskey(refl_data, (s, i)) && continue

      root_i = pos_roots[i]
      coroot_i = pos_coroots[i]

      pairing = zero(Int)
      copairing = zero(Int)
      @inbounds for j in 1:rk
        pairing += C[s, j] * root_i[j]
        copairing += coroot_i[j] * C[j, s]
      end
      pairing * copairing < 4 ||
        error("Non-finite Cartan data encountered while enumerating positive roots")

      new_root = copy(root_i)
      new_root[s] -= pairing
      new_coroot = copy(coroot_i)
      new_coroot[s] -= copairing
      _is_nonnegative(new_root) ||
        error("Positive-root reflection unexpectedly left the positive cone")
      idx = get(root_index, new_root, 0)
      if idx == 0
        push!(pos_roots, new_root)
        push!(pos_coroots, new_coroot)
        idx = length(pos_roots)
        root_index[new_root] = idx
      end
      refl_data[(s, i)] = UInt(idx)
      refl_data[(s, idx)] = UInt(i)
    end
    i += 1
  end

  return _sort_positive_root_data(pos_roots, pos_coroots, refl_data, rk)
end

_make_root_system(::Type{DT}) where {DT<:DynkinType} = _make_root_system_runtime(DT)

RootSystem(dt::DynkinType) = RootSystem(typeof(dt))

function Base.show(io::IO, RS::RootSystem{DT,R}) where {DT,R}
  print(
    io,
    "Root system of type $(_type_name(DT)), rank $R with $(n_positive_roots(RS)) positive roots",
  )
end

# ─── Accessors ───────────────────────────────────────────────────────────────

"""
    n_simple_roots(RS::RootSystem) -> Int

Return the number of simple roots, equal to the rank of the root system.

# Examples
```jldoctest
julia> using Semisimple

julia> n_simple_roots(RootSystem(TypeA{2}))
2
```
"""
n_simple_roots(RS::RootSystem{DT,R}) where {DT,R} = R

"""
    n_positive_roots(RS::RootSystem) -> Int

Return the number of positive roots.

# Examples
```jldoctest
julia> using Semisimple

julia> n_positive_roots(RootSystem(TypeA{2}))
3
```
"""
n_positive_roots(RS::RootSystem) = length(RS.positive_roots_list)

"""
    n_roots(RS::RootSystem) -> Int

Return the total number of roots (positive and negative).

# Examples
```jldoctest
julia> using Semisimple

julia> n_roots(RootSystem(TypeA{2}))
6
```
"""
n_roots(RS::RootSystem) = 2 * n_positive_roots(RS)

"""
    simple_root(RS::RootSystem{DT,R}, i) -> RootSpaceElem

Return the `i`-th simple root.

# Examples
```jldoctest
julia> using Semisimple

julia> simple_root(RootSystem(TypeA{2}), 1)
α1
```
"""
function simple_root(RS::RootSystem{DT,R}, i::Integer) where {DT,R}
  @boundscheck 1 <= i <= R || throw(BoundsError("simple root index $i out of range"))
  return RootSpaceElem{DT,R}(RS.positive_roots_list[i])
end

"""
    simple_roots(RS::RootSystem{DT,R}) -> Vector{RootSpaceElem}

Return all simple roots.

# Examples
```jldoctest
julia> using Semisimple

julia> simple_roots(RootSystem(TypeA{2}))
2-element Vector{RootSpaceElem{TypeA{2}, 2}}:
 α1
 α2
```
"""
simple_roots(RS::RootSystem{DT,R}) where {DT,R} = [
  RootSpaceElem{DT,R}(RS.positive_roots_list[i]) for i in 1:R
]

"""
    positive_root(RS::RootSystem{DT,R}, i) -> RootSpaceElem

Return the `i`-th positive root.

# Examples
```jldoctest
julia> using Semisimple

julia> positive_root(RootSystem(TypeA{2}), 3)
α1 + α2
```
"""
function positive_root(RS::RootSystem{DT,R}, i::Integer) where {DT,R}
  return RootSpaceElem{DT,R}(RS.positive_roots_list[i])
end

"""
    positive_roots(RS::RootSystem{DT,R}) -> Vector{RootSpaceElem}

Return all positive roots.

# Examples
```jldoctest
julia> using Semisimple

julia> length(positive_roots(RootSystem(TypeA{2})))
3
```
"""
positive_roots(RS::RootSystem{DT,R}) where {DT,R} = [
  RootSpaceElem{DT,R}(v) for v in RS.positive_roots_list
]

"""
    negative_root(RS::RootSystem{DT,R}, i) -> RootSpaceElem

Return the `i`-th negative root (negative of the `i`-th positive root).

# Examples
```jldoctest
julia> using Semisimple

julia> negative_root(RootSystem(TypeA{2}), 1)
-α1
```
"""
negative_root(RS::RootSystem{DT,R}, i::Integer) where {DT,R} = RootSpaceElem{DT,R}(
  -RS.positive_roots_list[i]
)

"""
    negative_roots(RS::RootSystem{DT,R}) -> Vector{RootSpaceElem}

Return all negative roots.

# Examples
```jldoctest
julia> using Semisimple

julia> length(negative_roots(RootSystem(TypeA{2})))
3
```
"""
negative_roots(RS::RootSystem{DT,R}) where {DT,R} = [
  RootSpaceElem{DT,R}(-v) for v in RS.positive_roots_list
]

"""
    roots(RS::RootSystem) -> Vector{RootSpaceElem}

Return all roots (positive followed by negative).

# Examples
```jldoctest
julia> using Semisimple

julia> length(roots(RootSystem(TypeA{2})))
6
```
"""
roots(RS::RootSystem) = vcat(positive_roots(RS), negative_roots(RS))

"""
    root(RS::RootSystem{DT,R}, i) -> RootSpaceElem

Return the `i`-th root. Indices 1..n_pos are positive roots,
n_pos+1..2*n_pos are negative roots.

# Examples
```jldoctest
julia> using Semisimple

julia> root(RootSystem(TypeA{2}), 4)
-α1
```
"""
function root(RS::RootSystem{DT,R}, i::Integer) where {DT,R}
  np = n_positive_roots(RS)
  if 1 <= i <= np
    return positive_root(RS, i)
  elseif np < i <= 2 * np
    return negative_root(RS, i - np)
  else
    throw(BoundsError("root index $i out of range"))
  end
end

# ─── Coroots ─────────────────────────────────────────────────────────────────

"""
    simple_coroots(RS::RootSystem{DT,R}) -> Vector{RootSpaceElem}

Return the simple coroots.

# Examples
```jldoctest
julia> using Semisimple

julia> simple_coroots(RootSystem(TypeA{2})) == simple_roots(RootSystem(TypeA{2}))
true
```
"""
simple_coroots(RS::RootSystem{DT,R}) where {DT,R} = [
  RootSpaceElem{DT,R}(RS.positive_coroots_list[i]) for i in 1:R
]

"""
    positive_coroots(RS::RootSystem{DT,R}) -> Vector{RootSpaceElem}

Return all positive coroots.

# Examples
```jldoctest
julia> using Semisimple

julia> length(positive_coroots(RootSystem(TypeB{2})))
4
```
"""
positive_coroots(RS::RootSystem{DT,R}) where {DT,R} = [
  RootSpaceElem{DT,R}(v) for v in RS.positive_coroots_list
]

# ─── Highest root ────────────────────────────────────────────────────────────

"""
    highest_root(RS::RootSystem{DT,R}) -> RootSpaceElem

Return the highest root. Positive roots are ordered by non-decreasing height,
so the highest root is always the last positive root.

# Examples
```jldoctest
julia> using Semisimple

julia> highest_root(RootSystem(TypeA{2}))
α1 + α2
```
"""
function highest_root(RS::RootSystem{DT,R}) where {DT,R}
  positive_root(RS, n_positive_roots(RS))
end

# ─── Highest coroot ──────────────────────────────────────────────────────────

"""
    highest_coroot(RS::RootSystem{DT,R}) -> RootSpaceElem

Return the highest coroot ``\\theta^\\vee``: the positive coroot of greatest height.
This is the coroot of the highest short root.

The index is precomputed when the root system is built and stored in
`RS.highest_coroot_idx`.

# Examples
```jldoctest
julia> using Semisimple

julia> highest_coroot(RootSystem(TypeA{2}))
α1 + α2
```
"""
function highest_coroot(RS::RootSystem{DT,R}) where {DT,R}
  RootSpaceElem{DT,R}(RS.positive_coroots_list[RS.highest_coroot_idx])
end

# ─── Highest short root ──────────────────────────────────────────────────────

"""
    highest_short_root(RS::RootSystem{DT,R}) -> RootSpaceElem

Return the highest short root: the positive root of minimal length that has
greatest height among all short positive roots.

For simply-laced types (A, D, E), every root has the same length, so this
coincides with `highest_root`.

The index equals `RS.highest_coroot_idx`, precomputed when the root system is built.

# Examples
```jldoctest
julia> using Semisimple

julia> RS = RootSystem(TypeB{2});

julia> coefficients(highest_short_root(RS))
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 1
 1

julia> RS_G2 = RootSystem(TypeG2);

julia> coefficients(highest_short_root(RS_G2))
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 2
 1
```
"""
function highest_short_root(RS::RootSystem{DT,R}) where {DT,R}
  positive_root(RS, RS.highest_coroot_idx)
end

# ─── Coxeter coefficients ────────────────────────────────────────────────

"""
    coxeter_coefficients(::Type{DT}) -> SVector{R,Int}
    coxeter_coefficients(dt::DT) -> SVector{R,Int}

Return the **Coxeter labels** (also called marks): the coefficients of the
highest root in the simple root basis:
``\\theta = \\sum_i m_i \\alpha_i``

These are not the Weyl group exponents; the degrees of fundamental invariants
are returned by [`degrees_fundamental_invariants`](@ref), and the exponents are
those degrees minus 1.

# Examples
```jldoctest
julia> using Semisimple

julia> coxeter_coefficients(TypeA{3})
3-element StaticArraysCore.SVector{3, Int64} with indices SOneTo(3):
 1
 1
 1

julia> coxeter_coefficients(TypeB{2})
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 1
 2
```
"""
coxeter_coefficients(::Type{TypeA{N}}) where {N} =
  (check_dynkin_type(TypeA{N}); SVector{N,Int}(ntuple(_ -> 1, N)))

coxeter_coefficients(::Type{TypeB{N}}) where {N} = SVector{N,Int}(
  (check_dynkin_type(TypeB{N}); ntuple(i -> i == 1 ? 1 : 2, Val(N)))
)

coxeter_coefficients(::Type{TypeC{N}}) where {N} = SVector{N,Int}(
  (check_dynkin_type(TypeC{N}); ntuple(i -> i == N ? 1 : 2, Val(N)))
)

function coxeter_coefficients(::Type{TypeD{N}}) where {N}
  check_dynkin_type(TypeD{N})
  return SVector{N,Int}(ntuple(i -> (i == 1 || i >= N - 1) ? 1 : 2, Val(N)))
end

coxeter_coefficients(::Type{TypeE{6}}) = SVector{6,Int}((1, 2, 2, 3, 2, 1))
coxeter_coefficients(::Type{TypeE{7}}) = SVector{7,Int}((2, 2, 3, 4, 3, 2, 1))
coxeter_coefficients(::Type{TypeE{8}}) = SVector{8,Int}((2, 3, 4, 6, 5, 4, 3, 2))
coxeter_coefficients(::Type{TypeF4}) = SVector{4,Int}((2, 3, 4, 2))
coxeter_coefficients(::Type{TypeG2}) = SVector{2,Int}((3, 2))

@generated function coxeter_coefficients(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  types = Ts.parameters
  all_coeffs = vcat([collect(coxeter_coefficients(T)) for T in types]...)
  R = length(all_coeffs)
  entries = Tuple(all_coeffs)
  return :(SVector{$R,Int}($entries))
end

function coxeter_coefficients(dt::DynkinType)
  return coxeter_coefficients(typeof(dt))
end

# ─── Dual Coxeter coefficients ────────────────────────────────────────────────

"""
    dual_coxeter_coefficients(::Type{DT}) -> SVector{R,Int}
    dual_coxeter_coefficients(dt::DT) -> SVector{R,Int}

Return the **dual Coxeter coefficients**: the coefficients of simple roots in the
highest short root of the dual root system (Langlands dual). The dual Coxeter number
is ``h^\\vee = 1 + \\sum_i n_i^\\vee``.

For simply-laced types (A, D, E) all roots have the same length, so these equal the
Coxeter coefficients. For B, C, F4, and G2 they differ.

# Examples
```jldoctest
julia> using Semisimple

julia> dual_coxeter_coefficients(TypeB{2})
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 1
 1

julia> dual_coxeter_coefficients(TypeG2)
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 1
 2
```
"""
dual_coxeter_coefficients(::Type{TypeA{N}}) where {N} =
  (check_dynkin_type(TypeA{N}); coxeter_coefficients(TypeA{N}))

# Dual of B_n is C_n; highest short root of C_n = e₁+e₂, coefficients [1,2,...,2,1]
function dual_coxeter_coefficients(::Type{TypeB{N}}) where {N}
  check_dynkin_type(TypeB{N})
  if N == 2
    return SVector{2,Int}((1, 1))
  else
    return SVector{N,Int}(ntuple(i -> (i == 1 || i == N) ? 1 : 2, Val(N)))
  end
end

# Dual of C_n is B_n; highest short root of B_n = e₁ = α₁+...+αₙ, coefficients [1,...,1]
function dual_coxeter_coefficients(::Type{TypeC{N}}) where {N}
  check_dynkin_type(TypeC{N})
  return SVector{N,Int}(ntuple(_ -> 1, N))
end

dual_coxeter_coefficients(::Type{TypeD{N}}) where {N} =
  (check_dynkin_type(TypeD{N}); coxeter_coefficients(TypeD{N}))
dual_coxeter_coefficients(::Type{TypeE{6}}) = coxeter_coefficients(TypeE{6})
dual_coxeter_coefficients(::Type{TypeE{7}}) = coxeter_coefficients(TypeE{7})
dual_coxeter_coefficients(::Type{TypeE{8}}) = coxeter_coefficients(TypeE{8})
# Dual of F₄ is F₄; highest short root of F₄ = α₁+2α₂+3α₃+2α₄, coefficients [1,2,3,2]
dual_coxeter_coefficients(::Type{TypeF4}) = SVector{4,Int}((1, 2, 3, 2))
# Dual of G₂ is G₂; highest short root of G₂∨ in coroot basis, coefficients [1,2]
dual_coxeter_coefficients(::Type{TypeG2}) = SVector{2,Int}((1, 2))

@generated function dual_coxeter_coefficients(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  types = Ts.parameters
  all_coeffs = vcat([dual_coxeter_coefficients(T) for T in types]...)
  R = length(all_coeffs)
  entries = Tuple(all_coeffs)
  return :(SVector{$R,Int}($entries))
end

function dual_coxeter_coefficients(dt::DynkinType)
  return dual_coxeter_coefficients(typeof(dt))
end

# ─── Coxeter number ────────────────────────────────────────────────────────

"""
    coxeter_number(::Type{DT}) -> Int
    coxeter_number(dt::DT) -> Int

Return the **Coxeter number** ``h`` of the Dynkin type, defined as
``h = 1 + \\sum_i m_i``
where ``m_i`` are the Coxeter coefficients (coefficients of the highest root).

The Coxeter number is the order of a Coxeter element (product of all simple reflections)
in the Weyl group.

# Examples
```jldoctest
julia> using Semisimple

julia> coxeter_number(TypeA{1})
2

julia> coxeter_number(TypeA{3})
4

julia> coxeter_number(TypeG2)
6
```
"""
coxeter_number(::Type{DT}) where {DT<:DynkinType} = 1 + sum(coxeter_coefficients(DT))
coxeter_number(dt::DynkinType) = coxeter_number(typeof(dt))

# ─── Dual Coxeter number ────────────────────────────────────────────────────

"""
    dual_coxeter_number(::Type{DT}) -> Int
    dual_coxeter_number(dt::DT) -> Int

Return the **dual Coxeter number** ``h^\\vee`` of the Dynkin type, which is the Coxeter
number of the Langlands dual root system.

# Examples
```jldoctest
julia> using Semisimple

julia> dual_coxeter_number(TypeA{1})
2

julia> dual_coxeter_number(TypeA{3})
4

julia> dual_coxeter_number(TypeB{2})
3

julia> dual_coxeter_number(TypeG2)
4
```
"""
dual_coxeter_number(::Type{DT}) where {DT<:DynkinType} =
  1 + sum(dual_coxeter_coefficients(DT))
dual_coxeter_number(dt::DynkinType) = dual_coxeter_number(typeof(dt))

# ─── Degrees of fundamental invariants ────────────────────────────────────────

"""
    degrees_fundamental_invariants(::Type{DT}) -> SVector{R,Int}
    degrees_fundamental_invariants(dt::DT) -> SVector{R,Int}

Return the degrees of the fundamental invariants of the Weyl group action on the
polynomial ring.

# Examples
```jldoctest
julia> using Semisimple

julia> degrees_fundamental_invariants(TypeA{2})
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 2
 3

julia> degrees_fundamental_invariants(TypeB{3})
3-element StaticArraysCore.SVector{3, Int64} with indices SOneTo(3):
 2
 4
 6

julia> degrees_fundamental_invariants(TypeD{4})
4-element StaticArraysCore.SVector{4, Int64} with indices SOneTo(4):
 2
 4
 6
 4

julia> degrees_fundamental_invariants(TypeG2)
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 2
 6
```
"""
degrees_fundamental_invariants(::Type{TypeA{N}}) where {N} =
  (check_dynkin_type(TypeA{N}); SVector{N,Int}(Tuple(2:(N + 1))))

# B_n and C_n: 2, 4, 6, ..., 2n
function degrees_fundamental_invariants(::Type{TypeB{N}}) where {N}
  check_dynkin_type(TypeB{N})
  return SVector{N,Int}(Tuple(2:2:(2N)))
end

function degrees_fundamental_invariants(::Type{TypeC{N}}) where {N}
  check_dynkin_type(TypeC{N})
  return SVector{N,Int}(Tuple(2:2:(2N)))
end

# D_n: 2, 4, ..., 2(n-1), n
function degrees_fundamental_invariants(::Type{TypeD{N}}) where {N}
  check_dynkin_type(TypeD{N})
  return SVector{N,Int}(Tuple(vcat(collect(2:2:(2N - 2)), [N])))
end

degrees_fundamental_invariants(::Type{TypeE{6}}) = SVector{6,Int}((2, 5, 6, 8, 9, 12))
degrees_fundamental_invariants(::Type{TypeE{7}}) = SVector{7,Int}((2, 6, 8, 10, 12, 14, 18))
degrees_fundamental_invariants(::Type{TypeE{8}}) = SVector{8,Int}((
  2, 8, 12, 14, 18, 20, 24, 30
))
degrees_fundamental_invariants(::Type{TypeF4}) = SVector{4,Int}((2, 6, 8, 12))
degrees_fundamental_invariants(::Type{TypeG2}) = SVector{2,Int}((2, 6))

@generated function degrees_fundamental_invariants(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  types = Ts.parameters
  all_degrees = vcat([degrees_fundamental_invariants(T) for T in types]...)
  R = length(all_degrees)
  entries = Tuple(all_degrees)
  return :(SVector{$R,Int}($entries))
end

function degrees_fundamental_invariants(dt::DynkinType)
  return degrees_fundamental_invariants(typeof(dt))
end

# ─── Root queries ────────────────────────────────────────────────────────────

"""
    is_root(RS::RootSystem{DT,R}, v::RootSpaceElem{DT,R}) -> Bool

Check whether `v` is a root.

# Examples
```jldoctest
julia> using Semisimple

julia> RS = RootSystem(TypeA{2});

julia> is_root(RS, RootSpaceElem(TypeA{2}, [1, 1]))
true
```
"""
function is_root(RS::RootSystem{DT,R}, v::RootSpaceElem{DT,R}) where {DT,R}
  return is_positive_root(RS, v) || is_positive_root(RS, -v)
end

const _positive_roots_set_cache = Dict{Type,Any}()

"""
    is_positive_root(RS::RootSystem{DT,R}, v::RootSpaceElem{DT,R}) -> Bool

Check whether `v` is a positive root.

# Examples
```jldoctest
julia> using Semisimple

julia> RS = RootSystem(TypeA{2});

julia> is_positive_root(RS, RootSpaceElem(TypeA{2}, [-1, 0]))
false
```
"""
function is_positive_root(RS::RootSystem{DT,R}, v::RootSpaceElem{DT,R}) where {DT,R}
  s = _typedict_get(_positive_roots_set_cache, DT)
  if s === nothing
    s = Set{SVector{R,Int}}(RS.positive_roots_list)
    _typedict_set!(_positive_roots_set_cache, DT, s)
  end
  return v.vec in (s::Set{SVector{R,Int}})
end

# ─── Inner product on root space ─────────────────────────────────────────────

"""
    dot(a::RootSpaceElem{DT,R}, b::RootSpaceElem{DT,R}) -> Rational{Int}

Inner product of two root space elements using the symmetrized Cartan form.

``(α, β) = αᵀ \\operatorname{diag}(d) C β``
"""
function dot(a::RootSpaceElem{DT,R}, b::RootSpaceElem{DT,R}) where {DT,R}
  B = cartan_bilinear_form(DT)
  return Rational{Int}(a.vec' * B * b.vec)
end
