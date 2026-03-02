# ═══════════════════════════════════════════════════════════════════════════════
#  Root systems — static root enumeration and root space elements
#
#  Roots are stored as SVector{R,Int} in the basis of simple roots.
#  All roots for a given Dynkin type are computed at compile time via
#  @generated functions, yielding a unique singleton per Dynkin type.
# ═══════════════════════════════════════════════════════════════════════════════

export RootSystem, RootSpaceElem
export simple_roots, simple_root, positive_roots, positive_root
export negative_roots, negative_root, roots, root
export n_roots, n_simple_roots, highest_root
export simple_coroots, positive_coroots
export is_root, is_positive_root, height
export dot, coefficients

# ═══════════════════════════════════════════════════════════════════════════════
#  RootSpaceElem — a vector in the root space (linear combination of simple roots)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RootSpaceElem{DT,R}

An element of the root space for Dynkin type `DT` of rank `R`,
stored as an `SVector{R,Int}` of coordinates in the simple root basis.
"""
struct RootSpaceElem{DT<:DynkinType,R}
  vec::SVector{R,Int}
end

function RootSpaceElem(::Type{DT}, v::SVector{R,Int}) where {DT<:DynkinType,R}
  @assert R == rank(DT) "Vector length $R does not match rank $(rank(DT))"
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
"""
coefficients(r::RootSpaceElem) = r.vec

Base.:+(a::RootSpaceElem{DT,R}, b::RootSpaceElem{DT,R}) where {DT,R} =
  RootSpaceElem{DT,R}(a.vec + b.vec)
Base.:-(a::RootSpaceElem{DT,R}, b::RootSpaceElem{DT,R}) where {DT,R} =
  RootSpaceElem{DT,R}(a.vec - b.vec)
Base.:-(a::RootSpaceElem{DT,R}) where {DT,R} = RootSpaceElem{DT,R}(-a.vec)
Base.:*(n::Integer, a::RootSpaceElem{DT,R}) where {DT,R} =
  RootSpaceElem{DT,R}(n * a.vec)
Base.:*(a::RootSpaceElem, n::Integer) = n * a
Base.:(==)(a::RootSpaceElem{DT,R}, b::RootSpaceElem{DT,R}) where {DT,R} =
  a.vec == b.vec
Base.hash(a::RootSpaceElem, h::UInt) = hash(a.vec, h)
Base.zero(::Type{RootSpaceElem{DT,R}}) where {DT,R} =
  RootSpaceElem{DT,R}(zero(SVector{R,Int}))
Base.iszero(a::RootSpaceElem) = iszero(a.vec)

"""
    height(r::RootSpaceElem) -> Int

Sum of coefficients in the simple root expansion.
"""
height(r::RootSpaceElem) = sum(r.vec)

function Base.show(io::IO, r::RootSpaceElem{DT,R}) where {DT,R}
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
    RootSystem{DT,R,N}

A root system for the Dynkin type `DT` of rank `R` with `N` positive roots.
All data is computed at compile time via a `@generated` constructor, so
there is exactly one `RootSystem` per Dynkin type — a compile-time singleton.

Fields:
- `positive_roots_list`: `NTuple{N, SVector{R,Int}}` of positive roots
- `positive_coroots_list`: `NTuple{N, SVector{R,Int}}` of positive coroots
- `refl`: `SMatrix{R,N,UInt}` reflection table — `refl[s, i]` = index of
          `s_s(α_i)` among positive roots, or 0 if the result is negative.
"""
struct RootSystem{DT<:DynkinType,R,N}
  positive_roots_list::NTuple{N,SVector{R,Int}}
  positive_coroots_list::NTuple{N,SVector{R,Int}}
  refl::SMatrix{R,N,UInt}
end

"""
    RootSystem(::Type{DT}) -> RootSystem{DT,R,N}

Return the root system for Dynkin type `DT`.  Root data (positive roots,
coroots, reflection table) is computed at compile time via `@generated`.
A single instance is cached per Dynkin type.
"""
const _root_system_cache = Dict{Type,Any}()

function RootSystem(::Type{DT}) where {DT<:DynkinType}
  return get!(_root_system_cache, DT) do
    _make_root_system(DT)
  end::RootSystem{DT,rank(DT),n_positive_roots(DT)}
end

@generated function _make_root_system(::Type{DT}) where {DT<:DynkinType}
  R = rank(DT)
  C_data = _cartan_matrix_data(DT)
  C = SMatrix{R,R,Int,R * R}(Tuple(C_data))
  pos_roots, pos_coroots, refl_mat = _compute_positive_roots_and_reflections(C, R)
  N = length(pos_roots)

  # Flatten data into tuples for embedding in the generated expression
  roots_tuple = Tuple(Tuple(v) for v in pos_roots)
  coroots_tuple = Tuple(Tuple(v) for v in pos_coroots)
  # refl_mat is Matrix{UInt} of size (R, N) — flatten column-major
  refl_entries = Tuple(UInt(refl_mat[i, j]) for j in 1:N for i in 1:R)

  return quote
    RootSystem{$DT,$R,$N}(
      $(roots_tuple),
      $(coroots_tuple),
      SMatrix{$R,$N,UInt,$(R * N)}($refl_entries),
    )
  end
end

RootSystem(dt::DynkinType) = RootSystem(typeof(dt))

function Base.show(io::IO, RS::RootSystem{DT,R}) where {DT,R}
  print(
    io,
    "Root system of type $(_type_name(DT)), rank $R with $(n_positive_roots(RS)) positive roots",
  )
end

# ─── Core computation of positive roots ─────────────────────────────────────

"""
Compute positive roots, coroots, and the reflection table from a Cartan matrix.

This uses the standard algorithm: start with simple roots and iteratively apply
simple reflections to discover new positive roots.

Returns:
- `pos_roots::Vector{SVector{R,Int}}` — positive roots in simple root coordinates
- `pos_coroots::Vector{SVector{R,Int}}` — positive coroots
- `refl::Matrix{UInt}` — reflection table
"""
function _compute_positive_roots_and_reflections(C::SMatrix{R,R,Int}, rk::Integer) where {R}
  # Start with simple roots (standard basis vectors)
  pos_roots = [SVector{R,Int}(ntuple(j -> Int(i == j), R)) for i in 1:rk]
  pos_coroots = [SVector{R,Int}(ntuple(j -> Int(i == j), R)) for i in 1:rk]

  # Build map from root vector to index
  root_index = Dict{SVector{R,Int},Int}()
  for i in 1:rk
    root_index[pos_roots[i]] = i
  end

  # Reflection table: refl[s, i] gives the index of s_s(α_i) in pos_roots,
  # or 0 if the result is a negative root
  # We'll grow this as we discover new roots
  refl_data = Dict{Tuple{Int,Int},UInt}()
  for s in 1:rk
    refl_data[(s, s)] = 0  # s_s(α_s) = -α_s  (negative)
  end

  i = 1
  while i <= length(pos_roots)
    for s in 1:rk
      haskey(refl_data, (s, i)) && continue

      root_i = pos_roots[i]
      coroot_i = pos_coroots[i]

      # ⟨αₛ∨, root_i⟩ = sum_j C[s,j] * root_i[j]
      pairing = sum(C[s, j] * root_i[j] for j in 1:rk)

      # ⟨coroot_i, αₛ⟩ = sum_j coroot_i[j] * C[j, s]
      copairing = sum(coroot_i[j] * C[j, s] for j in 1:rk)

      if pairing * copairing >= 4
        # Not a valid reflection (imaginary root situation for non-finite types)
        refl_data[(s, i)] = 0
        continue
      end

      # Reflected root: s_s(root_i) = root_i - pairing * α_s
      new_root = SVector{R,Int}(ntuple(j -> root_i[j] - pairing * (j == s ? 1 : 0), R))

      # Reflected coroot: s_s(coroot_i) = coroot_i - copairing * α_s∨
      new_coroot = SVector{R,Int}(
        ntuple(j -> coroot_i[j] - copairing * (j == s ? 1 : 0), R)
      )

      if all(>=(0), new_root)
        # Result is a positive root
        idx = get(root_index, new_root, 0)
        if idx == 0
          # New positive root discovered
          push!(pos_roots, new_root)
          push!(pos_coroots, new_coroot)
          idx = length(pos_roots)
          root_index[new_root] = idx
        end
        refl_data[(s, i)] = UInt(idx)
        refl_data[(s, idx)] = UInt(i)
      elseif all(<=(0), new_root)
        # Result is a negative root
        neg_root = -new_root
        idx = get(root_index, neg_root, 0)
        if idx == 0
          # This shouldn't happen for positive root reflections
          refl_data[(s, i)] = 0
        else
          refl_data[(s, i)] = 0  # mark as "goes to negative"
        end
      else
        refl_data[(s, i)] = 0
      end
    end
    i += 1
  end

  # Build the reflection matrix
  n_pos = length(pos_roots)
  refl = zeros(UInt, rk, n_pos)
  for ((s, i), v) in refl_data
    if 1 <= s <= rk && 1 <= i <= n_pos
      refl[s, i] = v
    end
  end

  return pos_roots, pos_coroots, refl
end

# ─── Accessors ───────────────────────────────────────────────────────────────

"""
    n_simple_roots(RS::RootSystem) -> Int

Return the number of simple roots, equal to the rank of the root system.
"""
n_simple_roots(RS::RootSystem{DT,R}) where {DT,R} = R

"""
    n_positive_roots(RS::RootSystem) -> Int

Return the number of positive roots.
"""
n_positive_roots(RS::RootSystem) = length(RS.positive_roots_list)

"""
    n_roots(RS::RootSystem) -> Int

Return the total number of roots (positive and negative).
"""
n_roots(RS::RootSystem) = 2 * n_positive_roots(RS)

"""
    simple_root(RS::RootSystem{DT,R}, i) -> RootSpaceElem

Return the `i`-th simple root.
"""
function simple_root(RS::RootSystem{DT,R}, i::Integer) where {DT,R}
  @boundscheck 1 <= i <= R || throw(BoundsError("simple root index $i out of range"))
  return RootSpaceElem{DT,R}(RS.positive_roots_list[i])
end

"""
    simple_roots(RS::RootSystem{DT,R}) -> Vector{RootSpaceElem}

Return all simple roots.
"""
simple_roots(RS::RootSystem{DT,R}) where {DT,R} =
  [RootSpaceElem{DT,R}(RS.positive_roots_list[i]) for i in 1:R]

"""
    positive_root(RS::RootSystem{DT,R}, i) -> RootSpaceElem

Return the `i`-th positive root.
"""
function positive_root(RS::RootSystem{DT,R}, i::Integer) where {DT,R}
  return RootSpaceElem{DT,R}(RS.positive_roots_list[i])
end

"""
    positive_roots(RS::RootSystem{DT,R}) -> Vector{RootSpaceElem}

Return all positive roots.
"""
positive_roots(RS::RootSystem{DT,R}) where {DT,R} =
  [RootSpaceElem{DT,R}(v) for v in RS.positive_roots_list]

"""
    negative_root(RS::RootSystem{DT,R}, i) -> RootSpaceElem

Return the `i`-th negative root (negative of the `i`-th positive root).
"""
negative_root(RS::RootSystem{DT,R}, i::Integer) where {DT,R} =
  RootSpaceElem{DT,R}(-RS.positive_roots_list[i])

"""
    negative_roots(RS::RootSystem{DT,R}) -> Vector{RootSpaceElem}

Return all negative roots.
"""
negative_roots(RS::RootSystem{DT,R}) where {DT,R} =
  [RootSpaceElem{DT,R}(-v) for v in RS.positive_roots_list]

"""
    roots(RS::RootSystem) -> Vector{RootSpaceElem}

Return all roots (positive followed by negative).
"""
roots(RS::RootSystem) = vcat(positive_roots(RS), negative_roots(RS))

"""
    root(RS::RootSystem{DT,R}, i) -> RootSpaceElem

Return the `i`-th root. Indices 1..n_pos are positive roots,
n_pos+1..2*n_pos are negative roots.
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
"""
simple_coroots(RS::RootSystem{DT,R}) where {DT,R} =
  [RootSpaceElem{DT,R}(RS.positive_coroots_list[i]) for i in 1:R]

"""
    positive_coroots(RS::RootSystem{DT,R}) -> Vector{RootSpaceElem}

Return all positive coroots.
"""
positive_coroots(RS::RootSystem{DT,R}) where {DT,R} =
  [RootSpaceElem{DT,R}(v) for v in RS.positive_coroots_list]

# ─── Highest root ────────────────────────────────────────────────────────────

"""
    highest_root(RS::RootSystem{DT,R}) -> RootSpaceElem

Return the highest root (the positive root with greatest height).
"""
function highest_root(RS::RootSystem{DT,R}) where {DT,R}
  best_idx = 1
  best_ht = height(positive_root(RS, 1))
  for i in 2:n_positive_roots(RS)
    h = sum(RS.positive_roots_list[i])
    if h > best_ht
      best_ht = h
      best_idx = i
    end
  end
  return positive_root(RS, best_idx)
end

# ─── Degrees of fundamental invariants ────────────────────────────────────────

"""
    degrees_fundamental_invariants(::Type{DT}) -> SVector{R,Int}
    degrees_fundamental_invariants(dt::DT) -> SVector{R,Int}

Return the degrees of the fundamental invariants of the Weyl group action on the
polynomial ring.

# Examples
```jldoctest
julia> using Lie

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
  SVector{N,Int}(Tuple(2:(N + 1)))

# B_n and C_n: 2, 4, 6, ..., 2n
function degrees_fundamental_invariants(::Type{TypeB{N}}) where {N}
  return SVector{N,Int}(Tuple(2:2:(2N)))
end

function degrees_fundamental_invariants(::Type{TypeC{N}}) where {N}
  return SVector{N,Int}(Tuple(2:2:(2N)))
end

# D_n: 2, 4, ..., 2(n-1), n
function degrees_fundamental_invariants(::Type{TypeD{N}}) where {N}
  return SVector{N,Int}(Tuple(vcat(collect(2:2:(2N - 2)), [N])))
end

degrees_fundamental_invariants(::Type{TypeE{6}}) = SVector{6,Int}((2, 5, 6, 8, 9, 12))
degrees_fundamental_invariants(::Type{TypeE{7}}) =
  SVector{7,Int}((2, 6, 8, 10, 12, 14, 18))
degrees_fundamental_invariants(::Type{TypeE{8}}) =
  SVector{8,Int}((2, 8, 12, 14, 18, 20, 24, 30))
degrees_fundamental_invariants(::Type{TypeF4}) = SVector{4,Int}((2, 6, 8, 12))
degrees_fundamental_invariants(::Type{TypeG2}) = SVector{2,Int}((2, 6))

@generated function degrees_fundamental_invariants(::Type{ProductDynkinType{Ts}}) where {Ts}
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
"""
function is_root(RS::RootSystem{DT,R}, v::RootSpaceElem{DT,R}) where {DT,R}
  return is_positive_root(RS, v) || is_positive_root(RS, -v)
end

"""
    is_positive_root(RS::RootSystem{DT,R}, v::RootSpaceElem{DT,R}) -> Bool

Check whether `v` is a positive root.
"""
function is_positive_root(RS::RootSystem{DT,R}, v::RootSpaceElem{DT,R}) where {DT,R}
  return v.vec in RS.positive_roots_list
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
