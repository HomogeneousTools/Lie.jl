# ═══════════════════════════════════════════════════════════════════════════════
#  Weight lattice elements — fundamental weights and weight operations
#
#  Weights are stored as SVector{R,Int} in the fundamental weight basis (ω₁,…,ωᵣ).
#  The relationship to simple roots is: ωᵢ = ∑ⱼ (C⁻¹)ᵢⱼ αⱼ
#  and: αᵢ = ∑ⱼ Cⱼᵢ ωⱼ  (i.e. ⟨αᵢ∨, ωⱼ⟩ = δᵢⱼ)
# ═══════════════════════════════════════════════════════════════════════════════

export WeightLatticeElem
export fundamental_weight, fundamental_weights, weyl_vector
export is_dominant, conjugate_dominant_weight, conjugate_dominant_weight_with_elem,
  conjugate_dominant_weight_with_length
export reflect

# ═══════════════════════════════════════════════════════════════════════════════
#  WeightLatticeElem
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WeightLatticeElem{DT,R}

An element of the weight lattice for Dynkin type `DT` of rank `R`,
stored as an `SVector{R,Int}` of coordinates in the fundamental weight basis
``(\\omega_1, \\ldots, \\omega_r)``.

The pairing with the i-th simple coroot is simply `w[i]`:
``\\langle \\alpha_i^\\vee, \\lambda \\rangle = \\lambda_i``

## Constructors

    WeightLatticeElem(::Type{DT}, v::AbstractVector{<:Integer})

When `v` has fewer entries than `rank(DT)`, the remaining coordinates are
silently filled with zeros. When `v` has more entries than `rank(DT)`, a
warning is emitted and only the first `rank(DT)` entries are used.

## Length handling

The `AbstractVector` constructor is meant as a convenience for interactive work.
For library code, tests, and reproducible computations, prefer the exact-length
`SVector` or `NTuple` constructors so dimension mismatches are caught
immediately. Padding can change the intended weight by implicitly adding zero
coordinates, while truncation discards trailing coordinates after emitting a
warning.

# Examples
```jldoctest
julia> using Lie

julia> WeightLatticeElem(TypeA{3}, [1, 2])   # padded with one zero
ω1 + 2ω2

julia> WeightLatticeElem(TypeA{3}, [1, 2, 3])  # exact length
ω1 + 2ω2 + 3ω3

julia> using Test

julia> @test_logs (:warn, r"truncating to first 3 entries") WeightLatticeElem(TypeA{3}, [1, 2, 3, 4])
ω1 + 2ω2 + 3ω3
```
"""
struct WeightLatticeElem{DT<:DynkinType,R}
  vec::SVector{R,Int}
end

function WeightLatticeElem(::Type{DT}, v::SVector{R,Int}) where {DT<:DynkinType,R}
  R == rank(DT) || throw(ArgumentError("Vector length $R does not match rank $(rank(DT))"))
  return WeightLatticeElem{DT,R}(v)
end

function WeightLatticeElem(::Type{DT}, v::NTuple{R,Int}) where {DT<:DynkinType,R}
  return WeightLatticeElem(DT, SVector{R,Int}(v))
end

function WeightLatticeElem(::Type{DT}, v::AbstractVector{<:Integer}) where {DT<:DynkinType}
  R = rank(DT)
  n = length(v)
  if n > R
    @warn "weight vector has length $n but $(DT) has rank $R; truncating to first $R entries"
    return WeightLatticeElem(DT, SVector{R,Int}(v[1:R]...))
  elseif n < R
    return WeightLatticeElem(DT, SVector{R,Int}(v..., ntuple(_ -> 0, R - n)...))
  else
    return WeightLatticeElem(DT, SVector{R,Int}(v...))
  end
end

coefficients(w::WeightLatticeElem) = w.vec
Base.getindex(w::WeightLatticeElem, i::Integer) = w.vec[i]

Base.:+(a::WeightLatticeElem{DT,R}, b::WeightLatticeElem{DT,R}) where {DT,R} = WeightLatticeElem{
  DT,R
}(
  a.vec + b.vec
)
Base.:-(a::WeightLatticeElem{DT,R}, b::WeightLatticeElem{DT,R}) where {DT,R} = WeightLatticeElem{
  DT,R
}(
  a.vec - b.vec
)
Base.:-(a::WeightLatticeElem{DT,R}) where {DT,R} = WeightLatticeElem{DT,R}(-a.vec)
Base.:*(n::Integer, a::WeightLatticeElem{DT,R}) where {DT,R} = WeightLatticeElem{DT,R}(
  n * a.vec
)
Base.:*(a::WeightLatticeElem, n::Integer) = n * a
Base.:(==)(a::WeightLatticeElem{DT,R}, b::WeightLatticeElem{DT,R}) where {DT,R} =
  a.vec == b.vec
Base.hash(a::WeightLatticeElem, h::UInt) = hash(a.vec, h)
Base.zero(::Type{WeightLatticeElem{DT,R}}) where {DT,R} = WeightLatticeElem{DT,R}(
  zero(SVector{R,Int})
)
Base.zero(::WeightLatticeElem{DT,R}) where {DT,R} = WeightLatticeElem{DT,R}(
  zero(SVector{R,Int})
)
Base.iszero(a::WeightLatticeElem) = iszero(a.vec)

function Base.show(io::IO, w::WeightLatticeElem{DT,R}) where {DT,R}
  if get(io, :compact, _compact_display[])
    print(io, _type_name(DT), "[", join(w.vec, ","), "]")
    return nothing
  end
  terms = String[]
  for i in 1:R
    c = w.vec[i]
    c == 0 && continue
    if c == 1
      push!(terms, "ω$i")
    elseif c == -1
      push!(terms, "-ω$i")
    else
      push!(terms, "$(c)ω$i")
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

# ─── Fundamental weights ────────────────────────────────────────────────────

"""
    fundamental_weight(::Type{DT}, i) -> WeightLatticeElem{DT}

Return the `i`-th fundamental weight ``\\omega_i``.

# Examples
```jldoctest
julia> using Lie

julia> fundamental_weight(TypeA{3}, 1)
ω1

julia> fundamental_weight(TypeB{2}, 2)
ω2
```
"""
function fundamental_weight(::Type{DT}, i::Integer) where {DT<:DynkinType}
  R = rank(DT)
  return WeightLatticeElem{DT,R}(SVector{R,Int}(ntuple(j -> Int(j == i), R)))
end

"""
    fundamental_weights(::Type{DT}) -> Vector{WeightLatticeElem{DT}}

Return all fundamental weights.

# Examples
```jldoctest
julia> using Lie

julia> fundamental_weights(TypeA{2})
2-element Vector{WeightLatticeElem{TypeA{2}, 2}}:
 ω1
 ω2
```
"""
function fundamental_weights(::Type{DT}) where {DT<:DynkinType}
  return [fundamental_weight(DT, i) for i in 1:rank(DT)]
end

"""
    weyl_vector(::Type{DT}) -> WeightLatticeElem{DT}

Return the Weyl vector
``\\rho = \\omega_1 + \\omega_2 + \\cdots + \\omega_r = \\frac{1}{2}\\sum_{\\alpha > 0} \\alpha``.

# Examples
```jldoctest
julia> using Lie

julia> weyl_vector(TypeA{3})
ω1 + ω2 + ω3
```
"""
function weyl_vector(::Type{DT}) where {DT<:DynkinType}
  R = rank(DT)
  return WeightLatticeElem{DT,R}(SVector{R,Int}(ntuple(j -> 1, R)))
end

# ─── Dominance ───────────────────────────────────────────────────────────────

"""
    is_dominant(w::WeightLatticeElem) -> Bool

A weight is dominant iff all its coordinates (pairings with simple coroots) are `>= 0`.

# Examples
```jldoctest
julia> using Lie

julia> is_dominant(fundamental_weight(TypeA{2}, 1))
true

julia> is_dominant(WeightLatticeElem(TypeA{2}, [-1, 1]))
false
```
"""
is_dominant(w::WeightLatticeElem) = all(>=(0), w.vec)

# ─── Conversion between root and weight coordinates ─────────────────────────

"""
    WeightLatticeElem(r::RootSpaceElem{DT,R}) -> WeightLatticeElem{DT,R}

Convert a root space element to weight coordinates.
Since ``\\alpha_i = \\sum_j C_{ji} \\omega_j``, the weight coordinates of
``v = \\sum_i v_i \\alpha_i`` are
``w_j = \\sum_i C_{ji} v_i = (Cv)_j``.

# Examples
```jldoctest
julia> using Lie

julia> WeightLatticeElem(RootSpaceElem(TypeA{2}, [1, 0]))
2ω1 - ω2
```
"""
function WeightLatticeElem(r::RootSpaceElem{DT,R}) where {DT,R}
  C = cartan_matrix(DT)
  w = C * r.vec
  return WeightLatticeElem{DT,R}(SVector{R,Int}(w))
end

"""
    RootSpaceElem(w::WeightLatticeElem{DT,R}) -> RootSpaceElem{DT,R}

Convert a weight to root coordinates.

Throws an `ArgumentError` when the weight does not lie in the root lattice,
since the inverse Cartan-matrix coordinates are then non-integral.

``v = C^{-1} w``

# Examples
```jldoctest
julia> using Lie

julia> RootSpaceElem(WeightLatticeElem(TypeA{2}, [2, -1]))
α1

julia> RootSpaceElem(fundamental_weight(TypeA{2}, 1))
ERROR: ArgumentError: Weight does not lie in the root lattice
```
"""
function RootSpaceElem(w::WeightLatticeElem{DT,R}) where {DT,R}
  Cinv = cartan_matrix_inverse(DT)
  v = Cinv * SVector{R,Rational{Int}}(w.vec)
  all(isinteger, v) || throw(ArgumentError("Weight does not lie in the root lattice"))
  return RootSpaceElem{DT,R}(SVector{R,Int}(Int.(v)))
end

# ─── Product-type helper layer ────────────────────────────────────────────────

function _product_component_weights(
  ::Type{PDT}, w::WeightLatticeElem{PDT,R}
) where {Ts,PDT<:ProductDynkinType{Ts},R}
  types = Ts.parameters
  ranks = component_ranks(PDT)
  offsets = component_offsets(PDT)
  weights = Any[]
  sizehint!(weights, length(ranks))
  for i in eachindex(ranks)
    T = types[i]
    r = ranks[i]
    offset = offsets[i]
    push!(weights, WeightLatticeElem(T, Int[w.vec[offset + j] for j in 1:r]))
  end
  return weights
end

function _product_embed_coords(
  ::Type{PDT}, i::Integer, v::AbstractVector{<:Integer}
) where {Ts,PDT<:ProductDynkinType{Ts}}
  offsets = component_offsets(PDT)
  ranks = component_ranks(PDT)
  1 <= i <= length(ranks) || throw(BoundsError(ranks, i))
  r = ranks[i]
  length(v) == r ||
    throw(ArgumentError("Vector length $(length(v)) does not match rank $r of factor $i"))

  R = rank(PDT)
  coords = zeros(Int, R)
  offset = offsets[i]
  @inbounds for j in 1:r
    coords[offset + j] = Int(v[j])
  end
  return SVector{R,Int}(Tuple(coords))
end

function _product_embed_weight(
  ::Type{PDT}, i::Integer, w::WeightLatticeElem
) where {Ts,PDT<:ProductDynkinType{Ts}}
  R = rank(PDT)
  return WeightLatticeElem{PDT,R}(_product_embed_coords(PDT, i, w.vec))
end

function _product_single_supported_component(
  w::WeightLatticeElem{PDT,R}
) where {Ts,PDT<:ProductDynkinType{Ts},R}
  weights = _product_component_weights(PDT, w)
  idx = 0
  component_weight = nothing
  for i in eachindex(weights)
    wi = weights[i]
    if !iszero(wi)
      idx == 0 || return nothing
      idx = i
      component_weight = wi
    end
  end
  return idx == 0 ? nothing : (idx, component_weight)
end

# ─── Reflect a weight by a simple reflection ────────────────────────────────

"""
    reflect(w::WeightLatticeElem{DT,R}, s::Integer) -> WeightLatticeElem{DT,R}

Reflect `w` by the `s`-th simple reflection:
``s_s(\\lambda) = \\lambda - \\langle \\alpha_s^\\vee, \\lambda \\rangle \\alpha_s``

In the fundamental weight basis,
``\\langle \\alpha_s^\\vee, \\lambda \\rangle = \\lambda_s`` and
``\\alpha_s = \\sum_j C_{js} \\omega_j``,
so the new weight has coordinates:
``(s_s(\\lambda))_j = \\lambda_j - \\lambda_s C_{js}``

# Examples
```jldoctest
julia> using Lie

julia> reflect(WeightLatticeElem(TypeA{2}, [2, 1]), 1)
-2ω1 + 3ω2
```
"""
function reflect(w::WeightLatticeElem{DT,R}, s::Integer) where {DT,R}
  C = cartan_matrix(DT)
  pairing = w.vec[s]  # Pairing with the s-th simple coroot.
  # Subtract pairing times the s-th simple root in the fundamental-weight basis.
  new_vec = SVector{R,Int}(ntuple(j -> w.vec[j] - pairing * C[j, s], R))
  return WeightLatticeElem{DT,R}(new_vec)
end

"""
    reflect(w::WeightLatticeElem{DT,R}, β::RootSpaceElem{DT,R}) -> WeightLatticeElem{DT,R}

Reflect `w` by the root `β`:
``s_\\beta(\\lambda) = \\lambda - \\langle \\beta^\\vee, \\lambda \\rangle \\beta``
where ``\\langle \\beta^\\vee, \\lambda \\rangle = 2(\\beta, \\lambda) / (\\beta, \\beta)``.

The argument ``\\beta`` must be an actual root of the root system.

# Examples
```jldoctest
julia> using Lie

julia> reflect(fundamental_weight(TypeA{2}, 1), simple_root(RootSystem(TypeA{2}), 1))
-ω1 + ω2
```
"""
function reflect(w::WeightLatticeElem{DT,R}, β::RootSpaceElem{DT,R}) where {DT,R}
  is_root(RootSystem(DT), β) || throw(ArgumentError("β must be a root"))
  C = cartan_matrix(DT)
  d = cartan_symmetrizer(DT)
  β_vec = β.vec
  # Compute (\beta, \lambda) and (\beta, \beta).
  numer = 0
  denom = 0
  @inbounds for i in 1:R
    numer += d[i] * β_vec[i] * w.vec[i]
    s = 0
    for j in 1:R
      s += C[i, j] * β_vec[j]
    end
    denom += d[i] * β_vec[i] * s
  end
  # beta in fundamental-weight coordinates.
  Cβ = C * β_vec
  coeff = div(2 * numer, denom)
  new_vec = SVector{R,Int}(ntuple(j -> @inbounds(w.vec[j] - coeff * Cβ[j]), R))
  return WeightLatticeElem{DT,R}(new_vec)
end

# ─── Conjugation to dominant chamber ────────────────────────────────────────

"""
    conjugate_dominant_weight(w::WeightLatticeElem{DT,R}) -> WeightLatticeElem{DT,R}

Return the unique dominant weight in the Weyl orbit of `w`.

# Examples
```jldoctest
julia> using Lie

julia> conjugate_dominant_weight(WeightLatticeElem(TypeA{2}, [-1, 1]))
ω1

julia> conjugate_dominant_weight(fundamental_weight(TypeA{3}, 1))
ω1
```
"""
@inline function conjugate_dominant_weight(w::WeightLatticeElem{DT,R}) where {DT,R}
  v = MVector{R,Int}(w.vec)
  C = cartan_matrix(DT)
  s = 1
  @inbounds while s <= R
    if v[s] < 0
      pairing = v[s]
      for j in 1:R
        v[j] -= pairing * C[j, s]
      end
      s = 1
    else
      s += 1
    end
  end
  return WeightLatticeElem{DT,R}(SVector{R,Int}(v))
end

"""
    conjugate_dominant_weight_with_elem(w::WeightLatticeElem{DT,R}) -> (WeightLatticeElem, Vector{Int})

Return the dominant weight and the sequence of simple reflections applied.

# Examples
```jldoctest
julia> using Lie

julia> conjugate_dominant_weight_with_elem(WeightLatticeElem(TypeA{2}, [-1, 1]))
(ω1, [1])
```
"""
function conjugate_dominant_weight_with_elem(w::WeightLatticeElem{DT,R}) where {DT,R}
  v = MVector{R,Int}(w.vec)
  C = cartan_matrix(DT)
  word = Int[]
  s = 1
  while s <= R
    if v[s] < 0
      pairing = v[s]
      for j in 1:R
        v[j] -= pairing * C[j, s]
      end
      push!(word, s)
      s = 1
    else
      s += 1
    end
  end
  return WeightLatticeElem{DT,R}(SVector{R,Int}(v)), word
end

"""
    conjugate_dominant_weight_with_length(w::WeightLatticeElem{DT,R}) -> (WeightLatticeElem, Int)

Return the dominant weight in the Weyl orbit of `w` together with the number
of simple reflections applied (i.e. the length of the Weyl group element
mapping `w` into the dominant chamber).

This is faster than [`conjugate_dominant_weight_with_elem`](@ref) because it
only tracks a counter instead of building the full word.

# Examples
```jldoctest
julia> using Lie

julia> conjugate_dominant_weight_with_length(WeightLatticeElem(TypeA{2}, [-1, 1]))
(ω1, 1)

julia> conjugate_dominant_weight_with_length(fundamental_weight(TypeA{3}, 1))
(ω1, 0)
```
"""
@inline function conjugate_dominant_weight_with_length(
  w::WeightLatticeElem{DT,R}
) where {DT,R}
  v = MVector{R,Int}(w.vec)
  C = cartan_matrix(DT)
  len = 0
  s = 1
  while s <= R
    if v[s] < 0
      pairing = v[s]
      for j in 1:R
        v[j] -= pairing * C[j, s]
      end
      len += 1
      s = 1
    else
      s += 1
    end
  end
  return WeightLatticeElem{DT,R}(SVector{R,Int}(v)), len
end

# ─── Inner products involving weights ────────────────────────────────────────

"""
    dot(r::RootSpaceElem{DT,R}, w::WeightLatticeElem{DT,R}) -> Rational{Int}

Compute the inner product ``(\\alpha, \\lambda)`` between a root ``\\alpha``
(in simple-root coordinates) and a weight ``\\lambda`` (in fundamental-weight coordinates).

Following OSCAR's convention:
``(\\alpha, \\lambda) = \\sum_i \\alpha_i d_i \\lambda_i``
where `d` is the Cartan symmetrizer.

This works because
``(\\alpha_i, \\omega_j) = d_i \\delta_{ij}``,
which follows from
``\\langle \\alpha_i^\\vee, \\omega_j \\rangle = \\delta_{ij}``
and
``\\alpha_i^\\vee = \\alpha_i / d_i``
in the bilinear-form sense.

# Examples
```jldoctest
julia> using Lie

julia> dot(simple_root(RootSystem(TypeA{2}), 1), fundamental_weight(TypeA{2}, 1))
1//1
```
"""
function dot(r::RootSpaceElem{DT,R}, w::WeightLatticeElem{DT,R}) where {DT,R}
  d = cartan_symmetrizer(DT)
  result = Rational{Int}(0)
  for i in 1:R
    result += r.vec[i] * d[i] * w.vec[i]
  end
  return result
end

function dot(w::WeightLatticeElem{DT,R}, r::RootSpaceElem{DT,R}) where {DT,R}
  return dot(r, w)
end

"""
    dot(w1::WeightLatticeElem{DT,R}, w2::WeightLatticeElem{DT,R}) -> Rational{Int}

Compute the inner product ``(\\lambda, \\mu)`` between two weights.
Both are given in fundamental-weight coordinates; the implementation converts
to root coordinates and applies the bilinear form there.

# Examples
```jldoctest
julia> using Lie

julia> dot(fundamental_weight(TypeA{2}, 1), fundamental_weight(TypeA{2}, 1))
2//3
```
"""
function dot(w1::WeightLatticeElem{DT,R}, w2::WeightLatticeElem{DT,R}) where {DT,R}
  Cinv = cartan_matrix_inverse(DT)
  w1_root = Cinv * SVector{R,Rational{Int}}(w1.vec)
  w2_root = Cinv * SVector{R,Rational{Int}}(w2.vec)
  B = cartan_bilinear_form(DT)
  return w1_root' * SVector{R,Rational{Int}}(B * SVector{R,Rational{Int}}(w2_root))
end
