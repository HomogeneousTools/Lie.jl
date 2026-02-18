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
stored as an `SVector{R,Int}` of coordinates in the fundamental weight basis (ω₁,…,ωᵣ).

The pairing with the i-th simple coroot is simply `w[i]`:
``⟨αᵢ∨, λ⟩ = λᵢ``
"""
struct WeightLatticeElem{DT<:DynkinType,R}
  vec::SVector{R,Int}
end

function WeightLatticeElem(::Type{DT}, v::SVector{R,Int}) where {DT<:DynkinType,R}
  @assert R == rank(DT) "Vector length $R does not match rank $(rank(DT))"
  return WeightLatticeElem{DT,R}(v)
end

function WeightLatticeElem(::Type{DT}, v::NTuple{R,Int}) where {DT<:DynkinType,R}
  return WeightLatticeElem(DT, SVector{R,Int}(v))
end

function WeightLatticeElem(::Type{DT}, v::AbstractVector{<:Integer}) where {DT<:DynkinType}
  R = rank(DT)
  return WeightLatticeElem(DT, SVector{R,Int}(v...))
end

coefficients(w::WeightLatticeElem) = w.vec
Base.getindex(w::WeightLatticeElem, i::Int) = w.vec[i]

Base.:+(a::WeightLatticeElem{DT,R}, b::WeightLatticeElem{DT,R}) where {DT,R} =
  WeightLatticeElem{DT,R}(a.vec + b.vec)
Base.:-(a::WeightLatticeElem{DT,R}, b::WeightLatticeElem{DT,R}) where {DT,R} =
  WeightLatticeElem{DT,R}(a.vec - b.vec)
Base.:-(a::WeightLatticeElem{DT,R}) where {DT,R} = WeightLatticeElem{DT,R}(-a.vec)
Base.:*(n::Integer, a::WeightLatticeElem{DT,R}) where {DT,R} =
  WeightLatticeElem{DT,R}(n * a.vec)
Base.:*(a::WeightLatticeElem, n::Integer) = n * a
Base.:(==)(a::WeightLatticeElem{DT,R}, b::WeightLatticeElem{DT,R}) where {DT,R} =
  a.vec == b.vec
Base.hash(a::WeightLatticeElem, h::UInt) = hash(a.vec, h)
Base.zero(::Type{WeightLatticeElem{DT,R}}) where {DT,R} =
  WeightLatticeElem{DT,R}(zero(SVector{R,Int}))
Base.zero(::WeightLatticeElem{DT,R}) where {DT,R} =
  WeightLatticeElem{DT,R}(zero(SVector{R,Int}))
Base.iszero(a::WeightLatticeElem) = iszero(a.vec)

function Base.show(io::IO, w::WeightLatticeElem{DT,R}) where {DT,R}
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

Return the `i`-th fundamental weight ωᵢ.

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
"""
function fundamental_weights(::Type{DT}) where {DT<:DynkinType}
  return [fundamental_weight(DT, i) for i in 1:rank(DT)]
end

"""
    weyl_vector(::Type{DT}) -> WeightLatticeElem{DT}

Return the Weyl vector ρ = ω₁ + ω₂ + ⋯ + ωᵣ = ½∑_{α>0} α.

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

A weight is dominant iff all its coordinates (pairings with simple coroots) are ≥ 0.

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
Since αᵢ = ∑ⱼ Cⱼᵢ ωⱼ, the weight coordinates of v = ∑ vᵢ αᵢ are:
``w_j = ∑_i C_{ji} v_i = (C v)_j``
"""
function WeightLatticeElem(r::RootSpaceElem{DT,R}) where {DT,R}
  C = cartan_matrix(DT)
  w = C * r.vec
  return WeightLatticeElem{DT,R}(SVector{R,Int}(w))
end

"""
    RootSpaceElem(w::WeightLatticeElem{DT,R}) -> RootSpaceElem{DT,R}

Convert a weight to root coordinates (may be rational; this function
rounds to Int which is valid only for weights in the root lattice).

``v = C^{-1} w``
"""
function RootSpaceElem(w::WeightLatticeElem{DT,R}) where {DT,R}
  Cinv = cartan_matrix_inverse(DT)
  v = Cinv * SVector{R,Rational{Int}}(w.vec)
  # Check integrality
  v_int = SVector{R,Int}(round.(Int, v))
  return RootSpaceElem{DT,R}(v_int)
end

# ─── Reflect a weight by a simple reflection ────────────────────────────────

"""
    reflect(w::WeightLatticeElem{DT,R}, s::Integer) -> WeightLatticeElem{DT,R}

Reflect `w` by the `s`-th simple reflection:
``s_s(λ) = λ - ⟨α_s∨, λ⟩ α_s``

In the fundamental weight basis, ⟨α_s∨, λ⟩ = λ_s and α_s = ∑_j C_{js} ω_j,
so the new weight has coordinates:
``(s_s(λ))_j = λ_j - λ_s C_{js}``
"""
function reflect(w::WeightLatticeElem{DT,R}, s::Integer) where {DT,R}
  C = cartan_matrix(DT)
  pairing = w.vec[s]  # = ⟨αₛ∨, λ⟩
  # Subtract pairing * (s-th column of C, which gives αₛ in ω-basis)
  new_vec = SVector{R,Int}(ntuple(j -> w.vec[j] - pairing * C[j, s], R))
  return WeightLatticeElem{DT,R}(new_vec)
end

"""
    reflect(w::WeightLatticeElem{DT,R}, β::RootSpaceElem{DT,R}) -> WeightLatticeElem{DT,R}

Reflect `w` by the root `β`:
``s_β(λ) = λ - ⟨β∨, λ⟩ β``
where `⟨β∨, λ⟩ = 2(β, λ)/(β, β)`.
"""
function reflect(w::WeightLatticeElem{DT,R}, β::RootSpaceElem{DT,R}) where {DT,R}
  # Convert w to root space, compute reflection, convert back
  # Or: use the weight-root pairing directly
  C = cartan_matrix(DT)
  # ⟨β∨, ωⱼ⟩ = 2(β, ωⱼ)/(β, β), and (β, ωⱼ) involves the bilinear form
  # Simpler: ⟨β∨, λ⟩ = ∑ᵢ β∨ᵢ λᵢ where β∨ = 2β/(β,β) in coroot coordinates
  # β∨ᵢ as simple coroot coords: β∨ = 2β/(β,β), but in weight pairing:
  # ⟨β∨, λ⟩ = ∑ᵢ βᵢ (Cλ)ᵢ ... no.
  # Actually: β = ∑ βᵢ αᵢ and λ = ∑ λⱼ ωⱼ, then
  # ⟨αᵢ∨, ωⱼ⟩ = δᵢⱼ, so ⟨β∨, λ⟩ = ∑ᵢ βᵢ^∨ λᵢ
  # where β∨ = (2/(β,β)) * diag(d) * β in simple coroot coords.
  # But for simple roots, this reduces to: ⟨αₛ∨, λ⟩ = λₛ.
  # For general β, we need: ⟨β∨, λ⟩ = ∑ β_coroot_i * λ_i.
  # Coroot coordinates: if β = ∑ bᵢαᵢ, then β∨ has coroot coords
  # β∨ᵢ = (d_i/d_β) bᵢ  where d_β relates to the root length.
  # This is simpler to compute via dot product.
  B = cartan_bilinear_form(DT)
  β_vec = β.vec
  Cinv_tr = cartan_matrix_inverse(DT)'
  w_root = Cinv_tr * SVector{R,Rational{Int}}(w.vec)
  β_dot_β = β_vec' * B * β_vec
  β_dot_w = β_vec' * B * w_root
  coeff = 2 * β_dot_w//β_dot_β
  # s_β(λ) = λ - coeff * β  (in root coords → convert to weight coords)
  new_root = w_root - Rational{Int}(coeff) .* SVector{R,Rational{Int}}(β_vec)
  # Convert back to weight coords
  C = cartan_matrix(DT)
  new_weight = SMatrix{R,R,Rational{Int}}(C)' * new_root
  return WeightLatticeElem{DT,R}(SVector{R,Int}(round.(Int, new_weight)))
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
function conjugate_dominant_weight(w::WeightLatticeElem{DT,R}) where {DT,R}
  v = MVector{R,Int}(w.vec)
  C = cartan_matrix(DT)
  s = 1
  while s <= R
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
function conjugate_dominant_weight_with_length(w::WeightLatticeElem{DT,R}) where {DT,R}
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

Compute the inner product `(α, λ)` between a root `α` (in simple root coords)
and a weight `λ` (in fundamental weight coords).

Following Oscar's convention:
``(α, λ) = ∑ᵢ αᵢ dᵢ λᵢ``
where `d` is the Cartan symmetrizer.

This works because `(αᵢ, ωⱼ) = dᵢ δᵢⱼ`, which follows from
`⟨αᵢ∨, ωⱼ⟩ = δᵢⱼ` and `αᵢ∨ = αᵢ/dᵢ` in the bilinear form sense.
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

Compute the inner product `(λ, μ)` between two weights.
Both in fundamental weight coords, convert to root coords and use the bilinear form.
"""
function dot(w1::WeightLatticeElem{DT,R}, w2::WeightLatticeElem{DT,R}) where {DT,R}
  Cinv = cartan_matrix_inverse(DT)
  w1_root = Cinv' * SVector{R,Rational{Int}}(w1.vec)
  w2_root = Cinv' * SVector{R,Rational{Int}}(w2.vec)
  B = cartan_bilinear_form(DT)
  return w1_root' * SVector{R,Rational{Int}}(B * SVector{R,Rational{Int}}(w2_root))
end
