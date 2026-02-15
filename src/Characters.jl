# ═══════════════════════════════════════════════════════════════════════════════
#  Weyl characters — elements of the representation ring (Grothendieck ring)
#
#  A WeylCharacter is a formal ℤ-linear combination of irreducible
#  representations, encoded as Dict{WeightLatticeElem{DT,R}, Int} mapping
#  dominant highest weights (in fundamental weight coordinates) to multiplicities.
#
#  Key algorithms:
#    • Freudenthal's formula — weight multiplicities of an irreducible
#    • Brauer–Klimyk — tensor product of a character with an irreducible
#    • Adams operators — ψᵏ via Freudenthal + weight scaling
#    • Symmetric / exterior powers — Newton–Girard recurrence
# ═══════════════════════════════════════════════════════════════════════════════

export WeylCharacter
export freudenthal_formula
export tensor_product, dual
export adams_operator, symmetric_power, exterior_power
export Sym, ⋀
export is_effective, is_irreducible, highest_weight
export character_from_weights
export add!, addmul!

# ═══════════════════════════════════════════════════════════════════════════════
#  WeylCharacter{DT,R} — element of the representation ring
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WeylCharacter{DT,R}

An element of the representation ring (Grothendieck ring) of a semisimple
Lie algebra of Dynkin type `DT` with rank `R`.

Stored as a `Dict{WeightLatticeElem{DT,R}, Int}` mapping dominant highest
weights to their integer multiplicities.  Positive multiplicities correspond
to actual representations; negative multiplicities arise in virtual differences.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> V = WeylCharacter(ω₁)
A2(1, 0)

julia> V + V == 2 * V
true
```
"""
struct WeylCharacter{DT<:DynkinType,R}
  # Mapping from dominant weight → multiplicity.
  # Zero-multiplicity entries are pruned.
  terms::Dict{WeightLatticeElem{DT,R},Int}
end

# ─── Constructors ────────────────────────────────────────────────────────────

"""
    WeylCharacter(λ::WeightLatticeElem{DT,R}, m::Integer=1)

Irreducible character ``m \\cdot \\mathrm{V}(λ)``. Requires `λ` to be dominant.
"""
function WeylCharacter(λ::WeightLatticeElem{DT,R}, m::Integer=1) where {DT,R}
  @assert is_dominant(λ) "Weight must be dominant"
  d = Dict{WeightLatticeElem{DT,R},Int}()
  iszero(m) || (d[λ] = Int(m))
  return WeylCharacter{DT,R}(d)
end

"""
    WeylCharacter(::Type{DT}) -> WeylCharacter{DT,R}

The zero virtual character (additive identity).
"""
function WeylCharacter(::Type{DT}) where {DT<:DynkinType}
  R = rank(DT)
  return WeylCharacter{DT,R}(Dict{WeightLatticeElem{DT,R},Int}())
end

# ─── Basic queries ───────────────────────────────────────────────────────────

Base.iszero(V::WeylCharacter) = isempty(V.terms)

Base.isone(V::WeylCharacter{DT,R}) where {DT,R} =
  length(V.terms) == 1 && haskey(V.terms, zero(WeightLatticeElem{DT,R})) &&
  V.terms[zero(WeightLatticeElem{DT,R})] == 1

"""
    is_effective(V::WeylCharacter) -> Bool

True when all multiplicities are non-negative, i.e. `V` corresponds to
an actual (not merely virtual) representation.
"""
is_effective(V::WeylCharacter) = all(>=(0), values(V.terms))

"""
    is_irreducible(V::WeylCharacter) -> Bool

True when `V` is a single irreducible with multiplicity 1.
"""
is_irreducible(V::WeylCharacter) =
  length(V.terms) == 1 && first(values(V.terms)) == 1

"""
    highest_weight(V::WeylCharacter{DT,R}) -> WeightLatticeElem{DT,R}

Return the highest weight of an irreducible virtual character.
Throws if `V` is not irreducible.
"""
function highest_weight(V::WeylCharacter{DT,R}) where {DT,R}
  is_irreducible(V) || error("Character is not irreducible")
  return first(keys(V.terms))
end

# ─── Iteration & collection ─────────────────────────────────────────────────

Base.iterate(V::WeylCharacter, args...) = iterate(V.terms, args...)
Base.length(V::WeylCharacter) = length(V.terms)
Base.pairs(V::WeylCharacter) = pairs(V.terms)
Base.keys(V::WeylCharacter) = keys(V.terms)
Base.values(V::WeylCharacter) = values(V.terms)

# ─── Display ─────────────────────────────────────────────────────────────────

function Base.show(io::IO, V::WeylCharacter{DT,R}) where {DT,R}
  if isempty(V.terms)
    print(io, "0")
    return nothing
  end
  type_str = _type_name(DT)
  first_term = true
  # Sort by weight for deterministic display
  for (w, m) in sort!(collect(V.terms); by=p -> p.first.vec, rev=true)
    wstr = "(" * join(w.vec, ", ") * ")"
    if first_term
      if m == 1
        print(io, type_str, wstr)
      elseif m == -1
        print(io, "-", type_str, wstr)
      else
        print(io, m, "*", type_str, wstr)
      end
      first_term = false
    else
      if m == 1
        print(io, " + ", type_str, wstr)
      elseif m == -1
        print(io, " - ", type_str, wstr)
      elseif m > 0
        print(io, " + ", m, "*", type_str, wstr)
      else  # m < 0, m ≠ -1
        print(io, " - ", -m, "*", type_str, wstr)
      end
    end
  end
end

# ─── Arithmetic ──────────────────────────────────────────────────────────────

function Base.:+(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}) where {DT,R}
  result = copy(V.terms)
  for (λ, m) in W.terms
    new_m = get(result, λ, 0) + m
    if iszero(new_m)
      delete!(result, λ)
    else
      result[λ] = new_m
    end
  end
  return WeylCharacter{DT,R}(result)
end

function Base.:-(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}) where {DT,R}
  result = copy(V.terms)
  for (λ, m) in W.terms
    new_m = get(result, λ, 0) - m
    if iszero(new_m)
      delete!(result, λ)
    else
      result[λ] = new_m
    end
  end
  return WeylCharacter{DT,R}(result)
end

Base.:-(V::WeylCharacter{DT,R}) where {DT,R} =
  WeylCharacter{DT,R}(Dict(λ => -m for (λ, m) in V.terms))

function Base.:*(a::Integer, V::WeylCharacter{DT,R}) where {DT,R}
  iszero(a) && return WeylCharacter(DT)
  return WeylCharacter{DT,R}(Dict(λ => a * m for (λ, m) in V.terms))
end

Base.:*(V::WeylCharacter, a::Integer) = a * V

Base.:(==)(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}) where {DT,R} =
  V.terms == W.terms

Base.hash(V::WeylCharacter, h::UInt) = hash(V.terms, h)

"""
    add!(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}) -> WeylCharacter{DT,R}

Add `W` into `V` in-place, modifying `V`. Returns `V`.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> V = WeylCharacter(ω₁); W = WeylCharacter(ω₁);

julia> add!(V, W) == 2 * WeylCharacter(ω₁)
true
```
"""
function add!(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}) where {DT,R}
  for (λ, m) in W.terms
    new_m = get(V.terms, λ, 0) + m
    if iszero(new_m)
      delete!(V.terms, λ)
    else
      V.terms[λ] = new_m
    end
  end
  return V
end

"""
    addmul!(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}, c::Integer) -> WeylCharacter{DT,R}

Add `c * W` into `V` in-place, modifying `V`. Returns `V`.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> V = WeylCharacter(TypeA{2}); W = WeylCharacter(ω₁);

julia> addmul!(V, W, 3) == 3 * WeylCharacter(ω₁)
true
```
"""
function addmul!(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}, c::Integer) where {DT,R}
  iszero(c) && return V
  for (λ, m) in W.terms
    new_m = get(V.terms, λ, 0) + c * m
    if iszero(new_m)
      delete!(V.terms, λ)
    else
      V.terms[λ] = new_m
    end
  end
  return V
end

# ─── Tensor product (multiplication in the representation ring) ──────────────

"""
    *(V::WeylCharacter, W::WeylCharacter) -> WeylCharacter

Tensor product of virtual characters, computed via Brauer–Klimyk.
Equivalent to the ring multiplication in the representation ring.
"""
function Base.:*(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}) where {DT,R}
  return tensor_product(V, W)
end

"""
    tensor_product(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}) -> WeylCharacter{DT,R}

Tensor product decomposition of two virtual characters.
Iterates over all pairs `(λ, m)` in `V` and `(μ, n)` in `W`, computes
`tensor_product(λ, μ)` via Brauer–Klimyk, and accumulates `m * n * result`.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> V = WeylCharacter(ω₁);

julia> V * V == Sym(2, ω₁) + ⋀(2, ω₁)
true
```
"""
function tensor_product(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}) where {DT,R}
  result = WeylCharacter(DT)
  for (λ, m) in V.terms
    for (μ, n) in W.terms
      t = tensor_product(λ, μ)
      addmul!(result, t, m * n)
    end
  end
  return result
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Freudenthal's recursion formula — weight multiplicities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    freudenthal_formula(λ::WeightLatticeElem{DT,R}) -> Dict{SVector{R,Int}, Int}

Compute the weight multiplicities of the irreducible representation ``\\mathrm{V}(λ)``
using Freudenthal's recursion formula.

Returns a dictionary mapping weights (in fundamental weight coordinates) to
their multiplicities. Only weights with non-zero multiplicity are included.

The recursion is:

``(⟨λ+ρ, λ+ρ⟩ - ⟨μ+ρ, μ+ρ⟩) \\, m(μ) = 2 \\sum_{α>0} \\sum_{k≥1} ⟨μ+kα, α⟩ \\, m(μ+kα)``

where the inner product `(·,·)` is the Weyl-group-invariant bilinear form
`B = diag(d) C` on the root space, extended to the weight lattice.

# Examples
```jldoctest
julia> using Lie; using StaticArrays

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> mults = freudenthal_formula(ω₁);

julia> mults[SVector(1, 0)]
1

julia> sum(values(mults))  # = dim V(ω₁) = 3
3
```
"""
function freudenthal_formula(λ::WeightLatticeElem{DT,R}) where {DT,R}
  @assert is_dominant(λ) "Weight must be dominant"

  RS = RootSystem(DT)
  C = cartan_matrix(DT)
  B = cartan_bilinear_form(DT)   # diag(d) * C — root-space bilinear form

  # Positive roots as weight-lattice vectors: α_ω[j] = Σᵢ C[j,i] α_root[i]
  # (i.e. α_ω = C α_root). This follows from αᵢ = Σⱼ C[j,i] ωⱼ (column i of C).
  n_pos = n_positive_roots(RS)
  α_w = Vector{SVector{R,Int}}(undef, n_pos)
  for k in 1:n_pos
    α_root = RS.positive_roots_list[k]
    α_w[k] = SVector{R,Int}(ntuple(j -> sum(C[j, i] * α_root[i] for i in 1:R), R))
  end

  # Simple roots in ω-coords (column s of C: αₛ = Σⱼ C[j,s] ωⱼ)
  simple_α_w = [SVector{R,Int}(ntuple(j -> C[j, s], R)) for s in 1:R]

  # ─── Inner products ─────────────────────────────────────────────────
  # All inner products use (u, v) = u_α^T B v_α  where _α denotes
  # coordinates in the simple-root basis and B = diag(d) C.
  #
  # Conversion: a weight μ with ω-coordinates μ_ω has α-coordinates
  #   μ_α = C⁻¹ μ_ω  (since ωⱼ = Σᵢ (C⁻¹)[i,j] αᵢ).
  #
  # Mixed-coordinate inner product (μ in ω-coords, α in root-coords):
  #   (μ, α) = (C⁻¹ μ_ω)^T B α = μ_ω^T C⁻ᵀ B α = μ_ω^T M α
  # where M = C⁻ᵀ B.
  #
  # Weight-space inner product (both in ω-coords):
  #   (μ, ν) = μ_ω^T B_ω ν_ω  where B_ω = C⁻ᵀ B C⁻¹.

  Cinv = cartan_matrix_inverse(DT)
  Cinv_r = SMatrix{R,R,Rational{Int}}(Cinv)
  B_r = SMatrix{R,R,Rational{Int}}(B)
  M = transpose(Cinv_r) * B_r             # C⁻ᵀ B
  B_omega = M * Cinv_r                           # C⁻ᵀ B C⁻¹

  # Precompute (α, α) for each positive root
  α_dot_α = Vector{Rational{Int}}(undef, n_pos)
  for k in 1:n_pos
    v = SVector{R,Rational{Int}}(RS.positive_roots_list[k])
    α_dot_α[k] = transpose(v) * B_r * v
  end

  # Precompute M * α_root for each positive root → gives the vector
  # such that (μ_ω, α) = μ_ω ⋅ Mα.
  Mα = Vector{SVector{R,Rational{Int}}}(undef, n_pos)
  for k in 1:n_pos
    Mα[k] = M * SVector{R,Rational{Int}}(RS.positive_roots_list[k])
  end

  λρ = λ + weyl_vector(DT)
  λρ_vec = SVector{R,Rational{Int}}(λρ.vec)
  first_term = transpose(λρ_vec) * B_omega * λρ_vec

  # Multiplicities: weight (ω-coords) → multiplicity
  multiplicities = Dict{SVector{R,Int},Int}()

  # BFS layer-by-layer: start from λ, subtract simple roots to find lower weights
  current = Dict{SVector{R,Int},Int}(λ.vec => 1)

  while !isempty(current)
    next = Dict{SVector{R,Int},Int}()

    for (μ_vec, m) in current
      if m != 0
        multiplicities[μ_vec] = m
        for α_s in simple_α_w
          next_vec = μ_vec - α_s
          haskey(next, next_vec) || (next[next_vec] = 0)
        end
      end
    end

    for μ_vec in keys(next)
      Σ = Rational{Int}(0)

      for k in 1:n_pos
        ν_vec = μ_vec + α_w[k]

        # (μ, α) = μ_ωᵀ Mα[k]
        μ_dot_α = sum(Rational{Int}(μ_vec[i]) * Mα[k][i] for i in 1:R)

        j = 1
        while true
          m_ν = get(multiplicities, ν_vec, 0)
          m_ν == 0 && break

          # (μ + jα, α) = (μ, α) + j (α, α)
          ip = μ_dot_α + j * α_dot_α[k]
          Σ += m_ν * ip

          ν_vec = ν_vec + α_w[k]
          j += 1
        end
      end

      if iszero(Σ)
        next[μ_vec] = 0
      else
        μρ_vec = SVector{R,Rational{Int}}(μ_vec + weyl_vector(DT).vec)
        second_term = transpose(μρ_vec) * B_omega * μρ_vec

        denom = first_term - second_term
        @assert denom != 0 "Denominator in Freudenthal's formula is zero"

        mult_rat = (2 * Σ) / denom
        @assert isinteger(mult_rat) "Freudenthal formula gave non-integer multiplicity: $mult_rat for μ=$μ_vec"
        next[μ_vec] = Int(mult_rat)
      end
    end

    current = next
  end

  return multiplicities
end

# ═══════════════════════════════════════════════════════════════════════════════
#  dot-reduce — Weyl group orbit reduction with sign
# ═══════════════════════════════════════════════════════════════════════════════

"""
    dot_reduce(λ::WeightLatticeElem{DT,R}) -> Tuple{Int, WeightLatticeElem{DT,R}}

Compute the "dot-action reduction" of `λ`:

Return `(ε, μ)` where:
- `μ` is the dominant weight such that `w ⋅ λ = μ` under the dot action
  `w ⋅ λ = w(λ + ρ) - ρ` for some Weyl group element `w`
- `ε = (-1)^{ℓ(w)}` is the sign of `w`, or `ε = 0` if `λ + ρ` is singular
  (lies on a Weyl chamber wall)

This is the key ingredient in the Brauer–Klimyk algorithm.
"""
function dot_reduce(λ::WeightLatticeElem{DT,R}) where {DT,R}
  C = cartan_matrix(DT)
  ε = 1
  v = MVector{R,Int}(λ.vec)

  # Iteratively reflect until dominant.
  # The "dot action" s_i · λ = s_i(λ + ρ) − ρ
  # In ω-coordinates: (λ + ρ)_i = λ_i + 1. The weight λ + ρ is regular
  # dominant iff all (λ_i + 1) > 0, i.e. λ_i ≥ 0 iff λ_i + 1 ≥ 1.
  # The dot-action reflection by s_i acts on (λ+ρ) as ordinary reflection,
  # then subtracts ρ. So the coefficient of ωᵢ in the dot action is:
  # new_λ_i = -1 - λ_i  if reflecting only node i with pairing λ_i + 1.
  #
  # More precisely: s_i · λ = s_i(λ + ρ) - ρ.
  # s_i(λ + ρ)_j = (λ + ρ)_j - (λ + ρ)_i * C[j,i]
  # So (s_i · λ)_j = (λ + ρ)_j - (λ + ρ)_i * C[j,i] - 1
  #                = λ_j + 1 - (λ_i + 1) * C[j,i] - 1
  #                = λ_j - (λ_i + 1) * C[j,i]
  #
  # In particular, (s_i · λ)_i = λ_i - (λ_i + 1) * 2 = -λ_i - 2

  while true
    done = true
    for s in 1:R
      c = v[s]  # = λ_s coordinate

      # (λ + ρ)_s = c + 1
      # If c + 1 == 0, i.e. c == -1, then λ+ρ is on the wall → singular
      c == -1 && return (0, WeightLatticeElem{DT,R}(zero(SVector{R,Int})))

      # If c + 1 < 0 (i.e. c ≤ -2), reflect
      if c <= -2
        ε = -ε
        pairing = c + 1  # = (λ+ρ)_s
        for j in 1:R
          v[j] -= pairing * C[j, s]
        end
        # After reflection: v[s] = c - pairing * 2 = c - (c+1)*2 = -c - 2
        done = false
        break
      end
    end
    done && break
  end

  return (ε, WeightLatticeElem{DT,R}(SVector{R,Int}(v)))
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Brauer–Klimyk algorithm — tensor product with an irreducible
# ═══════════════════════════════════════════════════════════════════════════════

"""
    brauer_klimyk(char::Dict{SVector{R,Int}, Int}, μ::WeightLatticeElem{DT,R}) -> WeylCharacter{DT,R}

Tensor the representation with weight multiplicities `char` (as from
[`freudenthal_formula`](@ref)) with the irreducible representation ``\\mathrm{V}(μ)``,
using the Brauer–Klimyk formula:

``\\mathrm{V} \\otimes \\mathrm{V}(μ) = \\sum_{\\text{weights } λ \\text{ of } \\mathrm{V}} m(λ) \\cdot ε(λ+μ) \\cdot \\mathrm{V}(ν(λ+μ))``

where `(ε, ν) = dot_reduce(μ + λ)`.
"""
function brauer_klimyk(
  char::Dict{SVector{R,Int},Int}, μ::WeightLatticeElem{DT,R}
) where {DT,R}
  @assert is_dominant(μ) "Weight μ must be dominant"

  result = Dict{WeightLatticeElem{DT,R},Int}()

  for (λ_vec, m) in char
    λ_wt = WeightLatticeElem{DT,R}(λ_vec)
    (ε, ν) = dot_reduce(μ + λ_wt)

    if ε == 1
      result[ν] = get(result, ν, 0) + m
    elseif ε == -1
      result[ν] = get(result, ν, 0) - m
    end
    # ε == 0: singular, skip
  end

  # Prune zeros
  filter!(p -> !iszero(p.second), result)
  return WeylCharacter{DT,R}(result)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Tensor product — irreducible ⊗ irreducible
# ═══════════════════════════════════════════════════════════════════════════════

# Cache for tensor products of irreducibles.
# Key: (DT, λ, μ), Value: WeylCharacter.
const _tensor_cache = Dict{Tuple{Type,Any,Any},Any}()

"""
    tensor_product(λ::WeightLatticeElem{DT,R}, μ::WeightLatticeElem{DT,R}) -> WeylCharacter{DT,R}

Decompose the tensor product ``\\mathrm{V}(λ) \\otimes \\mathrm{V}(μ)`` into irreducibles using the
Brauer–Klimyk algorithm. The Freudenthal formula is applied to whichever
factor has smaller dimension, for efficiency.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1); ω₂ = fundamental_weight(TypeA{2}, 2);

julia> tensor_product(ω₁, ω₁)
A2(2, 0) + A2(0, 1)
```
"""
function tensor_product(λ::WeightLatticeElem{DT,R}, μ::WeightLatticeElem{DT,R}) where {DT,R}
  @assert is_dominant(λ) "First weight must be dominant"
  @assert is_dominant(μ) "Second weight must be dominant"

  # Canonical ordering for cache: smaller dimension decomposes
  key = (DT, λ, μ)
  haskey(_tensor_cache, key) && return _tensor_cache[key]::WeylCharacter{DT,R}

  # Try reversed key too
  key_rev = (DT, μ, λ)
  haskey(_tensor_cache, key_rev) && return _tensor_cache[key_rev]::WeylCharacter{DT,R}

  # Brauer–Klimyk: decompose the smaller rep via Freudenthal
  if degree(λ) > degree(μ)
    result = brauer_klimyk(freudenthal_formula(μ), λ)
  else
    result = brauer_klimyk(freudenthal_formula(λ), μ)
  end

  _tensor_cache[key] = result
  return result
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Dual representation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    dual(λ::WeightLatticeElem{DT,R}) -> WeightLatticeElem{DT,R}

Highest weight of the contragredient (dual) representation: `λ* = -w₀(λ)`.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1); ω₂ = fundamental_weight(TypeA{2}, 2);

julia> dual(ω₁) == ω₂
true
```
"""
function dual(λ::WeightLatticeElem{DT,R}) where {DT,R}
  W = weyl_group(DT)
  w₀ = longest_element(W)
  return -(λ * w₀)
end

"""
    dual(V::WeylCharacter{DT,R}) -> WeylCharacter{DT,R}

Dual of a virtual character: each summand ``\\mathrm{V}(λ)`` maps to ``\\mathrm{V}(λ^*)``.
"""
function dual(V::WeylCharacter{DT,R}) where {DT,R}
  result = Dict{WeightLatticeElem{DT,R},Int}()
  for (λ, m) in V.terms
    λ_dual = dual(λ)
    result[λ_dual] = get(result, λ_dual, 0) + m
  end
  filter!(p -> !iszero(p.second), result)
  return WeylCharacter{DT,R}(result)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Adams operators
# ═══════════════════════════════════════════════════════════════════════════════

"""
    adams_operator(λ::WeightLatticeElem{DT,R}, k::Int) -> Dict{SVector{R,Int}, Int}

Compute the `k`-th Adams operator ``ψ^k(\\mathrm{V}(λ))``, returned as a dictionary of
weight multiplicities (not decomposed into irreducibles).

The Adams operator scales every weight by `k`: if ``\\mathrm{V}(λ)`` has weight
multiplicity ``m(μ)``, then ``ψ^k(\\mathrm{V}(λ))`` has ``m(μ)`` at weight ``kμ``.
"""
function adams_operator(λ::WeightLatticeElem{DT,R}, k::Int) where {DT,R}
  @assert k != 0 "Adams operator index must be non-zero"

  mults = freudenthal_formula(λ)
  return Dict{SVector{R,Int},Int}(k * μ => m for (μ, m) in mults)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Symmetric and exterior powers — Newton–Girard recurrence
# ═══════════════════════════════════════════════════════════════════════════════

# Caches: keyed by (DT, weight, power)
const _symmetric_power_cache = Dict{Tuple{Type,Any,Int},Any}()
const _exterior_power_cache = Dict{Tuple{Type,Any,Int},Any}()

"""
    symmetric_power(λ::WeightLatticeElem{DT,R}, k::Int) -> WeylCharacter{DT,R}

Compute the `k`-th symmetric power ``\\mathrm{Sym}^k \\mathrm{V}(λ)`` of the irreducible
representation with highest weight `λ`, using the Newton–Girard formula:

``k \\cdot \\mathrm{Sym}^k(\\mathrm{V}) = \\sum_{r=1}^{k} ψ^r(\\mathrm{V}) \\cdot \\mathrm{Sym}^{k-r}(\\mathrm{V})``

Results are memoized for efficiency in recursive calls.
"""
function symmetric_power(λ::WeightLatticeElem{DT,R}, k::Int) where {DT,R}
  @assert is_dominant(λ) "Weight must be dominant"
  k < 0 && return WeylCharacter(DT)
  k == 0 && return WeylCharacter(WeightLatticeElem{DT,R}(zero(SVector{R,Int})))
  k == 1 && return WeylCharacter(λ)

  cache_key = (DT, λ, k)
  haskey(_symmetric_power_cache, cache_key) &&
    return _symmetric_power_cache[cache_key]::WeylCharacter{DT,R}

  result = WeylCharacter(DT)

  for r in 1:k
    adams = adams_operator(λ, r)
    prev = symmetric_power(λ, k - r)

    # result += ψʳ(V) ⊗ Symᵏ⁻ʳ(V)
    # Brauer–Klimyk each term in prev against adams
    for (μ, m) in prev.terms
      bk = brauer_klimyk(adams, μ)
      addmul!(result, bk, m)
    end
  end

  # Divide by k (Newton–Girard normalization)
  for λv in keys(result.terms)
    q, r = divrem(result.terms[λv], k)
    @assert iszero(r) "Newton–Girard: non-integer coefficient after division by k=$k"
    result.terms[λv] = q
  end

  _symmetric_power_cache[cache_key] = result
  return result
end

"""
    exterior_power(λ::WeightLatticeElem{DT,R}, k::Int) -> WeylCharacter{DT,R}

Compute the `k`-th exterior power ``\\bigwedge^k \\mathrm{V}(λ)`` of the irreducible
representation with highest weight `λ`, using the Newton–Girard formula:

``k \\cdot \\bigwedge\\nolimits^k(\\mathrm{V}) = \\sum_{r=1}^{k} (-1)^{r-1} ψ^r(\\mathrm{V}) \\cdot \\bigwedge\\nolimits^{k-r}(\\mathrm{V})``

Results are memoized for efficiency in recursive calls.
"""
function exterior_power(λ::WeightLatticeElem{DT,R}, k::Int) where {DT,R}
  @assert is_dominant(λ) "Weight must be dominant"
  k < 0 && return WeylCharacter(DT)
  k == 0 && return WeylCharacter(WeightLatticeElem{DT,R}(zero(SVector{R,Int})))
  k == 1 && return WeylCharacter(λ)
  k > degree(λ) && return WeylCharacter(DT)

  cache_key = (DT, λ, k)
  haskey(_exterior_power_cache, cache_key) &&
    return _exterior_power_cache[cache_key]::WeylCharacter{DT,R}

  result = WeylCharacter(DT)

  for r in 1:k
    adams = adams_operator(λ, r)
    prev = exterior_power(λ, k - r)

    # result += (-1)^{r-1} * ψʳ(V) ⊗ ⋀^{k-r}(V)
    sign = iseven(r) ? -1 : 1

    for (μ, m) in prev.terms
      bk = brauer_klimyk(adams, μ)
      addmul!(result, bk, sign * m)
    end
  end

  # Divide by k (Newton–Girard normalization)
  for λv in keys(result.terms)
    q, r = divrem(result.terms[λv], k)
    @assert iszero(r) "Newton–Girard: non-integer coefficient after division by k=$k"
    result.terms[λv] = q
  end

  _exterior_power_cache[cache_key] = result
  return result
end

"""
    Sym(k::Int, λ::WeightLatticeElem) -> WeylCharacter

Shorthand for `symmetric_power(λ, k)`.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> Sym(2, ω₁)
A2(2, 0)

julia> degree(highest_weight(Sym(3, ω₁)))
10
```
"""
Sym(k::Int, λ::WeightLatticeElem) = symmetric_power(λ, k)

"""
    ⋀(k::Int, λ::WeightLatticeElem) -> WeylCharacter

Shorthand for `exterior_power(λ, k)`.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{3}, 1);

julia> ⋀(2, ω₁) == WeylCharacter(fundamental_weight(TypeA{3}, 2))
true
```
"""
⋀(k::Int, λ::WeightLatticeElem) = exterior_power(λ, k)

# ═══════════════════════════════════════════════════════════════════════════════
#  character_from_weights — reconstruct irreducible decomposition
# ═══════════════════════════════════════════════════════════════════════════════

"""
    character_from_weights(::Type{DT}, multiplicities::Dict{SVector{R,Int}, Int}) -> WeylCharacter{DT,R}

Given a dictionary of weight multiplicities (as from Freudenthal), decompose
the representation into a formal sum of irreducibles.

Uses the "peeling" algorithm: find the highest dominant weight, subtract
the Freudenthal multiplicities of that irreducible, repeat.
"""
function character_from_weights(
  ::Type{DT}, multiplicities::Dict{SVector{R,Int},Int}
) where {DT<:DynkinType,R}
  d = cartan_symmetrizer(DT)

  weights = Dict{WeightLatticeElem{DT,R},Int}()
  mults = copy(multiplicities)

  while !isempty(mults)
    # Find the highest weight: maximize ⟨λ, ρ⟩ = ∑ dᵢ λᵢ
    best = argmax(λ -> sum(d[i] * λ[i] for i in 1:R), keys(mults))

    @assert all(>=(0), best) "Multiplicity dictionary is not Weyl group invariant"

    coeff = mults[best]
    best_wt = WeightLatticeElem{DT,R}(best)
    weights[best_wt] = get(weights, best_wt, 0) + coeff

    # Subtract coeff copies of the Freudenthal multiplicities for V(best)
    sub_mults = freudenthal_formula(best_wt)
    for (μ, m) in sub_mults
      mults[μ] = get(mults, μ, 0) - coeff * m
      iszero(mults[μ]) && delete!(mults, μ)
    end
  end

  filter!(p -> !iszero(p.second), weights)
  return WeylCharacter{DT,R}(weights)
end
