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
export freudenthal_formula, weight_multiplicity
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
    WeylCharacter(::Type{DT}, v) -> WeylCharacter{DT,R}

Irreducible character ``\\mathrm{V}(λ)`` where `λ` is the dominant weight
with the given coordinates in the fundamental weight basis.

# Examples
```jldoctest
julia> using Lie

julia> WeylCharacter(TypeA{2}, [1, 0])
A2(1, 0)
```
"""
function WeylCharacter(::Type{DT}, v::AbstractVector{<:Integer}) where {DT<:DynkinType}
  return WeylCharacter(WeightLatticeElem(DT, v))
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

"""
    ^(V::WeylCharacter, n::Integer) -> WeylCharacter

Compute the `n`-th tensor power of `V`.

Uses right-to-left sequential multiplication `V * (V * (V * V))` rather
than repeated squaring, because the Brauer–Klimyk algorithm is faster
when one factor is small (the original irreducible) and the other grows.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> V = WeylCharacter(ω₁);

julia> V^2 == tensor_product(ω₁, ω₁)
true

julia> V^0 == WeylCharacter(zero(ω₁))
true
```
"""
function Base.:^(V::WeylCharacter{DT,R}, n::Integer) where {DT,R}
  n < 0 && throw(ArgumentError("WeylCharacter power requires n ≥ 0"))
  n == 0 && return WeylCharacter(WeightLatticeElem{DT,R}(zero(SVector{R,Int})))
  n == 1 && return V
  # Right-to-left sequential multiplication: V * (V * (V * ⋯))
  # This is more efficient for Brauer–Klimyk than repeated squaring,
  # because each intermediate product is tensored with the small factor V.
  result = V
  for _ in 2:n
    result = V * result
  end
  return result
end

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

julia> V^3  # right-to-left tensor power
A2(3, 0) + 2*A2(1, 1) + A2(0, 0)
```
"""
function tensor_product(V::WeylCharacter{DT,R}, W::WeylCharacter{DT,R}) where {DT,R}
  # Swap so the character with fewer terms is the outer loop (decomposed
  # via Freudenthal). This mirrors SageMath's product_on_basis optimization.
  if length(V.terms) > length(W.terms)
    V, W = W, V
  end
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

julia> ω₂ = fundamental_weight(TypeA{2}, 2);

julia> mults = freudenthal_formula(ω₁ + ω₂);  # adjoint of A₂

julia> length(mults)  # 7 distinct weights
7

julia> sum(values(mults))  # dim = 8
8
```
"""
function freudenthal_formula(λ::WeightLatticeElem{DT,R}) where {DT,R}
  @assert is_dominant(λ) "Weight must be dominant"

  # Check cache first
  cache_key = (DT, λ)
  haskey(_freudenthal_cache, cache_key) &&
    return _freudenthal_cache[cache_key]::Dict{SVector{R,Int},Int}

  RS = RootSystem(DT)
  C = cartan_matrix(DT)
  B = cartan_bilinear_form(DT)
  d = cartan_symmetrizer(DT)

  # Positive roots as weight-lattice vectors: α_ω = C α_root
  n_pos = n_positive_roots(RS)
  α_w = Vector{SVector{R,Int}}(undef, n_pos)
  for k in 1:n_pos
    α_root = RS.positive_roots_list[k]
    α_w[k] = SVector{R,Int}(ntuple(j -> sum(C[j, i] * α_root[i] for i in 1:R), R))
  end

  # Simple roots in ω-coords
  simple_α_w = SVector{R,SVector{R,Int}}(
    ntuple(s -> SVector{R,Int}(ntuple(j -> C[j, s], R)), R)
  )

  # ─── Integer inner products ──────────────────────────────────────────
  S, B_omega_S = omega_bilinear_form_scaled(DT)

  α_dot_α = Vector{Int}(undef, n_pos)
  for k in 1:n_pos
    v = RS.positive_roots_list[k]
    s = 0
    for j in 1:R, i in 1:R
      s += v[i] * B[i, j] * v[j]
    end
    α_dot_α[k] = s
  end

  dα = Vector{SVector{R,Int}}(undef, n_pos)
  for k in 1:n_pos
    v = RS.positive_roots_list[k]
    dα[k] = SVector{R,Int}(ntuple(i -> d[i] * v[i], R))
  end

  ρ_vec = weyl_vector(DT).vec
  λρ_vec = λ.vec + ρ_vec

  first_term_S = 0
  for j in 1:R, i in 1:R
    first_term_S += λρ_vec[i] * B_omega_S[i, j] * λρ_vec[j]
  end

  # Multiplicities: weight (ω-coords) → multiplicity
  multiplicities = Dict{SVector{R,Int},Int}()
  sizehint!(multiplicities, max(256, Int(min(degree(λ), big(1_000_000)))))

  # BFS layer-by-layer: start from λ, subtract simple roots
  current_layer = SVector{R,Int}[λ.vec]
  multiplicities[λ.vec] = 1

  while !isempty(current_layer)
    next_vecs = SVector{R,Int}[]

    for μ_vec in current_layer
      for s in 1:R
        next_vec = μ_vec - simple_α_w[s]
        if !haskey(multiplicities, next_vec)
          multiplicities[next_vec] = 0  # placeholder
          push!(next_vecs, next_vec)
        end
      end
    end

    current_layer = empty!(current_layer)

    for μ_vec in next_vecs
      Σ = 0

      for k in 1:n_pos
        ν_vec = μ_vec + α_w[k]

        μ_dot_α = 0
        for i in 1:R
          μ_dot_α += μ_vec[i] * dα[k][i]
        end

        j = 1
        while true
          m_ν = get(multiplicities, ν_vec, 0)
          m_ν == 0 && break

          ip = μ_dot_α + j * α_dot_α[k]
          Σ += m_ν * ip

          ν_vec = ν_vec + α_w[k]
          j += 1
        end
      end

      if !iszero(Σ)
        μρ_vec = μ_vec + ρ_vec
        second_term_S = 0
        for j in 1:R, i in 1:R
          second_term_S += μρ_vec[i] * B_omega_S[i, j] * μρ_vec[j]
        end

        denom_S = first_term_S - second_term_S
        @assert denom_S != 0 "Denominator in Freudenthal's formula is zero"

        numerator = 2 * S * Σ
        mult, rem = divrem(numerator, denom_S)
        @assert iszero(rem) "Freudenthal formula gave non-integer multiplicity for μ=$μ_vec"
        multiplicities[μ_vec] = mult
        push!(current_layer, μ_vec)
      end
    end
  end

  # Remove zero-multiplicity placeholders
  filter!(p -> p.second != 0, multiplicities)

  _freudenthal_cache[cache_key] = multiplicities
  return multiplicities
end

"""
    weight_multiplicity(λ::WeightLatticeElem{DT,R}, μ::WeightLatticeElem{DT,R}) -> Int

Return the multiplicity of weight `μ` in the irreducible representation
``\\mathrm{V}(λ)``.  This is a convenience wrapper around
[`freudenthal_formula`](@ref).

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1); ω₂ = fundamental_weight(TypeA{2}, 2);

julia> weight_multiplicity(ω₁ + ω₂, zero(ω₁))   # zero weight of adjoint
2
```
"""
function weight_multiplicity(
  λ::WeightLatticeElem{DT,R}, μ::WeightLatticeElem{DT,R}
) where {DT,R}
  mults = freudenthal_formula(λ)
  return get(mults, μ.vec, 0)
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
#  Littlewood–Richardson rule — tensor products in Type A
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _weight_to_partition(λ::WeightLatticeElem{TypeA{N},N}) -> Vector{Int}

Convert a dominant weight in fundamental weight coordinates to a partition
with ``N+1`` parts (for ``\\mathrm{GL}_{N+1}``).

For ``\\mathrm{A}_N``, the dominant weight ``λ = (λ_1, …, λ_N)`` in the
fundamental weight basis corresponds to the partition
``μ = (μ_1 ≥ μ_2 ≥ ⋯ ≥ μ_N ≥ 0)`` where ``μ_i = λ_i + λ_{i+1} + ⋯ + λ_N``
(partial sums from right to left), with ``μ_{N+1} = 0``.
"""
function _weight_to_partition(λ::WeightLatticeElem{TypeA{N},N}) where {N}
  p = Vector{Int}(undef, N + 1)
  p[N + 1] = 0
  s = 0
  for i in N:-1:1
    s += λ.vec[i]
    p[i] = s
  end
  return p
end

"""
    _partition_to_weight(::Type{TypeA{N}}, p::Vector{Int}) -> WeightLatticeElem{TypeA{N},N}

Convert a partition back to a dominant weight in the fundamental weight basis
for ``\\mathrm{SL}_{N+1}``. First reduces the partition by subtracting the
minimum part (to pass from ``\\mathrm{GL}`` to ``\\mathrm{SL}``), then computes
successive differences: ``λ_i = μ_i - μ_{i+1}``.
"""
function _partition_to_weight(::Type{TypeA{N}}, p::Vector{Int}) where {N}
  # Reduce: subtract the minimum part (SL quotient)
  m = length(p) >= N + 1 ? p[N + 1] : 0
  v = SVector{N,Int}(
    ntuple(N) do i
      pi = i <= length(p) ? p[i] - m : 0
      pi_next = i + 1 <= length(p) ? p[i + 1] - m : 0
      pi - pi_next
    end,
  )
  return WeightLatticeElem{TypeA{N},N}(v)
end

"""
    _lr_coefficients(α::Vector{Int}, β::Vector{Int}, n::Int) -> Dict{Vector{Int}, Int}

Compute all Littlewood–Richardson coefficients ``c^ν_{αβ}`` for partitions
`α` and `β`, where partitions have at most `n` parts.

Returns a dictionary mapping each partition `ν` (as `Vector{Int}`) to the
LR coefficient ``c^ν_{αβ}``.

The algorithm fills the skew shape ``ν / α`` with content `β` row by row,
enforcing:
1. **Semistandard**: entries weakly increase along rows, strictly increase
   down columns.
2. **Ballot (lattice word) condition**: reading the filling right-to-left,
   top-to-bottom, at every prefix the count of `j` ≥ count of `j+1`.

We enumerate valid fillings recursively row by row, which implicitly
determines the partition `ν`.
"""
function _lr_coefficients(α::Vector{Int}, β::Vector{Int}, n::Int)
  α = copy(α)
  β = copy(β)

  # Trim trailing zeros from β
  while length(β) > 0 && β[end] == 0
    pop!(β)
  end

  total = sum(β; init=0)
  nβ = length(β)

  # Pad α to length n
  while length(α) < n
    push!(α, 0)
  end

  if total == 0 || nβ == 0
    return Dict{Vector{Int},Int}(copy(α) => 1)
  end

  result = Dict{Vector{Int},Int}()

  # We enumerate valid fillings of the skew shape ν/α row by row, choosing
  # labels column-by-column (left to right, weakly increasing). The partition ν
  # is determined implicitly by where each row stops.
  #
  # Ballot condition optimization: within a row, labels are weakly increasing
  # (left→right), so the reading word (right→left) is weakly decreasing. This
  # means all (j+1)-labels from a row are read before any j-labels. The ballot
  # condition can therefore be checked at row boundaries only:
  #   counts_prev[j] ≥ counts_prev[j+1] + row_count[j+1]  for each j
  # where counts_prev[j] is the total count of label j from previous rows,
  # and row_count[j+1] is the number of (j+1)-labels in the current row.

  function enumerate_rows!(
    result::Dict{Vector{Int},Int},
    row::Int,
    counts::Vector{Int},      # counts[j] = total uses of label j in rows 1..row-1
    prev_labels::Vector{Int}, # prev_labels[c] = label at column c in row-1 (0 if none)
    ν_so_far::Vector{Int},    # ν[1..row-1]
    total_remaining::Int,
  )
    if total_remaining == 0
      ν = copy(ν_so_far)
      for i in row:n
        push!(ν, α[i])
      end
      result[ν] = get(result, ν, 0) + 1
      return nothing
    end

    if row > n
      return nothing
    end

    # Upper bound on ν[row]: at most α[row] + remaining cells, and ≤ ν[row-1]
    max_ν_row = α[row] + total_remaining
    if row > 1
      max_ν_row = min(max_ν_row, ν_so_far[row - 1])
    end

    function fill_row!(
      col::Int,             # current column to fill (1-indexed)
      min_label::Int,       # weakly increasing: min label at this column
      row_counts::Vector{Int},  # row_counts[j] = #j's placed in this row so far
      remaining::Int,
    )
      # Option 1: stop the row here (ν_row = col - 1)
      ν_row = col - 1
      if ν_row >= α[row] && (row == 1 || ν_row <= ν_so_far[row - 1])
        # Ballot condition: counts[j] ≥ counts[j+1] + row_counts[j+1]
        ballot_ok = true
        for j in 1:(nβ - 1)
          if counts[j] < counts[j + 1] + row_counts[j + 1]
            ballot_ok = false
            break
          end
        end

        if ballot_ok
          push!(ν_so_far, ν_row)

          for j in 1:nβ
            counts[j] += row_counts[j]
          end
          old_len = length(prev_labels)
          while length(prev_labels) < ν_row
            push!(prev_labels, 0)
          end

          enumerate_rows!(result, row + 1, counts, prev_labels, ν_so_far, remaining)

          # Restore state
          for j in 1:nβ
            counts[j] -= row_counts[j]
          end
          while length(prev_labels) > old_len
            pop!(prev_labels)
          end
          pop!(ν_so_far)
        end
      end

      (remaining == 0 || col > max_ν_row) && return nothing

      # Column-strict: label must exceed the label above
      col_min = col <= length(prev_labels) ? prev_labels[col] + 1 : 1
      effective_min = max(min_label, col_min)

      for j in effective_min:nβ
        row_counts[j] + counts[j] >= β[j] && continue

        row_counts[j] += 1
        old_prev = col <= length(prev_labels) ? prev_labels[col] : 0
        if col <= length(prev_labels)
          prev_labels[col] = j
        else
          while length(prev_labels) < col
            push!(prev_labels, 0)
          end
          prev_labels[col] = j
        end

        fill_row!(col + 1, j, row_counts, remaining - 1)

        prev_labels[col] = old_prev
        row_counts[j] -= 1
      end
    end

    row_counts = zeros(Int, nβ)
    fill_row!(α[row] + 1, 1, row_counts, total_remaining)
  end

  counts = zeros(Int, nβ)
  prev_labels = Int[]
  ν_so_far = Int[]
  enumerate_rows!(result, 1, counts, prev_labels, ν_so_far, total)

  return result
end

"""
    lr_tensor_product(λ::WeightLatticeElem{TypeA{N},N}, μ::WeightLatticeElem{TypeA{N},N}) -> WeylCharacter{TypeA{N},N}

Decompose ``\\mathrm{V}(λ) \\otimes \\mathrm{V}(μ)`` into irreducibles using
the Littlewood–Richardson rule. This is specific to type ``\\mathrm{A}`` and
is typically much faster than the general Brauer–Klimyk algorithm.

The algorithm converts highest weights to partitions, enumerates LR skew
tableaux of shape ``ν / α`` with content ``β`` (where ``α`` and ``β`` are the
partitions for ``λ`` and ``μ``), and reads off the multiplicities
``c^ν_{αβ}``.

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1); ω₂ = fundamental_weight(TypeA{2}, 2);

julia> lr_tensor_product(ω₁, ω₁)
A2(2, 0) + A2(0, 1)

julia> lr_tensor_product(ω₁, ω₂)
A2(1, 1) + A2(0, 0)

julia> lr_tensor_product(ω₁ + ω₂, ω₁)  # 8 ⊗ 3 in A₂
A2(2, 1) + A2(1, 0) + A2(0, 2)
```
"""
function lr_tensor_product(
  λ::WeightLatticeElem{TypeA{N},N}, μ::WeightLatticeElem{TypeA{N},N}
) where {N}
  @assert is_dominant(λ) "First weight must be dominant"
  @assert is_dominant(μ) "Second weight must be dominant"

  α = _weight_to_partition(λ)
  β = _weight_to_partition(μ)

  # Use N+1 parts for GL(N+1) → SL(N+1) reduction
  coeffs = _lr_coefficients(α, β, N + 1)

  result = Dict{WeightLatticeElem{TypeA{N},N},Int}()
  for (ν, c) in coeffs
    w = _partition_to_weight(TypeA{N}, ν)
    result[w] = get(result, w, 0) + c
  end

  filter!(p -> !iszero(p.second), result)
  return WeylCharacter{TypeA{N},N}(result)
end

export lr_tensor_product

# ═══════════════════════════════════════════════════════════════════════════════
#  Tensor product — irreducible ⊗ irreducible
# ═══════════════════════════════════════════════════════════════════════════════

# Cache for tensor products of irreducibles.
# Key: (DT, λ, μ), Value: WeylCharacter.
const _tensor_cache = Dict{Tuple{Type,Any,Any},Any}()

# Cache for Freudenthal weight multiplicities.
# Key: (DT, λ), Value: Dict{SVector{R,Int}, Int}.
const _freudenthal_cache = Dict{Tuple{Type,Any},Any}()

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

julia> ω₁ = fundamental_weight(TypeA{3}, 1); tensor_product(ω₁, ω₁)
A3(2, 0, 0) + A3(0, 1, 0)
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

  # Brauer–Klimyk: decompose the smaller rep via Freudenthal, tensor with
  # the larger one via the BK formula. Use degree() (Weyl dimension formula,
  # O(n_pos) BigInt mults) to decide which side is smaller, avoiding
  # computing Freudenthal for the large rep unnecessarily.
  if degree(λ) <= degree(μ)
    result = brauer_klimyk(freudenthal_formula(λ), μ)
  else
    result = brauer_klimyk(freudenthal_formula(μ), λ)
  end

  _tensor_cache[key] = result
  return result
end

"""
    tensor_product(λ::WeightLatticeElem{TypeA{N},N}, μ::WeightLatticeElem{TypeA{N},N}) -> WeylCharacter{TypeA{N},N}

Specialization for type ``\\mathrm{A}``: uses the Littlewood–Richardson rule
instead of Brauer–Klimyk, which is typically much faster.
"""
function tensor_product(
  λ::WeightLatticeElem{TypeA{N},N}, μ::WeightLatticeElem{TypeA{N},N}
) where {N}
  @assert is_dominant(λ) "First weight must be dominant"
  @assert is_dominant(μ) "Second weight must be dominant"

  key = (TypeA{N}, λ, μ)
  haskey(_tensor_cache, key) && return _tensor_cache[key]::WeylCharacter{TypeA{N},N}

  key_rev = (TypeA{N}, μ, λ)
  haskey(_tensor_cache, key_rev) && return _tensor_cache[key_rev]::WeylCharacter{TypeA{N},N}

  result = lr_tensor_product(λ, μ)

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

julia> [dual(fundamental_weight(TypeA{3}, i)) for i in 1:3]
3-element Vector{WeightLatticeElem{TypeA{3}, 3}}:
 ω3
 ω2
 ω1
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
representation with highest weight `λ`, using the Newton–Girard recurrence:

``k \\cdot \\mathrm{Sym}^k(\\mathrm{V}) = \\sum_{r=1}^{k} ψ^r(\\mathrm{V}) \\cdot \\mathrm{Sym}^{k-r}(\\mathrm{V})``

This is the representation-ring analogue of the classical identity
``k \\, h_k = \\sum_{r=1}^{k} p_r \\, h_{k-r}`` relating the complete
homogeneous symmetric polynomials ``h_k`` to the power-sum polynomials ``p_r``.
Here the Adams operator ``ψ^r`` plays the role of ``p_r``.

Results are memoized for efficiency in recursive calls.

# Examples
```jldoctest
julia> using Lie

julia> spin = fundamental_weight(TypeB{3}, 3);  # B₃ spin rep, dim 8

julia> Sym(6, spin)
B3(0, 0, 6) + B3(0, 0, 4) + B3(0, 0, 2) + B3(0, 0, 0)
```
"""
function symmetric_power(λ::WeightLatticeElem{DT,R}, k::Int) where {DT,R}
  @assert is_dominant(λ) "Weight must be dominant"
  k < 0 && return WeylCharacter(DT)
  k == 0 && return WeylCharacter(WeightLatticeElem{DT,R}(zero(SVector{R,Int})))
  k == 1 && return WeylCharacter(λ)

  cache_key = (DT, λ, k)
  haskey(_symmetric_power_cache, cache_key) &&
    return _symmetric_power_cache[cache_key]::WeylCharacter{DT,R}

  result = _symmetric_power_newton_girard(λ, k)

  _symmetric_power_cache[cache_key] = result
  return result
end

# Generic Newton–Girard: uses Brauer–Klimyk for each Adams ⊗ power term
function _symmetric_power_newton_girard(λ::WeightLatticeElem{DT,R}, k::Int) where {DT,R}
  result = WeylCharacter(DT)
  # Cache Freudenthal result: all Adams operators for V(λ) use the same weights
  mults = freudenthal_formula(λ)

  for r in 1:k
    adams = Dict{SVector{R,Int},Int}(r * μ => m for (μ, m) in mults)
    prev = symmetric_power(λ, k - r)

    for (μ, m) in prev.terms
      bk = brauer_klimyk(adams, μ)
      addmul!(result, bk, m)
    end
  end

  _newton_girard_divide!(result, k)
  return result
end

function _newton_girard_divide!(result::WeylCharacter, k::Int)
  for λv in keys(result.terms)
    q, r = divrem(result.terms[λv], k)
    @assert iszero(r) "Newton–Girard: non-integer coefficient after division by k=$k"
    result.terms[λv] = q
  end
end

"""
    exterior_power(λ::WeightLatticeElem{DT,R}, k::Int) -> WeylCharacter{DT,R}

Compute the `k`-th exterior power ``\\bigwedge^k \\mathrm{V}(λ)`` of the irreducible
representation with highest weight `λ`, using the Newton–Girard recurrence:

``k \\cdot \\bigwedge\\nolimits^k(\\mathrm{V}) = \\sum_{r=1}^{k} (-1)^{r-1} ψ^r(\\mathrm{V}) \\cdot \\bigwedge\\nolimits^{k-r}(\\mathrm{V})``

This is the representation-ring analogue of the classical identity
``k \\, e_k = \\sum_{r=1}^{k} (-1)^{r-1} p_r \\, e_{k-r}`` relating the
elementary symmetric polynomials ``e_k`` to the power-sum polynomials ``p_r``.
Here the Adams operator ``ψ^r`` plays the role of ``p_r``.

Results are memoized for efficiency in recursive calls.

# Examples
```jldoctest
julia> using Lie

julia> spin = fundamental_weight(TypeB{3}, 3);  # B₃ spin rep, dim 8

julia> ⋀(6, spin)
B3(1, 0, 0) + B3(0, 1, 0)
```
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

  result = _exterior_power_newton_girard(λ, k)

  _exterior_power_cache[cache_key] = result
  return result
end

# Generic Newton–Girard: uses Brauer–Klimyk for each Adams ⊗ power term
function _exterior_power_newton_girard(λ::WeightLatticeElem{DT,R}, k::Int) where {DT,R}
  result = WeylCharacter(DT)
  # Cache Freudenthal result: all Adams operators for V(λ) use the same weights
  mults = freudenthal_formula(λ)

  for r in 1:k
    adams = Dict{SVector{R,Int},Int}(r * μ => m for (μ, m) in mults)
    prev = exterior_power(λ, k - r)

    sign = iseven(r) ? -1 : 1
    for (μ, m) in prev.terms
      bk = brauer_klimyk(adams, μ)
      addmul!(result, bk, sign * m)
    end
  end

  _newton_girard_divide!(result, k)
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

Supports both effective (non-negative) and virtual (mixed sign) characters.

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

    @assert all(>=(0), best) "Highest remaining weight is not dominant — input is not Weyl-group invariant"

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
