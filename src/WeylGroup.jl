# ═══════════════════════════════════════════════════════════════════════════════
#  Weyl groups — group elements, multiplication, actions on roots and weights
#
#  Weyl group elements are stored as reduced words (sequences of simple
#  reflection indices).  The reflection table from RootSystem is used for
#  efficient multiplication and normalization.
# ═══════════════════════════════════════════════════════════════════════════════

export WeylGroup, WeylGroupElem
export weyl_group, root_system, gens, gen, longest_element
export word, weyl_order
export weyl_orbit, dominant_weights
export degree, weyl_dimension
export is_singular
export right_descent_set, left_descent_set
export bruhat_leq, bruhat_descendants
export right_coset_reps, left_coset_reps

# ═══════════════════════════════════════════════════════════════════════════════
#  WeylGroup
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WeylGroup{DT,R}

The Weyl group of a root system of Dynkin type `DT` with rank `R`.

Lie.jl writes Weyl group actions on the right: `λ * x` denotes the usual action
of the Weyl group element `x` on the weight or root `λ`.

# Examples
```jldoctest
julia> using Lie

julia> weyl_order(TypeA{2})
6
```
"""
struct WeylGroup{DT<:DynkinType,R}
  root_system::RootSystem{DT,R}
end

"""
    weyl_group(::Type{DT}) -> WeylGroup{DT}

Construct the Weyl group for the given Dynkin type.

# Examples
```jldoctest
julia> using Lie

julia> W = weyl_group(TypeA{2})
Weyl group of type A2
```
"""
function weyl_group(::Type{DT}) where {DT<:DynkinType}
  check_dynkin_type(DT)
  RS = RootSystem(DT)
  return WeylGroup{DT,rank(DT)}(RS)
end

weyl_group(dt::DynkinType) = weyl_group(typeof(dt))

"""
    root_system(W::WeylGroup) -> RootSystem

Return the root system underlying the Weyl group `W`.

# Examples
```jldoctest
julia> using Lie

julia> root_system(weyl_group(TypeA{2}))
Root system of type A2, rank 2 with 3 positive roots
```
"""
root_system(W::WeylGroup) = W.root_system

function Base.show(io::IO, W::WeylGroup{DT,R}) where {DT,R}
  print(io, "Weyl group of type $(_type_name(DT))")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  WeylGroupElem — stored as a reduced word in simple reflections
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WeylGroupElem{DT,R}

An element of the Weyl group, stored as a reduced word (vector of simple
reflection indices).

# Examples
```jldoctest
julia> using Lie

julia> W = weyl_group(TypeA{2});

julia> W([1, 2])
s1 * s2
```
"""
struct WeylGroupElem{DT<:DynkinType,R}
  parent::WeylGroup{DT,R}
  word::Vector{UInt8}  # reduced word in simple reflections
end

Base.parent(x::WeylGroupElem) = x.parent

"""
    word(x::WeylGroupElem) -> Vector{UInt8}

Return the reduced word of `x`.

# Examples
```jldoctest
julia> using Lie

julia> W = weyl_group(TypeA{2});

julia> word(W([1, 2]))
2-element Vector{UInt8}:
 0x01
 0x02
```
"""
word(x::WeylGroupElem) = x.word

"""
    Base.length(x::WeylGroupElem) -> Int

Return the length (number of simple reflections) of `x`.
"""
Base.length(x::WeylGroupElem) = length(x.word)

Base.:(==)(x::WeylGroupElem{DT,R}, y::WeylGroupElem{DT,R}) where {DT,R} =
  x.word == y.word
Base.hash(x::WeylGroupElem, h::UInt) = hash(x.word, h)

function Base.show(io::IO, x::WeylGroupElem)
  if isempty(x.word)
    print(io, "id")
  else
    print(io, join(["s$(i)" for i in x.word], " * "))
  end
end

# ─── Construction ────────────────────────────────────────────────────────────

"""
    (W::WeylGroup{DT,R})(word::Vector{<:Integer}; normalize=true) -> WeylGroupElem

Construct a Weyl group element from a word in simple reflections.
If `normalize=true`, reduces the word to short-lex normal form.

# Examples
```jldoctest
julia> using Lie

julia> W = weyl_group(TypeA{2});

julia> W([1, 1])
id
```
"""
function (W::WeylGroup{DT,R})(word_in::Vector{<:Integer}; normalize::Bool=true) where {DT,R}
  if !normalize
    return WeylGroupElem{DT,R}(W, UInt8.(word_in))
  end
  # Build up element one generator at a time using rmul!
  x = WeylGroupElem{DT,R}(W, UInt8[])
  for s in word_in
    rmul!(x, UInt8(s))
  end
  return x
end

"""
    one(W::WeylGroup) -> WeylGroupElem

Return the identity element.

# Examples
```jldoctest
julia> using Lie

julia> one(weyl_group(TypeA{2}))
id
```
"""
Base.one(W::WeylGroup{DT,R}) where {DT,R} = WeylGroupElem{DT,R}(W, UInt8[])

"""
    gen(W::WeylGroup, i) -> WeylGroupElem

Return the `i`-th simple reflection.

# Examples
```jldoctest
julia> using Lie

julia> gen(weyl_group(TypeA{2}), 1)
s1
```
"""
function gen(W::WeylGroup{DT,R}, i::Integer) where {DT,R}
  1 <= i <= R || throw(ArgumentError("Generator index $i out of range 1:$R"))
  return WeylGroupElem{DT,R}(W, UInt8[i])
end

"""
    gens(W::WeylGroup) -> Vector{WeylGroupElem}

Return all simple reflections.

# Examples
```jldoctest
julia> using Lie

julia> gens(weyl_group(TypeA{2}))
2-element Vector{WeylGroupElem{TypeA{2}, 2}}:
 s1
 s2
```
"""
gens(W::WeylGroup{DT,R}) where {DT,R} = [gen(W, i) for i in 1:R]

# ─── Right multiplication by a simple reflection ────────────────────────────

"""
    rmul!(x::WeylGroupElem, s::UInt8) -> WeylGroupElem

Multiply `x` from the right by the simple reflection `s`, maintaining
the reduced word in short-lex normal form.

Uses the reflection table from the root system.
"""
function rmul!(x::WeylGroupElem{DT,R}, s::UInt8) where {DT,R}
  W = parent(x)
  RS = W.root_system
  refl = RS.refl

  b, pos, letter = _explain_rmul(x, s, refl, R)
  if b
    insert!(x.word, pos, letter)
  else
    deleteat!(x.word, pos)
  end
  return x
end

"""
Internal: determines what right-multiplication by `s` does to word `x`.

Returns `(insert::Bool, position::Int, letter::UInt8)`:
- if `insert=true`: insert `letter` at `position`
- if `insert=false`: delete the element at `position`
"""
function _explain_rmul(x::WeylGroupElem, s::UInt8, refl::AbstractMatrix{UInt}, rk::Integer)
  insert_index = length(x.word) + 1
  insert_letter = s

  root = UInt(s)  # track which root s maps to
  for k in length(x.word):-1:1
    if x.word[k] == root
      # Found: xs_k = x with letter at k removed
      return false, k, x.word[k]
    end

    # Apply reflection s_{word[k]} to root
    root = refl[Int(x.word[k]), Int(root)]

    if iszero(root)
      # root is no longer a minimal root, meaning we found the best insertion point
      return true, insert_index, insert_letter
    end

    # Check if we have a better insertion point.
    # Since word[k] is a simple root, if root < word[k] it must also be simple.
    if root < x.word[k]
      insert_index = k
      insert_letter = UInt8(root)
    end
  end

  return true, insert_index, insert_letter
end

# ─── Group operations ───────────────────────────────────────────────────────

function Base.:*(x::WeylGroupElem{DT,R}, y::WeylGroupElem{DT,R}) where {DT,R}
  parent(x) === parent(y) ||
    throw(ArgumentError("Cannot multiply elements from different Weyl groups"))
  result = WeylGroupElem{DT,R}(parent(x), copy(x.word))
  for s in y.word
    rmul!(result, s)
  end
  return result
end

function Base.inv(x::WeylGroupElem{DT,R}) where {DT,R}
  W = parent(x)
  y = one(W)
  for s in Iterators.reverse(x.word)
    rmul!(y, s)
  end
  return y
end

Base.isone(x::WeylGroupElem) = isempty(x.word)

@inline function _rmul_simple(x::WeylGroupElem{DT,R}, s::Int) where {DT,R}
  y = WeylGroupElem{DT,R}(parent(x), copy(x.word))
  rmul!(y, UInt8(s))
  return y
end

function Base.:^(x::WeylGroupElem, n::Integer)
  W = parent(x)
  if n == 0
    return one(W)
  elseif n < 0
    return inv(x)^(-n)
  end
  result = one(W)
  for _ in 1:n
    result = result * x
  end
  return result
end

# ─── Action on roots ────────────────────────────────────────────────────────

"""
    *(r::RootSpaceElem{DT,R}, x::WeylGroupElem{DT,R}) -> RootSpaceElem{DT,R}

Right action of a Weyl group element on a root space element.
"""
function Base.:*(r::RootSpaceElem{DT,R}, x::WeylGroupElem{DT,R}) where {DT,R}
  C = cartan_matrix(DT)
  v = MVector{R,Int}(r.vec)
  for s in x.word
    # s_s(v) = v - ⟨αₛ∨, v⟩ αₛ  where ⟨αₛ∨, v⟩ = ∑ⱼ C[s,j] vⱼ
    pairing = sum(C[s, j] * v[j] for j in 1:R)
    v[s] -= pairing
  end
  return RootSpaceElem{DT,R}(SVector{R,Int}(v))
end

# ─── Action on weights ──────────────────────────────────────────────────────

"""
    *(w::WeightLatticeElem{DT,R}, x::WeylGroupElem{DT,R}) -> WeightLatticeElem{DT,R}

Right action of a Weyl group element on a weight.
"""
function Base.:*(w::WeightLatticeElem{DT,R}, x::WeylGroupElem{DT,R}) where {DT,R}
  C = cartan_matrix(DT)
  v = MVector{R,Int}(w.vec)
  for s in x.word
    pairing = v[s]  # ⟨αₛ∨, λ⟩ = λₛ in fundamental weight coords
    for j in 1:R
      v[j] -= pairing * C[j, s]
    end
  end
  return WeightLatticeElem{DT,R}(SVector{R,Int}(v))
end

# ─── Longest element ────────────────────────────────────────────────────────

const _longest_element_cache = Dict{Type,Any}()
const _longest_element_lock = ReentrantLock()

"""
    longest_element(W::WeylGroup{DT,R}) -> WeylGroupElem{DT,R}

Compute the longest element w0 of the Weyl group.
Uses the iterative algorithm: repeatedly find a simple reflection that increases length.
The result is cached per Dynkin type.

# Examples
```jldoctest
julia> using Lie

julia> W = weyl_group(TypeA{2});

julia> w0 = longest_element(W);

julia> length(w0)
3
```
"""
function longest_element(W::WeylGroup{DT,R}) where {DT,R}
  lock(_longest_element_lock) do
    get!(_longest_element_cache, DT) do
      w0 = one(W)
      wt = MVector{R,Int}(ntuple(j -> 1, R))
      C = cartan_matrix(DT)
      while true
        found = false
        for s in 1:R
          if wt[s] > 0
            rmul!(w0, UInt8(s))
            pairing = wt[s]
            for j in 1:R
              wt[j] -= pairing * C[j, s]
            end
            found = true
            break
          end
        end
        found || break
      end
      w0
    end::WeylGroupElem{DT,R}
  end
end

# ─── Descent sets and Bruhat tools ──────────────────────────────────────────

"""
    right_descent_set(w::WeylGroupElem) -> Vector{Int}

Return the right descent set of `w`, i.e. the indices `i` such that
`\\ell(ws_i) < \\ell(w)`.

# Examples
```jldoctest
julia> using Lie

julia> W = weyl_group(TypeA{2});

julia> right_descent_set(W([1, 2]))
1-element Vector{Int64}:
 2
```
"""
function right_descent_set(w::WeylGroupElem{DT,R}) where {DT,R}
  desc = Int[]
  lw = length(w)
  for i in 1:R
    length(_rmul_simple(w, i)) < lw && push!(desc, i)
  end
  return desc
end

"""
    left_descent_set(w::WeylGroupElem) -> Vector{Int}

Return the left descent set of `w`, i.e. the indices `i` such that
`\\ell(s_iw) < \\ell(w)`.

# Examples
```jldoctest
julia> using Lie

julia> W = weyl_group(TypeA{2});

julia> left_descent_set(W([1, 2]))
1-element Vector{Int64}:
 1
```
"""
function left_descent_set(w::WeylGroupElem)
  right_descent_set(inv(w))
end

"""
    bruhat_leq(x::WeylGroupElem, y::WeylGroupElem) -> Bool

Return whether `x \\le y` in the (strong) Bruhat order.

# Examples
```jldoctest
julia> using Lie

julia> W = weyl_group(TypeA{2});

julia> bruhat_leq(gen(W, 1), W([1, 2]))
true
```
"""
function bruhat_leq(x::WeylGroupElem{DT,R}, y::WeylGroupElem{DT,R}) where {DT,R}
  parent(x) === parent(y) ||
    throw(ArgumentError("Cannot compare elements from different Weyl groups"))

  lx = length(x)
  ly = length(y)
  lx > ly && return false
  x == y && return true

  # Standard recursive criterion via right descents.
  s = first(right_descent_set(y))
  ys = _rmul_simple(y, s)
  if s in right_descent_set(x)
    return bruhat_leq(_rmul_simple(x, s), ys)
  else
    return bruhat_leq(x, ys)
  end
end

"""
    bruhat_descendants(w::WeylGroupElem) -> Vector{WeylGroupElem}

Return the immediate Bruhat descendants obtained by right-multiplying by
simple reflections in the right descent set.

# Examples
```jldoctest
julia> using Lie

julia> W = weyl_group(TypeA{2});

julia> bruhat_descendants(W([1, 2]))
1-element Vector{WeylGroupElem{TypeA{2}, 2}}:
 s1
```
"""
function bruhat_descendants(w::WeylGroupElem)
  [_rmul_simple(w, s) for s in right_descent_set(w)]
end

"""
    right_coset_reps(W::WeylGroup, I::AbstractVector{<:Integer}) -> Vector{WeylGroupElem}

Enumerate minimal right coset representatives for `W/W_I`, where `W_I` is the
parabolic subgroup generated by simple reflections in `I`.

Uses a weight-orbit BFS: the weight ``λ_I = \\sum_{j \\notin I} ω_j`` has stabilizer
exactly ``W_I``, so its ``W``-orbit has size ``|W/W_I|``.  Enumerates that orbit
using ``O(|W/W_I| \\cdot R)`` weight reflections, independent of ``|W|``.

# Examples
```jldoctest
julia> using Lie

julia> length(right_coset_reps(weyl_group(TypeA{2}), [1]))
3
```
"""
function right_coset_reps(W::WeylGroup{DT,R}, I::AbstractVector{<:Integer}) where {DT,R}
  Iset = Set{Int}(Int.(I))
  for i in Iset
    1 <= i <= R || throw(ArgumentError("Parabolic index $i out of range 1:$R"))
  end

  # λ_I has coordinate 0 for j ∈ I and 1 for j ∉ I; its stabilizer is W_I.
  λ_I = WeightLatticeElem{DT,R}(SVector{R,Int}(ntuple(j -> j ∈ Iset ? 0 : 1, Val(R))))

  # BFS tracking (u, μ) where u = inv(w) and μ = w(λ_I).
  # Right-multiplying u by s_j (via _rmul_simple) corresponds to left-multiplying
  # w by s_j, and the weight updates as μ → reflect(μ, j).
  us = WeylGroupElem{DT,R}[]
  orbit_seen = Dict{SVector{R,Int},Nothing}()
  q = Pair{WeylGroupElem{DT,R},WeightLatticeElem{DT,R}}[]

  e = one(W)
  orbit_seen[λ_I.vec] = nothing
  push!(us, e)
  push!(q, e => λ_I)

  while !isempty(q)
    u, μ = popfirst!(q)
    for j in 1:R
      new_μ = reflect(μ, j)
      if !haskey(orbit_seen, new_μ.vec)
        new_u = _rmul_simple(u, j)
        orbit_seen[new_μ.vec] = nothing
        push!(us, new_u)
        push!(q, new_u => new_μ)
      end
    end
  end

  reps = [inv(u) for u in us]
  sort!(reps; by=w -> (length(w), Vector{UInt8}(w.word)))
  return reps
end

"""
    left_coset_reps(W::WeylGroup, I::AbstractVector{<:Integer}) -> Vector{WeylGroupElem}

Enumerate minimal left coset representatives for `W_I\\W`.

# Examples
```jldoctest
julia> using Lie

julia> length(left_coset_reps(weyl_group(TypeA{2}), [1]))
3
```
"""
function left_coset_reps(W::WeylGroup, I::AbstractVector{<:Integer})
  [inv(w) for w in right_coset_reps(W, I)]
end

# ─── Weyl group order ───────────────────────────────────────────────────────

"""
    weyl_order(::Type{DT}) -> BigInt

Return the order of the Weyl group of type `DT`.

# Examples
```jldoctest
julia> using Lie

julia> weyl_order(TypeA{3})
24

julia> weyl_order(TypeE{8})
696729600
```
"""
weyl_order(::Type{TypeA{N}}) where {N} =
  (check_dynkin_type(TypeA{N}); factorial(BigInt(N + 1)))
weyl_order(::Type{TypeB{N}}) where {N} =
  (check_dynkin_type(TypeB{N}); factorial(BigInt(N)) * BigInt(2)^N)
weyl_order(::Type{TypeC{N}}) where {N} =
  (check_dynkin_type(TypeC{N}); factorial(BigInt(N)) * BigInt(2)^N)
weyl_order(::Type{TypeD{N}}) where {N} =
  (check_dynkin_type(TypeD{N}); factorial(BigInt(N)) * BigInt(2)^(N - 1))
weyl_order(::Type{TypeE{6}}) = BigInt(51840)
weyl_order(::Type{TypeE{7}}) = BigInt(2903040)
weyl_order(::Type{TypeE{8}}) = BigInt(696729600)
weyl_order(::Type{TypeF4}) = BigInt(1152)
weyl_order(::Type{TypeG2}) = BigInt(12)

function weyl_order(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  return prod(weyl_order(T) for T in Ts.parameters)
end

weyl_order(dt::DynkinType) = weyl_order(typeof(dt))

# ─── Weyl orbit ──────────────────────────────────────────────────────────────

"""
    weyl_orbit(::Type{DT}, w::WeightLatticeElem{DT,R}) -> Vector{WeightLatticeElem{DT,R}}

Compute the full Weyl orbit of the weight `w`.

# Examples
```jldoctest
julia> using Lie

julia> length(weyl_orbit(TypeA{2}, fundamental_weight(TypeA{2}, 1)))
3
```
"""
function weyl_orbit(::Type{DT}, w::WeightLatticeElem{DT,R}) where {DT<:DynkinType,R}
  # Use weylloop for efficient traversal (no hash set)
  result = WeightLatticeElem{DT,R}[]
  weylloop(DT, Vector{Int}(w.vec)) do tmp
    push!(result, WeightLatticeElem{DT,R}(SVector{R,Int}(tmp)))
  end
  return result
end

function weyl_orbit(w::WeightLatticeElem{DT,R}) where {DT,R}
  return weyl_orbit(DT, w)
end

# ─── Dominant weights ────────────────────────────────────────────────────────

"""
    dominant_weights(::Type{DT}, hw::WeightLatticeElem{DT,R}) -> Vector{WeightLatticeElem{DT,R}}

Compute the dominant weights occurring in the irreducible representation
with highest weight `hw`, sorted by decreasing level below `hw`.

The level of `μ` below `hw` is the root-lattice height of `hw - μ`,
i.e. the sum of coefficients when `hw - μ` is written in the simple root basis.

# Examples
```jldoctest
julia> using Lie

julia> λ = fundamental_weight(TypeA{2}, 1) + fundamental_weight(TypeA{2}, 2);

julia> length(dominant_weights(λ))
2
```
"""
function dominant_weights(::Type{DT}, hw::WeightLatticeElem{DT,R}) where {DT<:DynkinType,R}
  is_dominant(hw) || throw(ArgumentError("Highest weight must be dominant"))
  RS = RootSystem(DT)
  C = cartan_matrix(DT)

  # Positive roots in weight coords: column j of C = weight coords of αⱼ
  # For root v = Σ vᵢ αᵢ: weight coord j = Σᵢ C[j,i] vᵢ = (Cv)ⱼ
  n_pos = n_positive_roots(RS)
  pos_roots_w = Vector{SVector{R,Int}}(undef, n_pos)
  for k in 1:n_pos
    α_root = RS.positive_roots_list[k]
    pos_roots_w[k] = SVector{R,Int}(ntuple(j -> sum(C[j, i] * α_root[i] for i in 1:R), R))
  end

  result = Set{SVector{R,Int}}([hw.vec])
  todo = [hw.vec]

  while !isempty(todo)
    new_todo = SVector{R,Int}[]
    for w in todo
      for α_w in pos_roots_w
        w_sub = w - α_w
        if all(>=(0), w_sub) && w_sub ∉ result
          push!(result, w_sub)
          push!(new_todo, w_sub)
        end
      end
    end
    todo = new_todo
  end

  # Compute level vector: transforms ω-coords to root height.
  # Level of μ below hw = dot(level_vec, hw - μ) / det(C)
  # For sorting we just use dot(level_vec, μ) (higher = closer to hw).
  Cinv = cartan_matrix_inverse(DT)
  level_vec = SVector{R,Rational{Int}}(
    ntuple(j -> sum(Cinv[i, j] for i in 1:R), R)
  )

  weights = [WeightLatticeElem{DT,R}(v) for v in result]
  sort!(weights; by=w -> -sum(w.vec[i] * level_vec[i] for i in 1:R))
  return weights
end

function dominant_weights(hw::WeightLatticeElem{DT,R}) where {DT,R}
  return dominant_weights(DT, hw)
end

# ─── Dimension of simple module (Weyl dimension formula) ────────────────────

const _weyl_dimension_data_cache = Dict{Type,Any}()
const _weyl_dimension_data_lock = ReentrantLock()

function _weyl_dimension_data_from_roots(d, pos_roots, ::Val{R}) where {R}
  denom = BigInt(1)
  scaled = Vector{SVector{R,Int}}(undef, length(pos_roots))
  for idx in eachindex(pos_roots)
    α = pos_roots[idx]
    dα = SVector{R,Int}(ntuple(i -> d[i] * α[i], Val(R)))
    scaled[idx] = dα
    ip = zero(Int)
    @inbounds for i in 1:R
      ip += dα[i]
    end
    Base.GMP.MPZ.mul_si!(denom, denom, ip)
  end
  return denom, Tuple(scaled)
end

function _weyl_dimension_data_compiletime(::Type{DT}) where {DT<:DynkinType}
  R = rank(DT)
  d = _cartan_symmetrizer_data(DT)
  C = _cartan_matrix_data(DT)
  C_sm = SMatrix{R,R,Int,R * R}(Tuple(C))
  pos_roots, _, _ = _compute_positive_roots_and_reflections(C_sm, R)
  denom, scaled = _weyl_dimension_data_from_roots(d, pos_roots, Val(R))
  return denom, Tuple(Tuple(v) for v in scaled)
end

function _weyl_dimension_data_cached(
  ::Type{DT}, ::Val{R}, ::Val{N}
) where {DT<:DynkinType,R,N}
  lock(_weyl_dimension_data_lock) do
    get!(_weyl_dimension_data_cache, DT) do
      d = _cartan_symmetrizer_data(DT)
      C = _cartan_matrix_data(DT)
      pos_roots = _positive_roots_runtime(C, R)
      _weyl_dimension_data_from_roots(
        d, pos_roots, Val(R)
      )
    end::Tuple{BigInt,NTuple{N,SVector{R,Int}}}
  end
end

@generated function _weyl_dimension_data(::Type{DT}) where {DT<:DynkinType}
  R = rank(DT)
  N = n_positive_roots(DT)
  if R > 9
    return :(_weyl_dimension_data_cached($DT, Val{$R}(), Val{$N}()))
  end
  denom, scaled = _weyl_dimension_data_compiletime(DT)
  return :(($denom, NTuple{$N,SVector{$R,Int}}($scaled)))
end

"""
    _weyl_denominator(::Type{DT}) -> BigInt

Compute the Weyl dimension denominator `∏_{α>0} ⟨ρ, α⟩`.
"""
function _weyl_denominator(::Type{DT}) where {DT<:DynkinType}
  return first(_weyl_dimension_data(DT))
end

"""
    _weyl_dim_scaled_roots(::Type{DT}) -> NTuple{N, SVector{R,Int}}

Return the symmetrizer-scaled positive root vectors `d .* α` for Dynkin type `DT`.
"""
function _weyl_dim_scaled_roots(::Type{DT}) where {DT<:DynkinType}
  return last(_weyl_dimension_data(DT))
end

"""
    degree(::Type{DT}, hw::WeightLatticeElem{DT,R}) -> BigInt
    degree(hw::WeightLatticeElem{DT,R}) -> BigInt

Dimension of the irreducible representation with highest weight `hw`,
computed via the Weyl dimension formula:

``\\dim \\mathrm{V}(λ) = \\prod_{α > 0} \\frac{⟨λ + ρ, α^\\vee⟩}{⟨ρ, α^\\vee⟩}``

Equivalently, using the invariant bilinear form,
``\\prod_{α>0} (λ+ρ,α)/(ρ,α)``.

The denominator and the symmetrizer-scaled root vectors are precomputed once per
Dynkin type. The numerator is computed as
a `BigInt` product of `Int`-valued inner products via in-place GMP arithmetic.

# Examples
```jldoctest
julia> using Lie

julia> degree(fundamental_weight(TypeA{3}, 1))
4

julia> degree(fundamental_weight(TypeB{3}, 3))
8

julia> degree(fundamental_weight(TypeE{8}, 8))
248

julia> [degree(fundamental_weight(TypeB{3}, i)) for i in 1:3]
3-element Vector{BigInt}:
  7
 21
  8
```
"""
function degree(
  ::Type{PDT}, hw::WeightLatticeElem{PDT,R}
) where {Ts,PDT<:ProductDynkinType{Ts},R}
  is_dominant(hw) || throw(ArgumentError("Highest weight must be dominant"))

  result = BigInt(1)
  for factor_weight in _product_component_weights(PDT, hw)
    result *= degree(factor_weight)
  end
  return result
end

function degree(::Type{DT}, hw::WeightLatticeElem{DT,R}) where {DT<:DynkinType,R}
  is_dominant(hw) || throw(ArgumentError("Highest weight must be dominant"))

  denom, dα_all = _weyl_dimension_data(DT)
  λ_ρ = hw + weyl_vector(DT)

  # Numerator: ∏_{α>0} ⟨λ+ρ, α⟩, computed in-place with GMP mul_si!
  numer = BigInt(1)
  for dα in dα_all
    ip = zero(Int)
    @inbounds for i in 1:R
      ip += λ_ρ.vec[i] * dα[i]
    end
    Base.GMP.MPZ.mul_si!(numer, numer, ip)
  end

  result, rem = divrem(numer, denom)
  iszero(rem) || throw(
    DomainError(
      (numerator=numer, denominator=denom),
      "Weyl dimension formula for type $(_type_name(DT)) and highest weight $hw gave the non-integer value $numer / $denom",
    ),
  )
  return result
end

function _degree_runtime(::Type{DT}, hw::WeightLatticeElem{DT,R}) where {DT<:DynkinType,R}
  return degree(DT, hw)
end

"""
    _positive_roots_runtime(C, R) -> Vector{Vector{Int}}

Compute positive roots using plain arrays (no StaticArrays).
"""
function _positive_roots_runtime(C::AbstractMatrix{Int}, R::Int)
  # Simple roots = standard basis vectors
  pos_roots = [zeros(Int, R) for _ in 1:R]
  for i in 1:R
    pos_roots[i][i] = 1
  end
  root_set = Set{Vector{Int}}(pos_roots)

  i = 1
  while i <= length(pos_roots)
    α = pos_roots[i]
    for s in 1:R
      pairing = zero(Int)
      @inbounds for j in 1:R
        pairing += C[s, j] * α[j]
      end
      # s_s(α) = α - pairing * eₛ
      new_root = copy(α)
      new_root[s] -= pairing
      if _is_nonnegative(new_root) && !(new_root in root_set)
        push!(pos_roots, new_root)
        push!(root_set, new_root)
      end
    end
    i += 1
  end
  sort!(pos_roots; by=_root_height)
  return pos_roots
end

function degree(hw::WeightLatticeElem{DT,R}) where {DT,R}
  return degree(DT, hw)
end

"""
    degree(::Type{DT}, v::AbstractVector{<:Integer}) -> BigInt

Dimension of the irreducible representation with highest weight given by
the integer vector `v` (in the fundamental weight basis).

This is a convenience wrapper: `degree(DT, v) == degree(WeightLatticeElem(DT, v))`.

# Examples
```jldoctest
julia> using Lie

julia> degree(TypeA{2}, [1, 0])  # standard representation of A2
3

julia> degree(TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1])  # adjoint of E8
248
```
"""
function degree(::Type{DT}, v::AbstractVector{<:Integer}) where {DT<:DynkinType}
  return degree(DT, WeightLatticeElem(DT, v))
end

degree(dt::DynkinType, v::AbstractVector{<:Integer}) = degree(typeof(dt), v)

"""
    weyl_dimension(λ::WeightLatticeElem) -> BigInt
    weyl_dimension(::Type{DT}, λ::WeightLatticeElem) -> BigInt
    weyl_dimension(::Type{DT}, v::AbstractVector{<:Integer}) -> BigInt
    weyl_dimension(dt::DynkinType, v) -> BigInt

Synonym for [`degree`](@ref). Computes the dimension of the irreducible
representation via the Weyl dimension formula.
"""
weyl_dimension(λ::WeightLatticeElem) = degree(λ)
weyl_dimension(::Type{DT}, λ::WeightLatticeElem) where {DT<:DynkinType} = degree(DT, λ)
weyl_dimension(::Type{DT}, v::AbstractVector{<:Integer}) where {DT<:DynkinType} = degree(
  DT, v
)
weyl_dimension(dt::DynkinType, v) = degree(typeof(dt), v)

# ─── Singularity ─────────────────────────────────────────────────────────────

"""
    is_singular(w::WeightLatticeElem{DT,R}) -> Bool

Check whether the weight `w` is singular, i.e. lies on some wall of a Weyl
chamber. Equivalently, `w` is singular iff `⟨α∨, w⟩ = 0` for some positive
root `α`.

For a dominant weight this simplifies to checking whether any fundamental
weight coordinate is zero. For a general weight, we first conjugate to the
dominant chamber.

# Examples
```jldoctest
julia> using Lie

julia> is_singular(fundamental_weight(TypeA{2}, 1))
true

julia> is_singular(weyl_vector(TypeA{2}))
false
```
"""
function is_singular(w::WeightLatticeElem{DT,R}) where {DT,R}
  dom = conjugate_dominant_weight(w)
  return any(i -> dom.vec[i] == 0, 1:R)
end

# ─── Borel–Weil–Bott ────────────────────────────────────────────────────────

"""
    borel_weil_bott(λ::WeightLatticeElem{DT,R}) -> Union{Nothing, Tuple{Int, WeightLatticeElem{DT,R}}}

Apply the Borel–Weil–Bott theorem to the weight `λ`.

!!! note "Package placement"
    This function is a preview implementation that properly belongs to
    `PartialFlagVarieties.jl`, an upcoming companion package. It is included here
    for convenience but is **not part of the public API of `Lie.jl`** and is not
    exported. Access it via `import Lie: borel_weil_bott`.

Compute `μ = λ + ρ` and find the unique Weyl group element `w` such that
`w(μ)` is dominant. If `μ` is singular (lies on a Weyl chamber wall),
all cohomology vanishes and we return `nothing`. Otherwise, return
`(d, w(μ) - ρ)` where `d = ℓ(w)` is the cohomological degree, meaning

``\\mathrm{H}^d(G/B, \\mathcal{L}_λ) \\cong \\mathrm{V}_{w(μ)-ρ}^*``

and all other cohomology groups vanish.

# Examples
```jldoctest
julia> using Lie; import Lie: borel_weil_bott

julia> borel_weil_bott(fundamental_weight(TypeA{2}, 1))
(0, ω1)

julia> borel_weil_bott(WeightLatticeElem(TypeA{2}, [-2, 1]))
(1, 0)

julia> borel_weil_bott(-weyl_vector(TypeA{2})) === nothing
true
```
"""
function borel_weil_bott(λ::WeightLatticeElem{DT,R}) where {DT,R}
  ρ = weyl_vector(DT)
  μ = λ + ρ

  # Move μ to the dominant chamber; the number of reflections is the degree
  μ_dom, d = conjugate_dominant_weight_with_length(μ)

  # If any coordinate of μ_dom is zero, λ + ρ lies on a Weyl chamber wall
  # (including the case μ = 0 when λ = -ρ), so all cohomology vanishes.
  any(==(0), μ_dom.vec) && return nothing

  return (d, μ_dom - ρ)
end
