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
export dim_of_simple_module
export is_singular, borel_weil_bott

# ═══════════════════════════════════════════════════════════════════════════════
#  WeylGroup
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WeylGroup{DT,R}

The Weyl group of a root system of Dynkin type `DT` with rank `R`.
"""
struct WeylGroup{DT<:DynkinType,R}
    root_system::RootSystem{DT,R}
end

"""
    weyl_group(::Type{DT}) -> WeylGroup{DT}

Construct the Weyl group for the given Dynkin type.
"""
function weyl_group(::Type{DT}) where {DT<:DynkinType}
    RS = RootSystem(DT)
    return WeylGroup{DT,rank(DT)}(RS)
end

weyl_group(dt::DynkinType) = weyl_group(typeof(dt))

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
"""
struct WeylGroupElem{DT<:DynkinType,R}
    parent::WeylGroup{DT,R}
    word::Vector{UInt8}  # reduced word in simple reflections
end

Base.parent(x::WeylGroupElem) = x.parent

"""
    word(x::WeylGroupElem) -> Vector{UInt8}

Return the reduced word of `x`.
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
"""
Base.one(W::WeylGroup{DT,R}) where {DT,R} = WeylGroupElem{DT,R}(W, UInt8[])

"""
    gen(W::WeylGroup, i) -> WeylGroupElem

Return the `i`-th simple reflection.
"""
function gen(W::WeylGroup{DT,R}, i::Integer) where {DT,R}
    @assert 1 <= i <= R
    return WeylGroupElem{DT,R}(W, UInt8[i])
end

"""
    gens(W::WeylGroup) -> Vector{WeylGroupElem}

Return all simple reflections.
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
function _explain_rmul(x::WeylGroupElem, s::UInt8, refl::Matrix{UInt}, rk::Int)
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
    @assert parent(x) === parent(y)
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

function Base.:^(x::WeylGroupElem, n::Int)
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

"""
    longest_element(W::WeylGroup{DT,R}) -> WeylGroupElem{DT,R}

Compute the longest element w₀ of the Weyl group.
Uses the iterative algorithm: repeatedly find a simple reflection that increases length.
"""
function longest_element(W::WeylGroup{DT,R}) where {DT,R}
    RS = W.root_system
    np = n_positive_roots(RS)

    w0 = one(W)
    # ρ in weight coords (all 1s)
    wt = MVector{R,Int}(ntuple(j -> 1, R))
    C = cartan_matrix(DT)

    while true
        found = false
        for s in 1:R
            if wt[s] > 0
                # Apply s-th reflection
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

    return w0
end

# ─── Weyl group order ───────────────────────────────────────────────────────

"""
    weyl_order(::Type{DT}) -> BigInt

Return the order of the Weyl group of type `DT`.
"""
function weyl_order(::Type{DT}) where {DT<:DynkinType}
    return _weyl_order(DT)
end

weyl_order(dt::DynkinType) = weyl_order(typeof(dt))

_weyl_order(::Type{TypeA{N}}) where {N} = factorial(BigInt(N + 1))
_weyl_order(::Type{TypeB{N}}) where {N} = factorial(BigInt(N)) * BigInt(2)^N
_weyl_order(::Type{TypeC{N}}) where {N} = factorial(BigInt(N)) * BigInt(2)^N
_weyl_order(::Type{TypeD{N}}) where {N} = factorial(BigInt(N)) * BigInt(2)^(N - 1)
_weyl_order(::Type{TypeE{6}}) = BigInt(51840)
_weyl_order(::Type{TypeE{7}}) = BigInt(2903040)
_weyl_order(::Type{TypeE{8}}) = BigInt(696729600)
_weyl_order(::Type{TypeF4}) = BigInt(1152)
_weyl_order(::Type{TypeG2}) = BigInt(12)

function _weyl_order(::Type{ProductDynkinType{Ts}}) where {Ts}
    return prod(_weyl_order(T) for T in Ts.parameters)
end

# ─── Weyl orbit ──────────────────────────────────────────────────────────────

"""
    weyl_orbit(::Type{DT}, w::WeightLatticeElem{DT,R}) -> Vector{WeightLatticeElem{DT,R}}

Compute the full Weyl orbit of the weight `w`.
"""
function weyl_orbit(::Type{DT}, w::WeightLatticeElem{DT,R}) where {DT<:DynkinType,R}
    C = cartan_matrix(DT)
    orbit = Set{SVector{R,Int}}()
    push!(orbit, w.vec)
    queue = [w.vec]

    while !isempty(queue)
        v = popfirst!(queue)
        for s in 1:R
            # Reflect by s-th simple reflection
            pairing = v[s]
            new_v = MVector{R,Int}(v)
            for j in 1:R
                new_v[j] -= pairing * C[j, s]
            end
            sv = SVector{R,Int}(new_v)
            if sv ∉ orbit
                push!(orbit, sv)
                push!(queue, sv)
            end
        end
    end

    return [WeightLatticeElem{DT,R}(v) for v in orbit]
end

function weyl_orbit(w::WeightLatticeElem{DT,R}) where {DT,R}
    return weyl_orbit(DT, w)
end

# ─── Dominant weights ────────────────────────────────────────────────────────

"""
    dominant_weights(::Type{DT}, hw::WeightLatticeElem{DT,R}) -> Vector{WeightLatticeElem{DT,R}}

Compute the dominant weights occurring in the irreducible representation
with highest weight `hw`, sorted by decreasing height.
"""
function dominant_weights(::Type{DT}, hw::WeightLatticeElem{DT,R}) where {DT<:DynkinType,R}
    @assert is_dominant(hw) "Highest weight must be dominant"
    RS = RootSystem(DT)

    pos_roots_w = [WeightLatticeElem(pr) for pr in positive_roots(RS)]

    result = Set{SVector{R,Int}}([hw.vec])
    todo = [hw]

    while !isempty(todo)
        new_todo = WeightLatticeElem{DT,R}[]
        for w in todo
            for α_w in pos_roots_w
                w_sub = w - α_w
                if is_dominant(w_sub) && w_sub.vec ∉ result
                    push!(result, w_sub.vec)
                    push!(new_todo, w_sub)
                end
            end
        end
        todo = new_todo
    end

    weights = [WeightLatticeElem{DT,R}(v) for v in result]
    sort!(weights; by=w -> -sum(w.vec))
    return weights
end

function dominant_weights(hw::WeightLatticeElem{DT,R}) where {DT,R}
    return dominant_weights(DT, hw)
end

# ─── Dimension of simple module (Weyl dimension formula) ────────────────────

"""
    dim_of_simple_module(::Type{DT}, hw::WeightLatticeElem{DT,R}) -> Int

Compute the dimension of the irreducible representation with highest weight `hw`
using the Weyl dimension formula:

``\\dim V(λ) = \\prod_{α > 0} \\frac{⟨λ + ρ, α⟩}{⟨ρ, α⟩}``
"""
function dim_of_simple_module(::Type{DT}, hw::WeightLatticeElem{DT,R}) where {DT<:DynkinType,R}
    @assert is_dominant(hw) "Highest weight must be dominant"
    RS = RootSystem(DT)
    ρ = weyl_vector(DT)
    hw_plus_ρ = hw + ρ

    num = 1 // 1
    den = 1 // 1

    for α in positive_roots(RS)
        num *= dot(hw_plus_ρ, α)
        den *= dot(ρ, α)
    end

    d = num // den
    @assert denominator(d) == 1 "Weyl dimension formula gave non-integer result"
    return Int(numerator(d))
end

function dim_of_simple_module(hw::WeightLatticeElem{DT,R}) where {DT,R}
    return dim_of_simple_module(DT, hw)
end

# ─── Singularity ─────────────────────────────────────────────────────────────

"""
    is_singular(w::WeightLatticeElem{DT,R}) -> Bool

Check whether the weight `w` is singular, i.e. lies on some wall of a Weyl
chamber. Equivalently, `w` is singular iff `⟨α∨, w⟩ = 0` for some positive
root `α`.

For a dominant weight this simplifies to checking whether any fundamental
weight coordinate is zero. For a general weight, we first conjugate to the
dominant chamber.
"""
function is_singular(w::WeightLatticeElem{DT,R}) where {DT,R}
    dom = conjugate_dominant_weight(w)
    return any(i -> dom.vec[i] == 0, 1:R)
end

# ─── Borel–Weil–Bott ────────────────────────────────────────────────────────

"""
    borel_weil_bott(λ::WeightLatticeElem{DT,R}) -> Union{Nothing, Tuple{Int, WeightLatticeElem{DT,R}}}

Apply the Borel–Weil–Bott theorem to the weight `λ`.

Compute `μ = λ + ρ` and find the unique Weyl group element `w` such that
`w(μ)` is dominant. If `μ` is singular (lies on a Weyl chamber wall),
all cohomology vanishes and we return `nothing`. Otherwise, return
`(d, w(μ) - ρ)` where `d = ℓ(w)` is the cohomological degree, meaning

``H^d(G/B, \\mathcal{L}_λ) \\cong V_{w(μ)-ρ}^*``

and all other cohomology groups vanish.

# Examples
```julia
julia> borel_weil_bott(fundamental_weight(TypeA{2}, 1))
(0, ω1)

julia> borel_weil_bott(WeightLatticeElem(TypeA{2}, [-2, 1]))
(1, 0)

julia> borel_weil_bott(-weyl_vector(TypeA{2}))
# nothing (singular case)
```
"""
function borel_weil_bott(λ::WeightLatticeElem{DT,R}) where {DT,R}
    ρ = weyl_vector(DT)
    μ = λ + ρ

    # Move μ to the dominant chamber; the number of reflections is the degree
    μ_dom, word = conjugate_dominant_weight_with_elem(μ)
    d = length(word)

    # If μ_dom is the zero weight, λ + ρ is singular → no cohomology
    iszero(μ_dom) && return nothing

    return (d, μ_dom - ρ)
end
