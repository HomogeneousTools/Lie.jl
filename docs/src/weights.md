# Weight lattice

Weights are elements of the weight lattice, expressed in the
**fundamental weight basis** ``(\omega_1, \ldots, \omega_r)``
where ``\langle \alpha_i^\vee, \omega_j \rangle = \delta_{ij}``.

## Creating weights

Weights are constructed with `fundamental_weight` or directly
from a coefficient vector using `WeightLatticeElem`:

```jldoctest weights
julia> using Semisimple

julia> ω1 = fundamental_weight(TypeA{3}, 1)
ω1

julia> ω2 = fundamental_weight(TypeA{3}, 2)
ω2

julia> ω3 = fundamental_weight(TypeA{3}, 3)
ω3

julia> ω1 + ω2
ω1 + ω2

julia> 2 * ω1
2ω1

julia> WeightLatticeElem(TypeA{3}, [3, 1, 0])
3ω1 + ω2
```

### All fundamental weights

```jldoctest weights
julia> fundamental_weights(TypeA{3})
3-element Vector{WeightLatticeElem{TypeA{3}, 3}}:
 ω1
 ω2
 ω3
```

### Weyl vector

The Weyl vector ``\rho = \omega_1 + \cdots + \omega_r``:

```jldoctest weights
julia> weyl_vector(TypeA{3})
ω1 + ω2 + ω3
```

```@docs
WeightLatticeElem
fundamental_weight
fundamental_weights
weyl_vector
```

## Display

Weights are printed in the fundamental weight basis by default:
`ω1`, `2ω1 + ω2`, `0`, etc.

### Per-call compact format

Pass `:compact => true` via `IOContext` to switch a single `show` call to the
concise coordinate form `DT[c₁,c₂,…]`:

```jldoctest weights
julia> show(IOContext(stdout, :compact => true), ω1)
A3[1,0,0]

julia> show(IOContext(stdout, :compact => true), 2*ω1 - ω3)
A3[2,0,-1]
```

The same compact form is used by [`RootSpaceElem`](@ref).

### Global compact toggle

Call [`compact_display!`](@ref) to make the compact form the session-wide
default for all `WeightLatticeElem` and `RootSpaceElem` output:

```jldoctest weights
julia> compact_display!(true)
true

julia> fundamental_weights(TypeA{3})
3-element Vector{WeightLatticeElem{TypeA{3}, 3}}:
 A3[1,0,0]
 A3[0,1,0]
 A3[0,0,1]

julia> compact_display!(false)   # restore default
false
```

```@docs
compact_display!
```

## Dominance

A weight is **dominant** when all its fundamental weight coordinates
are non-negative:

```jldoctest weights
julia> is_dominant(ω1)
true

julia> is_dominant(ω1 - 2 * ω2)
false
```

### Conjugation to the dominant chamber

Every weight is Weyl-conjugate to a unique dominant weight:

```jldoctest weights
julia> w = WeightLatticeElem(TypeA{3}, [-1, 2, 0]);

julia> is_dominant(w)
false

julia> conjugate_dominant_weight(w)
ω1 + ω2
```

```@docs
is_dominant
conjugate_dominant_weight
conjugate_dominant_weight_with_elem
conjugate_dominant_weight_with_length
```

## Reflections

Simple reflections act on weights by the formula
``s_i(\lambda) = \lambda - \langle \alpha_i^\vee, \lambda \rangle \alpha_i``,
which in the fundamental weight basis simplifies to
``(s_i(\lambda))_j = \lambda_j - \lambda_i C_{ji}``, because
``\alpha_i = \sum_j C_{ji}\omega_j``:

```jldoctest weights
julia> reflect(ω1, 1)   # reflection in the first simple root
-ω1 + ω2

julia> reflect(ω1, 2)   # unchanged because the pairing is zero
ω1
```

```@docs
reflect
```

## Inner products

Pairing of roots and weights, ``\langle \alpha^\vee, \lambda \rangle``,
and the weight-space inner product:

```jldoctest weights
julia> RS = RootSystem(TypeA{2});

julia> α1 = simple_root(RS, 1);

julia> ω1 = fundamental_weight(TypeA{2}, 1);

julia> ω2 = fundamental_weight(TypeA{2}, 2);

julia> dot(α1, ω1)   # simple-coroot pairing equals 1
1//1

julia> dot(α1, ω2)   # simple-coroot pairing equals 0
0//1

julia> dot(ω1, ω1)   # invariant bilinear form
2//3

julia> dot(ω1, ω2)
1//3
```

```@docs
dot
```

## Conversions

Weights and roots live in different coordinate systems.
Convert between them:

```jldoctest weights
julia> α1_as_weight = WeightLatticeElem(simple_root(RS, 1))
2ω1 - ω2

julia> ρ_as_root = RootSpaceElem(weyl_vector(TypeA{2}))
α1 + α2
```

Every weight lies in the rational span of the simple roots via ``C^{-1}``.
It lies in the root lattice only when those rational simple-root coordinates
are integral; otherwise `RootSpaceElem(w)` throws an `ArgumentError`.
