# Weight lattice

Weights are elements of the weight lattice, expressed in the
**fundamental weight basis** ``(\omega_1, \ldots, \omega_r)``
where ``\langle \alpha_i^\vee, \omega_j \rangle = \delta_{ij}``.

## Creating weights

Weights are constructed with `fundamental_weight` or directly
from a coefficient vector using `WeightLatticeElem`:

```jldoctest weights
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{3}, 1)
ω1

julia> ω₂ = fundamental_weight(TypeA{3}, 2)
ω2

julia> ω₃ = fundamental_weight(TypeA{3}, 3)
ω3

julia> ω₁ + ω₂
ω1 + ω2

julia> 2 * ω₁
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

## Dominance

A weight is **dominant** when all its fundamental weight coordinates
are non-negative:

```jldoctest weights
julia> is_dominant(ω₁)
true

julia> is_dominant(ω₁ - 2 * ω₂)
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
``s_i(\lambda)_j = \lambda_j - C_{ij} \lambda_i``:

```jldoctest weights
julia> reflect(ω₁, 1)   # s₁(ω₁) = ω₁ - ⟨α₁∨, ω₁⟩ α₁ = -ω₁ + ω₂
-ω1 + ω2

julia> reflect(ω₁, 2)   # s₂(ω₁) = ω₁ (orthogonal)
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

julia> α₁ = simple_root(RS, 1);

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> ω₂ = fundamental_weight(TypeA{2}, 2);

julia> dot(α₁, ω₁)   # ⟨α₁∨, ω₁⟩ = 1
1//1

julia> dot(α₁, ω₂)   # ⟨α₁∨, ω₂⟩ = 0
0//1

julia> dot(ω₁, ω₁)   # (ω₁, ω₁) in the Killing form
2//3

julia> dot(ω₁, ω₂)
1//3
```

```@docs
dot
```

## Conversions

Weights and roots live in different coordinate systems.
Convert between them:

```jldoctest weights
julia> α₁_as_weight = WeightLatticeElem(simple_root(RS, 1))
2ω1 - ω2

julia> ρ_as_root = RootSpaceElem(weyl_vector(TypeA{2}))
α1 + α2
```

For type ``\mathrm{A}_n``, every fundamental weight is in the root lattice
(with rational coefficients via ``C^{-1}``).
