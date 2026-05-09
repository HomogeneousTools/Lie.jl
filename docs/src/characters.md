# Characters and representations

A `WeylCharacter` is an element of the representation ring (Grothendieck ring)
of a semisimple Lie algebra. It stores a formal ``\mathbb{Z}``-linear
combination of irreducible representations, indexed by their dominant
highest weights.

## Constructing characters

```jldoctest chars
julia> using Lie, StaticArrays

julia> ω₁ = fundamental_weight(TypeA{3}, 1);

julia> ω₂ = fundamental_weight(TypeA{3}, 2);

julia> V = WeylCharacter(ω₁)   # irreducible V(ω₁)
A3(1, 0, 0)

julia> WeylCharacter(TypeA{3})   # zero character (additive identity)
0

julia> is_effective(V)
true

julia> is_irreducible(V)
true

julia> highest_weight(V)
ω1
```

```@docs
WeylCharacter
is_effective
is_irreducible
highest_weight
```

## Arithmetic

Characters support addition, subtraction, and scalar multiplication:

```jldoctest chars
julia> V₂ = WeylCharacter(ω₂);

julia> V + V₂
A3(1, 0, 0) + A3(0, 1, 0)

julia> 2 * V
2*A3(1, 0, 0)

julia> V + V == 2 * V
true
```

Characters also support multiplication (tensor product) and exponentiation
(tensor power):

```jldoctest chars
julia> V * V₂ == tensor_product(V, V₂)
true

julia> V^3 == V * V * V
true
```

```@docs
Base.:*(::WeylCharacter{DT,R}, ::WeylCharacter{DT,R}) where {DT,R}
Base.:^(::WeylCharacter{DT,R}, ::Integer) where {DT,R}
```

### In-place mutation

For performance-critical loops, use `add!` and `addmul!`:

```jldoctest chars
julia> result = WeylCharacter(TypeA{3});

julia> add!(result, V);

julia> result
A3(1, 0, 0)

julia> addmul!(result, V₂, 3);

julia> result
A3(1, 0, 0) + 3*A3(0, 1, 0)
```

```@docs
add!
addmul!
```

## Character data terminology

Lie.jl uses three related but distinct character representations:

| Name | Julia representation | Meaning |
|---|---|---|
| `WeylCharacter` | `Dict{WeightLatticeElem,Int}` in the `terms` field | irreducible decomposition in the representation ring |
| Full weight character | `Dict{SVector{R,Int},Int}` from [`freudenthal_formula`](@ref) | multiplicity of every weight |
| Dominant character | `Dict{SVector{R,Int},Int}` from [`dominant_character`](@ref) | multiplicity on dominant representatives of Weyl orbits |

## Character polynomials

A **character polynomial** of a finite-dimensional representation is a formal sum

$$\chi(\lambda) = \sum_{\mu \in P} m_\lambda(\mu) e^\mu$$

where $P$ is the weight lattice, $m_\lambda(\mu)$ is the multiplicity of weight $\mu$ in the representation with highest weight $\lambda$, and $e^\mu$ is a formal exponential. This encodes the weight multiplicities in a single object that respects the representation ring structure (tensor product → multiplication of characters).

Since character polynomials are **Weyl group invariant** (i.e., $\chi_\lambda(w \cdot \mu) = \chi_\lambda(\mu)$ for all $w \in W$), the full character is determined by its values on dominant weights. More precisely: each weight orbit under the Weyl group contains exactly one dominant weight, and all weights in an orbit have equal multiplicity.

```jldoctest chars
julia> ω₁_a2 = fundamental_weight(TypeA{2}, 1);

julia> full_mults = freudenthal_formula(ω₁_a2);

julia> length(full_mults)   # includes all W-orbit members
3

julia> dom_mults = dominant_character(ω₁_a2);

julia> length(dom_mults)   # only dominant weights
1
```

## Dominant character polynomials

The **dominant character polynomial** (or sometimes just "dominant character") is a more compact representation:

$$\chi_{\lambda, \text{dom}} = \sum_{\mu \in P^+ : m_\lambda(\mu) > 0} m_\lambda(\mu) e^\mu$$

where $P^+$ is the set of dominant weights. By W-invariance, this omits all non-dominant weights while retaining *all information* about the full character.

Explicitly: since each W-orbit contains a unique dominant representative, reconstructing the full character from the dominant character is straightforward — just apply the Weyl group orbit operator. This is what [`freudenthal_formula`](@ref) does internally.

`dominant_character` computes this using Freudenthal's recursion formula and returns a
`Dict{SVector{R,Int}, Int}` mapping dominant weight coordinates to multiplicities. This is the cached building block behind [`freudenthal_formula`](@ref), [`weight_multiplicity`](@ref), tensor products, Adams operators, and plethysms.

The relationship to the LiE computer algebra system: the "dominant character polynomial" used here corresponds to the output of the `domchar` command in LiE, which filters the full character to list only the dominant weights and uses the Weyl group orbit operator to go back and forth.

```jldoctest chars
julia> ω₁_a2 = fundamental_weight(TypeA{2}, 1);

julia> dc = dominant_character(ω₁_a2);

julia> length(dc)   # only 1 dominant weight for V(ω₁) of A₂
1

julia> dc[SVector(1, 0)]  # the highest weight itself
1
```

```@docs
dominant_character
```


## Freudenthal's formula

Compute the full weight multiplicity dictionary of an irreducible
representation ``\mathrm{V}(\lambda)``. Returns a
`Dict{SVector{R,Int}, Int}` mapping weight coordinates (in the
fundamental weight basis) to their multiplicities:

```jldoctest chars
julia> m = freudenthal_formula(ω₁);

julia> length(m)   # V(ω₁) of A₃ has 4 weights
4

julia> sum(values(m)) == degree(ω₁)   # total = dimension
true
```

The convenience function `weight_multiplicity` returns the multiplicity
of a single weight.  For example, the ``\mathrm{A}_2`` adjoint
representation ``\mathrm{V}(\omega_1 + \omega_2)`` has dimension 8, and
the zero weight has multiplicity 2:

```jldoctest chars
julia> adj = fundamental_weight(TypeA{2}, 1) + fundamental_weight(TypeA{2}, 2);

julia> weight_multiplicity(adj, zero(adj))
2
```

```@docs
freudenthal_formula
weight_multiplicity
```

## Tensor products

### Brauer–Klimyk (general)

The default tensor product algorithm works for all types:

```jldoctest chars
julia> tensor_product(ω₁, ω₁)   # V ⊗ V = Sym²V ⊕ ⋀²V
A3(2, 0, 0) + A3(0, 1, 0)

julia> tensor_product(ω₁, ω₂)
A3(1, 1, 0) + A3(0, 0, 1)
```

Tensor product of a weight with the trivial representation:

```jldoctest chars
julia> ω₃ = fundamental_weight(TypeA{3}, 3);

julia> tensor_product(ω₁, ω₃)   # V(ω₁) ⊗ V(ω₃) contains trivial
A3(1, 0, 1) + A3(0, 0, 0)
```

### Littlewood–Richardson (type A)

For type A, the Littlewood–Richardson rule provides a faster
tensor product algorithm.  It is used automatically by
`tensor_product` for type A inputs:

```jldoctest chars
julia> ω₁_a4 = fundamental_weight(TypeA{4}, 1);

julia> ω₂_a4 = fundamental_weight(TypeA{4}, 2);

julia> lr_tensor_product(ω₁_a4, ω₂_a4)
A4(1, 1, 0, 0) + A4(0, 0, 1, 0)
```

```jldoctest chars
julia> λ = WeightLatticeElem(TypeA{4}, [2, 1, 0, 0]);

julia> r = tensor_product(λ, λ);

julia> length(r.terms)   # number of irreducible components
11

julia> sum(m * degree(μ) for (μ, m) in r.terms) == degree(λ)^2
true
```

```@docs
tensor_product
lr_tensor_product
```

## Duality

The dual (contragredient) representation:

```jldoctest chars
julia> dual(WeylCharacter(ω₁))
A3(0, 0, 1)

julia> dual(WeylCharacter(ω₃))
A3(1, 0, 0)

julia> dual(WeylCharacter(ω₂))   # self-dual for A₃
A3(0, 1, 0)
```

```@docs
dual
```

## Adams operators

The Adams operator ``\psi^k`` scales every weight of ``\mathrm{V}(\lambda)``
by ``k``. The result is returned as a `Dict{SVector{R,Int}, Int}` mapping
weight coordinates to multiplicities (not decomposed into irreducibles).
Use [`character_from_weights`](@ref) to convert it to a `WeylCharacter`:

```jldoctest chars
julia> ω₁_a2 = fundamental_weight(TypeA{2}, 1);

julia> ψ₂ = adams_operator(ω₁_a2, 2);

julia> length(ψ₂)   # number of distinct weights
3
```

The Newton identity connects Adams operators to symmetric/exterior powers:
``\psi^2(\mathrm{V}) = \mathrm{Sym}^2(\mathrm{V}) - \bigwedge^2(\mathrm{V})``:

```jldoctest chars
julia> ψ₂_char = character_from_weights(TypeA{2}, ψ₂);

julia> ψ₂_char == Sym(2, ω₁_a2) - ⋀(2, ω₁_a2)
true
```

```@docs
adams_operator
character_from_weights
```

## Exterior powers

The ``k``-th exterior power ``\bigwedge^k \mathrm{V}`` is computed
via the **Newton–Girard recurrence**, which relates exterior powers to
Adams operators (power-sum symmetric functions):

```math
k \cdot \bigwedge\nolimits^k(\mathrm{V}) = \sum_{r=1}^{k} (-1)^{r-1}\, \psi^r(\mathrm{V}) \cdot \bigwedge\nolimits^{k-r}(\mathrm{V})
```

This is the representation-ring analogue of the classical identity
``k \, e_k = \sum_{r=1}^{k} (-1)^{r-1} p_r \, e_{k-r}``
relating elementary symmetric polynomials ``e_k`` to power-sum
polynomials ``p_r``.

Both `WeightLatticeElem` and `WeylCharacter` are accepted:

```jldoctest chars
julia> ⋀(2, V)   # ⋀²V(ω₁) of A₃ = V(ω₂)
A3(0, 1, 0)

julia> ⋀(3, V)   # ⋀³V(ω₁) of A₃ = V(ω₃)
A3(0, 0, 1)

julia> ⋀(4, V)   # top exterior power = trivial (det)
A3(0, 0, 0)

julia> ⋀(5, V)   # exceeds dim = 0
0
```

### Dimension identity

``\dim \bigwedge^k \mathrm{V} = \binom{\dim \mathrm{V}}{k}``:

```jldoctest chars
julia> r = ⋀(2, V);

julia> sum(m * degree(μ) for (μ, m) in r.terms) == binomial(4, 2)
true
```

### Type A fundamental representations

For ``\mathrm{A}_n``, ``\bigwedge^k \mathrm{V}(\omega_1) = \mathrm{V}(\omega_k)``:

```jldoctest chars
julia> ω = [fundamental_weight(TypeA{5}, i) for i in 1:5];

julia> all(⋀(k, WeylCharacter(ω[1])) == WeylCharacter(ω[k]) for k in 1:5)
true
```

### Non-minuscule exterior powers

```jldoctest chars
julia> adj = ω₁ + ω₃;   # adjoint of A₃ (15-dim)

julia> r = ⋀(2, adj);

julia> is_effective(r)
true

julia> sum(m * degree(μ) for (μ, m) in r.terms) == binomial(15, 2)
true
```

```@docs
exterior_power
⋀
```

## Symmetric powers

The ``k``-th symmetric power ``\mathrm{Sym}^k \mathrm{V}`` is
likewise computed via the **Newton–Girard recurrence**:

```math
k \cdot \mathrm{Sym}^k(\mathrm{V}) = \sum_{r=1}^{k} \psi^r(\mathrm{V}) \cdot \mathrm{Sym}^{k-r}(\mathrm{V})
```

This is the representation-ring analogue of the identity
``k \, h_k = \sum_{r=1}^{k} p_r \, h_{k-r}``
for complete homogeneous symmetric polynomials ``h_k``.

Both `WeightLatticeElem` and `WeylCharacter` are accepted:

```jldoctest chars
julia> Sym(2, V)   # Sym² of std rep of A₃
A3(2, 0, 0)

julia> Sym(0, V)   # Sym⁰ = trivial
A3(0, 0, 0)

julia> Sym(1, V)   # Sym¹ = identity
A3(1, 0, 0)
```

### Type A: always irreducible for ω₁

For ``\mathrm{A}_n``, ``\mathrm{Sym}^k \mathrm{V}(\omega_1) = \mathrm{V}(k\omega_1)``:

```jldoctest chars
julia> all(Sym(k, V) == WeylCharacter(k * ω₁) for k in 1:5)
true
```

### Dimension identity

``\dim \mathrm{Sym}^k \mathrm{V} = \binom{\dim \mathrm{V} + k - 1}{k}``:

```jldoctest chars
julia> r = Sym(3, V);

julia> sum(m * degree(μ) for (μ, m) in r.terms) == binomial(4 + 2, 3)
true
```

### Newton identity: V ⊗ V = Sym²V ⊕ ⋀²V

```jldoctest chars
julia> V * V == Sym(2, V) + ⋀(2, V)
true

julia> ω₁_g2 = fundamental_weight(TypeG2, 1);

julia> W_g2 = WeylCharacter(ω₁_g2);

julia> W_g2 * W_g2 == Sym(2, W_g2) + ⋀(2, W_g2)
true
```

### Reducible character example

`Sym` and `⋀` also accept general `WeylCharacter` values (not just
irreducibles). Scaling ``V \mapsto 2V`` adds a copy, so:
``\mathrm{Sym}^2(2V) = 3\,\mathrm{Sym}^2(V) \oplus \bigwedge^2(V)``:

```jldoctest chars
julia> symmetric_power(2 * V, 2) == 3 * Sym(2, V) + ⋀(2, V)
true
```

```@docs
symmetric_power
Sym
```

## Plethysm (Schur functors)

The **plethysm** ``s_\lambda(\mathrm{V}(\mu))`` applies the Schur functor
associated to a partition ``\lambda \vdash n`` to an irreducible
representation ``\mathrm{V}(\mu)``.  Symmetric and exterior powers are
special cases:

- ``s_{(n)} = \mathrm{Sym}^n``
- ``s_{(1^n)} = \bigwedge^n``

The general formula uses the **Murnaghan–Nakayama rule** for ``S_n``
characters and **Adams operators** (power-sum symmetric functions):

```math
s_\lambda(\mathrm{V}) = \frac{1}{n!} \sum_{\kappa \vdash n}
\chi^\lambda(\kappa) \cdot |\mathrm{Cl}(\kappa)| \cdot
\psi^{\kappa_1}(\mathrm{V}) \otimes \cdots \otimes \psi^{\kappa_m}(\mathrm{V})
```

```jldoctest chars
julia> plethysm([2], ω₁) == Sym(2, V)   # one-row partition = Sym
true

julia> plethysm([1, 1], ω₁) == ⋀(2, V)  # one-column partition = ⋀
true

julia> plethysm([2, 1], ω₁)               # mixed symmetry S_{(2,1)}
A3(1, 1, 0)

julia> degree(plethysm([2, 1], ω₁))       # mixed-symmetry representation of A₃
20
```

### Plethysm on non-simply-laced types

```jldoctest chars
julia> plethysm([2], ω₁_g2) == Sym(2, W_g2)
true

julia> plethysm([1, 1], ω₁_g2) == ⋀(2, W_g2)
true
```

```@docs
plethysm
```

## Reconstructing characters from weights

Given a weight multiplicity dictionary (e.g. from an Adams operator),
recover the Weyl character decomposition:

```jldoctest chars
julia> m = Dict(SVector(1, 0, 0) => 1, SVector(-1, 1, 0) => 1,
                SVector(0, -1, 1) => 1, SVector(0, 0, -1) => 1);

julia> V_rec = character_from_weights(TypeA{3}, m);

julia> is_irreducible(V_rec)
true

julia> highest_weight(V_rec) == ω₁
true
```

## Cross-type examples

### B₃: spin representation

```jldoctest chars
julia> ω₃_b3 = fundamental_weight(TypeB{3}, 3);

julia> degree(ω₃_b3)   # 8-dim spin rep
8

julia> spin = WeylCharacter(ω₃_b3);

julia> r = ⋀(2, spin);

julia> sum(m * degree(μ) for (μ, m) in r.terms) == binomial(8, 2)
true
```

### G₂: 7-dimensional representation

```jldoctest chars
julia> degree(ω₁_g2)
7

julia> r = Sym(2, W_g2);

julia> is_effective(r)
true

julia> sum(m * degree(μ) for (μ, m) in r.terms) == binomial(7 + 1, 2)
true
```

### E₈: adjoint representation

```jldoctest chars
julia> ω₈_e8 = fundamental_weight(TypeE{8}, 8);

julia> degree(ω₈_e8)   # 248-dim adjoint
248

julia> r = ⋀(2, WeylCharacter(ω₈_e8));

julia> length(r.terms)   # 2 irreducible components
2
```

## Representation invariants

Lie.jl provides several classical invariants attached to irreducible
representations.

### Dynkin index

The **Dynkin index** of ``\mathrm{V}(λ)`` is the proportionality constant
between the trace form on ``\mathrm{V}(λ)`` and the Killing form:

```math
I(λ) = \frac{\dim \mathrm{V}(λ) \cdot (λ,\, λ + 2ρ)}{2\,\dim \mathfrak{g}}
```

where the inner product is normalized so that long roots have squared
length&nbsp;2. For the adjoint representation, ``I(θ) = h^∨`` (the dual
Coxeter number).

```jldoctest chars
julia> dynkin_index(fundamental_weight(TypeA{3}, 1))
1//2

julia> adj = adjoint_representation(TypeA{3});

julia> dynkin_index(highest_weight(adj)) == dual_coxeter_number(TypeA{3})
true
```

### Casimir eigenvalue

The eigenvalue of the quadratic Casimir element on ``\mathrm{V}(λ)`` is
``c(λ) = 2(λ, λ + 2ρ)/(\theta,\theta)``, where ``\theta`` is the highest root.
This normalizes long roots to have squared length 2:

```jldoctest chars
julia> casimir_eigenvalue(fundamental_weight(TypeA{3}, 1))
15//4

julia> casimir_eigenvalue(fundamental_weight(TypeB{3}, 3))
21//4
```

### Congruency class

```jldoctest chars
julia> congruency_class(fundamental_weight(TypeA{3}, 1))
1

julia> congruency_class(fundamental_weight(TypeD{4}, 3))
(1, 0)
```

### Self-duality and Frobenius–Schur indicator

```jldoctest chars
julia> is_self_dual(fundamental_weight(TypeB{3}, 1))
true

julia> frobenius_schur_indicator(fundamental_weight(TypeB{3}, 1))
1

julia> frobenius_schur_indicator(fundamental_weight(TypeC{2}, 1))
-1

julia> frobenius_schur_indicator(fundamental_weight(TypeA{2}, 1))
0
```

### Adjoint representation

```jldoctest chars
julia> adj = adjoint_representation(TypeE{8});

julia> degree(adj)
248
```

```@docs
dynkin_index
casimir_eigenvalue
congruency_class
is_self_dual
frobenius_schur_indicator
adjoint_representation
```
