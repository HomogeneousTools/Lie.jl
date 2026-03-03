# Lie.jl

A Julia package for computations with semisimple Lie algebras: root systems,
Weyl groups, weight lattices, and representation-theoretic operations.
It is heavily optimized, and uses various Julia features making this possible.

Lie.jl is inspired by (and partially ported) from [LiE](http://wwwmathlabo.univ-poitiers.fr/~maavl/LiE/),
a computer algebra system for Lie group computations written in C.
Parts of the Lie.jl code have been written with the assistance of Claude Opus 4.6.

Similar features are also available in [SageMath](https://sagemath.org),
and a similar package is [LieART](https://lieart.hepforge.org/),
which runs on the proprietary [Mathematica](https://www.wolfram.com/mathematica/) software.

Note that Lie.jl is less feature-complete than any of the aforementioned packages.
Porting more features is planned.

## Features

- **Dynkin types** — Type-level classification (`TypeA{N}`, `TypeB{N}`, …, `TypeG2`, products) with text Dynkin diagrams
- **Cartan matrices** — Compile-time `@generated` Cartan matrices, symmetrisers, bilinear forms
- **Root systems** — Positive roots, coroots, reflection tables (immutable singletons)
- **Weight lattice** — Fundamental weights, Weyl vector, dominance, conjugation
- **Weyl groups** — Reduced words, multiplication via reflection tables, orbits, dimension formula
- **Characters** — Weyl characters (representation ring), Freudenthal formula, Brauer–Klimyk
  tensor products, Littlewood–Richardson (Type A), Adams operators, symmetric/exterior powers,
  Borel–Weil–Bott

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/…/Lie.jl")
```

## Quick start

```jldoctest quickstart
julia> using Lie

julia> import Lie: borel_weil_bott

julia> ω₁ = fundamental_weight(TypeA{3}, 1)
ω1

julia> degree(ω₁)   # dimension of the standard representation
4

julia> tensor_product(ω₁, ω₁)   # V(ω₁) ⊗ V(ω₁) = Sym²V ⊕ ⋀²V
A3(2, 0, 0) + A3(0, 1, 0)

julia> length(weyl_orbit(TypeA{3}, ω₁))
4

julia> borel_weil_bott(ω₁)
(0, ω1)
```

## Contents

```@contents
Pages = [
    "types.md",
    "roots.md",
    "weights.md",
    "weyl.md",
    "characters.md",
    "details.md",
]
Depth = 2
```
