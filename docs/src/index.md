```@raw html
<h1><img src="assets/favicon.svg" alt="Semisimple.jl icon" width="32" valign="middle" /> Semisimple.jl</h1>
```

A Julia package for computations with finite-dimensional complex semisimple
Lie algebras via their root data: root systems, Weyl groups, weight lattices,
and highest-weight representation-theoretic operations.
It is heavily optimized and uses Julia's type system to specialize many finite
root-data computations.

```@raw html
<p><a href="Semisimple.jl.pdf">Download the PDF manual</a>.</p>
```

## Features

- **Dynkin types** — Type-level classification (`TypeA{N}`, `TypeB{N}`, …, `TypeG2`, products) with text Dynkin diagrams
- **Cartan matrices** — Compile-time `@generated` Cartan matrices, symmetrisers, bilinear forms
- **Root systems** — Positive roots, coroots, reflection tables (immutable singletons)
- **Weight lattice** — Fundamental weights, Weyl vector, dominance, conjugation
- **Weyl groups** — Reduced words, multiplication via reflection tables, orbits, dimension formula
- **Characters** — Weyl characters (representation ring), Freudenthal formula, Brauer–Klimyk
  tensor products, Littlewood–Richardson (Type A), Adams operators, symmetric/exterior powers

## Installation

Semisimple.jl is registered in the Julia General registry:

```julia
using Pkg
Pkg.add("Semisimple")
```

## Quick start

```jldoctest quickstart
julia> using Semisimple

julia> ω1 = fundamental_weight(TypeA{3}, 1)
ω1

julia> degree(ω1)   # dimension of the standard representation
4

julia> V = WeylCharacter(ω1);

julia> tensor_product(V, V)   # V(ω1) ⊗ V(ω1) = Sym²V ⊕ ⋀²V
A3(2, 0, 0) + A3(0, 1, 0)

julia> Sym(2, V) + ⋀(2, V) == V * V   # Newton identity
true

julia> length(weyl_orbit(TypeA{3}, ω1))
4
```

## Contents

```@docs
Semisimple.Semisimple
```

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
