# Lie.jl

A Julia package for computations with finite-dimensional complex semisimple
Lie algebras via their root data: root systems, Weyl groups, weight lattices,
and highest-weight representation-theoretic operations.
It is heavily optimized and uses Julia's type system to specialize many finite
root-data computations.

```@raw html
<p><a href="Lie.jl.pdf">Download the PDF manual</a>.</p>
```

Lie.jl is a finite-type root-data and highest-weight package. It does not
construct concrete Lie algebra elements, brackets, Chevalley bases, ideals,
subalgebras, homomorphisms, arbitrary-field Lie algebras, or module
homomorphisms.

Lie.jl is inspired by (and partially ported) from [LiE](http://wwwmathlabo.univ-poitiers.fr/~maavl/LiE/),
a computer algebra system for Lie group computations written in C.
Parts of the Lie.jl code have been written with the assistance of Claude Opus 4.6.

Similar features are also available in [SageMath](https://sagemath.org),
and a similar package is [LieART](https://lieart.hepforge.org/),
which runs on the proprietary [Mathematica](https://www.wolfram.com/mathematica/) software.

Note that Lie.jl is less feature-complete than any of the aforementioned packages.
Porting more features is planned.

## Relationship to OSCAR

Lie.jl overlaps partly with OSCAR's [stable Lie Theory](https://docs.oscar-system.org/stable/LieTheory/intro/)
module, but the emphasis is different. OSCAR stable provides intentionally
minimal combinatorial scaffolding: Cartan matrices, root systems, Weyl groups,
and weight lattices, represented with OSCAR/AbstractAlgebra parent objects and
integer matrices. Lie.jl focuses on finite-type complex semisimple root data
with type-level Dynkin types, `StaticArrays`-based weights and roots, optimized
Weyl orbit traversal, and highest-weight representation-ring computations.

OSCAR's [experimental Lie Algebras](https://docs.oscar-system.org/stable/Experimental/LieAlgebras/introduction/)
module is broader on the algebraic side: it has concrete finite-dimensional Lie
algebra objects, brackets, ideals, subalgebras, homomorphisms, modules, and
module homomorphisms. That module is explicitly experimental, so its API carries
stability caveats. Use OSCAR when you need integrated algebraic objects; use
Lie.jl when you need lightweight, optimized highest-weight and character
computations.

## Features

- **Dynkin types** — Type-level classification (`TypeA{N}`, `TypeB{N}`, …, `TypeG2`, products) with text Dynkin diagrams
- **Cartan matrices** — Compile-time `@generated` Cartan matrices, symmetrisers, bilinear forms
- **Root systems** — Positive roots, coroots, reflection tables (immutable singletons)
- **Weight lattice** — Fundamental weights, Weyl vector, dominance, conjugation
- **Weyl groups** — Reduced words, multiplication via reflection tables, orbits, dimension formula
- **Characters** — Weyl characters (representation ring), Freudenthal formula, Brauer–Klimyk
  tensor products, Littlewood–Richardson (Type A), Adams operators, symmetric/exterior powers

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/HomogeneousTools/Lie.jl")
```

## Quick start

```jldoctest quickstart
julia> using Lie

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
Lie.Lie
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
