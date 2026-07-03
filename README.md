<h1><img src="docs/src/assets/favicon.svg" alt="Semisimple.jl icon" width="32" valign="middle"> Semisimple.jl</h1>

[![tests](https://github.com/HomogeneousTools/Semisimple.jl/actions/workflows/tests.yml/badge.svg)](https://github.com/HomogeneousTools/Semisimple.jl/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/HomogeneousTools/Semisimple.jl/graph/badge.svg)](https://codecov.io/gh/HomogeneousTools/Semisimple.jl)
[![Docs](https://img.shields.io/badge/docs-homogeneous.tools/Semisimple.jl-blue)](https://homogeneous.tools/Semisimple.jl/)
[![Release](https://img.shields.io/github/v/release/HomogeneousTools/Semisimple.jl?color=green)](https://github.com/HomogeneousTools/Semisimple.jl/releases)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia package for computations with representations of
finite-dimensional complex semisimple Lie algebras
via their root data: root systems, Weyl groups, weight lattices,
and highest-weight representation-theoretic operations.

## Features

| Module | Highlights |
|---|---|
| **DynkinTypes** | Type-level classification `TypeA{N}`, …, `TypeG2`, `ProductDynkinType` |
| **CartanMatrix** | Compile-time `@generated` Cartan matrices, symmetrizers, bilinear forms |
| **RootSystem** | Positive roots, coroots, reflection tables (immutable singletons) |
| **WeightLattice** | Fundamental weights, Weyl vector, dominance, conjugation to dominant chamber |
| **WeylGroup** | Reduced words, multiplication via reflection tables, orbits, Weyl dimension formula, Borel–Weil–Bott |
| **Characters** | `WeylCharacter` (representation ring), Freudenthal formula, Brauer–Klimyk tensor products, Adams operators, symmetric/exterior powers |

Semisimple.jl is a finite-type root-data and highest-weight package. It does not
construct concrete Lie algebra elements, brackets, Chevalley bases, ideals,
subalgebras, homomorphisms, arbitrary-field Lie algebras, or module
homomorphisms.

## Relationship to OSCAR

Semisimple.jl overlaps partly with OSCAR's [stable Lie Theory](https://docs.oscar-system.org/stable/LieTheory/intro/)
module, but the emphasis is different. OSCAR stable provides intentionally
minimal combinatorial scaffolding: Cartan matrices, root systems, Weyl groups,
and weight lattices, represented with OSCAR/AbstractAlgebra parent objects and
integer matrices. Semisimple.jl focuses on finite-type complex semisimple root data
with type-level Dynkin types, `StaticArrays`-based weights and roots, optimized
Weyl orbit traversal, and highest-weight representation-ring computations.

OSCAR's [experimental Lie Algebras](https://docs.oscar-system.org/stable/Experimental/LieAlgebras/introduction/)
module is broader on the algebraic side: it has concrete finite-dimensional Lie
algebra objects, brackets, ideals, subalgebras, homomorphisms, modules, and
module homomorphisms. That module is explicitly experimental, so its API carries
stability caveats. Use OSCAR when you need integrated algebraic objects; use
Semisimple.jl when you need lightweight, optimized highest-weight and character
computations.

## Related software

Semisimple.jl is inspired by (and partially ported from) [LiE](http://wwwmathlabo.univ-poitiers.fr/~maavl/LiE/),
a computer algebra system for Lie group computations written in C, and by similar
functionality available in [SageMath](https://sagemath.org) and [LieART](https://lieart.hepforge.org/)
(the latter targeting [Mathematica](https://www.wolfram.com/mathematica/)).
Note that Semisimple.jl is less feature-complete than any of these; porting more features is planned.

## Installation

Semisimple.jl is registered in the Julia General registry:

```julia
using Pkg
Pkg.add("Semisimple")
```

Or for local development:

```julia
julia --project=.
```

## Quick start

```julia
using Semisimple

# Fundamental weights of A₃
ω1 = fundamental_weight(TypeA{3}, 1)
ω2 = fundamental_weight(TypeA{3}, 2)

# Dimension of the standard representation
degree(ω1)   # 4

# Tensor product decomposition: V(ω1) ⊗ V(ω1) = Sym²V ⊕ ⋀²V
tensor_product(ω1, ω2)

# Symmetric and exterior powers
Sym(3, ω1)    # Sym³(standard rep)
⋀(2, ω1)      # ⋀²(standard rep) = V(ω2)

# Weyl orbit and dimension
length(weyl_orbit(TypeA{3}, ω1))   # 4
weyl_order(TypeA{3})               # 24

# Virtual (Weyl) characters and the representation ring
V = WeylCharacter(ω1)
2V * V                   # tensor product as ring multiplication
dual(V)                  # contragredient
```

## Running tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Coverage for the default Julia 1 CI run is uploaded automatically to
[Codecov](https://codecov.io/gh/HomogeneousTools/Semisimple.jl) by the `codecov` job in
`.github/workflows/tests.yml`.

## Running doctests

Doctests are embedded in docstrings throughout the source. Run them with:

```bash
julia --project=. -e 'using Documenter, Semisimple; doctest(Semisimple)'
```

Or via the docs build:

```bash
julia --project=docs/ docs/make.jl
```

## Running benchmarks

```bash
# Run benchmarks and save results to benchmark/results/<timestamp>.json
julia --project=. benchmark/bench.jl

# Run and compare against the previous saved baseline
julia --project=. benchmark/bench.jl --compare
```

Benchmark results track minimum time, allocation count, and memory usage across
explicit warm and cold benchmark categories. Results are saved as JSON for
regression tracking.

## Dependencies

- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) — fixed-size arrays for compile-time root/weight data
- [LRUCache.jl](https://github.com/JuliaCollections/LRUCache.jl) — bounded LRU caches for memoised computations
- [PrecompileTools.jl](https://github.com/JuliaLang/PrecompileTools.jl) — precompilation workloads for fast first-call latency
- [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl) — persistent per-environment configuration
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (stdlib)
- [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) — benchmarking (test/benchmark only)
- [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) — documentation and doctests (docs only)

Julia ≥ 1.9 required.

## License

See [LICENSE](LICENSE) for details.
