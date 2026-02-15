# Lie.jl

A Julia package for computations with semisimple Lie algebras: root systems,
Weyl groups, weight lattices, and representation-theoretic operations.

## Features

| Module | Highlights |
|---|---|
| **DynkinTypes** | Type-level classification `TypeA{N}`, …, `TypeG2`, `ProductDynkinType` |
| **CartanMatrix** | Compile-time `@generated` Cartan matrices, symmetrizers, bilinear forms |
| **RootSystem** | Positive roots, coroots, reflection tables (immutable singletons) |
| **WeightLattice** | Fundamental weights, Weyl vector, dominance, conjugation to dominant chamber |
| **WeylGroup** | Reduced words, multiplication via reflection tables, orbits, Weyl dimension formula, Borel–Weil–Bott |
| **Characters** | `WeylCharacter` (representation ring), Freudenthal formula, Brauer–Klimyk tensor products, Adams operators, symmetric/exterior powers |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/YourOrg/Lie.jl")  # adjust URL
```

Or for local development:

```julia
julia --project=.
```

## Quick start

```julia
using Lie

# Fundamental weights of A₃
ω₁ = fundamental_weight(TypeA{3}, 1)
ω₂ = fundamental_weight(TypeA{3}, 2)

# Dimension of the standard representation
degree(ω₁)   # 4

# Tensor product decomposition: V(ω₁) ⊗ V(ω₁) = Sym²V ⊕ ⋀²V
tensor_product(ω₁, ω₁)

# Symmetric and exterior powers
Sym(3, ω₁)    # Sym³(standard rep)
⋀(2, ω₁)      # ⋀²(standard rep) = V(ω₂)

# Weyl orbit and dimension
length(weyl_orbit(TypeA{3}, ω₁))   # 4
weyl_order(TypeA{3})                # 24

# Borel–Weil–Bott theorem
borel_weil_bott(ω₁)   # (0, ω₁) — degree 0, weight ω₁

# Virtual (Weyl) characters and the representation ring
V = WeylCharacter(ω₁)
V * V                    # tensor product as ring multiplication
dual(V)                  # contragredient
is_effective(V - V)      # false (zero character, but not a "negative" rep)
```

## Running tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Running doctests

Doctests are embedded in docstrings throughout the source. Run them with:

```bash
julia --project=. -e 'using Documenter, Lie; doctest(Lie)'
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
9 categories (w₀ action, conjugation, Weyl orbits, dimension formula,
Freudenthal, tensor products, exterior/symmetric powers, Borel–Weil–Bott,
infrastructure). Results are saved as JSON for regression tracking.

## Dependencies

- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) — fixed-size arrays for compile-time root/weight data
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (stdlib)
- [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) — benchmarking
- [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) — documentation and doctests

Julia ≥ 1.9 required.

## License

See [LICENSE](LICENSE) for details.
