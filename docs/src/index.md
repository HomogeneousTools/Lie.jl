# Lie.jl

A Julia package for computations with semisimple Lie algebras: root systems,
Weyl groups, weight lattices, and representation-theoretic operations.

## Features

- **Dynkin types**: type-level classification (`TypeA{N}`, `TypeB{N}`, …, `TypeG2`, products)
- **Cartan matrices**: compile-time `@generated` Cartan matrices, symmetrizers, bilinear forms
- **Root systems**: positive roots, coroots, reflection tables (immutable singletons)
- **Weight lattice**: fundamental weights, Weyl vector, dominance, conjugation
- **Weyl groups**: reduced words, multiplication via reflection tables, orbits, dimension formula
- **Characters**: Weyl characters (representation ring), Freudenthal formula, Brauer–Klimyk
  tensor products, Adams operators, symmetric/exterior powers, Borel–Weil–Bott

## Quick start

```julia
using Lie

# Fundamental weights of A₃
ω₁ = fundamental_weight(TypeA{3}, 1)
ω₂ = fundamental_weight(TypeA{3}, 2)

# Dimension of the standard representation
degree(ω₁)   # 4

# Tensor product decomposition
tensor_product(ω₁, ω₁)   # Sym² ⊕ ⋀²

# Weyl orbit
length(weyl_orbit(TypeA{3}, ω₁))   # 4

# Borel–Weil–Bott
borel_weil_bott(ω₁)   # (0, ω₁)
```

## API Reference

```@autodocs
Modules = [Lie]
```
