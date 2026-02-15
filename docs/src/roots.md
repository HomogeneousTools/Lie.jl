# Root systems

A root system is determined by its Dynkin type.  Lie.jl caches one immutable
`RootSystem` singleton per type, so constructing the same type twice returns
the same object.

## Creating a root system

```jldoctest roots
julia> using Lie

julia> RS = RootSystem(TypeA{3})
Root system of type A3, rank 3 with 6 positive roots

julia> n_roots(RS)
12

julia> n_simple_roots(RS)
3
```

```@docs
RootSystem
n_roots
n_simple_roots
```

## Simple and positive roots

Roots are stored in the simple root basis as `RootSpaceElem` values.
Use `coefficients` to extract the underlying coordinate vector:

```jldoctest roots
julia> α₁ = simple_root(RS, 1)
α1

julia> α₂ = simple_root(RS, 2)
α2

julia> α₁ + α₂
α1 + α2

julia> 2 * α₁
2α1

julia> coefficients(α₁)
3-element StaticArraysCore.SVector{3, Int64} with indices SOneTo(3):
 1
 0
 0
```

### Listing roots

```jldoctest roots
julia> length(positive_roots(RS))
6

julia> positive_root(RS, 3)
α3

julia> positive_root(RS, 4)
α1 + α2

julia> highest_root(RS)
α1 + α2 + α3
```

Negative roots are the negations of positive roots:

```jldoctest roots
julia> negative_root(RS, 1)
-α1
```

```@docs
RootSpaceElem
simple_roots
simple_root
positive_roots
positive_root
negative_roots
negative_root
roots
root
highest_root
coefficients
```

## Root queries

```jldoctest roots
julia> is_root(RS, α₁ + α₂)
true

julia> is_positive_root(RS, α₁ + α₂)
true

julia> is_root(RS, 2 * α₁)
false
```

```@docs
is_root
is_positive_root
```

## Height

The height of a root is the sum of its simple root coefficients:

```jldoctest roots
julia> height(α₁)
1

julia> height(α₁ + α₂)
2

julia> height(highest_root(RS))
3
```

```@docs
height
```

## Inner product

The inner product on the root space uses the symmetrised Cartan form
``(\alpha, \beta) = \alpha^T \operatorname{diag}(d) \, C \, \beta``:

```jldoctest roots
julia> dot(α₁, α₁)
2//1

julia> dot(α₁, α₂)
-1//1
```

## Coroots

For simply-laced types (A, D, E), coroots coincide with roots.
For non-simply-laced types the coroots are rescaled by the
symmetriser entries:

```jldoctest roots
julia> sc = simple_coroots(RS);

julia> sc[1]
α1

julia> length(positive_coroots(RS))
6
```

```@docs
simple_coroots
positive_coroots
```

## Examples

### A₂

```jldoctest roots
julia> RS₂ = RootSystem(TypeA{2})
Root system of type A2, rank 2 with 3 positive roots

julia> [positive_root(RS₂, i) for i in 1:3]
3-element Vector{RootSpaceElem{TypeA{2}, 2}}:
 α1
 α2
 α1 + α2

julia> highest_root(RS₂)
α1 + α2
```

### G₂

The exceptional type ``\mathrm{G}_2`` has 6 positive roots:

```jldoctest roots
julia> RS_G2 = RootSystem(TypeG2);

julia> n_positive_roots(TypeG2)
6

julia> highest_root(RS_G2)
3α1 + 2α2
```
