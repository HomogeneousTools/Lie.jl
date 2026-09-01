# Root systems

A root system is determined by its Dynkin type.  Semisimple.jl caches one immutable
`RootSystem` singleton per type, so constructing the same type twice returns
the same object.

## Creating a root system

```jldoctest roots
julia> using Semisimple

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
julia> α1 = simple_root(RS, 1)
α1

julia> α2 = simple_root(RS, 2)
α2

julia> α1 + α2
α1 + α2

julia> 2 * α1
2α1

julia> coefficients(α1)
3-element StaticArraysCore.SVector{3, Int64} with indices SOneTo(3):
 1
 0
 0
```

### Listing roots

Positive roots are returned in **non-decreasing order of height**:
indices `1` through `rank` are the simple roots
(height 1), and the last element is always the highest root.

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

Because the ordering is canonical, `highest_root` simply returns the last
positive root — no search is performed.

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
highest_short_root
coefficients
```

## Root queries

```jldoctest roots
julia> is_root(RS, α1 + α2)
true

julia> is_positive_root(RS, α1 + α2)
true

julia> is_root(RS, 2 * α1)
false
```

```@docs
is_root
is_positive_root
```

## Height

The height of a root is the sum of its simple root coefficients:

```jldoctest roots
julia> height(α1)
1

julia> height(α1 + α2)
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
julia> dot(α1, α1)
2//1

julia> dot(α1, α2)
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

## Coxeter invariants

The **highest root** is a fundamental invariant of the root system. When expressed in the simple root basis as
``\theta = \sum_i m_i \alpha_i``, the coefficients ``m_i`` are the **Coxeter labels** or
**marks** (returned by [`coxeter_coefficients`](@ref)).
Because positive roots are sorted by height, `highest_root(RS)` is simply `positive_root(RS, N)` where `N` is the
number of positive roots — no search needed.

The **highest short root** ``\theta_s`` is the short root of greatest height.  For simply-laced types
(A, D, E) it coincides with ``\theta``.  Its index in the positive root list is precomputed when the
root system is built and stored in `RS.highest_coroot_idx`.

The **highest coroot** (or **dominant coroot**) ``\theta^\vee`` is the positive coroot of greatest height.
It equals the coroot of the highest short root, and is stored at the same precomputed index
`RS.highest_coroot_idx`.

The **Coxeter number** ``h = 1 + \sum_i m_i`` is the order of a Coxeter element in the Weyl group.

For the **dual root system** (Langlands dual, swapping B↔C), the corresponding invariants are the
**dual Coxeter labels** and **dual Coxeter number** ``h^\vee``.

```jldoctest roots
julia> c_coeff = coxeter_coefficients(TypeA{3})
3-element StaticArraysCore.SVector{3, Int64} with indices SOneTo(3):
 1
 1
 1

julia> coxeter_number(TypeA{3})
4

julia> cartan_determinant(TypeA{3})
4
```

For multiply-laced types like ``\mathrm{G}_2``:

```jldoctest roots
julia> c_coeff_G2 = coxeter_coefficients(TypeG2)
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 3
 2

julia> coxeter_number(TypeG2)
6
```

```@docs
highest_coroot
coxeter_coefficients
dual_coxeter_coefficients
coxeter_number
dual_coxeter_number
degrees_fundamental_invariants
```

## Bourbaki tables

`bourbaki_table` collects the root-system data appearing in Bourbaki's finite-type
plates into one programmatic object. It uses intrinsic simple-root coordinates rather
than a separate Euclidean realization for each Dynkin family.

```jldoctest roots
julia> table = bourbaki_table(TypeA{2});

julia> table.exponents
2-element Vector{Int64}:
 1
 2

julia> table.root_lattice_quotient
1-element Vector{Int64}:
 3

julia> table.opposition_involution
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 2
 1
```

Display the table itself to see a Unicode plate rendered with PrettyTables.jl: the
Dynkin diagram, all positive roots and coroots, fundamental weights, Coxeter and
lattice invariants, Weyl-group data, and finite and affine Cartan matrices.

```@docs
BourbakiTable
bourbaki_table
```

## Examples

### A2

```jldoctest roots
julia> RS2 = RootSystem(TypeA{2})
Root system of type A2, rank 2 with 3 positive roots

julia> [positive_root(RS2, i) for i in 1:3]
3-element Vector{RootSpaceElem{TypeA{2}, 2}}:
 α1
 α2
 α1 + α2

julia> highest_root(RS2)
α1 + α2

julia> highest_short_root(RS2)
α1 + α2

julia> coefficients(highest_coroot(RS2))
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 1
 1
```

``\mathrm{A}_2`` is simply-laced, so the highest root, highest short root, and highest coroot all coincide.

### G2

The exceptional type ``\mathrm{G}_2`` has short roots of squared length ``2`` and long roots of squared length ``6``.
The highest short root and the highest long root are distinct:

```jldoctest roots
julia> RS_G2 = RootSystem(TypeG2);

julia> n_positive_roots(TypeG2)
6

julia> highest_root(RS_G2)
3α1 + 2α2

julia> highest_short_root(RS_G2)
2α1 + α2

julia> coefficients(highest_coroot(RS_G2))
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 2
 3
```

### B2

```jldoctest roots
julia> RS_B2 = RootSystem(TypeB{2});

julia> highest_root(RS_B2)
α1 + 2α2

julia> highest_short_root(RS_B2)
α1 + α2

julia> coefficients(highest_coroot(RS_B2))
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 2
 1
```
