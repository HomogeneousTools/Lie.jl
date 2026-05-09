# Dynkin types and Cartan matrices

## Dynkin types

Over ``\mathbb{C}``, every finite-dimensional semisimple Lie algebra is
classified up to isomorphism by its **Dynkin type**, a combinatorial datum that
encodes the root system.
Lie.jl represents types at the Julia type level — `TypeA{3}`, `TypeB{4}`, etc. —
so that rank and root counts are compile-time constants, enabling
zero-cost dispatch and `@generated` specialisation.

### Simple types

The classical families `TypeA{N}`, `TypeB{N}`, `TypeC{N}`, `TypeD{N}`,
and the exceptional types `TypeE{6}`, `TypeE{7}`, `TypeE{8}`, `TypeF4`, `TypeG2`:

```jldoctest types
julia> using Lie

julia> rank(TypeA{3})
3

julia> rank(TypeG2)
2

julia> n_positive_roots(TypeA{3})
6

julia> n_positive_roots(TypeE{8})
120
```

```@docs
DynkinType
SimpleDynkinType
TypeA
TypeB
TypeC
TypeD
TypeE
TypeF4
TypeG2
rank
n_positive_roots
```

### Dynkin diagrams

The function `dynkin_diagram` produces a text rendering of the Dynkin diagram
following Bourbaki labelling conventions. This includes the correct arrow
directions for non-simply-laced types and the branching for types D and E.

The return value is a [`DynkinDiagram`](@ref) object that pretty-prints
automatically in the REPL without requiring `println`:

```jldoctest types
julia> dynkin_diagram(TypeA{4})
○───○───○───○
1   2   3   4

julia> dynkin_diagram(TypeG2)
○≡<≡○
1   2
```

For types with branching (D and E), the diagram shows the fork:

```@example
using Lie  # hide
dynkin_diagram(TypeD{5})
```

```@example
using Lie  # hide
dynkin_diagram(TypeE{6})
```

```@docs
DynkinDiagram
dynkin_diagram
```

### Product types

Direct products of simple types are represented by `ProductDynkinType`.
The Dynkin diagram shows each component separately:

```jldoctest types
julia> PT = ProductDynkinType{Tuple{TypeA{2}, TypeB{2}}};

julia> rank(PT)
4

julia> n_components(PT)
2

julia> component_ranks(PT)
(2, 2)

julia> component_offsets(PT)
(0, 2)
```

```@example
using Lie  # hide
PT = ProductDynkinType{Tuple{TypeA{2}, TypeB{2}}}
dynkin_diagram(PT)
```

```@docs
ProductDynkinType
n_components
component_type
component_ranks
component_offsets
```

### Validation

Invalid ranks are caught at construction time:

```jldoctest types
julia> TypeA{0}()
ERROR: ArgumentError: TypeA{N} requires N ≥ 1, got N=0
[...]

julia> TypeD{3}()
ERROR: ArgumentError: TypeD{N} requires N ≥ 4, got N=3
[...]
```

## Cartan matrices

The Cartan matrix ``C_{ij} = \langle \alpha_i^\vee, \alpha_j \rangle``
is computed at compile time via `@generated` functions, returning a
`StaticArrays.SMatrix`. The conventions follow Bourbaki. With this convention,
column ``i`` of ``C`` gives the simple root ``\alpha_i`` in the fundamental
weight basis: ``\alpha_i = \sum_j C_{ji}\omega_j``.

```jldoctest types
julia> cartan_matrix(TypeA{3})
3×3 StaticArraysCore.SMatrix{3, 3, Int64, 9} with indices SOneTo(3)×SOneTo(3):
  2  -1   0
 -1   2  -1
  0  -1   2

julia> cartan_matrix(TypeG2)
2×2 StaticArraysCore.SMatrix{2, 2, Int64, 4} with indices SOneTo(2)×SOneTo(2):
  2  -3
 -1   2

julia> cartan_matrix(TypeB{2})
2×2 StaticArraysCore.SMatrix{2, 2, Int64, 4} with indices SOneTo(2)×SOneTo(2):
  2  -1
 -2   2
```

### Symmetriser

The vector ``d`` such that ``\operatorname{diag}(d) \cdot C`` is symmetric.
For simply-laced types (A, D, E) all entries are 1:

```jldoctest types
julia> cartan_symmetrizer(TypeA{3})
3-element StaticArraysCore.SVector{3, Int64} with indices SOneTo(3):
 1
 1
 1

julia> cartan_symmetrizer(TypeG2)
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 1
 3

julia> cartan_symmetrizer(TypeB{2})
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 2
 1
```

### Bilinear form and inverse

The symmetric bilinear form ``B = \operatorname{diag}(d) \cdot C`` and
the rational inverse ``C^{-1}``:

```jldoctest types
julia> cartan_bilinear_form(TypeA{3})
3×3 StaticArraysCore.SMatrix{3, 3, Int64, 9} with indices SOneTo(3)×SOneTo(3):
  2  -1   0
 -1   2  -1
  0  -1   2

julia> cartan_matrix_inverse(TypeA{3})
3×3 StaticArraysCore.SMatrix{3, 3, Rational{Int64}, 9} with indices SOneTo(3)×SOneTo(3):
 3//4  1//2  1//4
 1//2   1    1//2
 1//4  1//2  3//4
```

```@docs
cartan_matrix
cartan_symmetrizer
cartan_bilinear_form
cartan_matrix_inverse
cartan_determinant
```

## Dimension

The dimension of the semisimple Lie algebra equals
``r + 2n`` where ``r`` is the rank and ``n`` is the number of positive roots:

```jldoctest types
julia> dimension(TypeA{3})   # dim sl₄(C) = 15
15

julia> dimension(TypeE{8})   # dim(E₈) = 248
248

julia> dimension(ProductDynkinType{Tuple{TypeA{1}, TypeA{1}}}())   # dim(sl₂(C) ⊕ sl₂(C)) = 6
6
```

```@docs
dimension
```

### Scaled bilinear form

The scaled bilinear form on the fundamental weight basis, used internally
by Freudenthal's formula:

```@docs
omega_bilinear_form_scaled
```

### Connection index

The determinant of the Cartan matrix is the **connection index**, which gives the index
of the root lattice $Q$ in the weight lattice $P$:

$$\det(C) = [P : Q]$$

For simply-laced types (A, D, E), the index varies by type. For multiply-laced types,
the determinant encodes how the roots and weights are related:

```jldoctest types
julia> using Lie

julia> cartan_determinant(TypeA{3})   # A₃: det = 4 = n+1
4

julia> cartan_determinant(TypeB{2})   # B₂ multiply-laced
2

julia> cartan_determinant(TypeG2)
1
```
