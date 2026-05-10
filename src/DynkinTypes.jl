# ═══════════════════════════════════════════════════════════════════════════════
#  Dynkin types — encoded at the type level for compile-time specialization
# ═══════════════════════════════════════════════════════════════════════════════

export DynkinType, SimpleDynkinType, ProductDynkinType
export TypeA, TypeB, TypeC, TypeD, TypeE, TypeF4, TypeG2
export rank, n_positive_roots, dimension
export n_components, component_type, component_ranks, component_offsets
export dynkin_diagram, DynkinDiagram

"""
    DynkinType

Abstract supertype for finite Dynkin types (simple and semisimple).

# Examples
```jldoctest
julia> using Lie

julia> TypeA{2} <: DynkinType
true
```
"""
abstract type DynkinType end

"""
    SimpleDynkinType <: DynkinType

Abstract supertype for simple (irreducible) finite Dynkin types.

# Examples
```jldoctest
julia> using Lie

julia> TypeG2 <: SimpleDynkinType
true
```
"""
abstract type SimpleDynkinType <: DynkinType end

# ─── Classical families ─────────────────────────────────────────────────────

"""
    TypeA{N} <: SimpleDynkinType

Dynkin type ``\\mathrm{A}_N``: the root-system type of
``\\mathfrak{sl}_{N+1}(\\mathbb{C})`` and groups isogenous to
``\\mathrm{SL}_{N+1}``. Valid for ``N \\ge 1``.

# Examples
```jldoctest
julia> using Lie

julia> rank(TypeA{3})
3
```
"""
struct TypeA{N} <: SimpleDynkinType
  function TypeA{N}() where {N}
    N::Int
    N >= 1 || throw(ArgumentError("TypeA{N} requires N ≥ 1, got N=$N"))
    new{N}()
  end
end
TypeA(n::Integer) = TypeA{n}()

"""
    TypeB{N} <: SimpleDynkinType

Dynkin type ``\\mathrm{B}_N``: the root-system type of
``\\mathfrak{so}_{2N+1}(\\mathbb{C})`` and groups isogenous to
``\\mathrm{Spin}_{2N+1}`` or ``\\mathrm{SO}_{2N+1}``. Valid for ``N \\ge 2``.

# Examples
```jldoctest
julia> using Lie

julia> rank(TypeB{3})
3
```
"""
struct TypeB{N} <: SimpleDynkinType
  function TypeB{N}() where {N}
    N::Int
    N >= 2 || throw(ArgumentError("TypeB{N} requires N ≥ 2, got N=$N"))
    new{N}()
  end
end
TypeB(n::Integer) = TypeB{n}()

"""
    TypeC{N} <: SimpleDynkinType

Dynkin type ``\\mathrm{C}_N``: the root-system type of
``\\mathfrak{sp}_{2N}(\\mathbb{C})`` and groups isogenous to
``\\mathrm{Sp}_{2N}``. Valid for ``N \\ge 2``.

# Examples
```jldoctest
julia> using Lie

julia> rank(TypeC{3})
3
```
"""
struct TypeC{N} <: SimpleDynkinType
  function TypeC{N}() where {N}
    N::Int
    N >= 2 || throw(ArgumentError("TypeC{N} requires N ≥ 2, got N=$N"))
    new{N}()
  end
end
TypeC(n::Integer) = TypeC{n}()

"""
    TypeD{N} <: SimpleDynkinType

Dynkin type ``\\mathrm{D}_N``: the root-system type of
``\\mathfrak{so}_{2N}(\\mathbb{C})`` and groups isogenous to
``\\mathrm{Spin}_{2N}`` or ``\\mathrm{SO}_{2N}``. Valid for ``N \\ge 3``.

# Examples
```jldoctest
julia> using Lie

julia> rank(TypeD{4})
4
```
"""
struct TypeD{N} <: SimpleDynkinType
  function TypeD{N}() where {N}
    N::Int
    N >= 3 || throw(ArgumentError("TypeD{N} requires N ≥ 3, got N=$N"))
    new{N}()
  end
end
TypeD(n::Integer) = TypeD{n}()

# ─── Exceptional types ──────────────────────────────────────────────────────

"""
    TypeE{N} <: SimpleDynkinType

Exceptional Dynkin type ``\\mathrm{E}_N`` for ``N \\in \\{6,7,8\\}``.

# Examples
```jldoctest
julia> using Lie

julia> n_positive_roots(TypeE{6})
36
```
"""
struct TypeE{N} <: SimpleDynkinType
  function TypeE{N}() where {N}
    N::Int
    N in (6, 7, 8) || throw(ArgumentError("TypeE{N} requires N ∈ {6,7,8}, got N=$N"))
    new{N}()
  end
end
TypeE(n::Integer) = TypeE{n}()

"""
    TypeF4 <: SimpleDynkinType

Exceptional Dynkin type ``\\mathrm{F}_4``.

# Examples
```jldoctest
julia> using Lie

julia> rank(TypeF4)
4
```
"""
struct TypeF4 <: SimpleDynkinType end

"""
    TypeG2 <: SimpleDynkinType

Exceptional Dynkin type ``\\mathrm{G}_2``.

# Examples
```jldoctest
julia> using Lie

julia> rank(TypeG2)
2
```
"""
struct TypeG2 <: SimpleDynkinType end

# ─── Product (semisimple) types ──────────────────────────────────────────────

"""
    ProductDynkinType{Ts} <: DynkinType

Product of simple Dynkin types, representing a semisimple Lie algebra.
`Ts` is a `Tuple` type of `SimpleDynkinType` subtypes.

# Examples
```jldoctest
julia> using Lie

julia> ProductDynkinType{Tuple{TypeA{3}, TypeD{5}, TypeE{6}}}()   # A3 × D5 × E6
A3 × D5 × E6
```
"""
struct ProductDynkinType{Ts<:Tuple} <: DynkinType
  function ProductDynkinType{Ts}() where {Ts<:Tuple}
    # Validate all components are SimpleDynkinType
    for T in Ts.parameters
      T <: SimpleDynkinType || throw(ArgumentError("$T is not a SimpleDynkinType"))
      check_dynkin_type(T)
    end
    new{Ts}()
  end
end

"""
    ProductDynkinType(types::SimpleDynkinType...)

Convenience constructor for product types from instances.
"""
function ProductDynkinType(types::SimpleDynkinType...)
  Ts = Tuple{typeof.(types)...}
  return ProductDynkinType{Ts}()
end

# ─── Type-level validation ─────────────────────────────────────────────────────

is_valid_dynkin_type(::Type{TypeA{N}}) where {N} = N >= 1
is_valid_dynkin_type(::Type{TypeB{N}}) where {N} = N >= 2
is_valid_dynkin_type(::Type{TypeC{N}}) where {N} = N >= 2
is_valid_dynkin_type(::Type{TypeD{N}}) where {N} = N >= 3
is_valid_dynkin_type(::Type{TypeE{N}}) where {N} = N in (6, 7, 8)
is_valid_dynkin_type(::Type{TypeF4}) = true
is_valid_dynkin_type(::Type{TypeG2}) = true

function is_valid_dynkin_type(::Type{ProductDynkinType{Ts}}) where {Ts}
  all(T <: SimpleDynkinType && is_valid_dynkin_type(T) for T in Ts.parameters)
end

_invalid_dynkin_type_message(::Type{TypeA{N}}) where {N} = "TypeA{$N} requires N ≥ 1, got N=$N"
_invalid_dynkin_type_message(::Type{TypeB{N}}) where {N} = "TypeB{$N} requires N ≥ 2, got N=$N"
_invalid_dynkin_type_message(::Type{TypeC{N}}) where {N} = "TypeC{$N} requires N ≥ 2, got N=$N"
_invalid_dynkin_type_message(::Type{TypeD{N}}) where {N} = "TypeD{$N} requires N ≥ 3, got N=$N"
_invalid_dynkin_type_message(::Type{TypeE{N}}) where {N} = "TypeE{$N} requires N ∈ {6,7,8}, got N=$N"
_invalid_dynkin_type_message(::Type{TypeF4}) = "TypeF4 is valid"
_invalid_dynkin_type_message(::Type{TypeG2}) = "TypeG2 is valid"

function _invalid_dynkin_type_message(::Type{ProductDynkinType{Ts}}) where {Ts}
  problems = String[]
  for T in Ts.parameters
    if !(T <: SimpleDynkinType)
      push!(problems, "$T is not a SimpleDynkinType")
    elseif !is_valid_dynkin_type(T)
      push!(problems, _invalid_dynkin_type_message(T))
    end
  end
  isempty(problems) && return "Invalid ProductDynkinType{$Ts}"
  return "Invalid ProductDynkinType component(s): $(join(problems, "; "))"
end

@inline function check_dynkin_type(::Type{DT}) where {DT<:DynkinType}
  is_valid_dynkin_type(DT) || throw(ArgumentError(_invalid_dynkin_type_message(DT)))
  return DT
end

# ─── Rank ────────────────────────────────────────────────────────────────────

"""
    rank(::Type{DT}) where DT<:DynkinType -> Int

Return the rank (dimension of the Cartan subalgebra) of the Dynkin type `DT`.
This is a compile-time constant.

# Examples
```jldoctest
julia> using Lie

julia> rank(TypeA{3})
3

julia> rank(TypeE{8})
8
```
"""
rank(::Type{TypeA{N}}) where {N} = (check_dynkin_type(TypeA{N}); N)
rank(::Type{TypeB{N}}) where {N} = (check_dynkin_type(TypeB{N}); N)
rank(::Type{TypeC{N}}) where {N} = (check_dynkin_type(TypeC{N}); N)
rank(::Type{TypeD{N}}) where {N} = (check_dynkin_type(TypeD{N}); N)
rank(::Type{TypeE{N}}) where {N} = (check_dynkin_type(TypeE{N}); N)
rank(::Type{TypeF4}) = 4
rank(::Type{TypeG2}) = 2

function rank(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  return sum(rank, fieldtypes(Ts))
end

# Instance versions
rank(dt::DynkinType) = rank(typeof(dt))

# ─── Number of positive roots ───────────────────────────────────────────────

"""
    n_positive_roots(::Type{DT}) -> Int

Number of positive roots for a simple Dynkin type.

# Examples
```jldoctest
julia> using Lie

julia> n_positive_roots(TypeA{3})
6

julia> n_positive_roots(TypeE{8})
120
```
"""
n_positive_roots(::Type{TypeA{N}}) where {N} =
  (check_dynkin_type(TypeA{N}); N * (N + 1) ÷ 2)
n_positive_roots(::Type{TypeB{N}}) where {N} = (check_dynkin_type(TypeB{N}); N^2)
n_positive_roots(::Type{TypeC{N}}) where {N} = (check_dynkin_type(TypeC{N}); N^2)
n_positive_roots(::Type{TypeD{N}}) where {N} = (check_dynkin_type(TypeD{N}); N * (N - 1))
n_positive_roots(::Type{TypeE{6}}) = 36
n_positive_roots(::Type{TypeE{7}}) = 63
n_positive_roots(::Type{TypeE{8}}) = 120
n_positive_roots(::Type{TypeF4}) = 24
n_positive_roots(::Type{TypeG2}) = 6

function n_positive_roots(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  return sum(n_positive_roots, fieldtypes(Ts))
end

n_positive_roots(dt::DynkinType) = n_positive_roots(typeof(dt))

"""
    dimension(::Type{DT}) -> Int
    dimension(dt::DynkinType) -> Int

Return the dimension of the semisimple Lie algebra of type `DT`.

For a semisimple Lie algebra of rank ``r`` with ``n`` positive roots,
the dimension is ``r + 2n`` (Cartan subalgebra plus positive and negative
root spaces).

Note: for the adjoint representation dimension (same number), use
[`degree`](@ref) on [`adjoint_representation`](@ref).

# Examples
```jldoctest
julia> using Lie

julia> dimension(TypeA{3})  # sl_4(C) has dimension 15
15

julia> dimension(TypeE{8})  # e_8(C) has dimension 248
248

julia> dimension(ProductDynkinType{Tuple{TypeA{1}, TypeA{1}}}())  # sl_2(C) ⊕ sl_2(C)
6
```
"""
dimension(::Type{DT}) where {DT<:DynkinType} = rank(DT) + 2 * n_positive_roots(DT)
dimension(dt::DynkinType) = dimension(typeof(dt))

# ─── Component access for product types ─────────────────────────────────────

"""
    n_components(::Type{ProductDynkinType{Ts}}) -> Int

Number of simple factors in a product type.

# Examples
```jldoctest
julia> using Lie

julia> n_components(ProductDynkinType{Tuple{TypeA{2}, TypeB{2}}})
2
```
"""
function n_components(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  return length(fieldtypes(Ts))
end

n_components(::Type{<:SimpleDynkinType}) = 1
n_components(dt::DynkinType) = n_components(typeof(dt))

"""
    component_type(::Type{ProductDynkinType{Ts}}, i::Integer) -> Type

Return the `i`-th simple Dynkin type in a product.

# Examples
```jldoctest
julia> using Lie

julia> PT = ProductDynkinType{Tuple{TypeA{2}, TypeB{2}}};

julia> component_type(PT, 1)
TypeA{2}

julia> component_type(PT, 2)
TypeB{2}
```
"""
function component_type(::Type{ProductDynkinType{Ts}}, i::Integer) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  return fieldtypes(Ts)[i]
end

"""
    component_ranks(::Type{ProductDynkinType{Ts}}) -> Tuple

Return a tuple of ranks of the components.

# Examples
```jldoctest
julia> using Lie

julia> component_ranks(ProductDynkinType{Tuple{TypeA{2}, TypeB{3}}})
(2, 3)
```
"""
function component_ranks(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  return map(rank, fieldtypes(Ts))
end

"""
    component_offsets(::Type{ProductDynkinType{Ts}}) -> Tuple

Return a tuple of starting index offsets for each component in the product type.
The i-th component occupies indices offset[i]+1 : offset[i]+rank(component_i).

# Examples
```jldoctest
julia> using Lie

julia> component_offsets(ProductDynkinType{Tuple{TypeA{2}, TypeB{3}}})
(0, 2)
```
"""
function component_offsets(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  rs = map(rank, fieldtypes(Ts))
  return ntuple(i -> i == 1 ? 0 : sum(rs[j] for j in 1:(i - 1)), length(rs))
end

# ─── Display ─────────────────────────────────────────────────────────────────

_type_name(::Type{TypeA{N}}) where {N} = "A$N"
_type_name(::Type{TypeB{N}}) where {N} = "B$N"
_type_name(::Type{TypeC{N}}) where {N} = "C$N"
_type_name(::Type{TypeD{N}}) where {N} = "D$N"
_type_name(::Type{TypeE{N}}) where {N} = "E$N"
_type_name(::Type{TypeF4}) = "F4"
_type_name(::Type{TypeG2}) = "G2"

@generated function _type_name(::Type{ProductDynkinType{Ts}}) where {Ts}
  names = [_type_name(T) for T in Ts.parameters]
  return join(names, " × ")
end

Base.show(io::IO, dt::SimpleDynkinType) = print(io, _type_name(typeof(dt)))
Base.show(io::IO, dt::ProductDynkinType) = print(io, _type_name(typeof(dt)))

# ─── Dynkin diagrams ─────────────────────────────────────────────────────────

"""
    DynkinDiagram

A pretty-printable wrapper around the text rendering of a Dynkin diagram.

Displaying a `DynkinDiagram` in the REPL renders the diagram automatically
without requiring `println`. Call `string` to recover the raw string.

See also: [`dynkin_diagram`](@ref).

# Examples
```jldoctest
julia> using Lie

julia> occursin(Char(10), string(dynkin_diagram(TypeA{2})))
true
```
"""
struct DynkinDiagram
  str::String
end

Base.show(io::IO, d::DynkinDiagram) = print(io, d.str)
Base.show(io::IO, ::MIME"text/plain", d::DynkinDiagram) = print(io, d.str)
Base.:(==)(a::DynkinDiagram, b::DynkinDiagram) = a.str == b.str
Base.:(==)(d::DynkinDiagram, s::AbstractString) = d.str == s
Base.:(==)(s::AbstractString, d::DynkinDiagram) = s == d.str
Base.string(d::DynkinDiagram) = d.str
Base.hash(d::DynkinDiagram, h::UInt) = hash(d.str, h)

"""
    dynkin_diagram(::Type{DT}) -> DynkinDiagram
    dynkin_diagram(dt::DynkinType) -> DynkinDiagram

Return the Dynkin diagram for the given type as a [`DynkinDiagram`](@ref),
following Bourbaki conventions. The result pretty-prints automatically in
the REPL; call `string` to recover the raw multi-line string.

# Examples
```jldoctest
julia> using Lie

julia> dynkin_diagram(TypeA{4})
○───○───○───○
1   2   3   4

julia> dynkin_diagram(TypeB{3})
○───○═>═○
1   2   3

julia> dynkin_diagram(TypeG2)
○≡<≡○
1   2
```
"""
function dynkin_diagram(::Type{TypeA{N}}) where {N}
  check_dynkin_type(TypeA{N})
  nodes = join(fill("○", N), "───")
  labels = join([lpad(string(i), 1) for i in 1:N], "   ")
  return DynkinDiagram(nodes * "\n" * labels)
end

function dynkin_diagram(::Type{TypeB{N}}) where {N}
  check_dynkin_type(TypeB{N})
  # B_n: ○───○───…───○=>=○  (double bond with arrow to last)
  if N == 2
    nodes = "○═>═○"
  else
    nodes = join(fill("○", N - 1), "───") * "═>═○"
  end
  labels = join([lpad(string(i), 1) for i in 1:N], "   ")
  return DynkinDiagram(nodes * "\n" * labels)
end

function dynkin_diagram(::Type{TypeC{N}}) where {N}
  check_dynkin_type(TypeC{N})
  # C_n: ○───○───…───○=<=○  (double bond with arrow from last)
  if N == 2
    nodes = "○═<═○"
  else
    nodes = join(fill("○", N - 1), "───") * "═<═○"
  end
  labels = join([lpad(string(i), 1) for i in 1:N], "   ")
  return DynkinDiagram(nodes * "\n" * labels)
end

function dynkin_diagram(::Type{TypeD{N}}) where {N}
  check_dynkin_type(TypeD{N})
  # D_n: linear chain 1..N-2, then fork to N-1 and N at node N-2
  # Bourbaki layout:
  #          ○ N
  #         /
  # ○──○──...──○──○
  # 1  2     N-2 N-1
  prefix = " "^(4 * (N - 2)) * "○ $N"
  fork = " "^(4 * (N - 2) - 1) * "/"
  main = join(fill("○", N - 1), "───")
  main_labels = join([lpad(string(i), 1) for i in 1:(N - 1)], "   ")
  return DynkinDiagram(prefix * "\n" * fork * "\n" * main * "\n" * main_labels)
end

function dynkin_diagram(::Type{TypeE{N}}) where {N}
  check_dynkin_type(TypeE{N})
  # E_n (n=6,7,8): linear chain 1,3,4,5,...,n with node 2 branching from node 4
  # Bourbaki:
  #         ○ 2
  #         |
  # ○───○───○───○───○  (for E6: nodes 1,3,4,5,6)
  # 1   3   4   5   6
  n_main = N - 1  # nodes on main chain: 1, 3, 4, 5, ..., N
  main = join(fill("○", n_main), "───")
  main_labels_arr = [1; collect(3:N)]
  main_labels = join([lpad(string(i), 1) for i in main_labels_arr], "   ")
  # Node 2 branches from the 3rd position (node 4, which is at index 3 in main chain)
  indent = 8  # position of node 4 = 2 nodes * 4 chars each
  top = " "^indent * "○ 2"
  branch = " "^indent * "|"
  return DynkinDiagram(top * "\n" * branch * "\n" * main * "\n" * main_labels)
end

function dynkin_diagram(::Type{TypeF4})
  return DynkinDiagram("○───○═>═○───○\n1   2   3   4")
end

function dynkin_diagram(::Type{TypeG2})
  return DynkinDiagram("○≡<≡○\n1   2")
end

function dynkin_diagram(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  diagrams = [dynkin_diagram(T).str for T in Ts.parameters]
  labels = [_type_name(T) for T in Ts.parameters]
  parts = [labels[i] * ":\n" * diagrams[i] for i in eachindex(diagrams)]
  return DynkinDiagram(join(parts, "\n\n"))
end

dynkin_diagram(dt::DynkinType) = dynkin_diagram(typeof(dt))
