# ═══════════════════════════════════════════════════════════════════════════════
#  Dynkin types — encoded at the type level for compile-time specialization
# ═══════════════════════════════════════════════════════════════════════════════

export DynkinType, SimpleDynkinType, ProductDynkinType
export TypeA, TypeB, TypeC, TypeD, TypeE, TypeF4, TypeG2
export rank, n_positive_roots, is_valid_dynkin_type
export n_components, component_type, component_ranks, component_offsets

"""
    DynkinType

Abstract supertype for all Dynkin types (simple and semisimple).
"""
abstract type DynkinType end

"""
    SimpleDynkinType <: DynkinType

Abstract supertype for simple (irreducible) Dynkin types.
"""
abstract type SimpleDynkinType <: DynkinType end

# ─── Classical families ─────────────────────────────────────────────────────

"""
    TypeA{N} <: SimpleDynkinType

Dynkin type ``A_N`` (``\\mathrm{SL}_{N+1}``). Valid for ``N \\ge 1``.
"""
struct TypeA{N} <: SimpleDynkinType
    function TypeA{N}() where {N}
        N::Int
        N >= 1 || throw(ArgumentError("TypeA{N} requires N ≥ 1, got N=$N"))
        new{N}()
    end
end
TypeA(n::Int) = TypeA{n}()

"""
    TypeB{N} <: SimpleDynkinType

Dynkin type ``B_N`` (``\\mathrm{SO}_{2N+1}``). Valid for ``N \\ge 2``.
"""
struct TypeB{N} <: SimpleDynkinType
    function TypeB{N}() where {N}
        N::Int
        N >= 2 || throw(ArgumentError("TypeB{N} requires N ≥ 2, got N=$N"))
        new{N}()
    end
end
TypeB(n::Int) = TypeB{n}()

"""
    TypeC{N} <: SimpleDynkinType

Dynkin type ``C_N`` (``\\mathrm{Sp}_{2N}``). Valid for ``N \\ge 2``.
"""
struct TypeC{N} <: SimpleDynkinType
    function TypeC{N}() where {N}
        N::Int
        N >= 2 || throw(ArgumentError("TypeC{N} requires N ≥ 2, got N=$N"))
        new{N}()
    end
end
TypeC(n::Int) = TypeC{n}()

"""
    TypeD{N} <: SimpleDynkinType

Dynkin type ``D_N`` (``\\mathrm{SO}_{2N}``). Valid for ``N \\ge 4``.
"""
struct TypeD{N} <: SimpleDynkinType
    function TypeD{N}() where {N}
        N::Int
        N >= 4 || throw(ArgumentError("TypeD{N} requires N ≥ 4, got N=$N"))
        new{N}()
    end
end
TypeD(n::Int) = TypeD{n}()

# ─── Exceptional types ──────────────────────────────────────────────────────

"""
    TypeE{N} <: SimpleDynkinType

Dynkin type ``E_N`` for ``N \\in \\{6,7,8\\}``.
"""
struct TypeE{N} <: SimpleDynkinType
    function TypeE{N}() where {N}
        N::Int
        N in (6, 7, 8) || throw(ArgumentError("TypeE{N} requires N ∈ {6,7,8}, got N=$N"))
        new{N}()
    end
end
TypeE(n::Int) = TypeE{n}()

"""
    TypeF4 <: SimpleDynkinType

Dynkin type ``F_4``.
"""
struct TypeF4 <: SimpleDynkinType end

"""
    TypeG2 <: SimpleDynkinType

Dynkin type ``G_2``.
"""
struct TypeG2 <: SimpleDynkinType end

# ─── Product (semisimple) types ──────────────────────────────────────────────

"""
    ProductDynkinType{Ts} <: DynkinType

Product of simple Dynkin types, representing a semisimple Lie algebra.
`Ts` is a `Tuple` type of `SimpleDynkinType` subtypes.

# Example
```julia
ProductDynkinType{Tuple{TypeA{3}, TypeD{5}, TypeE{6}}}()   # A₃ × D₅ × E₆
```
"""
struct ProductDynkinType{Ts<:Tuple} <: DynkinType
    function ProductDynkinType{Ts}() where {Ts<:Tuple}
        # Validate all components are SimpleDynkinType
        for T in Ts.parameters
            T <: SimpleDynkinType || throw(ArgumentError("$T is not a SimpleDynkinType"))
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

# ─── Rank ────────────────────────────────────────────────────────────────────

"""
    rank(::Type{DT}) where DT<:DynkinType -> Int

Return the rank (dimension of the Cartan subalgebra) of the Dynkin type `DT`.
This is a compile-time constant.
"""
rank(::Type{TypeA{N}}) where {N} = N
rank(::Type{TypeB{N}}) where {N} = N
rank(::Type{TypeC{N}}) where {N} = N
rank(::Type{TypeD{N}}) where {N} = N
rank(::Type{TypeE{N}}) where {N} = N
rank(::Type{TypeF4}) = 4
rank(::Type{TypeG2}) = 2

@generated function rank(::Type{ProductDynkinType{Ts}}) where {Ts}
    return sum(rank(T) for T in Ts.parameters)
end

# Instance versions
rank(dt::DynkinType) = rank(typeof(dt))

# ─── Number of positive roots ───────────────────────────────────────────────

"""
    n_positive_roots(::Type{DT}) -> Int

Number of positive roots for a simple Dynkin type.
"""
n_positive_roots(::Type{TypeA{N}}) where {N} = N * (N + 1) ÷ 2
n_positive_roots(::Type{TypeB{N}}) where {N} = N^2
n_positive_roots(::Type{TypeC{N}}) where {N} = N^2
n_positive_roots(::Type{TypeD{N}}) where {N} = N * (N - 1)
n_positive_roots(::Type{TypeE{6}}) = 36
n_positive_roots(::Type{TypeE{7}}) = 63
n_positive_roots(::Type{TypeE{8}}) = 120
n_positive_roots(::Type{TypeF4}) = 24
n_positive_roots(::Type{TypeG2}) = 6

@generated function n_positive_roots(::Type{ProductDynkinType{Ts}}) where {Ts}
    return sum(n_positive_roots(T) for T in Ts.parameters)
end

n_positive_roots(dt::DynkinType) = n_positive_roots(typeof(dt))

# ─── Component access for product types ─────────────────────────────────────

"""
    n_components(::Type{ProductDynkinType{Ts}}) -> Int

Number of simple factors in a product type.
"""
@generated function n_components(::Type{ProductDynkinType{Ts}}) where {Ts}
    return length(Ts.parameters)
end

n_components(::Type{<:SimpleDynkinType}) = 1
n_components(dt::DynkinType) = n_components(typeof(dt))

"""
    component_type(::Type{ProductDynkinType{Ts}}, i) -> Type

Return the i-th simple Dynkin type in a product.
"""
@generated function component_type(::Type{ProductDynkinType{Ts}}, ::Val{I}) where {Ts, I}
    return Ts.parameters[I]
end

"""
    component_ranks(::Type{ProductDynkinType{Ts}}) -> Tuple

Return a tuple of ranks of the components.
"""
@generated function component_ranks(::Type{ProductDynkinType{Ts}}) where {Ts}
    return Tuple(rank(T) for T in Ts.parameters)
end

"""
    component_offsets(::Type{ProductDynkinType{Ts}}) -> Tuple

Return a tuple of starting index offsets for each component in the product type.
The i-th component occupies indices offset[i]+1 : offset[i]+rank(component_i).
"""
@generated function component_offsets(::Type{ProductDynkinType{Ts}}) where {Ts}
    ranks = [rank(T) for T in Ts.parameters]
    offsets = cumsum([0; ranks[1:end-1]])
    return Tuple(offsets)
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
