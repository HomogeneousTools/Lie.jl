# Implementation details

This page covers performance considerations, caching mechanisms, precompilation,
and other implementation details of Semisimple.jl.

## Caching

Semisimple.jl uses several internal caches to avoid recomputing expensive results. Understanding
these caches is important for benchmarking and memory management.

### Available caches

Semisimple.jl maintains ten internal caches. Six are unbounded `Dict` caches for
small singletons and lookup tables; four are bounded `LRU` caches (from
[LRUCache.jl](https://github.com/JuliaCollections/LRUCache.jl)) whose total
memory budget is configurable at runtime via [`configure_caches!`](@ref).

| Cache | Variable | Type | Purpose |
|-------|----------|------|--------|
| Root system | `Semisimple._root_system_cache` | `Dict` | Singleton `RootSystem` instances per Dynkin type |
| Positive roots set | `Semisimple._positive_roots_set_cache` | `Dict` | Fast `is_positive_root` lookup sets |
| Longest Weyl element | `Semisimple._longest_element_cache` | `Dict` | Cached longest element `w₀` per Dynkin type |
| Coset representatives | `Semisimple._coset_reps_cache` | `Dict` | Weyl orbit coset reps for exceptional types |
| Dominant character (type) | `Semisimple._dominant_character_type_cache` | `Dict` | Type-level Freudenthal intermediates |
| Weyl dimension data | `Semisimple._weyl_dimension_data_cache` | `Dict` | Dimension formula denominator and scaled roots |
| Dominant character | `Semisimple._dominant_character_cache` | `LRU` | Dominant weight multiplicities from Freudenthal's formula |
| Tensor product | `Semisimple._tensor_cache` | `LRU` | Tensor product decompositions |
| Symmetric power | `Semisimple._symmetric_power_cache` | `LRU` | Symmetric power decompositions |
| Exterior power | `Semisimple._exterior_power_cache` | `LRU` | Exterior power decompositions |

The six `Dict` caches are unbounded and persist for the lifetime of the Julia
session.  The four `LRU` caches have a configurable memory budget (default:
25 % of system RAM, minimum 256 MiB) and automatically evict least-recently-used
entries when the budget is exceeded.

!!! note "Why the dominant character cache matters"
    The dominant character cache is the main performance lever for downstream
    operations. Tensor products, symmetric/exterior powers, and plethysms call
    [`dominant_character`](@ref) repeatedly for the same highest weights.

### Inspecting caches

Use [`cache_info`](@ref) to get a snapshot of cache occupancy:

```julia
using Semisimple

# Snapshot before any work
info = cache_info()
println("Tensor cache: ", info.tensor.length, " entries (max ", info.tensor.maxsize, " bytes)")

# Populate some caches by doing computations
ω₁ = fundamental_weight(TypeE{8}, 1)
freudenthal_formula(ω₁)
tensor_product(ω₁, ω₁)

# Snapshot after
info = cache_info()
println("Dominant character cache: ", info.dominant_character.length, " entries")
println("Tensor cache: ", info.tensor.length, " entries")
```

### Clearing caches

Use [`clear_caches!`](@ref) (or its alias [`clear_all_caches!`](@ref)) to
empty every cache at once:

```julia
using Semisimple

# Do some computations
ω₁ = fundamental_weight(TypeA{2}, 1)
tensor_product(ω₁, ω₁)
freudenthal_formula(ω₁)
symmetric_power(ω₁, 3)

# Clear everything
clear_caches!()
```

This is particularly useful for:
- **Benchmarking cold-start performance** — measure how long operations take without cached results
- **Memory management** — free memory after large computations (e.g., after computing many E₈ tensor products)
- **Reproducible testing** — ensure tests start from a clean state

Individual cache variables have underscored names and are internal
implementation details. Prefer the public [`clear_caches!`](@ref) and
[`configure_caches!`](@ref) APIs.

### Configuring cache budgets

Use [`configure_caches!`](@ref) to resize the LRU caches at runtime.  The
`budget` (in bytes) controls the total memory envelope; the four fraction
arguments determine how it is divided:

```julia
using Semisimple

# Give caches 512 MiB total
configure_caches!(budget = 512 * 1024^2)

# Custom split: 50 % tensor, 30 % dominant, 10 % each for Sym/⋀
configure_caches!(
  budget = 512 * 1024^2,
  dominant_frac = 0.30,
  tensor_frac = 0.50,
  sym_power_frac = 0.10,
  ext_power_frac = 0.10,
)
```

Default fractions: dominant 30 %, tensor 40 %, symmetric 15 %, exterior 15 %.
The default total budget is 25 % of system RAM (minimum 256 MiB).  These
defaults can also be set persistently via Julia's `Preferences.jl`
(keys: `cache_budget`, `dominant_frac`, `tensor_frac`, `sym_power_frac`,
`ext_power_frac`).

### Cache invalidation

Caches are **never invalidated by code changes** — all cached functions are
pure (same inputs always produce same outputs).  However, cached entries can
disappear in three ways:

- You explicitly clear a cache (via [`clear_caches!`](@ref) or `empty!(...)`)
- An LRU cache evicts least-recently-used entries when its memory budget is exceeded
- Your Julia session ends

Automatic eviction only affects the four bounded LRU caches.  The six
unbounded `Dict` caches persist until cleared or session end.

This design is safe because:
- Dynkin types are immutable compile-time constants
- Weights are immutable `SVector` objects
- All cached functions are pure — re-computing an evicted entry always gives the same result

## Precompilation

Semisimple.jl precompiles many commonly-used methods to reduce first-call latency. When you
load the package with `using Semisimple`, the precompilation work has already been done.

### What gets precompiled

The package precompiles the following operations for all simple Dynkin types up to rank 9
(plus the exceptional types):

**Dynkin types precompiled:**
- `TypeA{1}` through `TypeA{9}`
- `TypeB{2}` through `TypeB{9}`
- `TypeC{2}` through `TypeC{9}`
- `TypeD{4}` through `TypeD{9}`
- `TypeE{6}`, `TypeE{7}`, `TypeE{8}`
- `TypeF4`
- `TypeG2`

**Operations precompiled:**
- `cartan_matrix`, `cartan_symmetrizer`, `cartan_bilinear_form`, `cartan_matrix_inverse`
- `_make_root_system` (internal root system construction)
- `_weyl_denominator`, `_weyl_dim_scaled_roots` (Weyl dimension formula internals)
- `degree` (representation dimension)
- `conjugate_dominant_weight` (dominant weight conjugation)
- `conjugate_dominant_weight_with_length` (Borel–Weil–Bott hot path)
- `weyl_orbit` (Weyl orbit generation)
- Weyl group actions (`*` operator for roots and weights with Weyl elements)
- `freudenthal_formula` (weight multiplicities)
- `dot_reduce` (weight normalization)
- `lr_tensor_product` (Littlewood–Richardson rule for Type A)

### Why precompilation matters

Without precompilation, the first call to a method triggers just-in-time (JIT) compilation,
which can take hundreds of milliseconds. With precompilation, these methods are ready to use
immediately:

```julia
using Semisimple

# First call is fast due to precompilation
@time degree(fundamental_weight(TypeE{8}, 1))  # ~0.001s

# Without precompilation, this would take ~0.5s for the first call
```

### What is NOT precompiled

Operations involving:
- **Product Dynkin types** (e.g., `ProductDynkinType{Tuple{TypeA{2}, TypeB{3}}}`)
- **Rank ≥ 10 simple types** (e.g., `TypeA{15}`)
- **Specific high-dimensional computations** (e.g., `tensor_product(ω₇, ω₇)` for E₈)

These will experience first-call latency but will be fast on subsequent calls (after JIT compilation).

## Performance characteristics

### Compile-time vs. run-time

Semisimple.jl leverages Julia's type system and `@generated` functions to move many computations
to compile time:

| Compile-Time (Type-Level) | Run-Time |
|---------------------------|----------|
| Dynkin type classification | Weight coordinate values |
| Rank of Dynkin type | Weight lattice arithmetic |
| Cartan matrix entries | Weyl orbit traversal |
| Root system enumeration | Freudenthal recursion |
| Weyl denominator product | Character multiplication |

This means that `cartan_matrix(TypeE{8})` produces a compile-time constant `SMatrix`
that is embedded directly into your compiled code — there's no matrix allocation at runtime.

### Memory usage

| Operation | Memory Footprint | Notes |
|-----------|-----------------|-------|
| `RootSystem{TypeE{8}}` | ~15 KB | Singleton, cached per type |
| `WeightLatticeElem` | 8R bytes | R = rank; stored as `SVector{R,Int}` |
| `WeylGroupElem` | ~40 + L bytes | Word stored as `Vector{UInt8}`; L = word length |
| `WeylCharacter` | ~24 + 40N bytes | N = number of terms in the character |
| Freudenthal cache (E₈ adjoint) | ~40 KB | 3,875 weight multiplicities |

For large-scale computations (e.g., thousands of E₈ tensor products), the
character-related LRU caches will automatically evict old entries once their
memory budget is reached.  Use [`configure_caches!`](@ref) to increase the
budget, or [`clear_caches!`](@ref) to free memory immediately.

### Asymptotic complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `degree(λ)` | O(N²) | N = number of positive roots |
| `freudenthal_formula(λ)` | O(M·N) | M = |{μ : μ ≤ λ}| dominant weights |
| `tensor_product(λ, μ)` (BK) | O(M·W·d) | W = Weyl group order, d = dim V(smaller weight) |
| `tensor_product(λ, μ)` (LR, Type A) | O(n³) | n = max(|λ|, |μ|); much faster than BK |
| `symmetric_power(λ, k)` | O(k²·T) | T = cost of one tensor product |
| `weyl_orbit(λ)` | O(W·R·R) | W = orbit size ≤ Weyl order, R = rank |

For reproducible performance measurements, see the benchmark scripts in
`benchmark/`.

## Type stability

Semisimple.jl is designed for **complete type stability**:

```julia
using Semisimple

ω₁ = fundamental_weight(TypeE{8}, 1)
typeof(ω₁)  # WeightLatticeElem{TypeE{8}, 8} — concrete type

ch = freudenthal_formula(ω₁)
typeof(ch)  # Dict{SVector{8, Int64}, Int64} — concrete type

result = tensor_product(ω₁, ω₁)
typeof(result)  # WeylCharacter{TypeE{8}, 8} — concrete type
```

All public APIs return concrete types, enabling aggressive compiler optimizations.
There are **no type instabilities** in hot paths.

## Numerical precision

All computations use **exact integer arithmetic** — there are no floating-point operations:

- Weights are `SVector{R, Int}` — exact integer vectors
- Multiplicities are `Int` — exact integer counts
- Dimensions are computed exactly (Weyl dimension formula uses `BigInt` for large products)
- Inner products use scaled integer forms to avoid division

This means:
- **No numerical stability concerns** — safe for arbitrarily large representations
- **Overflow protection** — dimension computations automatically promote to `BigInt` when needed

Example:
```julia
julia> ω7 = fundamental_weight(TypeE{8}, 7);

julia> degree(ω7)  # 147,250 × 2⁶⁰ — too large for Int64
170141183460469137866240

julia> typeof(degree(ω7))
BigInt
```

## Thread safety

!!! warning "Some caches are not thread-safe"
    The small `Dict` singleton/type-data caches are protected by locks where
    they are populated through public APIs. The bounded `LRU` character caches
    are not synchronized, so concurrent cache-populating calls such as
    [`dominant_character`](@ref), [`tensor_product`](@ref),
    [`symmetric_power`](@ref), or [`exterior_power`](@ref) can race.

    **Safe:** Using Semisimple.jl from a single thread (the default)

    **Safe:** Read-only operations from multiple threads after warming up caches

    **Unsafe:** Calling LRU-cache-populating representation operations from
    multiple threads simultaneously

If you need parallel computation, populate caches in a single-threaded warm-up phase,
then perform read-only operations in parallel.

## Comparison with LiE

Semisimple.jl reimplements many algorithms from the [LiE computer algebra system](http://wwwmathlabo.univ-poitiers.fr/~maavl/LiE/).
Key differences:

| Aspect | LiE (C) | Semisimple.jl (Julia) |
|--------|---------|----------------|
| **Language** | C (CWEB literate programming) | Julia (pure Julia) |
| **Type system** | Runtime `group` structs | Compile-time Dynkin type parameters |
| **Cartan matrices** | Runtime matrix allocation | Compile-time `SMatrix` constants |
| **Caching** | Permanent "long-life" objects | Bounded `LRU` caches + `Dict` singletons |
| **Hot performance** | Fast (compiled C) | Fast (JIT-compiled, with caching) |
| **Cold performance** | Instant (no compilation) | Slow first call (JIT overhead) |

For hot operations (cached, precompiled), Semisimple.jl matches or exceeds LiE's performance.
For cold operations, LiE is faster due to no JIT compilation delay.

## Implementation philosophy

Semisimple.jl follows these design principles:

1. **Type-level dispatch** — Use Julia's type system to specialize code for each Dynkin type
2. **Compile-time constants** — Leverage `@generated` functions to embed mathematical data
3. **Immutability** — All core types are immutable for thread safety and optimization
4. **Caching** — Trade memory for speed by memoizing expensive computations
5. **Minimal dependencies** — StaticArrays.jl, LRUCache.jl, PrecompileTools.jl, Preferences.jl, and LinearAlgebra (stdlib)
6. **Pure Julia** — No C/Fortran, enabling introspection and compilation to other targets

These principles enable aggressive compiler optimizations while maintaining mathematical rigor.

### Weyl orbit traversal

Weyl orbits are computed by the internal module `Weylloop.jl`, which
implements LiE-style systematic orbit traversal.  Rather than a hash-set BFS
that scales with orbit size, it converts weight coordinates to the **ε-basis**
where classical Weyl subgroups act as permutations (type A) or
permutations + sign flips (types B/C/D).  Orbits are enumerated via
lexicographic permutation generation and Gray-code sign flips, eliminating
the ``O(|\text{orbit}|)`` hash-set overhead that would otherwise dominate for
large orbits (e.g., E₈ orbits with millions of elements).  For exceptional
types, precomputed coset representatives reduce the problem to the classical
case.

## API reference

```@docs
clear_all_caches!
clear_caches!
configure_caches!
cache_info
```
## Internals reference

These are internal functions not part of the public API.  They are
documented here for contributors and advanced users.

### Root system internals

```@docs
Semisimple._root_system_cache
Semisimple._compute_positive_roots_and_reflections
Semisimple._make_root_system_runtime
```

### Weyl group internals

```@docs
Semisimple._weyl_denominator
Semisimple._weyl_dim_scaled_roots
Semisimple._explain_rmul
Semisimple.weylloop
Semisimple._positive_roots_runtime
```

### Cache internals

```@docs
Semisimple._apply_cache_preferences!
```

### Character internals

```@docs
Semisimple.dot_reduce
Semisimple.brauer_klimyk
Semisimple._brauer_klimyk_dominant
Semisimple._vdecomp
Semisimple._tensor_characters
```

### Littlewood–Richardson internals (type A)

```@docs
Semisimple._weight_to_partition
Semisimple._partition_to_weight
Semisimple._lr_coefficients
Semisimple._n_tableaux
```

### Plethysm internals (Murnaghan–Nakayama)

```@docs
Semisimple._partitions
Semisimple._classord
Semisimple._mn_char_val
Semisimple._mn_recurse!
```
