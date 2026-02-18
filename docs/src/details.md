# Implementation Details

This page covers performance considerations, caching mechanisms, precompilation,
and other implementation details of Lie.jl.

## Caching

Lie.jl uses several internal caches to avoid recomputing expensive results. Understanding
these caches is important for benchmarking and memory management.

### Available Caches

Lie.jl maintains six internal caches:

| Cache | Module Variable | Purpose |
|-------|----------------|---------|
| Root system cache | `Lie._root_system_cache` | Singleton `RootSystem` instances per Dynkin type |
| Longest Weyl element cache | `Lie._longest_element_cache` | Cached longest element `w₀` per Dynkin type |
| Dominant character cache | `Lie._dominant_character_cache` | Dominant weight multiplicities from Freudenthal's formula |
| Tensor product cache | `Lie._tensor_cache` | Tensor product decompositions |
| Symmetric power cache | `Lie._symmetric_power_cache` | Symmetric power decompositions |
| Exterior power cache | `Lie._exterior_power_cache` | Exterior power decompositions |

All caches are global `Dict` objects that persist for the lifetime of the Julia session.

!!! note "Why the dominant character cache matters"
    Benchmarks show that the dominant character cache (formerly called the
    Freudenthal cache) provides a **2×–30× speedup** for downstream operations.
    Tensor products see 5×–30× improvement, symmetric/exterior powers 1.4×–14×,
    and plethysms 2.9×–5.7×. This is because many operations (Newton–Girard
    recurrence, Brauer–Klimyk, plethysm) call [`dominant_character`](@ref)
    repeatedly for the same highest weights.

### Inspecting Caches

You can inspect cache contents and sizes at any time:

```julia
using Lie

# Check cache sizes
println("Dominant character cache: ", length(Lie._dominant_character_cache), " entries")
println("Tensor cache: ", length(Lie._tensor_cache), " entries")

# Populate some caches by doing computations
ω₁ = fundamental_weight(TypeE{8}, 1)
freudenthal_formula(ω₁)
tensor_product(ω₁, ω₁)

# Check again
println("Dominant character cache: ", length(Lie._dominant_character_cache), " entries")
println("Tensor cache: ", length(Lie._tensor_cache), " entries")

# Inspect cache keys (to see what's cached)
for key in keys(Lie._tensor_cache)
    println("Cached tensor product: ", key)
end
```

### Clearing Caches

#### Clearing All Caches

Use [`clear_all_caches!`](@ref) to empty all caches at once:

```julia
using Lie

# Do some computations
ω₁ = fundamental_weight(TypeA{2}, 1)
tensor_product(ω₁, ω₁)
freudenthal_formula(ω₁)
symmetric_power(ω₁, 3)

# Clear everything
clear_all_caches!()
```

This is particularly useful for:
- **Benchmarking cold-start performance** — measure how long operations take without cached results
- **Memory management** — free memory after large computations (e.g., after computing many E₈ tensor products)
- **Reproducible testing** — ensure tests start from a clean state

#### Clearing Individual Caches

You can also clear caches individually using `empty!`:

```julia
using Lie

# Clear only the Freudenthal cache
empty!(Lie._dominant_character_cache)

# Clear only the tensor product cache
empty!(Lie._tensor_cache)

# Clear only the symmetric power cache
empty!(Lie._symmetric_power_cache)

# Clear only the exterior power cache
empty!(Lie._exterior_power_cache)

# Clear only the root system cache (rarely needed)
empty!(Lie._root_system_cache)

# Clear only the longest element cache (rarely needed)
empty!(Lie._longest_element_cache)
```

!!! tip "When to clear individual caches"
    The root system and longest element caches are typically small and cheap to populate,
    so there's rarely a reason to clear them. The character-related caches
    (dominant character, tensor, symmetric/exterior power) can grow large and may
    benefit from selective clearing between different computation phases.

### Cache Invalidation

Caches are **never automatically invalidated**. Once a result is computed and cached,
it persists until:
- You explicitly clear the cache (via `clear_all_caches!()` or `empty!(...)`)
- Your Julia session ends

This is safe because:
- Dynkin types are immutable compile-time constants
- Weights are immutable `SVector` objects
- All cached functions are pure (same inputs always produce same outputs)

## Precompilation

Lie.jl precompiles many commonly-used methods to reduce first-call latency. When you
load the package with `using Lie`, the precompilation work has already been done.

### What Gets Precompiled

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
- `weyl_orbit` (Weyl orbit generation)
- Weyl group actions (`*` operator for roots and weights with Weyl elements)
- `freudenthal_formula` (weight multiplicities)
- `dot_reduce` (weight normalization)
- `lr_tensor_product` (Littlewood–Richardson rule for Type A)

### Why Precompilation Matters

Without precompilation, the first call to a method triggers just-in-time (JIT) compilation,
which can take hundreds of milliseconds. With precompilation, these methods are ready to use
immediately:

```julia
using Lie

# First call is fast due to precompilation
@time degree(fundamental_weight(TypeE{8}, 1))  # ~0.001s

# Without precompilation, this would take ~0.5s for the first call
```

### What Is NOT Precompiled

Operations involving:
- **Product Dynkin types** (e.g., `ProductDynkinType{Tuple{TypeA{2}, TypeB{3}}}`)
- **Rank ≥ 10 simple types** (e.g., `TypeA{15}`)
- **Specific high-dimensional computations** (e.g., `tensor_product(ω₇, ω₇)` for E₈)

These will experience first-call latency but will be fast on subsequent calls (after JIT compilation).

## Performance Characteristics

### Compile-Time vs. Run-Time

Lie.jl leverages Julia's type system and `@generated` functions to move many computations
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

### Memory Usage

| Operation | Memory Footprint | Notes |
|-----------|-----------------|-------|
| `RootSystem{TypeE{8}}` | ~15 KB | Singleton, cached per type |
| `WeightLatticeElem` | 8R bytes | R = rank; stored as `SVector{R,Int}` |
| `WeylGroupElem` | 8L bytes | L = length of reduced word |
| `WeylCharacter` | ~24 + 40N bytes | N = number of terms in the character |
| Freudenthal cache (E₈ adjoint) | ~40 KB | 3,875 weight multiplicities |

For large-scale computations (e.g., thousands of E₈ tensor products), the character-related
caches can grow to hundreds of megabytes. Use [`clear_all_caches!`](@ref) periodically
if memory becomes a concern.

### Asymptotic Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `degree(λ)` | O(N²) | N = number of positive roots |
| `freudenthal_formula(λ)` | O(M·N) | M = |{μ : μ ≤ λ}| dominant weights |
| `tensor_product(λ, μ)` (BK) | O(M·W·d) | W = Weyl group order, d = dim V(smaller weight) |
| `tensor_product(λ, μ)` (LR, Type A) | O(n³) | n = max(|λ|, |μ|); much faster than BK |
| `symmetric_power(λ, k)` | O(k²·T) | T = cost of one tensor product |
| `weyl_orbit(λ)` | O(W·R·R) | W = orbit size ≤ Weyl order, R = rank |

For E₈:
- Weyl group order: 696,729,600
- Positive roots: 120
- Typical Freudenthal run (e.g., fundamental weight): 0.01–1s
- Hot tensor product (both weights small): 0.0001–0.1s
- Cold tensor product (one large): 1–100s

## Type Stability

Lie.jl is designed for **complete type stability**:

```julia
using Lie

ω₁ = fundamental_weight(TypeE{8}, 1)
typeof(ω₁)  # WeightLatticeElem{TypeE{8}, 8} — concrete type

ch = freudenthal_formula(ω₁)
typeof(ch)  # Dict{SVector{8, Int64}, Int64} — concrete type

result = tensor_product(ω₁, ω₁)
typeof(result)  # WeylCharacter{TypeE{8}, 8} — concrete type
```

All public APIs return concrete types, enabling aggressive compiler optimizations.
There are **no type instabilities** in hot paths.

## Numerical Precision

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
julia> ω₇ = fundamental_weight(TypeE{8}, 7);

julia> degree(ω₇)  # 147,250 × 2⁶⁰ — too large for Int64
170141183460469137866240

julia> typeof(degree(ω₇))
BigInt
```

## Thread Safety

!!! warning "Caches are NOT thread-safe"
    The internal caches are ordinary `Dict` objects without synchronization.
    Concurrent writes from multiple threads can lead to race conditions.

    **Safe:** Using Lie.jl from a single thread (the default)

    **Safe:** Read-only operations from multiple threads after warming up caches

    **Unsafe:** Calling cache-populating operations (e.g., `freudenthal_formula`,
    `tensor_product`) from multiple threads simultaneously

If you need parallel computation, populate caches in a single-threaded warm-up phase,
then perform read-only operations in parallel.

## Comparison with LiE

Lie.jl reimplements many algorithms from the [LiE computer algebra system](http://wwwmathlabo.univ-poitiers.fr/~maavl/LiE/).
Key differences:

| Aspect | LiE (C) | Lie.jl (Julia) |
|--------|---------|----------------|
| **Language** | C (CWEB literate programming) | Julia (pure Julia) |
| **Type system** | Runtime `group` structs | Compile-time Dynkin type parameters |
| **Cartan matrices** | Runtime matrix allocation | Compile-time `SMatrix` constants |
| **Caching** | Permanent "long-life" objects | `Dict` caches |
| **Hot performance** | Fast (compiled C) | Fast (JIT-compiled, with caching) |
| **Cold performance** | Instant (no compilation) | Slow first call (JIT overhead) |

For hot operations (cached, precompiled), Lie.jl matches or exceeds LiE's performance.
For cold operations, LiE is faster due to no JIT compilation delay.

## Implementation Philosophy

Lie.jl follows these design principles:

1. **Type-level dispatch** — Use Julia's type system to specialize code for each Dynkin type
2. **Compile-time constants** — Leverage `@generated` functions to embed mathematical data
3. **Immutability** — All core types are immutable for thread safety and optimization
4. **Caching** — Trade memory for speed by memoizing expensive computations
5. **Zero dependencies** — Only StaticArrays.jl and LinearAlgebra stdlib
6. **Pure Julia** — No C/Fortran, enabling introspection and compilation to other targets

These principles enable aggressive compiler optimizations while maintaining mathematical rigor.

## API Reference

```@docs
clear_all_caches!
```
