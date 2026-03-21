# ═══════════════════════════════════════════════════════════════════════════════
#  CacheConfig — runtime & preferences-based cache management
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _apply_cache_preferences!()

Read Preferences-based cache settings and resize caches accordingly.
Called from `__init__()` so that user preferences take effect on load.

Recognised preferences (set via `Preferences.set_preferences!`):

| Key                   | Type    | Default               |
|:----------------------|:--------|:----------------------|
| `cache_budget`        | `Int`   | 25% of RAM (≥256 MiB) |
| `dominant_frac`       | `Float64` | 0.30                |
| `tensor_frac`         | `Float64` | 0.40                |
| `sym_power_frac`      | `Float64` | 0.15                |
| `ext_power_frac`      | `Float64` | 0.15                |
"""
function _apply_cache_preferences!()
  budget = @load_preference("cache_budget", _default_cache_budget())::Int
  dom_frac = @load_preference("dominant_frac", _DEFAULT_DOMINANT_FRAC)::Float64
  ten_frac = @load_preference("tensor_frac", _DEFAULT_TENSOR_FRAC)::Float64
  sym_frac = @load_preference("sym_power_frac", _DEFAULT_SYM_POWER_FRAC)::Float64
  ext_frac = @load_preference("ext_power_frac", _DEFAULT_EXT_POWER_FRAC)::Float64

  _resize_caches!(budget, dom_frac, ten_frac, sym_frac, ext_frac)
  return nothing
end

function _resize_caches!(budget::Int, dom_frac::Float64, ten_frac::Float64,
  sym_frac::Float64, ext_frac::Float64)
  LRUCache.resize!(_dominant_character_cache; maxsize=_cache_maxsize(budget, dom_frac))
  LRUCache.resize!(_tensor_cache; maxsize=_cache_maxsize(budget, ten_frac))
  LRUCache.resize!(_symmetric_power_cache; maxsize=_cache_maxsize(budget, sym_frac))
  LRUCache.resize!(_exterior_power_cache; maxsize=_cache_maxsize(budget, ext_frac))
  return nothing
end

"""
    configure_caches!(; budget=nothing, dominant_frac=nothing, tensor_frac=nothing,
                        sym_power_frac=nothing, ext_power_frac=nothing)

Resize the LRU caches at runtime. Unspecified keyword arguments retain their
current values.  The `budget` is in **bytes** and controls the total memory
envelope; the four fraction arguments determine how it is divided among the
caches (they need not sum to 1 — each cache is sized independently as
`budget * frac`).

# Examples
```jldoctest
julia> using Lie

julia> configure_caches!(budget = 512 * 1024^2)  # 512 MiB total
```
"""
function configure_caches!(;
  budget::Union{Int,Nothing}=nothing,
  dominant_frac::Union{Float64,Nothing}=nothing,
  tensor_frac::Union{Float64,Nothing}=nothing,
  sym_power_frac::Union{Float64,Nothing}=nothing,
  ext_power_frac::Union{Float64,Nothing}=nothing,
)
  b = something(budget, _default_cache_budget())
  df = something(dominant_frac, _DEFAULT_DOMINANT_FRAC)
  tf = something(tensor_frac, _DEFAULT_TENSOR_FRAC)
  sf = something(sym_power_frac, _DEFAULT_SYM_POWER_FRAC)
  ef = something(ext_power_frac, _DEFAULT_EXT_POWER_FRAC)

  _resize_caches!(b, df, tf, sf, ef)
  return nothing
end

"""
    clear_caches!()

Empty every internal cache in Lie.jl (both bounded Dict caches and
LRU caches).

# Examples
```jldoctest
julia> using Lie

julia> ω₁ = fundamental_weight(TypeA{2}, 1);

julia> tensor_product(ω₁, ω₁);  # populates caches

julia> clear_caches!()
```
"""
function clear_caches!()
  # Bounded singletons (Dict)
  empty!(_root_system_cache)
  empty!(_positive_roots_set_cache)
  empty!(_longest_element_cache)
  empty!(_coset_reps_cache)
  # LRU caches
  empty!(_dominant_character_cache)
  empty!(_tensor_cache)
  empty!(_symmetric_power_cache)
  empty!(_exterior_power_cache)
  return nothing
end

"""
    cache_info() -> NamedTuple

Return a snapshot of the current cache occupancy.  Each entry is a
`NamedTuple` with fields `length` (number of entries) and `maxsize`
(capacity in bytes).

# Examples
```jldoctest
julia> using Lie

julia> info = cache_info();

julia> info.tensor.length >= 0
true
```
"""
function cache_info()
  return (
    root_system=(length=length(_root_system_cache),),
    positive_roots_set=(length=length(_positive_roots_set_cache),),
    longest_element=(length=length(_longest_element_cache),),
    coset_reps=(length=length(_coset_reps_cache),),
    dominant_character=(length=length(_dominant_character_cache), maxsize=_dominant_character_cache.maxsize),
    tensor=(length=length(_tensor_cache), maxsize=_tensor_cache.maxsize),
    symmetric_power=(length=length(_symmetric_power_cache), maxsize=_symmetric_power_cache.maxsize),
    exterior_power=(length=length(_exterior_power_cache), maxsize=_exterior_power_cache.maxsize),
  )
end
