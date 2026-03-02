# ═══════════════════════════════════════════════════════════════════════════════
#  tensor_product benchmark: Lie.jl vs Oscar.jl
#
#  Compares Lie.jl's `tensor_product` (Brauer–Klimyk on Freudenthal character,
#  with LR shortcut for Type A) against Oscar.jl's `tensor_product_decomposition`
#  (Klimyk's formula over heap-allocated WeightLatticeElems, no persistent
#  cross-call cache).
#
#  §1 — general tensor product: cases from bench.jl §6.
#  §2 — Type A LR vs BK vs Oscar: cases from bench.jl §7.
#
#  Usage:
#    julia --project=. benchmark/tensor_product.jl
#    julia --project=. benchmark/tensor_product.jl --no-oscar
#
#  Oscar requirement:
#    Oscar.jl is NOT in Lie.jl's Project.toml (heavyweight dependency).
#    To benchmark Oscar either:
#      (a) activate a separate environment that has Oscar installed, or
#      (b) temporarily  pkg> add Oscar  in the current project.
#    If Oscar cannot be loaded the Oscar columns are skipped gracefully.
#
#  Caching notes:
#    Lie.jl    — cache cleared *inside* every timed call via clear_all_caches!()
#                (both tensor_product / lr_tensor_product / brauer_klimyk paths).
#                This measures pure cold computation comparable to Oscar.
#    Oscar     — no persistent cross-call cache; every call is "cold".
#                The RootSystem object is pre-constructed once outside timing
#                so its internal caches (positive_roots, weyl_group, …) are
#                warm — analogous to Lie.jl's type specialisation / JIT.
# ═══════════════════════════════════════════════════════════════════════════════

using BenchmarkTools
using Printf
using Lie
using StaticArrays
using Statistics: median

const NO_OSCAR = "--no-oscar" in ARGS

# ─── Try to load Oscar ────────────────────────────────────────────────────────

const OSCAR_AVAILABLE = if NO_OSCAR
  @info "Oscar benchmarks disabled via --no-oscar."
  false
else
  try
    @eval Main begin
      using Oscar: Oscar
    end
    true
  catch e
    @warn "Oscar.jl not available — comparison will be Lie-only." exception = e
    false
  end
end

# ─── Formatting helpers ───────────────────────────────────────────────────────

function fmt_bytes(b)
  b == 0 && return "0 B"
  b < 1024 && return @sprintf("%d B", b)
  b < 1024^2 && return @sprintf("%.1f KiB", b / 1024)
  b < 1024^3 && return @sprintf("%.1f MiB", b / 1024^2)
  return @sprintf("%.1f GiB", b / 1024^3)
end

fmt_time(ns) =
  if ns < 1e3
    @sprintf("%.0f ns", ns)
  elseif ns < 1e6
    @sprintf("%.1f μs", ns / 1e3)
  elseif ns < 1e9
    @sprintf("%.2f ms", ns / 1e6)
  else
    @sprintf("%.2f s", ns / 1e9)
  end

# ─── Oscar type mapping ───────────────────────────────────────────────────────

oscar_cartan(::Type{TypeA{N}}) where {N} = (:A, N)
oscar_cartan(::Type{TypeB{N}}) where {N} = (:B, N)
oscar_cartan(::Type{TypeC{N}}) where {N} = (:C, N)
oscar_cartan(::Type{TypeD{N}}) where {N} = (:D, N)
oscar_cartan(::Type{TypeE{6}}) = (:E, 6)
oscar_cartan(::Type{TypeE{7}}) = (:E, 7)
oscar_cartan(::Type{TypeE{8}}) = (:E, 8)
oscar_cartan(::Type{TypeF4}) = (:F, 4)
oscar_cartan(::Type{TypeG2}) = (:G, 2)

# ─── Benchmark kernel functions ───────────────────────────────────────────────

# Cold Lie.jl: clear cache inside timing so every sample is a fresh computation.
function _bench_lie_tensor_cold(
  ::Type{DT}, c1::NTuple{R,Int}, c2::NTuple{R,Int}
) where {DT,R}
  Lie.clear_all_caches!()
  λ = WeightLatticeElem(DT, SVector{R,Int}(c1))
  μ = WeightLatticeElem(DT, SVector{R,Int}(c2))
  tensor_product(λ, μ)
end

# Cold LR: LR is combinatorial (no Freudenthal cache) but we clear anyway for
# consistency with the "cold" contract used throughout this file.
function _bench_lie_lr_cold(
  ::Type{TypeA{N}}, c1::NTuple{N,Int}, c2::NTuple{N,Int}
) where {N}
  Lie.clear_all_caches!()
  λ = WeightLatticeElem(TypeA{N}, SVector{N,Int}(c1))
  μ = WeightLatticeElem(TypeA{N}, SVector{N,Int}(c2))
  lr_tensor_product(λ, μ)
end

# Cold BK: Freudenthal cache cleared before computing the smaller character.
function _bench_lie_bk_cold(
  ::Type{TypeA{N}}, c1::NTuple{N,Int}, c2::NTuple{N,Int}
) where {N}
  Lie.clear_all_caches!()
  λ = WeightLatticeElem(TypeA{N}, SVector{N,Int}(c1))
  μ = WeightLatticeElem(TypeA{N}, SVector{N,Int}(c2))
  if Lie.degree(λ) > Lie.degree(μ)
    Lie.brauer_klimyk(Lie.freudenthal_formula(μ), λ)
  else
    Lie.brauer_klimyk(Lie.freudenthal_formula(λ), μ)
  end
end

# Oscar: RootSystem pre-constructed outside timing (warm, analogous to JIT).
function _bench_oscar_tensor(oscar_R, ivec1::Vector{Int}, ivec2::Vector{Int})
  Main.Oscar.tensor_product_decomposition(oscar_R, ivec1, ivec2)
end

# ─── Result storage ───────────────────────────────────────────────────────────

struct TPResult
  label::String
  lie_ns::Float64
  lie_allocs::Int
  oscar_ns::Float64
  oscar_allocs::Int
  n_irreps::Int
end

struct LRBKResult
  label::String
  lr_ns::Float64
  lr_allocs::Int
  bk_ns::Float64
  bk_allocs::Int
  oscar_ns::Float64
  oscar_allocs::Int
  n_irreps::Int
end

const TP_RESULTS = TPResult[]
const LRBK_RESULTS = LRBKResult[]

# ─── Section-header printers ──────────────────────────────────────────────────

function section_header_tp(title)
  println()
  println("="^110)
  println("  ", title)
  println("="^110)
  if OSCAR_AVAILABLE
    @printf(
      "  %-54s  %-21s  %-21s  %s\n",
      "case", "Lie.jl (cold min)", "Oscar (min)", "speedup Oscar/Lie",
    )
    println("  ", "-"^104)
  else
    @printf("  %-54s  %-21s\n", "case", "Lie.jl (cold min)")
    println("  ", "-"^78)
  end
end

function section_header_lrbk(title)
  println()
  println("="^126)
  println("  ", title)
  println("="^126)
  if OSCAR_AVAILABLE
    @printf(
      "  %-46s  %-21s  %-21s  %-21s  %s\n",
      "case", "Lie LR (cold min)", "Lie BK (cold min)", "Oscar (min)",
      "LR/Oscar speedup",
    )
    println("  ", "-"^120)
  else
    @printf(
      "  %-46s  %-21s  %-21s  %s\n",
      "case", "Lie LR (cold min)", "Lie BK (cold min)", "LR/BK speedup",
    )
    println("  ", "-"^94)
  end
end

# ─── Per-case runners ─────────────────────────────────────────────────────────

"""
    run_tensor_case(label, DT, c1, c2; samples, samples_oscar, skip_large_oscar)

Benchmark `tensor_product(V(c1), V(c2))` for Dynkin type `DT`.
Lie.jl is always benchmarked cold (cache cleared per sample).
"""
function run_tensor_case(
  label, ::Type{DT}, c1, c2;
  samples=20,
  samples_oscar=20,
  skip_large_oscar=false,
) where {DT}
  R_n = rank(DT)
  c1_tup = NTuple{R_n,Int}(c1)
  c2_tup = NTuple{R_n,Int}(c2)

  # warmup: compile all code paths and populate any warm cache used outside timing
  _bench_lie_tensor_cold(DT, c1_tup, c2_tup)
  tp = tensor_product(
    WeightLatticeElem(DT, SVector{R_n,Int}(c1_tup)),
    WeightLatticeElem(DT, SVector{R_n,Int}(c2_tup)),
  )
  n_irreps = length(tp.terms)

  b_lie = @benchmark _bench_lie_tensor_cold($DT, $c1_tup, $c2_tup) evals = 1 samples =
    samples

  b_oscar = nothing
  if OSCAR_AVAILABLE && !skip_large_oscar
    try
      ct = oscar_cartan(DT)
      oscar_R = Main.Oscar.root_system(ct...)
      iv1 = collect(Int, c1_tup)
      iv2 = collect(Int, c2_tup)
      _bench_oscar_tensor(oscar_R, iv1, iv2)   # warmup — fills RootSystem caches
      b_oscar = @benchmark _bench_oscar_tensor($oscar_R, $iv1, $iv2) evals = 1 samples =
        samples_oscar
    catch e
      @warn "Oscar benchmark failed for \"$label\"" exception = e
    end
  end

  lie_ns = Float64(minimum(b_lie).time)
  lie_a = minimum(b_lie).allocs
  osc_ns = b_oscar === nothing ? NaN : Float64(minimum(b_oscar).time)
  osc_a = b_oscar === nothing ? 0 : minimum(b_oscar).allocs

  lie_str = @sprintf("%-9s %6d alloc", fmt_time(lie_ns), lie_a)
  osc_str = if b_oscar !== nothing
    @sprintf("%-9s %6d alloc", fmt_time(osc_ns), osc_a)
  elseif OSCAR_AVAILABLE
    skip_large_oscar ? "    (skipped)" : "     (failed)"
  else
    "—"
  end

  meta = @sprintf("irreps=%d", n_irreps)

  if OSCAR_AVAILABLE
    spstr =
      (!isnan(osc_ns) && lie_ns > 0) ? @sprintf("%6.1fx", osc_ns / lie_ns) : "     —"
    @printf(
      "  %-54s  %-21s  %-21s  %s  [%s]\n",
      label, lie_str, osc_str, spstr, meta,
    )
  else
    @printf("  %-54s  %-21s  [%s]\n", label, lie_str, meta)
  end

  push!(TP_RESULTS, TPResult(label, lie_ns, lie_a, osc_ns, osc_a, n_irreps))
end

"""
    run_lr_bk_case(label, ::Type{TypeA{N}}, c1, c2; samples, samples_oscar)

3-way benchmark for Type A: LR (cold) vs BK (cold) vs Oscar.
"""
function run_lr_bk_case(
  label, ::Type{TypeA{N}}, c1, c2;
  samples=100,
  samples_oscar=50,
  skip_large_oscar=false,
) where {N}
  c1_tup = NTuple{N,Int}(c1)
  c2_tup = NTuple{N,Int}(c2)

  # warmup
  _bench_lie_lr_cold(TypeA{N}, c1_tup, c2_tup)
  _bench_lie_bk_cold(TypeA{N}, c1_tup, c2_tup)
  tp = lr_tensor_product(
    WeightLatticeElem(TypeA{N}, SVector{N,Int}(c1_tup)),
    WeightLatticeElem(TypeA{N}, SVector{N,Int}(c2_tup)),
  )
  n_irreps = length(tp.terms)

  b_lr = @benchmark _bench_lie_lr_cold($(TypeA{N}), $c1_tup, $c2_tup) evals = 1 samples =
    samples
  b_bk = @benchmark _bench_lie_bk_cold($(TypeA{N}), $c1_tup, $c2_tup) evals = 1 samples =
    samples

  b_oscar = nothing
  if OSCAR_AVAILABLE && !skip_large_oscar
    try
      oscar_R = Main.Oscar.root_system(:A, N)
      iv1 = collect(Int, c1_tup)
      iv2 = collect(Int, c2_tup)
      _bench_oscar_tensor(oscar_R, iv1, iv2)   # warmup
      b_oscar = @benchmark _bench_oscar_tensor($oscar_R, $iv1, $iv2) evals = 1 samples =
        samples_oscar
    catch e
      @warn "Oscar benchmark failed for \"$label\"" exception = e
    end
  end

  lr_ns = Float64(minimum(b_lr).time)
  lr_a = minimum(b_lr).allocs
  bk_ns = Float64(minimum(b_bk).time)
  bk_a = minimum(b_bk).allocs
  osc_ns = b_oscar === nothing ? NaN : Float64(minimum(b_oscar).time)
  osc_a = b_oscar === nothing ? 0 : minimum(b_oscar).allocs

  lr_str = @sprintf("%-9s %6d alloc", fmt_time(lr_ns), lr_a)
  bk_str = @sprintf("%-9s %6d alloc", fmt_time(bk_ns), bk_a)
  osc_str = if b_oscar !== nothing
    @sprintf("%-9s %6d alloc", fmt_time(osc_ns), osc_a)
  elseif OSCAR_AVAILABLE
    skip_large_oscar ? "    (skipped)" : "     (failed)"
  else
    "—"
  end

  meta = @sprintf("irreps=%d", n_irreps)

  if OSCAR_AVAILABLE
    spstr =
      (!isnan(osc_ns) && lr_ns > 0) ? @sprintf("%6.1fx", osc_ns / lr_ns) : "     —"
    @printf(
      "  %-46s  %-21s  %-21s  %-21s  %s  [%s]\n",
      label, lr_str, bk_str, osc_str, spstr, meta,
    )
  else
    lr_bk_str = lr_ns > 0 ? @sprintf("%6.1fx", bk_ns / lr_ns) : "     —"
    @printf(
      "  %-46s  %-21s  %-21s  %s  [%s]\n",
      label, lr_str, bk_str, lr_bk_str, meta,
    )
  end

  push!(
    LRBK_RESULTS,
    LRBKResult(label, lr_ns, lr_a, bk_ns, bk_a, osc_ns, osc_a, n_irreps),
  )
end

# ═══════════════════════════════════════════════════════════════════════════════
#  §1  General tensor product  (bench.jl §6 cases)
# ═══════════════════════════════════════════════════════════════════════════════

section_header_tp("§1  General tensor product  (bench.jl §6 cases)")

run_tensor_case("A₃  V(ω₁)⊗V(ω₁)", TypeA{3}, [1, 0, 0], [1, 0, 0])
run_tensor_case("A₃  V(ω₁)⊗V(ω₂)", TypeA{3}, [1, 0, 0], [0, 1, 0])
run_tensor_case("A₃  V(ω₂)⊗V(ω₂)", TypeA{3}, [0, 1, 0], [0, 1, 0])
run_tensor_case("A₄  V(ω₁)⊗V(ω₁)", TypeA{4}, [1, 0, 0, 0], [1, 0, 0, 0])
run_tensor_case("B₃  V(ω₁)⊗V(ω₁)", TypeB{3}, [1, 0, 0], [1, 0, 0])
run_tensor_case("B₃  V(ω₃)⊗V(ω₃)  (spinor)", TypeB{3}, [0, 0, 1], [0, 0, 1])
run_tensor_case("C₃  V(ω₁)⊗V(ω₁)", TypeC{3}, [1, 0, 0], [1, 0, 0])
run_tensor_case("D₄  V(ω₁)⊗V(ω₃)", TypeD{4}, [1, 0, 0, 0], [0, 0, 1, 0])
run_tensor_case("G₂  V(ω₁)⊗V(ω₁)", TypeG2, [1, 0], [1, 0])
run_tensor_case("G₂  V(ω₂)⊗V(ω₂)  (adj)", TypeG2, [0, 1], [0, 1])
run_tensor_case("F₄  V(ω₄)⊗V(ω₄)", TypeF4, [0, 0, 0, 1], [0, 0, 0, 1])
run_tensor_case(
  "E₆  V(ω₁)⊗V(ω₆)",
  TypeE{6}, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1],
)
run_tensor_case(
  "E₆  V(ω₁)⊗V(ω₁)",
  TypeE{6}, [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0];
  samples=10, samples_oscar=10,
)
run_tensor_case(
  "E₈  V(ω₈)⊗V(ω₈)  (adj)",
  TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1];
  samples=10, samples_oscar=10,
)
run_tensor_case(
  "E₈  V(ω₁)⊗V(ω₈)",
  TypeE{8}, [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1];
  samples=10, samples_oscar=10,
)
run_tensor_case(
  "E₈  V(ω₁)⊗V(ω₄)",
  TypeE{8}, [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0];
  samples=5, samples_oscar=5,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  §2  Type A — LR vs BK vs Oscar  (bench.jl §7 cases)
# ═══════════════════════════════════════════════════════════════════════════════

section_header_lrbk("§2  Type A — LR vs BK vs Oscar  (bench.jl §7 cases)")

run_lr_bk_case("A₃  V(ω₁)⊗V(ω₁)", TypeA{3}, [1, 0, 0], [1, 0, 0])
run_lr_bk_case("A₃  V(ω₂)⊗V(ω₂)", TypeA{3}, [0, 1, 0], [0, 1, 0])
run_lr_bk_case("A₃  V(2ω₁)⊗V(2ω₁)", TypeA{3}, [2, 0, 0], [2, 0, 0])
run_lr_bk_case("A₅  V(ω₁)⊗V(ω₁)", TypeA{5}, [1, 0, 0, 0, 0], [1, 0, 0, 0, 0])
run_lr_bk_case("A₅  V(ω₂)⊗V(ω₂)", TypeA{5}, [0, 1, 0, 0, 0], [0, 1, 0, 0, 0])
run_lr_bk_case("A₅  V(2ω₁)⊗V(2ω₁)", TypeA{5}, [2, 0, 0, 0, 0], [2, 0, 0, 0, 0])
run_lr_bk_case(
  "A₅  V(ω₁+ω₂)⊗V(ω₁+ω₄)",
  TypeA{5}, [1, 1, 0, 0, 0], [1, 0, 0, 1, 0],
)
run_lr_bk_case(
  "A₇  V(ω₁)⊗V(ω₁)",
  TypeA{7}, [1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0],
)
run_lr_bk_case(
  "A₇  V(ω₂)⊗V(ω₂)",
  TypeA{7}, [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
)
run_lr_bk_case(
  "A₇  V(ω₃)⊗V(ω₃)",
  TypeA{7}, [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
)
run_lr_bk_case(
  "A₉  V(ω₁)⊗V(ω₁)",
  TypeA{9}, [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
)
run_lr_bk_case(
  "A₉  V(ω₂)⊗V(ω₂)",
  TypeA{9}, [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0],
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════════

println()
println("="^110)
println("  Summary")
println("="^110)
@printf("  §1 general tensor product cases: %d\n", length(TP_RESULTS))
@printf("  §2 Type A LR vs BK cases:        %d\n", length(LRBK_RESULTS))

if OSCAR_AVAILABLE
  valid_tp = filter(r -> !isnan(r.oscar_ns) && r.lie_ns > 0, TP_RESULTS)
  if !isempty(valid_tp)
    ratios = [r.oscar_ns / r.lie_ns for r in valid_tp]
    println()
    @printf("  §1 Oscar vs Lie.jl cold tensor_product — ratio (>1 = Lie faster):\n")
    @printf(
      "    min = %.2fx,  median = %.1fx,  max = %.0fx\n",
      minimum(ratios), median(ratios), maximum(ratios),
    )
    @printf("  Lie.jl faster on %d / %d cases\n", count(>(1.0), ratios), length(ratios))
  end

  valid_lr = filter(r -> !isnan(r.oscar_ns) && r.lr_ns > 0, LRBK_RESULTS)
  valid_bk = filter(r -> !isnan(r.oscar_ns) && r.bk_ns > 0, LRBK_RESULTS)
  if !isempty(valid_lr)
    ratios_lr = [r.oscar_ns / r.lr_ns for r in valid_lr]
    ratios_bk = [r.oscar_ns / r.bk_ns for r in valid_bk]
    println()
    @printf("  §2 Oscar vs Lie.jl LR (cold) — ratio (>1 = LR faster):\n")
    @printf(
      "    min = %.2fx,  median = %.1fx,  max = %.0fx\n",
      minimum(ratios_lr), median(ratios_lr), maximum(ratios_lr),
    )
    @printf("  §2 Oscar vs Lie.jl BK (cold) — ratio (>1 = BK faster):\n")
    @printf(
      "    min = %.2fx,  median = %.1fx,  max = %.0fx\n",
      minimum(ratios_bk), median(ratios_bk), maximum(ratios_bk),
    )
  end
end

println()
