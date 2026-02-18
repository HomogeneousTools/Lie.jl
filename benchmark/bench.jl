# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmarks for Lie.jl — with regression tracking
#
#  Usage:
#    julia --project=. benchmark/bench.jl               # run + save results
#    julia --project=. benchmark/bench.jl --compare      # compare vs last saved
#    julia --project=. benchmark/bench.jl --save-only    # save without comparing
#
#  Results are saved to benchmark/results/<timestamp>.json
#  The most recent saved result is also copied as benchmark/results/latest.json
# ═══════════════════════════════════════════════════════════════════════════════

using BenchmarkTools
using Printf
using Lie
using StaticArrays
using Dates

# ─── CLI ────────────────────────────────────────────────────────────────────

const COMPARE = "--compare" in ARGS
const SAVE_ONLY = "--save-only" in ARGS

# ─── Result storage ─────────────────────────────────────────────────────────

struct BenchResult
  name::String
  category::String
  min_time_ns::Float64
  median_time_ns::Float64
  mean_time_ns::Float64
  min_allocs::Int
  min_memory::Int      # bytes
  median_allocs::Int
  median_memory::Int   # bytes
end

const ALL_RESULTS = BenchResult[]

# ─── Helpers ─────────────────────────────────────────────────────────────────

function header(name)
  println("\n", "="^80)
  println("  ", name)
  println("="^80)
end

function report(name, b::BenchmarkTools.Trial; category="", extra="")
  t_min = minimum(b).time / 1e6
  t_med = median(b).time / 1e6
  t_mean = mean(b).time / 1e6
  a_min = minimum(b).allocs
  m_min = minimum(b).memory
  a_med = Int(round(median(b).allocs))
  m_med = Int(round(median(b).memory))

  @printf("  %-55s %9.3f ms  %6d allocs  %8s", name, t_min, a_min, fmt_bytes(m_min))
  extra != "" && print("  $extra")
  println()

  push!(
    ALL_RESULTS,
    BenchResult(name, category, minimum(b).time, median(b).time,
      mean(b).time, a_min, m_min, a_med, m_med),
  )
end

function fmt_bytes(b)
  b < 1024 && return @sprintf("%d B", b)
  b < 1024^2 && return @sprintf("%.1f KiB", b / 1024)
  b < 1024^3 && return @sprintf("%.1f MiB", b / 1024^2)
  return @sprintf("%.1f GiB", b / 1024^3)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Apply longest Weyl element to all positive roots
# ═══════════════════════════════════════════════════════════════════════════════

header("1. w₀ action on all positive roots")

function bench_w0_on_roots(::Type{DT}) where {DT}
  W = weyl_group(DT)
  RS = root_system(W)
  w0 = longest_element(W)
  return [α * w0 for α in positive_roots(RS)]
end

for DT in [TypeA{4}, TypeA{6}, TypeB{4}, TypeB{6}, TypeC{5}, TypeD{5}, TypeD{7},
  TypeE{6}, TypeE{7}, TypeE{8}, TypeF4, TypeG2]
  bench_w0_on_roots(DT)
  b = @benchmark bench_w0_on_roots($DT) evals = 1 samples = 300
  report("$(sprint(show,DT())): w₀ · $(n_positive_roots(DT)) roots", b;
    category="w0_action")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  2. conjugate_dominant_weight over a box of weights
# ═══════════════════════════════════════════════════════════════════════════════

header("2. conjugate_dominant_weight on a box of weights")

function bench_conj_dom_box(::Type{DT}, bound) where {DT}
  R = rank(DT)
  count = 0
  for coords in Iterators.product(ntuple(_ -> (-bound):bound, R)...)
    w = WeightLatticeElem(DT, SVector{R,Int}(coords))
    conjugate_dominant_weight(w)
    count += 1
  end
  return count
end

for (DT, bound) in [(TypeA{3}, 5), (TypeA{4}, 3), (TypeB{3}, 5), (TypeB{4}, 3),
  (TypeC{3}, 5), (TypeD{4}, 3), (TypeD{5}, 2),
  (TypeE{6}, 2), (TypeE{7}, 1), (TypeF4, 3), (TypeG2, 8)]
  R = rank(DT)
  n = (2 * bound + 1)^R
  bench_conj_dom_box(DT, bound)
  b = @benchmark bench_conj_dom_box($DT, $bound) evals = 1 samples = 50
  report("$(sprint(show,DT())): box [-$bound,$bound]^$R ($n wts)", b;
    category="conj_dom_weight")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  3. Weyl orbit of interesting weights
# ═══════════════════════════════════════════════════════════════════════════════

header("3. Weyl orbit computation")

function bench_weyl_orbit(::Type{DT}, coords) where {DT}
  R = rank(DT)
  w = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return weyl_orbit(w)
end

orbit_cases = [
  (TypeA{4}, [1, 1, 1, 1], "ρ"),
  (TypeA{6}, [1, 0, 0, 0, 0, 0], "ω₁"),
  (TypeB{4}, [1, 1, 1, 1], "ρ"),
  (TypeB{4}, [0, 0, 0, 1], "ω₄"),
  (TypeC{4}, [1, 0, 0, 0], "ω₁"),
  (TypeD{5}, [1, 1, 1, 1, 1], "ρ"),
  (TypeD{5}, [0, 0, 0, 1, 0], "ω₄"),
  (TypeE{6}, [1, 1, 1, 1, 1, 1], "ρ"),
  (TypeE{7}, [1, 0, 0, 0, 0, 0, 0], "ω₁"),
  (TypeE{8}, [1, 0, 0, 0, 0, 0, 0, 0], "ω₁"),
  (TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], "ω₈"),
  (TypeF4, [1, 1, 1, 1], "ρ"),
  (TypeF4, [1, 0, 0, 0], "ω₁"),
  (TypeG2, [2, 3], "2ω₁+3ω₂"),
  (TypeG2, [1, 1], "ρ"),
]

for (DT, coords, label) in orbit_cases
  orb = bench_weyl_orbit(DT, coords)
  b = @benchmark bench_weyl_orbit($DT, $coords) evals = 1 samples = 50
  report("$(sprint(show,DT())): orbit($label), |W·λ|=$(length(orb))", b;
    category="weyl_orbit")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  4. Weyl dimension formula
# ═══════════════════════════════════════════════════════════════════════════════

header("4. Weyl dimension formula")

function bench_degree_fund(::Type{DT}) where {DT}
  R = rank(DT)
  return [degree(fundamental_weight(DT, i)) for i in 1:R]
end

for DT in [TypeA{4}, TypeA{6}, TypeA{8}, TypeB{4}, TypeB{5}, TypeC{4}, TypeC{5},
  TypeD{5}, TypeD{6}, TypeE{6}, TypeE{7}, TypeE{8}, TypeF4, TypeG2]
  bench_degree_fund(DT)
  b = @benchmark bench_degree_fund($DT) evals = 1 samples = 500
  report("$(sprint(show,DT())): degree(ωᵢ), i=1…$(rank(DT))", b;
    category="dimension_formula")
end

# High-weight representations
function bench_degree_hw(::Type{DT}, coords) where {DT}
  R = rank(DT)
  hw = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return degree(hw)
end

hw_cases = [
  (TypeA{6}, [3, 2, 1, 0, 1, 2], "3ω₁+2ω₂+ω₃+ω₅+2ω₆"),
  (TypeB{4}, [2, 1, 1, 2], "2ω₁+ω₂+ω₃+2ω₄"),
  (TypeD{6}, [1, 1, 1, 1, 1, 1], "ρ"),
  (TypeE{6}, [2, 1, 0, 0, 1, 2], "2ω₁+ω₂+ω₅+2ω₆"),
  (TypeE{8}, [2, 0, 0, 0, 0, 0, 0, 1], "2ω₁+ω₈"),
  (TypeE{8}, [1, 1, 1, 1, 1, 1, 1, 1], "ρ"),
  (TypeF4, [2, 2, 2, 2], "2ρ"),
  # Lübeck (arXiv:2601.18786) — high-dimensional pairs with equal degree
  (TypeA{10}, [0, 9, 0, 0, 0, 0, 0, 0, 0, 0], "9ω₂ [Lübeck A₁₀]"),
  (TypeA{15}, [0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "14ω₂ [Lübeck A₁₅]"),
  (TypeB{8}, [0, 14, 0, 0, 0, 0, 0, 0], "14ω₂ [Lübeck B₈]"),
  (TypeB{10}, [0, 18, 0, 0, 0, 0, 0, 0, 0, 0], "18ω₂ [Lübeck B₁₀]"),
  (TypeD{8}, [0, 13, 0, 0, 0, 0, 0, 0], "13ω₂ [Lübeck D₈]"),
  (TypeD{10}, [0, 17, 0, 0, 0, 0, 0, 0, 0, 0], "17ω₂ [Lübeck D₁₀]"),
  (TypeE{7}, [0, 0, 0, 1, 1, 0, 0], "ω₄+ω₅ [Lübeck E₇]"),
  (TypeE{8}, [1, 0, 1, 0, 0, 0, 0, 0], "ω₁+ω₃ [Lübeck E₈]"),
]

for (DT, coords, label) in hw_cases
  d = bench_degree_hw(DT, coords)
  b = @benchmark bench_degree_hw($DT, $coords) evals = 1 samples = 500
  ds = d < 10^15 ? string(d) : "≈10^$(round(log10(Float64(d)), digits=1))"
  report("$(sprint(show,DT())): degree($label) = $ds", b;
    category="dimension_formula")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  5. Freudenthal formula (weight multiplicities)
# ═══════════════════════════════════════════════════════════════════════════════

header("5. Freudenthal formula")

function bench_freudenthal(::Type{DT}, coords) where {DT}
  R = rank(DT)
  Lie.clear_all_caches!()
  hw = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return freudenthal_formula(hw)
end

freudenthal_cases = [
  (TypeA{3}, [1, 1, 1], "ρ (dim 20)"),
  (TypeA{3}, [2, 0, 2], "2ω₁+2ω₃ (dim 50)"),
  (TypeA{4}, [1, 0, 0, 1], "ω₁+ω₄ (adj, dim 24)"),
  (TypeA{4}, [2, 0, 0, 0], "2ω₁ (dim 15)"),
  (TypeA{4}, [2, 1, 1, 2], "2ω₁+ω₂+ω₃+2ω₄ (dim 6125)"),
  (TypeB{3}, [0, 1, 0], "ω₂ (dim 21)"),
  (TypeB{3}, [1, 0, 0], "ω₁ (dim 7)"),
  (TypeB{4}, [0, 0, 0, 1], "ω₄ (spinor, dim 16)"),
  (TypeC{3}, [1, 0, 0], "ω₁ (dim 6)"),
  (TypeD{4}, [0, 0, 1, 0], "ω₃ (spinor, dim 8)"),
  (TypeD{4}, [1, 0, 0, 0], "ω₁ (dim 8)"),
  (TypeG2, [1, 0], "ω₁ (dim 7)"),
  (TypeG2, [0, 1], "ω₂ (dim 14)"),
  (TypeG2, [1, 1], "ω₁+ω₂ (dim 64)"),
  (TypeG2, [2, 1], "2ω₁+ω₂ (dim 189)"),
  (TypeF4, [0, 0, 0, 1], "ω₄ (dim 26)"),
  (TypeF4, [1, 0, 0, 1], "ω₁+ω₄ [Lübeck] (dim 1053)"),
  (TypeE{6}, [1, 0, 0, 0, 0, 0], "ω₁ (dim 27)"),
  (TypeE{6}, [0, 0, 0, 0, 0, 1], "ω₆ (dim 27)"),
  (TypeE{6}, [0, 0, 1, 0, 0, 0], "ω₃ [Lübeck] (dim 351)"),
  (TypeE{7}, [1, 0, 0, 0, 0, 0, 0], "ω₁ (dim 133)"),
  (TypeE{7}, [0, 0, 0, 1, 1, 0, 0], "ω₄+ω₅ [Lübeck] (dim 1903725824)"),
  (TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], "ω₈ (dim 248)"),
  (TypeE{8}, [1, 0, 1, 0, 0, 0, 0, 0], "ω₁+ω₃ [Lübeck] (dim 8634368000)"),
]

# Per-case sample counts: large E₇/E₈ Lübeck representations are expensive
# when the Freudenthal cache is cleared each iteration.
freudenthal_samples = Dict{Tuple{Type,Vector{Int}},Int}(
  (TypeE{7}, [0, 0, 0, 1, 1, 0, 0]) => 5,
  (TypeE{8}, [1, 0, 1, 0, 0, 0, 0, 0]) => 5,
)

for (DT, coords, label) in freudenthal_cases
  bench_freudenthal(DT, coords)
  nsamp = get(freudenthal_samples, (DT, coords), 50)
  b = @benchmark bench_freudenthal($DT, $coords) evals = 1 samples = nsamp
  report("$(sprint(show,DT())): $label", b; category="freudenthal")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  6. Tensor product decomposition
# ═══════════════════════════════════════════════════════════════════════════════

header("6. Tensor product decomposition")

function bench_tensor(::Type{DT}, c1, c2) where {DT}
  R = rank(DT)
  Lie.clear_all_caches!()
  λ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(c1)))
  μ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(c2)))
  return tensor_product(λ, μ)
end

tensor_cases = [
  (TypeA{3}, [1, 0, 0], [1, 0, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeA{3}, [1, 0, 0], [0, 1, 0], "V(ω₁)⊗V(ω₂)"),
  (TypeA{3}, [0, 1, 0], [0, 1, 0], "V(ω₂)⊗V(ω₂)"),
  (TypeA{4}, [1, 0, 0, 0], [1, 0, 0, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeB{3}, [1, 0, 0], [1, 0, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeB{3}, [0, 0, 1], [0, 0, 1], "V(ω₃)⊗V(ω₃)"),
  (TypeC{3}, [1, 0, 0], [1, 0, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeD{4}, [1, 0, 0, 0], [0, 0, 1, 0], "V(ω₁)⊗V(ω₃)"),
  (TypeG2, [1, 0], [1, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeG2, [0, 1], [0, 1], "V(ω₂)⊗V(ω₂)"),
  (TypeF4, [0, 0, 0, 1], [0, 0, 0, 1], "V(ω₄)⊗V(ω₄)"),
  (TypeE{6}, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], "V(ω₁)⊗V(ω₆)"),
  (TypeE{6}, [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], "V(ω₈)⊗V(ω₈)"),
  (TypeE{8}, [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], "V(ω₁)⊗V(ω₈)"),
  (TypeE{8}, [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], "V(ω₁)⊗V(ω₄)"),
]

for (DT, c1, c2, label) in tensor_cases
  bench_tensor(DT, c1, c2)
  b = @benchmark bench_tensor($DT, $c1, $c2) evals = 1 samples = 20
  tp = bench_tensor(DT, c1, c2)
  report("$(sprint(show,DT())): $label → $(length(tp.terms)) irreps", b;
    category="tensor_product")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  7. Littlewood–Richardson vs Brauer–Klimyk (Type A tensor products)
# ═══════════════════════════════════════════════════════════════════════════════

header("7. Littlewood–Richardson vs Brauer–Klimyk (Type A)")

function bench_lr(::Type{TypeA{N}}, c1, c2) where {N}
  λ = WeightLatticeElem(TypeA{N}, SVector{N,Int}(Tuple(c1)))
  μ = WeightLatticeElem(TypeA{N}, SVector{N,Int}(Tuple(c2)))
  return lr_tensor_product(λ, μ)
end

function bench_bk(::Type{DT}, c1, c2) where {DT}
  R = rank(DT)
  Lie.clear_all_caches!()
  λ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(c1)))
  μ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(c2)))
  if Lie.degree(λ) > Lie.degree(μ)
    Lie.brauer_klimyk(Lie.freudenthal_formula(μ), λ)
  else
    Lie.brauer_klimyk(Lie.freudenthal_formula(λ), μ)
  end
end

lr_bk_cases = [
  (TypeA{3}, [1, 0, 0], [1, 0, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeA{3}, [0, 1, 0], [0, 1, 0], "V(ω₂)⊗V(ω₂)"),
  (TypeA{3}, [2, 0, 0], [2, 0, 0], "V(2ω₁)⊗V(2ω₁)"),
  (TypeA{5}, [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeA{5}, [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], "V(ω₂)⊗V(ω₂)"),
  (TypeA{5}, [2, 0, 0, 0, 0], [2, 0, 0, 0, 0], "V(2ω₁)⊗V(2ω₁)"),
  (TypeA{5}, [1, 1, 0, 0, 0], [1, 0, 0, 1, 0], "V(ω₁+ω₂)⊗V(ω₁+ω₄)"),
  (TypeA{7}, [1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeA{7}, [0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], "V(ω₂)⊗V(ω₂)"),
  (TypeA{7}, [0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], "V(ω₃)⊗V(ω₃)"),
  (TypeA{9}, [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], "V(ω₁)⊗V(ω₁)"),
  (TypeA{9}, [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], "V(ω₂)⊗V(ω₂)"),
]

for (DT, c1, c2, label) in lr_bk_cases
  bench_lr(DT, c1, c2)
  bench_bk(DT, c1, c2)

  b_lr = @benchmark bench_lr($DT, $c1, $c2) evals = 1 samples = 200
  b_bk = @benchmark bench_bk($DT, $c1, $c2) evals = 1 samples = 200

  t_lr = minimum(b_lr).time / 1e3
  t_bk = minimum(b_bk).time / 1e3
  speedup = t_bk / t_lr

  report("$(sprint(show,DT())): LR  $label", b_lr; category="lr_vs_bk")
  report("$(sprint(show,DT())): BK  $label", b_bk; category="lr_vs_bk",
    extra=@sprintf("speedup: %.1f×", speedup))
end

# ═══════════════════════════════════════════════════════════════════════════════
#  8. Exterior and symmetric powers
# ═══════════════════════════════════════════════════════════════════════════════

header("8. Exterior and symmetric powers")

function bench_exterior(::Type{DT}, coords, k) where {DT}
  R = rank(DT)
  Lie.clear_all_caches!()
  λ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return ⋀(k, λ)
end

function bench_symmetric(::Type{DT}, coords, k) where {DT}
  R = rank(DT)
  Lie.clear_all_caches!()
  λ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return Sym(k, λ)
end

ext_cases = [
  (TypeA{4}, [1, 0, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], 3, "⋀³V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], 4, "⋀⁴V(ω₁)"),
  (TypeA{7}, [1, 0, 0, 0, 0, 0, 0], 4, "⋀⁴V(ω₁)"),
  (TypeA{7}, [1, 0, 0, 0, 0, 0, 0], 6, "⋀⁶V(ω₁)"),
  (TypeA{4}, [1, 1, 0, 0], 3, "⋀³V(ω₁+ω₂)"),
  (TypeA{4}, [1, 1, 0, 0], 4, "⋀⁴V(ω₁+ω₂)"),
  (TypeB{3}, [1, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeB{3}, [0, 0, 1], 2, "⋀²V(ω₃)"),
  (TypeB{3}, [0, 0, 1], 4, "⋀⁴V(ω₃)"),
  (TypeB{4}, [1, 0, 0, 0], 4, "⋀⁴V(ω₁)"),
  (TypeC{3}, [1, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeD{4}, [1, 0, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeD{4}, [1, 0, 0, 0], 4, "⋀⁴V(ω₁)"),
  (TypeG2, [1, 0], 2, "⋀²V(ω₁)"),
  (TypeG2, [1, 0], 3, "⋀³V(ω₁)"),
  (TypeG2, [1, 0], 4, "⋀⁴V(ω₁)"),
  (TypeE{6}, [1, 0, 0, 0, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], 2, "⋀²V(ω₈)"),
  (TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], 3, "⋀³V(ω₈)"),
  (TypeE{8}, [1, 0, 0, 0, 0, 0, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeE{8}, [1, 0, 0, 0, 0, 0, 0, 0], 3, "⋀³V(ω₁)"),
]

for (DT, coords, k, label) in ext_cases
  bench_exterior(DT, coords, k)
  b = @benchmark bench_exterior($DT, $coords, $k) evals = 1 samples = 10
  r = bench_exterior(DT, coords, k)
  report("$(sprint(show,DT())): $label → $(length(r.terms)) irreps", b;
    category="exterior_power")
end

sym_cases = [
  (TypeA{3}, [1, 0, 0], 3, "Sym³V(ω₁)"),
  (TypeA{3}, [1, 0, 0], 4, "Sym⁴V(ω₁)"),
  (TypeA{3}, [1, 0, 0], 5, "Sym⁵V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], 3, "Sym³V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], 5, "Sym⁵V(ω₁)"),
  (TypeA{3}, [1, 1, 0], 3, "Sym³V(ω₁+ω₂)"),
  (TypeA{3}, [2, 0, 0], 4, "Sym⁴V(2ω₁)"),
  (TypeB{2}, [1, 0], 3, "Sym³V(ω₁)"),
  (TypeB{3}, [1, 0, 0], 2, "Sym²V(ω₁)"),
  (TypeB{3}, [0, 0, 1], 3, "Sym³V(ω₃)"),
  (TypeC{3}, [1, 0, 0], 3, "Sym³V(ω₁)"),
  (TypeG2, [1, 0], 2, "Sym²V(ω₁)"),
  (TypeG2, [1, 0], 3, "Sym³V(ω₁)"),
  (TypeG2, [1, 0], 4, "Sym⁴V(ω₁)"),
  (TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], 3, "Sym³V(ω₈)"),
  (TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], 4, "Sym⁴V(ω₈)"),
]

for (DT, coords, k, label) in sym_cases
  bench_symmetric(DT, coords, k)
  b = @benchmark bench_symmetric($DT, $coords, $k) evals = 1 samples = 10
  r = bench_symmetric(DT, coords, k)
  report("$(sprint(show,DT())): $label → $(length(r.terms)) irreps", b;
    category="symmetric_power")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  9. Borel–Weil–Bott
# ═══════════════════════════════════════════════════════════════════════════════

header("9. Borel–Weil–Bott theorem")

# ─── Box benchmark (like conjugate_dominant_weight) ──────────────────────────

function bench_bwb_box(::Type{DT}, bound) where {DT}
  R = rank(DT)
  count = 0
  for coords in Iterators.product(ntuple(_ -> (-bound):bound, R)...)
    λ = WeightLatticeElem(DT, SVector{R,Int}(coords))
    borel_weil_bott(λ)
    count += 1
  end
  return count
end

bwb_box_cases = [
  (TypeA{3}, 5), (TypeA{4}, 3), (TypeA{6}, 2),
  (TypeB{3}, 5), (TypeB{4}, 3),
  (TypeC{3}, 5),
  (TypeD{4}, 3), (TypeD{5}, 2),
  (TypeE{6}, 2), (TypeE{7}, 1), (TypeE{8}, 1),
  (TypeF4, 3), (TypeG2, 8),
]

for (DT, bound) in bwb_box_cases
  R = rank(DT)
  n = (2 * bound + 1)^R
  bench_bwb_box(DT, bound)
  b = @benchmark bench_bwb_box($DT, $bound) evals = 1 samples = 30
  report("$(sprint(show,DT())): box [-$bound,$bound]^$R ($n wts)", b; category="bwb")
end

# ─── Deep non-dominant weights ───────────────────────────────────────────────

function bench_bwb_deep(::Type{DT}, weights) where {DT}
  for w in weights
    borel_weil_bott(w)
  end
end

function make_deep_weights(::Type{DT}, n, scale) where {DT}
  R = rank(DT)
  weights = WeightLatticeElem{DT,R}[]
  for i in 1:n
    coords = SVector{R,Int}(ntuple(j -> -scale * (1 + (i * j) % 7), R))
    push!(weights, WeightLatticeElem(DT, coords))
  end
  return weights
end

bwb_deep_cases = [
  (TypeA{3}, 500, 20, "500 deep wts (scale=20)"),
  (TypeA{6}, 500, 10, "500 deep wts (scale=10)"),
  (TypeB{4}, 500, 15, "500 deep wts (scale=15)"),
  (TypeD{5}, 500, 10, "500 deep wts (scale=10)"),
  (TypeE{6}, 200, 10, "200 deep wts (scale=10)"),
  (TypeE{7}, 200, 8, "200 deep wts (scale=8)"),
  (TypeE{8}, 200, 5, "200 deep wts (scale=5)"),
  (TypeF4, 500, 15, "500 deep wts (scale=15)"),
  (TypeG2, 500, 30, "500 deep wts (scale=30)"),
]

for (DT, n, scale, label) in bwb_deep_cases
  weights = make_deep_weights(DT, n, scale)
  bench_bwb_deep(DT, weights)
  b = @benchmark bench_bwb_deep($DT, $weights) evals = 1 samples = 50
  report("$(sprint(show,DT())): $label", b; category="bwb")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  10. Plethysm
# ═══════════════════════════════════════════════════════════════════════════════

header("10. Plethysm")

function bench_plethysm(::Type{DT}, coords, partition) where {DT}
  R = rank(DT)
  Lie.clear_all_caches!()
  λ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return plethysm(partition, λ)
end

plethysm_cases = [
  # Symmetric powers via plethysm (should match symmetric_power)
  (TypeA{3}, [1, 0, 0], [3], "S₍₃₎V(ω₁) = Sym³"),
  (TypeA{3}, [1, 0, 0], [4], "S₍₄₎V(ω₁) = Sym⁴"),
  (TypeA{4}, [1, 0, 0, 0], [3], "S₍₃₎V(ω₁) = Sym³"),
  # Exterior powers via plethysm (should match exterior_power)
  (TypeA{3}, [1, 0, 0], [1, 1, 1], "S₍₁₁₁₎V(ω₁) = ⋀³"),
  (TypeA{4}, [1, 0, 0, 0], [1, 1, 1], "S₍₁₁₁₎V(ω₁) = ⋀³"),
  (TypeA{4}, [1, 0, 0, 0], [1, 1, 1, 1], "S₍₁₁₁₁₎V(ω₁) = ⋀⁴"),
  # Mixed symmetry (hook partitions)
  (TypeA{3}, [1, 0, 0], [2, 1], "S₍₂,₁₎V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], [2, 1], "S₍₂,₁₎V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], [2, 1, 1], "S₍₂,₁,₁₎V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], [2, 2], "S₍₂,₂₎V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], [3, 1], "S₍₃,₁₎V(ω₁)"),
  # Non-type-A
  (TypeB{3}, [1, 0, 0], [2, 1], "S₍₂,₁₎V(ω₁)"),
  (TypeG2, [1, 0], [2, 1], "S₍₂,₁₎V(ω₁)"),
  (TypeG2, [1, 0], [3], "S₍₃₎V(ω₁) = Sym³"),
  # Larger representations
  (TypeA{3}, [1, 1, 0], [2, 1], "S₍₂,₁₎V(ω₁+ω₂)"),
  (TypeA{3}, [2, 0, 0], [2, 1], "S₍₂,₁₎V(2ω₁)"),
]

for (DT, coords, partition, label) in plethysm_cases
  bench_plethysm(DT, coords, partition)
  b = @benchmark bench_plethysm($DT, $coords, $partition) evals = 1 samples = 10
  r = bench_plethysm(DT, coords, partition)
  report("$(sprint(show,DT())): $label → $(length(r.terms)) irreps", b;
    category="plethysm")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Save results
# ═══════════════════════════════════════════════════════════════════════════════

function save_results(results::Vector{BenchResult})
  results_dir = joinpath(@__DIR__, "results")
  mkpath(results_dir)

  timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
  filepath = joinpath(results_dir, "$timestamp.json")
  latest = joinpath(results_dir, "latest.json")

  # Write simple JSON manually (no JSON dependency needed)
  open(filepath, "w") do io
    println(io, "{")
    println(io, "  \"timestamp\": \"$timestamp\",")
    println(io, "  \"julia_version\": \"$(VERSION)\",")
    println(io, "  \"results\": [")
    for (i, r) in enumerate(results)
      comma = i < length(results) ? "," : ""
      println(io, "    {")
      println(io, "      \"name\": $(repr(r.name)),")
      println(io, "      \"category\": $(repr(r.category)),")
      println(io, "      \"min_time_ns\": $(r.min_time_ns),")
      println(io, "      \"median_time_ns\": $(r.median_time_ns),")
      println(io, "      \"mean_time_ns\": $(r.mean_time_ns),")
      println(io, "      \"min_allocs\": $(r.min_allocs),")
      println(io, "      \"min_memory\": $(r.min_memory),")
      println(io, "      \"median_allocs\": $(r.median_allocs),")
      println(io, "      \"median_memory\": $(r.median_memory)")
      println(io, "    }$comma")
    end
    println(io, "  ]")
    println(io, "}")
  end

  # Update latest copy
  isfile(latest) && rm(latest)
  cp(filepath, latest)

  println("\n📊 Results saved to: $filepath")
  return filepath
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Compare against previous results
# ═══════════════════════════════════════════════════════════════════════════════

function load_results_json(filepath::String)
  content = read(filepath, String)
  # Simple JSON parser for our known format
  results = Dict{String,Float64}()
  for m in
      eachmatch(r"\"name\":\s*\"([^\"]+)\"[^}]*\"min_time_ns\":\s*([0-9.e+\-]+)", content)
    results[m.captures[1]] = parse(Float64, m.captures[2])
  end
  return results
end

function compare_results_from_data(
  current::Vector{BenchResult}, baseline::Dict{String,Float64}, baseline_path::String
)
  println("\n", "="^80)
  println("  Regression comparison vs: $baseline_path")
  println("="^80)
  @printf("  %-55s %10s  %10s  %8s\n", "Benchmark", "Baseline", "Current", "Ratio")
  println("  ", "-"^90)

  regressions = 0
  improvements = 0

  for r in current
    if haskey(baseline, r.name)
      old_ns = baseline[r.name]
      new_ns = r.min_time_ns
      ratio = new_ns / old_ns

      marker = if ratio > 1.15
        regressions += 1
        " ⚠️  REGRESSION"
      elseif ratio < 0.85
        improvements += 1
        " ✅ IMPROVED"
      else
        ""
      end

      @printf("  %-55s %9.3f ms  %9.3f ms  %7.2fx%s\n",
        r.name, old_ns / 1e6, new_ns / 1e6, ratio, marker)
    else
      @printf("  %-55s %10s  %9.3f ms  %8s\n",
        r.name, "NEW", r.min_time_ns / 1e6, "—")
    end
  end

  println()
  println("  Summary: $improvements improved, $regressions regressions, ",
    "$(length(current) - improvements - regressions) unchanged")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

println("\n", "="^80)
println("  All benchmarks complete. $(length(ALL_RESULTS)) benchmarks recorded.")
println("="^80)

# Load baseline BEFORE saving new results (so we don't compare against ourselves)
baseline_path = if COMPARE
  lp = joinpath(@__DIR__, "results", "latest.json")
  isfile(lp) ? lp : nothing
else
  nothing
end
baseline_data = isnothing(baseline_path) ? nothing : load_results_json(baseline_path)

saved_path = save_results(ALL_RESULTS)

if COMPARE
  if !isnothing(baseline_data)
    compare_results_from_data(ALL_RESULTS, baseline_data, baseline_path)
  else
    println("⚠️  No previous results to compare against.")
  end
end
