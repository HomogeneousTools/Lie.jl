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
  hw = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return freudenthal_formula(hw)
end

freudenthal_cases = [
  (TypeA{3}, [1, 1, 1], "ρ (dim 20)"),
  (TypeA{3}, [2, 0, 2], "2ω₁+2ω₃ (dim 50)"),
  (TypeA{4}, [1, 0, 0, 1], "ω₁+ω₄ (adj, dim 24)"),
  (TypeA{4}, [2, 0, 0, 0], "2ω₁ (dim 15)"),
  (TypeB{3}, [0, 1, 0], "ω₂ (dim 21)"),
  (TypeB{3}, [1, 0, 0], "ω₁ (dim 7)"),
  (TypeB{4}, [0, 0, 0, 1], "ω₄ (spinor, dim 16)"),
  (TypeC{3}, [1, 0, 0], "ω₁ (dim 6)"),
  (TypeD{4}, [0, 0, 1, 0], "ω₃ (spinor, dim 8)"),
  (TypeD{4}, [1, 0, 0, 0], "ω₁ (dim 8)"),
  (TypeG2, [1, 0], "ω₁ (dim 7)"),
  (TypeG2, [0, 1], "ω₂ (dim 14)"),
  (TypeG2, [1, 1], "ω₁+ω₂ (dim 64)"),
  (TypeF4, [0, 0, 0, 1], "ω₄ (dim 26)"),
  (TypeE{6}, [1, 0, 0, 0, 0, 0], "ω₁ (dim 27)"),
  (TypeE{6}, [0, 0, 0, 0, 0, 1], "ω₆ (dim 27)"),
  (TypeE{7}, [1, 0, 0, 0, 0, 0, 0], "ω₁ (dim 133)"),
  (TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], "ω₈ (dim 248)"),
]

for (DT, coords, label) in freudenthal_cases
  bench_freudenthal(DT, coords)
  b = @benchmark bench_freudenthal($DT, $coords) evals = 1 samples = 50
  report("$(sprint(show,DT())): $label", b; category="freudenthal")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  6. Tensor product decomposition
# ═══════════════════════════════════════════════════════════════════════════════

header("6. Tensor product decomposition")

function bench_tensor(::Type{DT}, c1, c2) where {DT}
  R = rank(DT)
  empty!(Lie._tensor_cache)
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
]

for (DT, c1, c2, label) in tensor_cases
  bench_tensor(DT, c1, c2)
  b = @benchmark bench_tensor($DT, $c1, $c2) evals = 1 samples = 20
  tp = bench_tensor(DT, c1, c2)
  report("$(sprint(show,DT())): $label → $(length(tp.terms)) irreps", b;
    category="tensor_product")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  7. Exterior and symmetric powers
# ═══════════════════════════════════════════════════════════════════════════════

header("7. Exterior and symmetric powers")

function bench_exterior(::Type{DT}, coords, k) where {DT}
  R = rank(DT)
  empty!(Lie._exterior_power_cache)
  empty!(Lie._tensor_cache)
  λ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return ⋀(k, λ)
end

function bench_symmetric(::Type{DT}, coords, k) where {DT}
  R = rank(DT)
  empty!(Lie._symmetric_power_cache)
  empty!(Lie._tensor_cache)
  λ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return Sym(k, λ)
end

ext_cases = [
  (TypeA{4}, [1, 0, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], 3, "⋀³V(ω₁)"),
  (TypeA{4}, [1, 0, 0, 0], 4, "⋀⁴V(ω₁)"),
  (TypeB{3}, [1, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeB{3}, [0, 0, 1], 2, "⋀²V(ω₃)"),
  (TypeC{3}, [1, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeD{4}, [1, 0, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeG2, [1, 0], 2, "⋀²V(ω₁)"),
  (TypeG2, [1, 0], 3, "⋀³V(ω₁)"),
  (TypeE{6}, [1, 0, 0, 0, 0, 0], 2, "⋀²V(ω₁)"),
  (TypeE{8}, [0, 0, 0, 0, 0, 0, 0, 1], 2, "⋀²V(ω₈)"),
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
  (TypeB{2}, [1, 0], 3, "Sym³V(ω₁)"),
  (TypeB{3}, [1, 0, 0], 2, "Sym²V(ω₁)"),
  (TypeG2, [1, 0], 2, "Sym²V(ω₁)"),
  (TypeG2, [1, 0], 3, "Sym³V(ω₁)"),
]

for (DT, coords, k, label) in sym_cases
  bench_symmetric(DT, coords, k)
  b = @benchmark bench_symmetric($DT, $coords, $k) evals = 1 samples = 10
  r = bench_symmetric(DT, coords, k)
  report("$(sprint(show,DT())): $label → $(length(r.terms)) irreps", b;
    category="symmetric_power")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  8. Borel–Weil–Bott
# ═══════════════════════════════════════════════════════════════════════════════

header("8. Borel–Weil–Bott theorem")

function bench_bwb(::Type{DT}, coords) where {DT}
  R = rank(DT)
  λ = WeightLatticeElem(DT, SVector{R,Int}(Tuple(coords)))
  return borel_weil_bott(λ)
end

bwb_cases = [
  (TypeA{4}, [-3, 2, -1, 4], "generic A₄ weight"),
  (TypeA{6}, [-2, 3, -1, 2, -4, 1], "generic A₆ weight"),
  (TypeB{4}, [-2, 1, -3, 2], "generic B₄ weight"),
  (TypeC{4}, [-1, 3, -2, 1], "generic C₄ weight"),
  (TypeD{5}, [-3, 1, 2, -4, 1], "generic D₅ weight"),
  (TypeE{6}, [-2, 1, -1, 3, -2, 1], "generic E₆ weight"),
  (TypeE{7}, [-3, 2, -1, 1, -2, 3, -1], "generic E₇ weight"),
  (TypeE{8}, [-5, 3, -2, -3, 5, -8, 2, 1], "deep E₈ weight"),
  (TypeF4, [-2, 3, -1, 2], "generic F₄ weight"),
  (TypeG2, [-3, 5], "generic G₂ weight"),
]

for (DT, coords, label) in bwb_cases
  result = bench_bwb(DT, coords)
  d_str = result === nothing ? "singular" : "ℓ=$(result[1])"
  b = @benchmark bench_bwb($DT, $coords) evals = 1 samples = 500
  report("$(sprint(show,DT())): $label ($d_str)", b; category="bwb")
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

function compare_results(current::Vector{BenchResult}, baseline_path::String)
  baseline = load_results_json(baseline_path)

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

saved_path = save_results(ALL_RESULTS)

if COMPARE
  latest = joinpath(@__DIR__, "results", "latest.json")
  if isfile(latest) && latest != saved_path
    compare_results(ALL_RESULTS, latest)
  else
    println("⚠️  No previous results to compare against.")
  end
end
