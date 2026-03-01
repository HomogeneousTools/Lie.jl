# ═══════════════════════════════════════════════════════════════════════════════
#  dominant_character benchmark: Lie.jl vs Oscar.jl
#
#  Compares Lie.jl's `dominant_character` (Freudenthal + Moody-Patera grouping,
#  SVector/static-dispatch, global cache) against Oscar.jl's implementation
#  (same Moody-Patera algorithm but over heap-allocated WeightLatticeElems with
#  matrix-group stabiliser orbits, no persistent cross-call cache).
#
#  Usage:
#    julia --project=. benchmark/dominant_character.jl
#    julia --project=. benchmark/dominant_character.jl --no-oscar
#
#  Oscar requirement:
#    Oscar.jl is NOT in Lie.jl's Project.toml (heavyweight dependency).
#    To benchmark Oscar either:
#      (a) activate a separate environment that has Oscar installed, or
#      (b) temporarily  pkg> add Oscar  in the current project.
#    If Oscar cannot be loaded the Oscar columns are skipped gracefully.
#
#  Caching notes:
#    Lie.jl    — cache cleared *inside* the timed call via clear_all_caches!()
#                This measures pure computation time with no persistent cross-call cache,
#                directly comparable to Oscar.jl.
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

fmt_dim(d) = d < BigInt(10)^9 ? string(Int(d)) : @sprintf("~%.2e", Float64(d))

# ─── Section-header printer ───────────────────────────────────────────────────

function section_header(title)
  println()
  println("="^110)
  println("  ", title)
  println("="^110)
  if OSCAR_AVAILABLE
    @printf(
      "  %-54s  %-21s  %-21s  %s\n",
      "case", "Lie.jl (min)", "Oscar (min)", "speedup Oscar/Lie",
    )
    println("  ", "-"^104)
  else
    @printf("  %-54s  %-21s\n", "case", "Lie.jl (min)")
    println("  ", "-"^78)
  end
end

# ─── Benchmark kernel functions ───────────────────────────────────────────────

# Clear cache *inside* the function — the full clear cost is included in timing.
function _bench_lie_cold(::Type{DT}, coords::NTuple{R,Int}) where {DT,R}
  Lie.clear_all_caches!()
  dominant_character(WeightLatticeElem(DT, SVector{R,Int}(coords)))
end

# Oscar RootSystem pre-constructed outside timing.
function _bench_oscar(oscar_R, ivec::Vector{Int})
  Main.Oscar.dominant_character(oscar_R, ivec)
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

# ─── Result storage ───────────────────────────────────────────────────────────

struct DCResult
  label::String
  lie_cold_ns::Float64
  lie_cold_allocs::Int
  lie_warm_ns::Float64
  lie_warm_allocs::Int
  oscar_ns::Float64
  oscar_allocs::Int
  n_dom_weights::Int
  rep_dim_str::String
end

const ALL_RESULTS = DCResult[]

# ─── Per-case runner ──────────────────────────────────────────────────────────

"""
    run_case(label, ::Type{DT}, coords; samples_cold, samples_oscar,
             skip_large_oscar)

Benchmark `dominant_character` for the representation of `DT` with highest
weight given by `coords` (fundamental-weight coordinates).

- `coords` may be a `Tuple` or `Vector` of `Int`; it is normalised internally.
- `skip_large_oscar`: if true, Oscar is skipped for this case only (useful for
  representations whose dominant character Oscar computes very slowly).
"""
function run_case(
  label, ::Type{DT}, coords;
  samples_cold=50,
  samples_oscar=50,
  skip_large_oscar=false,
) where {DT}
  R_n = rank(DT)
  coords_tup = NTuple{R_n,Int}(coords)   # accept Vector or Tuple

  # ── Warmup + populate cache ───────────────────────────────────────────────
  _bench_lie_cold(DT, coords_tup)                    # compile + fill cache
  dom = dominant_character(WeightLatticeElem(DT, SVector{R_n,Int}(coords_tup)))
  n_dom = length(dom)
  rep_dim = degree(WeightLatticeElem(DT, SVector{R_n,Int}(coords_tup)))

  # ── Lie cold ──────────────────────────────────────────────────────────────
  b_cold = @benchmark _bench_lie_cold($DT, $coords_tup) evals = 1 samples = samples_cold

  # ── Oscar ─────────────────────────────────────────────────────────────────
  b_oscar = nothing
  if OSCAR_AVAILABLE && !skip_large_oscar
    try
      ct = oscar_cartan(DT)
      oscar_R = Main.Oscar.root_system(ct...)
      ivec = collect(Int, coords_tup)
      _bench_oscar(oscar_R, ivec)   # warmup — fills RootSystem internal caches
      b_oscar = @benchmark _bench_oscar($oscar_R, $ivec) evals = 1 samples = samples_oscar
    catch e
      @warn "Oscar benchmark failed for \"$label\"" exception = e
    end
  end

  # ── Extract minimum-trial values ──────────────────────────────────────────
  cold_ns = Float64(minimum(b_cold).time)
  cold_a = minimum(b_cold).allocs
  osc_ns = b_oscar === nothing ? NaN : Float64(minimum(b_oscar).time)
  osc_a = b_oscar === nothing ? 0 : minimum(b_oscar).allocs

  # ── Format columns ────────────────────────────────────────────────────────
  cold_str = @sprintf("%-9s %6d alloc", fmt_time(cold_ns), cold_a)
  osc_str = if b_oscar !== nothing
    @sprintf("%-9s %6d alloc", fmt_time(osc_ns), osc_a)
  elseif OSCAR_AVAILABLE
    skip_large_oscar ? "(skipped)" : "  (failed)"
  else
    "—"
  end

  meta = @sprintf(" dim=%s dom=%d", fmt_dim(rep_dim), n_dom)

  if OSCAR_AVAILABLE
    spstr =
      (!isnan(osc_ns) && cold_ns > 0) ?
      @sprintf("%6.1fx", osc_ns / cold_ns) : "     —"
    @printf("  %-54s  %-21s  %-21s  %s  [%s]\n",
      label, cold_str, osc_str, spstr, meta)
  else
    @printf("  %-54s  %-21s  [%s]\n", label, cold_str, meta)
  end

  push!(
    ALL_RESULTS,
    DCResult(
      label, cold_ns, cold_a, NaN, 0, osc_ns, osc_a, n_dom, fmt_dim(rep_dim)
    ),
  )
end

# Helper: unit-vector tuple of length n with a 1 at position i
e_tup(n, i) = NTuple{n,Int}(j == i ? 1 : 0 for j in 1:n)

# ═══════════════════════════════════════════════════════════════════════════════
#  §1  Small rank  (A₁–A₃, B₂, B₃, G₂)
# ═══════════════════════════════════════════════════════════════════════════════

section_header("§1  Small rank  (A₁–A₃, B₂, B₃, G₂)")

# ── A₁: trivial structure (dom = 1 always), tests raw overhead ───────────────
for n in (1, 2, 5, 10, 20, 50)
  run_case(@sprintf("A₁  %dω₁", n), TypeA{1}, (n,);
    samples_cold=n <= 10 ? 200 : 50, samples_oscar=n <= 10 ? 80 : 20)
end

# ── A₂ ────────────────────────────────────────────────────────────────────────
run_case("A₂  ω₁", TypeA{2}, (1, 0))
run_case("A₂  ω₁+ω₂  (adj)", TypeA{2}, (1, 1))
run_case("A₂  2ω₁", TypeA{2}, (2, 0))
run_case("A₂  3ω₁+3ω₂", TypeA{2}, (3, 3))
run_case("A₂  5ω₁+5ω₂", TypeA{2}, (5, 5))
run_case("A₂  10ω₁+10ω₂", TypeA{2}, (10, 10);
  samples_cold=20, samples_oscar=20)

# ── A₃ ────────────────────────────────────────────────────────────────────────
run_case("A₃  ω₁", TypeA{3}, (1, 0, 0))
run_case("A₃  ω₂", TypeA{3}, (0, 1, 0))
run_case("A₃  ρ = ω₁+ω₂+ω₃", TypeA{3}, (1, 1, 1))
run_case("A₃  2ω₁+ω₂+2ω₃", TypeA{3}, (2, 1, 2))
run_case("A₃  3ω₂", TypeA{3}, (0, 3, 0))
run_case("A₃  5ω₁+5ω₂+5ω₃", TypeA{3}, (5, 5, 5);
  samples_cold=10, samples_oscar=10)

# ── B₂ ────────────────────────────────────────────────────────────────────────
run_case("B₂  ω₁", TypeB{2}, (1, 0))
run_case("B₂  ω₂  (spinor)", TypeB{2}, (0, 1))
run_case("B₂  ρ = ω₁+ω₂", TypeB{2}, (1, 1))
run_case("B₂  3ω₁+3ω₂", TypeB{2}, (3, 3))
run_case("B₂  5ω₁+5ω₂", TypeB{2}, (5, 5);
  samples_cold=20, samples_oscar=20)

# ── B₃ ────────────────────────────────────────────────────────────────────────
run_case("B₃  ω₁", TypeB{3}, (1, 0, 0))
run_case("B₃  ω₂  (adj)", TypeB{3}, (0, 1, 0))
run_case("B₃  ω₃  (spinor)", TypeB{3}, (0, 0, 1))
run_case("B₃  ρ", TypeB{3}, (1, 1, 1);
  samples_cold=30, samples_oscar=30)
run_case("B₃  3ω₁+3ω₂+3ω₃", TypeB{3}, (3, 3, 3);
  samples_cold=10, samples_oscar=10)

# ── G₂ ────────────────────────────────────────────────────────────────────────
run_case("G₂  ω₁", TypeG2, (1, 0))
run_case("G₂  ω₂  (adj)", TypeG2, (0, 1))
run_case("G₂  ρ = ω₁+ω₂", TypeG2, (1, 1))
run_case("G₂  3ω₁+2ω₂", TypeG2, (3, 2);
  samples_cold=30, samples_oscar=30)
run_case("G₂  5ω₁+4ω₂", TypeG2, (5, 4);
  samples_cold=20, samples_oscar=20)

# ═══════════════════════════════════════════════════════════════════════════════
#  §2  Medium rank  (A₄–A₆, B₄, C₄, D₄–D₅)
# ═══════════════════════════════════════════════════════════════════════════════

section_header("§2  Medium rank  (A₄–A₆, B₄, C₄, D₄–D₅)")

# ── A₄ ────────────────────────────────────────────────────────────────────────
run_case("A₄  ω₁", TypeA{4}, (1, 0, 0, 0))
run_case("A₄  ω₂", TypeA{4}, (0, 1, 0, 0))
run_case("A₄  ρ", TypeA{4}, (1, 1, 1, 1);
  samples_cold=20, samples_oscar=20)
run_case("A₄  2ω₁+ω₂+ω₃+2ω₄", TypeA{4}, (2, 1, 1, 2);
  samples_cold=10, samples_oscar=10)

# ── A₅ ────────────────────────────────────────────────────────────────────────
run_case("A₅  ω₁", TypeA{5}, (1, 0, 0, 0, 0))
run_case("A₅  ω₃", TypeA{5}, (0, 0, 1, 0, 0))
run_case("A₅  ρ", TypeA{5}, (1, 1, 1, 1, 1);
  samples_cold=10, samples_oscar=10)

# ── A₆ ────────────────────────────────────────────────────────────────────────
run_case("A₆  ω₁", TypeA{6}, (1, 0, 0, 0, 0, 0))
run_case("A₆  ω₃", TypeA{6}, (0, 0, 1, 0, 0, 0))
run_case("A₆  ρ", TypeA{6}, (1, 1, 1, 1, 1, 1);
  samples_cold=5, samples_oscar=5)

# ── B₄ ────────────────────────────────────────────────────────────────────────
run_case("B₄  ω₁", TypeB{4}, (1, 0, 0, 0))
run_case("B₄  ω₂", TypeB{4}, (0, 1, 0, 0))
run_case("B₄  ω₄  (spinor)", TypeB{4}, (0, 0, 0, 1))
run_case("B₄  ρ", TypeB{4}, (1, 1, 1, 1);
  samples_cold=20, samples_oscar=20)

# ── C₄ ────────────────────────────────────────────────────────────────────────
run_case("C₄  ω₁", TypeC{4}, (1, 0, 0, 0))
run_case("C₄  ω₂", TypeC{4}, (0, 1, 0, 0))
run_case("C₄  ω₄", TypeC{4}, (0, 0, 0, 1))
run_case("C₄  ρ", TypeC{4}, (1, 1, 1, 1);
  samples_cold=20, samples_oscar=20)

# ── D₄ ────────────────────────────────────────────────────────────────────────
run_case("D₄  ω₁", TypeD{4}, (1, 0, 0, 0))
run_case("D₄  ω₃  (left spinor)", TypeD{4}, (0, 0, 1, 0))
run_case("D₄  ω₄  (right spinor)", TypeD{4}, (0, 0, 0, 1))
run_case("D₄  ρ", TypeD{4}, (1, 1, 1, 1);
  samples_cold=20, samples_oscar=20)

# ── D₅ ────────────────────────────────────────────────────────────────────────
run_case("D₅  ω₁", TypeD{5}, (1, 0, 0, 0, 0))
run_case("D₅  ω₄  (spinor)", TypeD{5}, (0, 0, 0, 1, 0))
run_case("D₅  ρ", TypeD{5}, (1, 1, 1, 1, 1);
  samples_cold=10, samples_oscar=10)

# ═══════════════════════════════════════════════════════════════════════════════
#  §3  Exceptional types  (F₄, E₆, E₇, E₈)
# ═══════════════════════════════════════════════════════════════════════════════

section_header("§3  Exceptional types  (F₄, E₆, E₇, E₈)")

# ── F₄ ────────────────────────────────────────────────────────────────────────
run_case("F₄  ω₁  (adj)", TypeF4, (1, 0, 0, 0))
run_case("F₄  ω₄", TypeF4, (0, 0, 0, 1))
run_case("F₄  2ω₄", TypeF4, (0, 0, 0, 2);
  samples_cold=20, samples_oscar=20)
run_case("F₄  ω₁+ω₄  [Lübeck]", TypeF4, (1, 0, 0, 1);
  samples_cold=10, samples_oscar=10)

# ── E₆ ────────────────────────────────────────────────────────────────────────
run_case("E₆  ω₁", TypeE{6}, (1, 0, 0, 0, 0, 0))
run_case("E₆  ω₂  (adj)", TypeE{6}, (0, 1, 0, 0, 0, 0))
run_case("E₆  ω₃  [Lübeck]", TypeE{6}, (0, 0, 1, 0, 0, 0);
  samples_cold=10, samples_oscar=10)
run_case("E₆  ρ", TypeE{6}, (1, 1, 1, 1, 1, 1);
  samples_cold=5, samples_oscar=5)

# ── E₇ ────────────────────────────────────────────────────────────────────────
run_case("E₇  ω₁  (adj)", TypeE{7}, (1, 0, 0, 0, 0, 0, 0))
run_case("E₇  ω₇", TypeE{7}, (0, 0, 0, 0, 0, 0, 1))
run_case("E₇  ω₄+ω₅  [Lübeck]", TypeE{7}, (0, 0, 0, 1, 1, 0, 0);
  samples_cold=5, samples_oscar=5)

# ── E₈ ────────────────────────────────────────────────────────────────────────
run_case("E₈  ω₈  (adj)", TypeE{8}, (0, 0, 0, 0, 0, 0, 0, 1))
run_case("E₈  ω₁", TypeE{8}, (1, 0, 0, 0, 0, 0, 0, 0);
  samples_cold=5, samples_oscar=5)
run_case("E₈  ω₁+ω₃  [Lübeck]", TypeE{8}, (1, 0, 1, 0, 0, 0, 0, 0);
  samples_cold=3, samples_oscar=3, skip_large_oscar=true)

# ═══════════════════════════════════════════════════════════════════════════════
#  §4  Rank-10 standard types  (A₁₀, B₁₀, C₁₀, D₁₀)
# ═══════════════════════════════════════════════════════════════════════════════

section_header("§4  Rank-10 standard types  (A₁₀, B₁₀, C₁₀, D₁₀)")

# A₁₀  — all 10 fundamental weights
for i in 1:10
  run_case(@sprintf("A₁₀  ω%d", i), TypeA{10}, e_tup(10, i);
    samples_cold=20, samples_oscar=20)
end

# B₁₀  — selected; skip ω₁₀ (spinor, 2^10 = 1024 weights)
for (i, skip_osc) in [(1, false), (2, false), (5, false), (9, false), (10, true)]
  run_case(@sprintf("B₁₀  ω%d%s", i, i == 10 ? "  (spinor)" : ""),
    TypeB{10}, e_tup(10, i);
    samples_cold=20, samples_oscar=20, skip_large_oscar=skip_osc)
end

# C₁₀  — selected
for i in (1, 2, 5, 9, 10)
  run_case(@sprintf("C₁₀  ω%d", i), TypeC{10}, e_tup(10, i);
    samples_cold=20, samples_oscar=20)
end

# D₁₀  — ω₉ and ω₁₀ are half-spinors (2^8 = 256 dim); skip Oscar
for (i, skip_osc) in [(1, false), (2, false), (5, false), (9, true), (10, true)]
  run_case(@sprintf("D₁₀  ω%d%s", i, i in (9, 10) ? "  (spinor)" : ""),
    TypeD{10}, e_tup(10, i);
    samples_cold=20, samples_oscar=20, skip_large_oscar=skip_osc)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  §5  Scaling with weight height  (n·ω₁ for A₃, B₃, G₂)
# ═══════════════════════════════════════════════════════════════════════════════

section_header("§5  Scaling with weight height  (n·ω₁  for A₃, B₃, G₂)")

for (DT, name) in [(TypeA{3}, "A₃"), (TypeB{3}, "B₃"), (TypeG2, "G₂")]
  R_n = rank(DT)
  for n in (1, 2, 3, 5, 8, 12, 20)
    coords = e_tup(R_n, 1) .* n
    sc = if n <= 5
      100
    elseif n <= 12
      30
    else
      10
    end
    so = if n <= 5
      30
    elseif n <= 12
      10
    else
      3
    end
    run_case(
      @sprintf("%s  %dω₁", name, n),
      DT, coords;
      samples_cold=sc, samples_oscar=so,
    )
  end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  §6  Scaling with rank  (ρ = (1,1,…,1) for classical series)
# ═══════════════════════════════════════════════════════════════════════════════

section_header("§6  Scaling with rank  (ρ = (1,…,1)  for A₂–A₈, B₂–B₆, C₃–C₆, D₄–D₇)")

for DT in [
  TypeA{2}, TypeA{3}, TypeA{4}, TypeA{5}, TypeA{6}, TypeA{7}, TypeA{8},
  TypeB{2}, TypeB{3}, TypeB{4}, TypeB{5}, TypeB{6},
  TypeC{3}, TypeC{4}, TypeC{5}, TypeC{6},
  TypeD{4}, TypeD{5}, TypeD{6}, TypeD{7},
]
  R_n = rank(DT)
  coords = ntuple(_ -> 1, R_n)
  run_case(
    @sprintf("%-4s  ρ  (rank %d)", sprint(show, DT()), R_n),
    DT, coords;
    samples_cold=max(3, 40 - 3 * R_n),
    samples_oscar=max(3, 15 - R_n),
  )
end

# ═══════════════════════════════════════════════════════════════════════════════
#  §7  Algorithmically demanding cases
# ═══════════════════════════════════════════════════════════════════════════════

section_header("§7  Algorithmically demanding  (large mults or many dominant weights)")

# Large number of dominant weights
run_case("A₃  3ω₂             (many dom wts)", TypeA{3}, (0, 3, 0);
  samples_cold=30, samples_oscar=30)
run_case("A₄  ω₂+ω₃", TypeA{4}, (0, 1, 1, 0);
  samples_cold=20, samples_oscar=20)
run_case("A₅  ω₁+ω₂+ω₄+ω₅", TypeA{5}, (1, 1, 0, 1, 1);
  samples_cold=10, samples_oscar=10)

# B/C mid-rank generics
run_case("B₃  2ω₁+ω₂+2ω₃", TypeB{3}, (2, 1, 2);
  samples_cold=20, samples_oscar=20)
run_case("C₃  2ω₁+ω₂+2ω₃", TypeC{3}, (2, 1, 2);
  samples_cold=20, samples_oscar=20)
run_case("B₄  ρ = ω₁+ω₂+ω₃+ω₄", TypeB{4}, (1, 1, 1, 1);
  samples_cold=10, samples_oscar=10)
run_case("D₅  ω₂+ω₃+ω₄+ω₅", TypeD{5}, (0, 1, 1, 1, 1);
  samples_cold=10, samples_oscar=10)

# Exceptional
run_case("G₂  4ω₁+3ω₂", TypeG2, (4, 3);
  samples_cold=20, samples_oscar=20)
run_case("F₄  ω₂+ω₃", TypeF4, (0, 1, 1, 0);
  samples_cold=10, samples_oscar=10)
run_case("E₆  2ω₁", TypeE{6}, (2, 0, 0, 0, 0, 0);
  samples_cold=10, samples_oscar=10)
run_case("E₇  ω₆", TypeE{7}, (0, 0, 0, 0, 0, 1, 0);
  samples_cold=5, samples_oscar=5)
run_case("E₈  ω₇", TypeE{8}, (0, 0, 0, 0, 0, 0, 1, 0);
  samples_cold=5, samples_oscar=5)

# ═══════════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════════

println()
println("="^110)
println("  Summary")
println("="^110)
println("  Total cases benchmarked: $(length(ALL_RESULTS))")

valid_cold = filter(r -> !isnan(r.lie_cold_ns), ALL_RESULTS)
valid_both = filter(r -> !isnan(r.lie_cold_ns) && !isnan(r.oscar_ns), ALL_RESULTS)

if OSCAR_AVAILABLE && !isempty(valid_both)
  ratios = [r.oscar_ns / r.lie_cold_ns for r in valid_both if r.lie_cold_ns > 0]
  println()
  @printf("  Oscar vs Lie.jl cold — ratio (>1 means Lie faster):\n")
  @printf("    min = %.2fx,  median = %.1fx,  max = %.0fx\n",
    minimum(ratios), median(ratios), maximum(ratios))
  nfaster = count(>(1.0), ratios)
  @printf("  Lie.jl faster cold on %d / %d cases\n", nfaster, length(ratios))
end
println()
