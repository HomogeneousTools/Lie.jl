# ═══════════════════════════════════════════════════════════════════════════════
#  Weylloop — systematic Weyl orbit traversal without hash sets
#
#  Ported from the LiE implementation by M.A.A. van Leeuwen, A.M. Cohen,
#  and B. Lisser.  The key idea: convert to an ε-basis where a large
#  classical Weyl subgroup acts by permutations (type A) or permutations +
#  sign changes (types B/C/D).  Orbits of the classical subgroup are
#  enumerated via lexicographic permutation generation (Nextperm) and
#  Gray-code sign flips — no hash set needed.  For exceptional types,
#  coset representatives of W / W_classical are precomputed via a Coxeter
#  element and stored as matrices.
#
#  This eliminates the O(|orbit|) hash overhead that dominates large orbits
#  (e.g. E₈ with 5 million orbit points).
#
#  All transforms and inner-loop functions dispatch on the Dynkin type,
#  allowing the compiler to inline transforms and unroll small fixed-size
#  loops (rank ≤ 8).  Workspace uses stack-allocated MVector where possible.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── ε-basis transforms per simple type ─────────────────────────────────────
#
# Each pair (_w2e!, _e2w!) converts between fundamental-weight coords (w) and
# ε-coords (e).  They dispatch on Type{DT} so the compiler knows the rank
# and can unroll the (tiny) loops.

# Type A: ε_i = ω_i + ⋯ + ω_r (partial sums from right), ε_{r+1} = 0.
@inline function _w2e!(::Type{TypeA{N}}, e, w) where {N}
  s = 0
  @inbounds e[N + 1] = 0
  @inbounds for i in N:-1:1
    s += w[i]
    e[i] = s
  end
end
@inline function _e2w!(::Type{TypeA{N}}, w, e) where {N}
  @inbounds for i in 1:N
    w[i] = e[i] - e[i + 1]
  end
end

# Type B: ε_i = 2ω_i + ⋯ + 2ω_{r-1} + ω_r for i < r, ε_r = ω_r.
@inline function _w2e!(::Type{TypeB{N}}, e, w) where {N}
  @inbounds s = w[N]
  @inbounds e[N] = s
  @inbounds for i in (N - 1):-1:1
    s += 2 * w[i]
    e[i] = s
  end
end
@inline function _e2w!(::Type{TypeB{N}}, w, e) where {N}
  @inbounds for i in 1:(N - 1)
    w[i] = (e[i] - e[i + 1]) ÷ 2
  end
  @inbounds w[N] = e[N]
end

# Type C: ε_i = ω_i + ⋯ + ω_r.
@inline function _w2e!(::Type{TypeC{N}}, e, w) where {N}
  @inbounds s = w[N]
  @inbounds e[N] = s
  @inbounds for i in (N - 1):-1:1
    s += w[i]
    e[i] = s
  end
end
@inline function _e2w!(::Type{TypeC{N}}, w, e) where {N}
  @inbounds for i in 1:(N - 1)
    w[i] = e[i] - e[i + 1]
  end
  @inbounds w[N] = e[N]
end

# Type D: ε_r = ω_r - ω_{r-1}, then ε_i = 2ω_i + ⋯ + 2ω_{r-2} + ω_{r-1} + ω_r.
@inline function _w2e!(::Type{TypeD{N}}, e, w) where {N}
  @inbounds s = w[N] - w[N - 1]
  @inbounds e[N] = s
  @inbounds for i in (N - 1):-1:1
    s += 2 * w[i]
    e[i] = s
  end
end
@inline function _e2w!(::Type{TypeD{N}}, w, e) where {N}
  @inbounds for i in 1:(N - 1)
    w[i] = (e[i] - e[i + 1]) ÷ 2
  end
  @inbounds w[N] = w[N - 1] + e[N]
end

# Type E₆ (LiE's w2eE68, fully unrolled).
@inline function _w2e!(::Type{TypeE{6}}, e, w)
  @inbounds begin
    s = w[2] - w[3]
    e[5] = s
    sumsum = 4 * w[1] - s
    s += 2 * w[3]
    e[4] = s
    sumsum += s
    s += 2 * w[4]
    e[3] = s
    sumsum += s
    s += 2 * w[5]
    e[2] = s
    sumsum += s
    s += 2 * w[6]
    e[1] = s
    sumsum += s
    e[6] = -sumsum
  end
end
@inline function _e2w!(::Type{TypeE{6}}, w, e)
  @inbounds begin
    s = e[6]
    s += e[1]
    w[6] = (e[1] - e[2]) ÷ 2
    s += e[2]
    w[5] = (e[2] - e[3]) ÷ 2
    s += e[3]
    w[4] = (e[3] - e[4]) ÷ 2
    s += e[4]
    w[3] = (e[4] - e[5]) ÷ 2
    w[2] = w[3] + e[5]
    w[1] = (e[5] - s) ÷ 4
  end
end

# Type E₈ (LiE's w2eE68, fully unrolled).
@inline function _w2e!(::Type{TypeE{8}}, e, w)
  @inbounds begin
    s = w[2] - w[3]
    e[8] = s
    sumsum = 4 * w[1] - s
    s += 2 * w[3]
    e[7] = s
    sumsum += s
    s += 2 * w[4]
    e[6] = s
    sumsum += s
    s += 2 * w[5]
    e[5] = s
    sumsum += s
    s += 2 * w[6]
    e[4] = s
    sumsum += s
    s += 2 * w[7]
    e[3] = s
    sumsum += s
    s += 2 * w[8]
    e[2] = s
    sumsum += s
    e[1] = -sumsum
  end
end
@inline function _e2w!(::Type{TypeE{8}}, w, e)
  @inbounds begin
    s = e[1]
    s += e[2]
    w[8] = (e[2] - e[3]) ÷ 2
    s += e[3]
    w[7] = (e[3] - e[4]) ÷ 2
    s += e[4]
    w[6] = (e[4] - e[5]) ÷ 2
    s += e[5]
    w[5] = (e[5] - e[6]) ÷ 2
    s += e[6]
    w[4] = (e[6] - e[7]) ÷ 2
    s += e[7]
    w[3] = (e[7] - e[8]) ÷ 2
    w[2] = w[3] + e[8]
    w[1] = (e[8] - s) ÷ 4
  end
end

# Type E₇ (LiE's w2eE7).
@inline function _w2e!(::Type{TypeE{7}}, e, w)
  @inbounds begin
    s = 0
    e[8] = 0
    for i in 7:-1:3
      s += w[i]
      e[i] = s
    end
    e[2] = s + w[1]
    e[1] = e[7] + e[6] + e[5] - e[4] - e[3] - e[2] - 2 * w[2]
  end
end
@inline function _e2w!(::Type{TypeE{7}}, w, e)
  @inbounds begin
    w[1] = e[2] - e[3]
    for i in 3:7
      w[i] = e[i] - e[i + 1]
    end
    w[2] = (e[8] + e[7] + e[6] + e[5] - e[4] - e[3] - e[2] - e[1]) ÷ 2
  end
end

# Type F₄ (LiE's w2eF4).
@inline function _w2e!(::Type{TypeF4}, e, w)
  @inbounds begin
    e[4] = w[3]
    e[3] = e[4] + 2 * w[2]
    e[2] = e[3] + 2 * w[1]
    e[1] = -2 * w[4] - e[2] - e[3] - e[4]
  end
end
@inline function _e2w!(::Type{TypeF4}, w, e)
  @inbounds begin
    w[3] = e[4]
    w[2] = (e[3] - e[4]) ÷ 2
    w[1] = (e[2] - e[3]) ÷ 2
    w[4] = -(e[1] + e[2] + e[3] + e[4]) ÷ 2
  end
end

# Type G₂ (LiE's w2eG2).
@inline function _w2e!(::Type{TypeG2}, e, w)
  @inbounds begin
    e[3] = 0
    e[2] = w[2]
    e[1] = -w[1] - w[2]
  end
end
@inline function _e2w!(::Type{TypeG2}, w, e)
  @inbounds begin
    w[2] = e[2] - e[3]
    w[1] = e[3] - w[2] - e[1]
  end
end

# ─── Weylloop parameters (compile-time constants per type) ──────────────────

#   subtype: :A (permutations only), :B (perm + all signs), :D (perm + even signs)
_weylloop_subtype(::Type{TypeA{N}}) where {N} = :A
_weylloop_subtype(::Type{TypeB{N}}) where {N} = :B
_weylloop_subtype(::Type{TypeC{N}}) where {N} = :B
_weylloop_subtype(::Type{TypeD{N}}) where {N} = :D
_weylloop_subtype(::Type{TypeE{6}}) = :D
_weylloop_subtype(::Type{TypeE{7}}) = :A
_weylloop_subtype(::Type{TypeE{8}}) = :D
_weylloop_subtype(::Type{TypeF4}) = :B
_weylloop_subtype(::Type{TypeG2}) = :A

_weylloop_eps_dim(::Type{TypeA{N}}) where {N} = N + 1
_weylloop_eps_dim(::Type{TypeB{N}}) where {N} = N
_weylloop_eps_dim(::Type{TypeC{N}}) where {N} = N
_weylloop_eps_dim(::Type{TypeD{N}}) where {N} = N
_weylloop_eps_dim(::Type{TypeE{6}}) = 6
_weylloop_eps_dim(::Type{TypeE{7}}) = 8
_weylloop_eps_dim(::Type{TypeE{8}}) = 8
_weylloop_eps_dim(::Type{TypeF4}) = 4
_weylloop_eps_dim(::Type{TypeG2}) = 3

_weylloop_perm_size(::Type{TypeA{N}}) where {N} = N + 1
_weylloop_perm_size(::Type{TypeB{N}}) where {N} = N
_weylloop_perm_size(::Type{TypeC{N}}) where {N} = N
_weylloop_perm_size(::Type{TypeD{N}}) where {N} = N
_weylloop_perm_size(::Type{TypeE{6}}) = 5
_weylloop_perm_size(::Type{TypeE{7}}) = 8
_weylloop_perm_size(::Type{TypeE{8}}) = 8
_weylloop_perm_size(::Type{TypeF4}) = 4
_weylloop_perm_size(::Type{TypeG2}) = 3

# ─── Nextperm — lexicographic next permutation (increasing order) ───────────

# Generates the next permutation in increasing lexicographic order.
# Returns true if a next permutation was found, false at the last
# (fully decreasing) permutation.
@inline function _nextperm!(w::AbstractVector{<:Integer}, ::Val{N}) where {N}
  N <= 1 && return false
  # Find last ascent: rightmost i where w[i] < w[i+1]
  i = N - 1
  @inbounds while i >= 1 && w[i] >= w[i + 1]
    i -= 1
  end
  i < 1 && return false
  # Find rightmost j > i with w[j] > w[i]
  j = N
  @inbounds while w[i] >= w[j]
    j -= 1
  end
  # Swap w[i] and w[j]
  @inbounds w[i], w[j] = w[j], w[i]
  # Reverse suffix starting at i+1
  lo, hi = i + 1, N
  @inbounds while lo < hi
    w[lo], w[hi] = w[hi], w[lo]
    lo += 1
    hi -= 1
  end
  return true
end

# ─── normalform — canonical representative of a classical suborbit ──────────

# Bring ε-vector to normal form (canonical representative).
# LiE convention: entries in weakly INCREASING order.
# Type A: sort increasingly, then subtract w[1] so first entry becomes 0.
# Types B/C: strip signs, sort increasingly.
# Type D: strip signs (count parity), sort increasingly, negate w[1]
#   if odd parity and w[1] ≠ 0.
@inline function _normalform!(
  w::AbstractVector{<:Integer}, ::Val{N}, subtype::Symbol
) where {N}
  if subtype == :A
    sort!(@view(w[1:N]))
    @inbounds if w[1] != 0
      c = w[1]
      for i in 1:N
        w[i] -= c
      end
    end
  else
    parity = 0
    @inbounds for i in 1:N
      if w[i] < 0
        w[i] = -w[i]
        parity += 1
      end
    end
    sort!(@view(w[1:N]))
    @inbounds if subtype == :D && isodd(parity) && w[1] != 0
      w[1] = -w[1]
    end
  end
end

# ─── Weyl group matrix from a reduced word ──────────────────────────────────

function _weyl_matrix(::Type{DT}, word::Vector{<:Integer}) where {DT}
  R = rank(DT)
  C = cartan_matrix(DT)
  M = Matrix{Int}(_I, R, R)
  @inbounds for k in word
    for i in 1:R
      pairing = M[i, k]
      iszero(pairing) && continue
      for j in 1:R
        M[i, j] -= pairing * C[j, k]
      end
    end
  end
  return M
end

# ─── Coset representative matrices (cached) ─────────────────────────────────

const _coset_reps_cache = Dict{Type,Vector{Matrix{Int}}}()

function _coset_reps(::Type{DT}) where {DT}
  haskey(_coset_reps_cache, DT) && return _coset_reps_cache[DT]::Vector{Matrix{Int}}
  reps = _build_coset_reps(DT)
  _coset_reps_cache[DT] = reps
  return reps
end

# Classical types: identity only.
_build_coset_reps(::Type{TypeA{N}}) where {N} = [Matrix{Int}(_I, N, N)]
_build_coset_reps(::Type{TypeB{N}}) where {N} = [Matrix{Int}(_I, N, N)]
_build_coset_reps(::Type{TypeC{N}}) where {N} = [Matrix{Int}(_I, N, N)]
_build_coset_reps(::Type{TypeD{N}}) where {N} = [Matrix{Int}(_I, N, N)]

# E₆: cox=[6,2,5,4,3,1], cox_order=12, X = {id, [2,4,3,1], [1,4,5,2,4,3,1]}
function _build_coset_reps(::Type{TypeE{6}})
  DT = TypeE{6}
  cox = _weyl_matrix(DT, [6, 2, 5, 4, 3, 1])
  X = [
    Matrix{Int}(_I, 6, 6),
    _weyl_matrix(DT, [2, 4, 3, 1]),
    _weyl_matrix(DT, [1, 4, 5, 2, 4, 3, 1]),
  ]
  return _build_from_cox_and_X(6, cox, 12, X)
end

# E₇: cox=[1,2,3,4,5,6,7], cox_order=18
#   x₁ = [7,6,5,4,3,2,1,5,3,4,2], X = {id, x₁, x₁², x₁⁵}
function _build_coset_reps(::Type{TypeE{7}})
  DT = TypeE{7}
  cox = _weyl_matrix(DT, [1, 2, 3, 4, 5, 6, 7])
  x1 = _weyl_matrix(DT, [7, 6, 5, 4, 3, 2, 1, 5, 3, 4, 2])
  x2 = x1 * x1
  x5 = x2 * x2 * x1
  X = [Matrix{Int}(_I, 7, 7), x1, x2, x5]
  return _build_from_cox_and_X(7, cox, 18, X)
end

# E₈: cox=[7,5,3,2,8,6,4,1], cox_order=15
#   x₁ = [7,4,3,2,4,5,6,1,3,4,5,2,4,3,1]
#   x₂ = [8,6,7,5,6,3,4,5,2,4,3,1]
#   X = {id, x₁, x₁³, x₁⁴, x₂, x₂², x₂³, x₂⁴, x₂⁵}
function _build_coset_reps(::Type{TypeE{8}})
  DT = TypeE{8}
  cox = _weyl_matrix(DT, [7, 5, 3, 2, 8, 6, 4, 1])
  x1 = _weyl_matrix(DT, [7, 4, 3, 2, 4, 5, 6, 1, 3, 4, 5, 2, 4, 3, 1])
  x1_2 = x1 * x1
  x1_3 = x1_2 * x1
  x1_4 = x1_2 * x1_2
  x2 = _weyl_matrix(DT, [8, 6, 7, 5, 6, 3, 4, 5, 2, 4, 3, 1])
  x2_2 = x2 * x2
  x2_3 = x2_2 * x2
  x2_4 = x2_3 * x2
  x2_5 = x2_4 * x2
  X = [
    Matrix{Int}(_I, 8, 8), x1, x1_3, x1_4,
    x2, x2_2, x2_3, x2_4, x2_5,
  ]
  return _build_from_cox_and_X(8, cox, 15, X)
end

# F₄: cox=[1,2,3,4], cox_order=3, X={id}
function _build_coset_reps(::Type{TypeF4})
  DT = TypeF4
  cox = _weyl_matrix(DT, [1, 2, 3, 4])
  X = [Matrix{Int}(_I, 4, 4)]
  return _build_from_cox_and_X(4, cox, 3, X)
end

# G₂: cox=[1,2], cox_order=2, X={id}
function _build_coset_reps(::Type{TypeG2})
  DT = TypeG2
  cox = _weyl_matrix(DT, [1, 2])
  X = [Matrix{Int}(_I, 2, 2)]
  return _build_from_cox_and_X(2, cox, 2, X)
end

# Build coset reps: {cox^k * X_j : 0 ≤ k < cox_order, j ∈ X}, deduplicated.
function _build_from_cox_and_X(
  R::Integer, cox_mat::Matrix{Int}, cox_order::Integer, X_mats::Vector{Matrix{Int}}
)
  all_reps = Matrix{Int}[]
  cur = Matrix{Int}(_I, R, R)
  for k in 0:(cox_order - 1)
    for X in X_mats
      push!(all_reps, cur * X)
    end
    if k < cox_order - 1
      cur = cur * cox_mat
    end
  end
  unique!(all_reps)
  return all_reps
end

# ─── Weylloop — the main orbit traversal ────────────────────────────────────

"""
    weylloop(action!, ::Type{DT}, v::AbstractVector{<:Integer})

Call `action!(w)` once for each weight `w` in the Weyl orbit of `v`,
where `v` and `w` are in the fundamental weight (ω) basis.

Uses LiE's ε-basis algorithm: converts to ε-coordinates where a classical
subgroup acts by permutations (type A) or permutations + sign changes
(types B/C/D), then enumerates orbits systematically via lexicographic
permutation generation and Gray-code sign flips — with no hash set or BFS.

For exceptional types, coset representatives W / W_classical are precomputed
as matrices.

All transforms dispatch on the Dynkin type, enabling the compiler to inline
them and unroll fixed-size loops.  Workspace vectors use stack-allocated
`MVector` from StaticArrays.

`action!` receives a mutable workspace vector; it must NOT hold a reference
to this vector across calls (copy if needed).
"""
function weylloop(
  action!::F, ::Type{DT}, v::AbstractVector{<:Integer}
) where {F,DT<:SimpleDynkinType}
  R = rank(DT)
  subtype = _weylloop_subtype(DT)
  ED = _weylloop_eps_dim(DT)
  PS = _weylloop_perm_size(DT)

  # Stack-allocated workspace
  tmp_w = MVector{R,Int}(undef)    # weight-coord scratch for _e2w!
  alt_w = MVector{R,Int}(undef)    # scratch for matrix multiply

  coset_reps = _coset_reps(DT)

  # Tabulate suborbit representatives:
  # For each coset rep, compute v * rep in weight coords, convert to ε, normalise.
  suborbit_eps = Vector{MVector{ED,Int}}(undef, 0)
  for rep in coset_reps
    # Multiply v * rep (row-vector × matrix)
    @inbounds for j in 1:R
      s = 0
      for i in 1:R
        s += v[i] * rep[i, j]
      end
      alt_w[j] = s
    end
    # Convert to ε-coords
    e_tmp = MVector{ED,Int}(undef)
    _w2e!(DT, e_tmp, alt_w)
    _normalform!(e_tmp, Val(PS), subtype)
    push!(suborbit_eps, e_tmp)
  end

  # Remove duplicate suborbits (sort + unique)
  sort!(suborbit_eps)
  unique!(suborbit_eps)

  # Traverse each suborbit
  _weylloop_suborbits!(action!, DT, suborbit_eps, tmp_w, subtype, Val(PS))
end

# Inner loop separated for type specialization.  The compiler sees DT, Val{PS},
# and subtype as constants, so _e2w! and _nextperm! are fully inlined.
@inline function _weylloop_suborbits!(
  action!::F, ::Type{DT}, suborbit_eps, tmp_w, subtype::Symbol, ::Val{PS}
) where {F,DT,PS}
  inx = MVector{PS,Int}(undef)

  for e_rep in suborbit_eps
    w = MVector(e_rep)  # copy into mutable stack vector

    if subtype == :A
      # ── Type A: permutations only ──────────────────────────────────
      @inbounds while true
        _e2w!(DT, tmp_w, w)
        action!(tmp_w)
        _nextperm!(w, Val(PS)) || break
      end
    else
      # ── Types B/D: permutations + sign changes ────────────────────
      @inbounds alternate = (subtype == :D && w[1] != 0)

      @inbounds while true  # permutation loop
        # Build index of nonzero entries
        n_nonzero = 0
        for i in 1:PS
          if w[i] != 0
            n_nonzero += 1
            inx[n_nonzero] = i
          end
        end

        # Sign-change inner loop (Gray code, following LiE exactly)
        signcount = UInt(0)
        while true
          _e2w!(DT, tmp_w, w)
          action!(tmp_w)

          # Generate next sign combination
          ii = 0  # 0-based index into inx
          bits = signcount
          signcount += 1
          if alternate
            w[1] = -w[1]
            ii += 1
          end
          while ii < n_nonzero && isodd(bits)
            bits >>= 1
            ii += 1
          end
          if ii >= n_nonzero
            # Done with sign combinations; restore last sign and break
            ii -= 1
            if ii >= 0
              w[inx[ii + 1]] = -w[inx[ii + 1]]
            end
            break
          end
          w[inx[ii + 1]] = -w[inx[ii + 1]]
        end

        # Generate next permutation
        # For type D with alternate, w[1] may be negative; temporarily fix
        minus = alternate && w[1] < 0
        if minus
          w[1] = -w[1]
        end
        has_next = _nextperm!(w, Val(PS))
        if minus
          w[1] = -w[1]
        end
        has_next || break
      end
    end
  end
end

# ─── Product type support ───────────────────────────────────────────────────

function weylloop(
  action!::F, ::Type{ProductDynkinType{T}}, v::AbstractVector{<:Integer}
) where {F,T}
  DT = ProductDynkinType{T}
  R = rank(DT)

  # Split v into factor components and collect orbits for each factor
  factors = T.parameters
  n_factors = length(factors)
  offsets = Vector{Int}(undef, n_factors + 1)
  offsets[1] = 0
  for i in 1:n_factors
    offsets[i + 1] = offsets[i] + rank(factors[i])
  end

  factor_orbits = Vector{Vector{Vector{Int}}}(undef, n_factors)
  for i in 1:n_factors
    FDT = typeof(factors[i])
    r_f = rank(FDT)
    v_f = Vector{Int}(v[(offsets[i] + 1):offsets[i + 1]])
    orbits_f = Vector{Int}[]
    weylloop(FDT, v_f) do tmp
      push!(orbits_f, Vector{Int}(tmp))
    end
    factor_orbits[i] = orbits_f
  end

  # Cartesian product
  tmp_full = Vector{Int}(undef, R)
  _product_weylloop_recurse!(action!, tmp_full, factor_orbits, offsets, 1, n_factors)
end

function _product_weylloop_recurse!(
  action!, tmp::Vector{Int}, factor_orbits, offsets, depth, n_factors
)
  if depth > n_factors
    action!(tmp)
    return nothing
  end
  r_f = offsets[depth + 1] - offsets[depth]
  for orb_pt in factor_orbits[depth]
    for j in 1:r_f
      tmp[offsets[depth] + j] = orb_pt[j]
    end
    _product_weylloop_recurse!(action!, tmp, factor_orbits, offsets, depth + 1, n_factors)
  end
end
