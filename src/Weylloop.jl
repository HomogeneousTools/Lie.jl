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
# ═══════════════════════════════════════════════════════════════════════════════

# ─── ε-basis transforms per simple type ─────────────────────────────────────

# Type A: ε_i = ω_i + ⋯ + ω_r (partial sums from right), ε_{r+1} = 0.
function _w2e_A!(e::AbstractVector, w::AbstractVector, r::Int)
  s = 0
  e[r + 1] = 0
  for i in r:-1:1
    s += w[i]
    e[i] = s
  end
end
function _e2w_A!(w::AbstractVector, e::AbstractVector, r::Int)
  for i in 1:r
    w[i] = e[i] - e[i + 1]
  end
end

# Type B: ε_i = 2ω_i + ⋯ + 2ω_{r-1} + ω_r for i < r, ε_r = ω_r.
function _w2e_B!(e::AbstractVector, w::AbstractVector, r::Int)
  s = w[r]
  e[r] = s
  for i in (r - 1):-1:1
    s += 2 * w[i]
    e[i] = s
  end
end
function _e2w_B!(w::AbstractVector, e::AbstractVector, r::Int)
  for i in 1:(r - 1)
    w[i] = (e[i] - e[i + 1]) ÷ 2
  end
  w[r] = e[r]
end

# Type C: ε_i = ω_i + ⋯ + ω_r.
function _w2e_C!(e::AbstractVector, w::AbstractVector, r::Int)
  s = w[r]
  e[r] = s
  for i in (r - 1):-1:1
    s += w[i]
    e[i] = s
  end
end
function _e2w_C!(w::AbstractVector, e::AbstractVector, r::Int)
  for i in 1:(r - 1)
    w[i] = e[i] - e[i + 1]
  end
  w[r] = e[r]
end

# Type D: ε_r = ω_r - ω_{r-1}, then ε_i = 2ω_i + ⋯ + 2ω_{r-2} + ω_{r-1} + ω_r.
function _w2e_D!(e::AbstractVector, w::AbstractVector, r::Int)
  s = w[r] - w[r - 1]
  e[r] = s
  for i in (r - 1):-1:1
    s += 2 * w[i]
    e[i] = s
  end
end
function _e2w_D!(w::AbstractVector, e::AbstractVector, r::Int)
  for i in 1:(r - 1)
    w[i] = (e[i] - e[i + 1]) ÷ 2
  end
  w[r] = w[r - 1] + e[r]
end

# Type E₆/E₈ (LiE's w2eE68).
# LiE uses 0-indexed arrays.  Key mapping to 1-indexed Julia:
#   C: e[u-1] = w[1]-w[2]         →  Julia: e[u] = w[2]-w[3]   (u = perm_size)
#   C: e[u-i] for i=2..rnk-1      →  Julia: e[u-i+2] for i=3..r
#   C: e[rnk==6 ? 5 : 0] = extra  →  Julia: e[r==6 ? 6 : 1]
# Here u = perm_size (5 for E₆, 8 for E₈).
function _w2e_E68!(e::AbstractVector, w::AbstractVector, r::Int)
  u = (r == 6) ? 5 : 8  # perm_size
  s = w[2] - w[3]
  e[u] = s
  sumsum = 4 * w[1] - s
  for i in 3:r
    s += 2 * w[i]
    e[u - i + 2] = s
    sumsum += s
  end
  e[r == 6 ? 6 : 1] = -sumsum
end
function _e2w_E68!(w::AbstractVector, e::AbstractVector, r::Int)
  u = (r == 6) ? 5 : 8
  s = e[r == 6 ? 6 : 1]
  for i in r:-1:3
    s += e[u - i + 2]
    w[i] = (e[u - i + 2] - e[u - i + 3]) ÷ 2
  end
  w[2] = w[3] + e[u]
  w[1] = (e[u] - s) ÷ 4
end

# Type E₇ (LiE's w2eE7).
function _w2e_E7!(e::AbstractVector, w::AbstractVector, ::Int)
  s = 0
  e[8] = 0
  for i in 7:-1:3
    s += w[i]
    e[i] = s
  end
  e[2] = s + w[1]
  e[1] = e[7] + e[6] + e[5] - e[4] - e[3] - e[2] - 2 * w[2]
end
function _e2w_E7!(w::AbstractVector, e::AbstractVector, ::Int)
  w[1] = e[2] - e[3]
  for i in 3:7
    w[i] = e[i] - e[i + 1]
  end
  w[2] = (e[8] + e[7] + e[6] + e[5] - e[4] - e[3] - e[2] - e[1]) ÷ 2
end

# Type F₄ (LiE's w2eF4).
function _w2e_F4!(e::AbstractVector, w::AbstractVector, ::Int)
  e[4] = w[3]
  e[3] = e[4] + 2 * w[2]
  e[2] = e[3] + 2 * w[1]
  e[1] = -2 * w[4] - e[2] - e[3] - e[4]
end
function _e2w_F4!(w::AbstractVector, e::AbstractVector, ::Int)
  w[3] = e[4]
  w[2] = (e[3] - e[4]) ÷ 2
  w[1] = (e[2] - e[3]) ÷ 2
  w[4] = -(e[1] + e[2] + e[3] + e[4]) ÷ 2
end

# Type G₂ (LiE's w2eG2).
function _w2e_G2!(e::AbstractVector, w::AbstractVector, ::Int)
  e[3] = 0
  e[2] = w[2]
  e[1] = -w[1] - w[2]
end
function _e2w_G2!(w::AbstractVector, e::AbstractVector, ::Int)
  w[2] = e[2] - e[3]
  w[1] = e[3] - w[2] - e[1]
end

# ─── Dispatch table ──────────────────────────────────────────────────────────

# Returns (to_e!, from_e!, subtype, eps_dim, perm_size) for a simple type.
#   subtype: :A (permutations only), :B (perm + all signs), :D (perm + even signs)
function _weylloop_params(::Type{TypeA{N}}) where {N}
  (_w2e_A!, _e2w_A!, :A, N + 1, N + 1)
end
function _weylloop_params(::Type{TypeB{N}}) where {N}
  (_w2e_B!, _e2w_B!, :B, N, N)
end
function _weylloop_params(::Type{TypeC{N}}) where {N}
  (_w2e_C!, _e2w_C!, :B, N, N)
end
function _weylloop_params(::Type{TypeD{N}}) where {N}
  (_w2e_D!, _e2w_D!, :D, N, N)
end
function _weylloop_params(::Type{TypeE{6}})
  (_w2e_E68!, _e2w_E68!, :D, 6, 5)
end
function _weylloop_params(::Type{TypeE{7}})
  (_w2e_E7!, _e2w_E7!, :A, 8, 8)
end
function _weylloop_params(::Type{TypeE{8}})
  (_w2e_E68!, _e2w_E68!, :D, 8, 8)
end
function _weylloop_params(::Type{TypeF4})
  (_w2e_F4!, _e2w_F4!, :B, 4, 4)
end
function _weylloop_params(::Type{TypeG2})
  (_w2e_G2!, _e2w_G2!, :A, 3, 3)
end

# ─── Nextperm — lexicographic next permutation (increasing order) ───────────

# Generates the next permutation in increasing lexicographic order.
# Returns true if a next permutation was found, false at the last
# (fully decreasing) permutation.
function _nextperm!(w::AbstractVector{Int}, n::Int)
  n <= 1 && return false
  # Find last ascent: rightmost i where w[i] < w[i+1]
  i = n - 1
  while i >= 1 && w[i] >= w[i + 1]
    i -= 1
  end
  i < 1 && return false
  # Find rightmost j > i with w[j] > w[i]
  j = n
  while w[i] >= w[j]
    j -= 1
  end
  # Swap w[i] and w[j]
  w[i], w[j] = w[j], w[i]
  # Reverse suffix starting at i+1
  lo, hi = i + 1, n
  while lo < hi
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
function _normalform!(w::AbstractVector{Int}, n::Int, subtype::Symbol)
  if subtype == :A
    sort!(@view(w[1:n]))
    if w[1] != 0
      c = w[1]
      for i in 1:n
        w[i] -= c
      end
    end
  else
    parity = 0
    for i in 1:n
      if w[i] < 0
        w[i] = -w[i]
        parity += 1
      end
    end
    sort!(@view(w[1:n]))
    if subtype == :D && isodd(parity) && w[1] != 0
      w[1] = -w[1]
    end
  end
end

# ─── Weyl group matrix from a reduced word ──────────────────────────────────

function _weyl_matrix(::Type{DT}, word::Vector{Int}) where {DT}
  R = rank(DT)
  C = cartan_matrix(DT)
  M = Matrix{Int}(_I, R, R)
  for k in word
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
  R::Int, cox_mat::Matrix{Int}, cox_order::Int, X_mats::Vector{Matrix{Int}}
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
    weylloop(action!, ::Type{DT}, v::AbstractVector{Int})

Call `action!(w)` once for each weight `w` in the Weyl orbit of `v`,
where `v` and `w` are in the fundamental weight (ω) basis.

Uses LiE's ε-basis algorithm: converts to ε-coordinates where a classical
subgroup acts by permutations (type A) or permutations + sign changes
(types B/C/D), then enumerates orbits systematically via lexicographic
permutation generation and Gray-code sign flips — with no hash set or BFS.

For exceptional types, coset representatives W / W_classical are precomputed
as matrices.

`action!` receives a mutable workspace vector; it must NOT hold a reference
to this vector across calls (copy if needed).
"""
function weylloop(
  action!::F, ::Type{DT}, v::AbstractVector{Int}
) where {F,DT<:SimpleDynkinType}
  R = rank(DT)
  to_e!, from_e!, subtype, eps_dim, perm_size = _weylloop_params(DT)

  # Workspace vectors
  tmp_w = Vector{Int}(undef, R)    # weight-coord scratch for from_e!
  alt_w = Vector{Int}(undef, R)    # scratch for matrix multiply

  coset_reps = _coset_reps(DT)

  # Tabulate suborbit representatives:
  # For each coset rep, compute v * rep in weight coords, convert to ε, normalise.
  suborbit_eps = Vector{Vector{Int}}(undef, 0)
  for rep in coset_reps
    # Multiply v * rep (row-vector × matrix)
    for j in 1:R
      s = 0
      for i in 1:R
        s += v[i] * rep[i, j]
      end
      alt_w[j] = s
    end
    # Convert to ε-coords
    e_tmp = Vector{Int}(undef, eps_dim)
    to_e!(e_tmp, alt_w, R)
    _normalform!(e_tmp, perm_size, subtype)
    push!(suborbit_eps, e_tmp)
  end

  # Remove duplicate suborbits (sort + unique)
  sort!(suborbit_eps)
  unique!(suborbit_eps)

  # Traverse each suborbit
  inx = Vector{Int}(undef, perm_size)  # indices of nonzero entries

  for e_rep in suborbit_eps
    w = copy(e_rep)

    if subtype == :A
      # ── Type A: permutations only ──────────────────────────────────
      while true
        from_e!(tmp_w, w, R)
        action!(tmp_w)
        _nextperm!(w, perm_size) || break
      end
    else
      # ── Types B/D: permutations + sign changes ────────────────────
      alternate = (subtype == :D && w[1] != 0)

      while true  # permutation loop
        # Build index of nonzero entries
        n_nonzero = 0
        for i in 1:perm_size
          if w[i] != 0
            n_nonzero += 1
            inx[n_nonzero] = i
          end
        end

        # Sign-change inner loop (Gray code, following LiE exactly)
        signcount = UInt(0)
        while true
          from_e!(tmp_w, w, R)
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
              w[inx[ii + 1]] = -w[inx[ii + 1]]  # +1 for 1-based inx
            end
            break
          end
          w[inx[ii + 1]] = -w[inx[ii + 1]]  # +1 for 1-based inx
        end

        # Generate next permutation
        # For type D with alternate, w[1] may be negative; temporarily fix
        minus = alternate && w[1] < 0
        if minus
          w[1] = -w[1]
        end
        has_next = _nextperm!(w, perm_size)
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
  action!::F, ::Type{ProductDynkinType{T}}, v::AbstractVector{Int}
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
      push!(orbits_f, copy(tmp))
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
