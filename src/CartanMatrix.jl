# ═══════════════════════════════════════════════════════════════════════════════
#  Cartan matrices — compile-time specialized via @generated functions
#
#  Convention: (a_{ij}) = ⟨αᵢ∨, αⱼ⟩  (following Bourbaki / Oscar)
# ═══════════════════════════════════════════════════════════════════════════════

export cartan_matrix, cartan_symmetrizer, cartan_bilinear_form, cartan_matrix_inverse
export omega_bilinear_form_scaled, cartan_determinant
export sub_dynkin_type, sub_dynkin_ordering, sub_dynkin_type_with_ordering

# ─── Type A ──────────────────────────────────────────────────────────────────
# A_n: tridiagonal, 2 on diagonal, -1 on super/sub-diagonal

# Runtime fallback for any DynkinType with rank ≥ 17 — avoids @generated stall
function _cartan_matrix_runtime(::Type{DT}) where {DT<:DynkinType}
  check_dynkin_type(DT)
  R = rank(DT)
  C = _cartan_matrix_data(DT)
  return SMatrix{R,R,Int,R*R}(Tuple(C[i, j] for j in 1:R for i in 1:R))
end

@generated function cartan_matrix(::Type{TypeA{N}}) where {N}
  N >= 1 || return :(throw(ArgumentError($(_invalid_dynkin_type_message(TypeA{N})))))
  N >= 17 && return :(_cartan_matrix_runtime($(TypeA{N})))
  entries = Int[]
  for j in 1:N, i in 1:N
    if i == j
      push!(entries, 2)
    elseif abs(i - j) == 1
      push!(entries, -1)
    else
      push!(entries, 0)
    end
  end
  return :(SMatrix{$N,$N,Int,$(N * N)}($(Tuple(entries))))
end

# ─── Type B ──────────────────────────────────────────────────────────────────
# B_n: like A but C[n, n-1] = -2

@generated function cartan_matrix(::Type{TypeB{N}}) where {N}
  N >= 2 || return :(throw(ArgumentError($(_invalid_dynkin_type_message(TypeB{N})))))
  N >= 17 && return :(_cartan_matrix_runtime($(TypeB{N})))
  entries = Int[]
  for j in 1:N, i in 1:N
    if i == j
      push!(entries, 2)
    elseif i == N && j == N - 1
      push!(entries, -2)
    elseif abs(i - j) == 1
      push!(entries, -1)
    else
      push!(entries, 0)
    end
  end
  return :(SMatrix{$N,$N,Int,$(N * N)}($(Tuple(entries))))
end

# ─── Type C ──────────────────────────────────────────────────────────────────
# C_n: like A but C[n-1, n] = -2

@generated function cartan_matrix(::Type{TypeC{N}}) where {N}
  N >= 2 || return :(throw(ArgumentError($(_invalid_dynkin_type_message(TypeC{N})))))
  N >= 17 && return :(_cartan_matrix_runtime($(TypeC{N})))
  entries = Int[]
  for j in 1:N, i in 1:N
    if i == j
      push!(entries, 2)
    elseif i == N - 1 && j == N
      push!(entries, -2)
    elseif abs(i - j) == 1
      push!(entries, -1)
    else
      push!(entries, 0)
    end
  end
  return :(SMatrix{$N,$N,Int,$(N * N)}($(Tuple(entries))))
end

# ─── Type D ──────────────────────────────────────────────────────────────────
# D_n: like A_{n-2} extended with branching at node n-2
#   Dynkin: 1 - 2 - ... - (n-2) < (n-1)
#                                  \ n
#   C[n-2,n-1] = C[n-1,n-2] = -1
#   C[n-2,n] = C[n,n-2] = -1
#   C[n-1,n] = C[n,n-1] = 0

@generated function cartan_matrix(::Type{TypeD{N}}) where {N}
  N >= 3 || return :(throw(ArgumentError($(_invalid_dynkin_type_message(TypeD{N})))))
  N >= 17 && return :(_cartan_matrix_runtime($(TypeD{N})))
  entries = Int[]
  for j in 1:N, i in 1:N
    if i == j
      push!(entries, 2)
    elseif i <= N - 2 && j <= N - 2 && abs(i - j) == 1
      push!(entries, -1)
    elseif (i == N - 2 && j == N - 1) || (i == N - 1 && j == N - 2)
      push!(entries, -1)
    elseif (i == N - 2 && j == N) || (i == N && j == N - 2)
      push!(entries, -1)
    else
      push!(entries, 0)
    end
  end
  return :(SMatrix{$N,$N,Int,$(N * N)}($(Tuple(entries))))
end

# ─── Type E ──────────────────────────────────────────────────────────────────
# E: nodes 1-3-4-5-..., with node 2 branching off node 4
#   (Bourbaki labeling used by Oscar)

function _E8_cartan()
  # E8 Cartan matrix (Bourbaki labeling):
  #   1 - 3 - 4 - 5 - 6 - 7 - 8
  #           |
  #           2
  C = zeros(Int, 8, 8)
  for i in 1:8
    C[i, i] = 2
  end
  # edges: 1-3, 3-4, 4-5, 5-6, 6-7, 7-8, 2-4
  edges = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (2, 4)]
  for (a, b) in edges
    C[a, b] = -1
    C[b, a] = -1
  end
  return C
end

@generated function cartan_matrix(::Type{TypeE{N}}) where {N}
  N in (6, 7, 8) ||
    return :(throw(ArgumentError($(_invalid_dynkin_type_message(TypeE{N})))))
  C8 = _E8_cartan()
  C = C8[1:N, 1:N]
  entries = Tuple(C[i, j] for j in 1:N for i in 1:N)
  return :(SMatrix{$N,$N,Int,$(N * N)}($entries))
end

# ─── Type F4 ─────────────────────────────────────────────────────────────────
# F4: 1 - 2 >=> 3 - 4

function cartan_matrix(::Type{TypeF4})
  #  F₄ Cartan matrix (Bourbaki): 1 - 2 =>= 3 - 4
  #  C = [2 -1 0 0; -1 2 -1 0; 0 -2 2 -1; 0 0 -1 2]
  #  Stored column-major:
  SMatrix{4,4,Int,16}((
    2, -1, 0, 0,   # column 1
    -1, 2, -2, 0,   # column 2
    0, -1, 2, -1,   # column 3
    0, 0, -1, 2,    # column 4
  ))
end

# ─── Type G2 ─────────────────────────────────────────────────────────────────
# G2: 1 <<< 2   (C[1,2] = -3, C[2,1] = -1)

function cartan_matrix(::Type{TypeG2})
  #  G₂ Cartan matrix: C = [2 -3; -1 2]
  #  Column-major: col1=[2,-1], col2=[-3,2]
  SMatrix{2,2,Int,4}((
    2, -1,   # column 1
    -3, 2,    # column 2
  ))
end

# ─── Product types ───────────────────────────────────────────────────────────

# Value-level Cartan matrix builders.
#
# These take the rank as an ordinary `Int`, so each builder is compiled exactly
# once instead of once per `TypeX{N}` instantiation.  The `_cartan_matrix_data`
# methods below are one-line forwarders that only carry the family dispatch.
function _cartan_matrix_data_a(n::Int)
  C = zeros(Int, n, n)
  for i in 1:n
    C[i, i] = 2
  end
  for i in 1:(n - 1)
    C[i, i + 1] = -1
    C[i + 1, i] = -1
  end
  return C
end

function _cartan_matrix_data_b(n::Int)
  C = _cartan_matrix_data_a(n)
  C[n, n - 1] = -2
  return C
end

function _cartan_matrix_data_c(n::Int)
  C = _cartan_matrix_data_a(n)
  C[n - 1, n] = -2
  return C
end

function _cartan_matrix_data_d(n::Int)
  C = zeros(Int, n, n)
  for i in 1:n
    C[i, i] = 2
  end
  for i in 1:(n - 3)
    C[i, i + 1] = -1
    C[i + 1, i] = -1
  end
  C[n - 2, n - 1] = -1
  C[n - 1, n - 2] = -1
  C[n - 2, n] = -1
  C[n, n - 2] = -1
  return C
end

_cartan_matrix_data_e(n::Int) = _E8_cartan()[1:n, 1:n]

_cartan_matrix_data(::Type{TypeA{N}}) where {N} =
  (check_dynkin_type(TypeA{N}); _cartan_matrix_data_a(N))
_cartan_matrix_data(::Type{TypeB{N}}) where {N} =
  (check_dynkin_type(TypeB{N}); _cartan_matrix_data_b(N))
_cartan_matrix_data(::Type{TypeC{N}}) where {N} =
  (check_dynkin_type(TypeC{N}); _cartan_matrix_data_c(N))
_cartan_matrix_data(::Type{TypeD{N}}) where {N} =
  (check_dynkin_type(TypeD{N}); _cartan_matrix_data_d(N))
_cartan_matrix_data(::Type{TypeE{N}}) where {N} =
  (check_dynkin_type(TypeE{N}); _cartan_matrix_data_e(N))

function _cartan_matrix_data(::Type{TypeF4})
  # Bourbaki: 1 - 2 >=> 3 - 4
  # C[3,2] = -2, C[2,3] = -1  (arrow points from short to long)
  return [2 -1 0 0; -1 2 -1 0; 0 -2 2 -1; 0 0 -1 2]
end

function _cartan_matrix_data(::Type{TypeG2})
  return [2 -3; -1 2]
end

function _cartan_matrix_data(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  types = Ts.parameters
  R = sum(rank(T) for T in types)
  C = zeros(Int, R, R)
  offset = 0
  for T in types
    r = rank(T)
    C_block = _cartan_matrix_data(T)
    C[(offset + 1):(offset + r), (offset + 1):(offset + r)] .= C_block
    offset += r
  end
  return C
end

"""
    cartan_matrix(::Type{ProductDynkinType{Ts}})

Block-diagonal Cartan matrix for a product of simple types.

# Examples
```jldoctest
julia> using Semisimple

julia> cartan_matrix(TypeA{2}) == [2 -1; -1 2]
true

julia> cartan_matrix(TypeG2) == [2 -3; -1 2]
true
```
"""
@generated function cartan_matrix(::Type{ProductDynkinType{Ts}}) where {Ts}
  types = Ts.parameters
  R = sum(rank(T) for T in types)
  R >= 17 && return :(_cartan_matrix_runtime($(ProductDynkinType{Ts})))
  C = zeros(Int, R, R)
  offset = 0
  for T in types
    r = rank(T)
    # Build the Cartan matrix for this component at compile time
    C_block = _cartan_matrix_data(T)
    C[(offset + 1):(offset + r), (offset + 1):(offset + r)] .= C_block
    offset += r
  end
  entries = Tuple(C[i, j] for j in 1:R for i in 1:R)
  return :(SMatrix{$R,$R,Int,$(R * R)}($entries))
end

# Instance dispatch
cartan_matrix(dt::DynkinType) = cartan_matrix(typeof(dt))

# ─── Cartan symmetrizer ─────────────────────────────────────────────────────
# d_i such that d_i * C_{ij} = d_j * C_{ji}  (symmetrizes the Cartan matrix)

# Value-level core: computed from the Cartan matrix alone, compiled once.
function _cartan_symmetrizer_from(C::Matrix{Int})
  N = size(C, 1)
  d = ones(Rational{Int}, N)
  visited = falses(N)
  queue = [1]
  visited[1] = true
  while !isempty(queue)
    i = popfirst!(queue)
    for j in 1:N
      if !visited[j] && C[i, j] != 0
        # d[i] * C[i,j] = d[j] * C[j,i]  =>  d[j] = d[i] * C[i,j] / C[j,i]
        d[j] = d[i] * C[i, j]//C[j, i]
        d[j] > 0 || error("Cartan symmetrizer must stay positive for finite Dynkin types")
        visited[j] = true
        push!(queue, j)
      end
    end
  end
  # Scale to positive integers with minimum 1
  lcm_den = lcm(denominator.(d)...)
  d_int = Int.(d .* lcm_den)
  g = gcd(d_int...)
  d_int .= d_int .÷ g
  return d_int
end

function _cartan_symmetrizer_data(::Type{DT}) where {DT<:SimpleDynkinType}
  return _cartan_symmetrizer_from(_cartan_matrix_data(DT))
end

function _cartan_symmetrizer_data(::Type{ProductDynkinType{Ts}}) where {Ts}
  types = Ts.parameters
  d = Int[]
  for T in types
    append!(d, _cartan_symmetrizer_data(T))
  end
  return d
end

"""
    cartan_symmetrizer(::Type{DT}) -> SVector

Return the symmetrizer `d` such that `diag(d) * C` is symmetric, where `C` is
the Cartan matrix of `DT`. Entries are positive integers with gcd 1.

# Examples
```jldoctest
julia> using Semisimple

julia> cartan_symmetrizer(TypeB{3}) == [2, 2, 1]
true

julia> cartan_symmetrizer(TypeG2) == [1, 3]
true
```
"""
@generated function cartan_symmetrizer(::Type{DT}) where {DT<:SimpleDynkinType}
  N = rank(DT)
  if N >= 17
    return quote
      d = _cartan_symmetrizer_data($DT)
      SVector{$N,Int}(Tuple(d))
    end
  end
  d = _cartan_symmetrizer_data(DT)
  entries = Tuple(d)
  return :(SVector{$N,Int}($entries))
end

@generated function cartan_symmetrizer(::Type{ProductDynkinType{Ts}}) where {Ts}
  types = Ts.parameters
  R = sum(rank(T) for T in types)
  if R >= 17
    return quote
      d = _cartan_symmetrizer_data($(ProductDynkinType{Ts}))
      SVector{$R,Int}(Tuple(d))
    end
  end
  d = Int[]
  for T in types
    d_T = _cartan_symmetrizer_data(T)
    append!(d, d_T)
  end
  entries = Tuple(d)
  return :(SVector{$R,Int}($entries))
end

cartan_symmetrizer(dt::DynkinType) = cartan_symmetrizer(typeof(dt))

# ─── Symmetric bilinear form ────────────────────────────────────────────────

function _cartan_bilinear_form_runtime(::Type{DT}) where {DT<:DynkinType}
  check_dynkin_type(DT)
  R = rank(DT)
  C = _cartan_matrix_data_full(DT)
  d = collect(_cartan_symmetrizer_data(DT))
  result = [d[i] * C[i, j] for i in 1:R, j in 1:R]
  return SMatrix{R,R,Int,R*R}(Tuple(result[i, j] for j in 1:R for i in 1:R))
end

function _cartan_matrix_inverse_runtime(::Type{DT}) where {DT<:DynkinType}
  check_dynkin_type(DT)
  R = rank(DT)
  C = Rational{Int}.(_cartan_matrix_data_full(DT))
  Cinv = inv(C)
  return SMatrix{R,R,Rational{Int},R*R}(Tuple(Cinv[i, j] for j in 1:R for i in 1:R))
end

function _omega_bilinear_form_scaled_runtime(::Type{DT}) where {DT<:DynkinType}
  check_dynkin_type(DT)
  R = rank(DT)
  C = Rational{Int}.(_cartan_matrix_data_full(DT))
  Cinv = inv(C)
  d_data = collect(_cartan_symmetrizer_data(DT))
  B = zeros(Rational{Int}, R, R)
  for j in 1:R, i in 1:R
    B[i, j] = d_data[i] * C[i, j]
  end
  B_omega = transpose(Cinv) * B * Cinv
  S = 1
  for j in 1:R, i in 1:R
    S = lcm(S, denominator(B_omega[i, j]))
  end
  B_omega_S = Int.(B_omega * S)
  return (S, SMatrix{R,R,Int,R*R}(Tuple(B_omega_S[i, j] for j in 1:R for i in 1:R)))
end

"""
    cartan_bilinear_form(::Type{DT}) -> SMatrix

Return the symmetrized Cartan matrix `diag(d) * C`, which is a symmetric
positive-definite matrix defining the inner product on the root space.

# Examples
```jldoctest
julia> using Semisimple

julia> cartan_bilinear_form(TypeB{2})
2×2 StaticArraysCore.SMatrix{2, 2, Int64, 4} with indices SOneTo(2)×SOneTo(2):
  4  -2
 -2   2
```
"""
@generated function cartan_bilinear_form(::Type{DT}) where {DT<:DynkinType}
  R = rank(DT)
  R >= 17 && return :(_cartan_bilinear_form_runtime($DT))
  C = _cartan_matrix_data(DT)
  d = _cartan_symmetrizer_data(DT)
  entries = Tuple(d[i] * C[i, j] for j in 1:R for i in 1:R)
  return :(SMatrix{$R,$R,Int,$(R * R)}($entries))
end

@generated function cartan_bilinear_form(::Type{ProductDynkinType{Ts}}) where {Ts}
  types = Ts.parameters
  R = sum(rank(T) for T in types)
  R >= 17 && return :(_cartan_bilinear_form_runtime($(ProductDynkinType{Ts})))
  C = zeros(Int, R, R)
  d_all = Int[]
  offset = 0
  for T in types
    r = rank(T)
    C_block = _cartan_matrix_data(T)
    C[(offset + 1):(offset + r), (offset + 1):(offset + r)] .= C_block
    append!(d_all, _cartan_symmetrizer_data(T))
    offset += r
  end
  entries = Tuple(d_all[i] * C[i, j] for j in 1:R for i in 1:R)
  return :(SMatrix{$R,$R,Int,$(R * R)}($entries))
end

cartan_bilinear_form(dt::DynkinType) = cartan_bilinear_form(typeof(dt))

# ─── Inverse Cartan matrix (rational) ───────────────────────────────────────

# Full Cartan matrix data helper (works for both simple and product)
function _cartan_matrix_data_full(::Type{DT}) where {DT<:SimpleDynkinType}
  return _cartan_matrix_data(DT)
end

function _cartan_matrix_data_full(::Type{ProductDynkinType{Ts}}) where {Ts}
  types = Ts.parameters
  R = sum(rank(T) for T in types)
  C = zeros(Int, R, R)
  offset = 0
  for T in types
    r = rank(T)
    C_block = _cartan_matrix_data(T)
    C[(offset + 1):(offset + r), (offset + 1):(offset + r)] .= C_block
    offset += r
  end
  return C
end

"""
    cartan_matrix_inverse(::Type{DT}) -> SMatrix{R,R,Rational{Int}}

Return the inverse of the Cartan matrix over the rationals.

# Examples
```jldoctest
julia> using Semisimple

julia> cartan_matrix_inverse(TypeA{2})
2×2 StaticArraysCore.SMatrix{2, 2, Rational{Int64}, 4} with indices SOneTo(2)×SOneTo(2):
 2//3  1//3
 1//3  2//3
```
"""
@generated function cartan_matrix_inverse(::Type{DT}) where {DT<:DynkinType}
  R = rank(DT)
  R >= 17 && return :(_cartan_matrix_inverse_runtime($DT))
  C = _cartan_matrix_data_full(DT)
  # Compute inverse over Rational
  Crat = Rational{Int}.(C)
  Cinv = inv(Crat)
  entries = Tuple(Cinv[i, j] for j in 1:R for i in 1:R)
  return :(SMatrix{$R,$R,Rational{Int},$(R * R)}($entries))
end

cartan_matrix_inverse(dt::DynkinType) = cartan_matrix_inverse(typeof(dt))

# ─── Scaled bilinear form in ω-coordinates (compile-time) ────────────────────

"""
    omega_bilinear_form_scaled(::Type{DT}) -> Tuple{Int, SMatrix{R,R,Int}}

Return ``(S, B_{\\omega,S})`` where
``B_{\\omega,S} = S (C^{-1})^{\\mathsf T} B C^{-1}``
is the bilinear form
in the fundamental weight basis, scaled by the smallest positive integer `S`
that makes all entries integral.  This is a compile-time constant.

# Examples
```jldoctest
julia> using Semisimple

julia> first(omega_bilinear_form_scaled(TypeA{2}))
3
```
"""
@generated function omega_bilinear_form_scaled(::Type{DT}) where {DT<:DynkinType}
  R = rank(DT)
  R >= 17 && return :(_omega_bilinear_form_scaled_runtime($DT))
  C = Rational{Int}.(_cartan_matrix_data_full(DT))
  Cinv = inv(C)
  d_data = _cartan_symmetrizer_data(DT)
  B = zeros(Rational{Int}, R, R)
  for j in 1:R, i in 1:R
    B[i, j] = d_data[i] * C[i, j]
  end
  B_omega = transpose(Cinv) * B * Cinv

  S = 1
  for j in 1:R, i in 1:R
    S = lcm(S, denominator(B_omega[i, j]))
  end
  B_omega_S = Int.(B_omega * S)
  entries = Tuple(B_omega_S[i, j] for j in 1:R for i in 1:R)
  return :(($S, SMatrix{$R,$R,Int,$(R * R)}($entries)))
end

omega_bilinear_form_scaled(dt::DynkinType) = omega_bilinear_form_scaled(typeof(dt))

# ─── Cartan matrix determinant (connection index) ────────────────────────────
# Pre-computed for all simple types; for product types, take the product.

"""
    cartan_determinant(::Type{DT}) -> Int
    cartan_determinant(dt::DT) -> Int

Compute the determinant of the Cartan matrix of the Dynkin type `DT`.

For semisimple Lie algebras, this determinant equals the **connection index**,
which measures the index of the root lattice in the weight lattice.

This is a compile-time constant based on hardcoded values for simple types.

# Examples
```jldoctest
julia> using Semisimple

julia> cartan_determinant(TypeA{3})
4

julia> cartan_determinant(TypeB{3})
2

julia> cartan_determinant(TypeG2)
1
```
"""
cartan_determinant(::Type{TypeA{N}}) where {N} = (check_dynkin_type(TypeA{N}); N + 1)
cartan_determinant(::Type{TypeB{N}}) where {N} = (check_dynkin_type(TypeB{N}); 2)
cartan_determinant(::Type{TypeC{N}}) where {N} = (check_dynkin_type(TypeC{N}); 2)
cartan_determinant(::Type{TypeD{N}}) where {N} = (check_dynkin_type(TypeD{N}); 4)
cartan_determinant(::Type{TypeE{6}}) = 3
cartan_determinant(::Type{TypeE{7}}) = 2
cartan_determinant(::Type{TypeE{8}}) = 1
cartan_determinant(::Type{TypeF4}) = 1
cartan_determinant(::Type{TypeG2}) = 1

function cartan_determinant(::Type{ProductDynkinType{Ts}}) where {Ts}
  check_dynkin_type(ProductDynkinType{Ts})
  return prod(cartan_determinant, fieldtypes(Ts))
end

function cartan_determinant(dt::DynkinType)
  return cartan_determinant(typeof(dt))
end

# ─── Identifying a Cartan matrix ─────────────────────────────────────────────
#  The inverse direction of the above: recover a Dynkin type from a Cartan
#  matrix.  Only the sub-diagram functions below need this, so the classifier
#  itself stays private; its (family, rank) pairs are a third encoding of a
#  Dynkin type and are not worth letting out of this file.  The algorithm
#  follows OSCAR.jl's cartan_type_with_ordering
#  (Oscar.jl/src/LieTheory/CartanMatrix.jl).

# Classify a Cartan matrix as (family, rank) pairs, one per connected component,
# together with a permutation putting the rows and columns in Bourbaki order.
# Components are found by adjacency and then told apart by their graph
# structure: path versus branching, and edge multiplicities.
function _classify_cartan_matrix(C::AbstractMatrix{<:Integer})
  rk = size(C, 1)
  size(C, 1) == size(C, 2) || throw(
    ArgumentError("Cartan matrix must be square, got $(size(C, 1))×$(size(C, 2))")
  )

  type = Tuple{Symbol,Int}[]
  ord = sizehint!(Int[], rk)

  # Build adjacency list
  adj = [[j for j in 1:rk if i != j && C[i, j] != 0] for i in 1:rk]

  done = falses(rk)

  for v0 in 1:rk
    done[v0] && continue

    # ── Rank 1: isolated node ─────────────────────────────────────────
    if isempty(adj[v0])
      push!(type, (:A, 1))
      push!(ord, v0)
      done[v0] = true
      continue
    end

    # ── Rank 2: pair of nodes ─────────────────────────────────────────
    if length(adj[v0]) == 1 && length(adj[only(adj[v0])]) == 1
      v1 = only(adj[v0])
      bond = C[v0, v1] * C[v1, v0]
      if bond == 1
        push!(type, (:A, 2))
        push!(ord, v0, v1)
      elseif C[v0, v1] == -2
        # v0 is the short-root side → C_2 convention
        push!(type, (:C, 2))
        push!(ord, v0, v1)
      elseif C[v1, v0] == -2
        push!(type, (:B, 2))
        push!(ord, v0, v1)
      elseif C[v0, v1] == -3
        push!(type, (:G, 2))
        push!(ord, v0, v1)
      elseif C[v1, v0] == -3
        push!(type, (:G, 2))
        push!(ord, v1, v0)
      else
        error("Could not identify rank-2 Cartan matrix component")
      end
      done[v0] = true
      done[v1] = true
      continue
    end

    # ── Rank > 2: DFS to find the whole component ────────────────────
    comp = [v0]
    todo = [v0]
    done[v0] = true
    while !isempty(todo)
      v = pop!(todo)
      for w in adj[v]
        if !done[w]
          push!(comp, w)
          push!(todo, w)
          done[w] = true
        end
      end
    end
    sort!(comp)
    len_comp = length(comp)

    # Find degree-3 node (branching → D or E)
    deg3 = findfirst(v -> length(adj[v]) == 3, comp)

    if isnothing(deg3)
      # ── Path graph: A, B, C, or F ─────────────────────────────────
      # Find the start of the path (a leaf with simply-laced left neighbor)
      start = 0
      for v1 in filter(v -> length(adj[v]) == 1, comp)
        v2 = only(adj[v1])
        C[v1, v2] * C[v2, v1] == 1 || continue   # skip right end of B/C
        if len_comp == 4
          v3 = only(filter(!=(v1), adj[v2]))
          C[v2, v3] == -1 || continue               # skip right end of F
        end
        start = v1
        break
      end
      @assert start != 0 "Could not find start of path in component $comp"

      # Trace the path
      path = [start, only(adj[start])]
      for _ in 1:(len_comp - 2)
        push!(path, only(filter(!=(path[end - 1]), adj[path[end]])))
      end

      # Determine type from last edge
      if len_comp == 4 && C[path[3], path[2]] == -2
        push!(type, (:F, 4))
      elseif C[path[end - 1], path[end]] == -2
        push!(type, (:C, len_comp))
      elseif C[path[end], path[end - 1]] == -2
        push!(type, (:B, len_comp))
      else
        push!(type, (:A, len_comp))
      end
      append!(ord, path)
    else
      # ── Branching: D or E ──────────────────────────────────────────
      v_deg3 = comp[deg3]

      # Find the three paths from the branch node
      paths = [[v_deg3, v_n] for v_n in adj[v_deg3]]
      for path in paths
        while length(adj[path[end]]) == 2
          push!(path, only(filter(!=(path[end - 1]), adj[path[end]])))
        end
        popfirst!(path)  # remove the branch node itself
      end
      sort!(paths; by=length)

      @assert sum(length, paths) + 1 == len_comp

      if length(paths[2]) == 1
        # ── D type: two short arms of length 1 ──────────────────────
        push!(type, (:D, len_comp))
        if len_comp == 4
          push!(ord, only(paths[1]), v_deg3, only(paths[2]), only(paths[3]))
        else
          append!(ord, reverse!(paths[3]))
          push!(ord, v_deg3, only(paths[1]), only(paths[2]))
        end
      elseif length(paths[2]) == 2
        # ── E type: arms of length 1, 2, and 2/3/4 ─────────────────
        push!(type, (:E, len_comp))
        push!(ord, paths[2][2], only(paths[1]), paths[2][1], v_deg3)
        append!(ord, paths[3])
      else
        error("Could not identify branching Cartan matrix of rank $len_comp")
      end
    end
  end

  return type, ord
end

# ─── Sub-diagrams ────────────────────────────────────────────────────────────

"""
    sub_dynkin_type_with_ordering(DT, vertices) -> Type{<:DynkinType}, Vector{Int}

The Dynkin type of the sub-diagram of `DT` induced on `vertices`, together with `vertices`
reordered so that the `i`-th entry is the vertex of `DT` playing the role of the `i`-th
simple root of that type.

`DT` is a [`DynkinType`](@ref), given either as a type or as an instance, and `vertices` are
numbered à la Bourbaki.  Both return values are needed whenever data indexed by the
sub-diagram has to be transported to or from the ambient diagram; use
[`sub_dynkin_type`](@ref) or [`sub_dynkin_ordering`](@ref) when only one of the two is
wanted.

The sub-diagram induced on the unmarked simple roots of a parabolic subgroup
``\\mathrm{P} \\subseteq \\mathrm{G}`` is the Dynkin diagram of the semisimple part
``[\\mathrm{L}, \\mathrm{L}]`` of its Levi factor, which is the main use of this function.

# Examples
```jldoctest
julia> using Semisimple

julia> sub_dynkin_type_with_ordering(TypeA{5}, [2, 3, 4, 5])
(TypeA{4}, [2, 3, 4, 5])
```

Type ``\\mathrm{A}`` needs no reordering, but ``\\mathrm{D}`` does: removing the first
vertex of ``\\mathrm{D}_4`` leaves an ``\\mathrm{A}_3`` whose middle vertex is the former
branch vertex.

```jldoctest
julia> using Semisimple

julia> sub_dynkin_type_with_ordering(TypeD{4}, [2, 3, 4])
(TypeA{3}, [3, 2, 4])
```
"""
function sub_dynkin_type_with_ordering(::Type{DT}, vertices) where {DT<:DynkinType}
  nodes = collect(Int, vertices)
  isempty(nodes) && throw(ArgumentError("the vertex set needs to be non-empty"))
  allunique(nodes) || throw(ArgumentError("repeated vertex in $vertices"))
  all(node -> 1 <= node <= rank(DT), nodes) ||
    throw(ArgumentError("vertex out of range for $(_type_name(DT))"))
  described, ordering = _classify_cartan_matrix(cartan_matrix(DT)[nodes, nodes])
  factors = DataType[_simple_dynkin_type(fam, rk) for (fam, rk) in described]
  return _combine_dynkin_factors(factors), nodes[ordering]
end

sub_dynkin_type_with_ordering(dt::DynkinType, vertices) =
  sub_dynkin_type_with_ordering(typeof(dt), vertices)

"""
    sub_dynkin_type(DT, vertices) -> Type{<:DynkinType}

The Dynkin type of the sub-diagram of `DT` induced on `vertices`, numbered à la Bourbaki.

Saves the caller from assembling the sub-Cartan matrix by hand.  See
[`sub_dynkin_type_with_ordering`](@ref) for how the vertices of the sub-diagram sit inside
the ambient one.

# Examples
```jldoctest
julia> using Semisimple

julia> sub_dynkin_type(TypeA{5}, [2, 3, 4, 5])
TypeA{4}

julia> sub_dynkin_type(TypeA{5}, [1, 2, 4, 5])
ProductDynkinType{Tuple{TypeA{2}, TypeA{2}}}
```

The classical Levi factors of ``\\mathrm{E}_8``:

```jldoctest
julia> using Semisimple

julia> sub_dynkin_type(TypeE{8}, [1, 2, 3, 4, 5, 6, 7])
TypeE{7}

julia> sub_dynkin_type(TypeE{8}, [2, 3, 4, 5, 6, 7, 8])
TypeD{7}

julia> sub_dynkin_type(TypeE{8}, [1, 3, 4, 5, 6, 7, 8])
TypeA{7}
```
"""
sub_dynkin_type(::Type{DT}, vertices) where {DT<:DynkinType} =
  first(sub_dynkin_type_with_ordering(DT, vertices))

sub_dynkin_type(dt::DynkinType, vertices) = sub_dynkin_type(typeof(dt), vertices)

"""
    sub_dynkin_ordering(DT, vertices) -> Vector{Int}

`vertices` reordered so that the `i`-th entry is the vertex of `DT` playing the role of the
`i`-th simple root of [`sub_dynkin_type`](@ref).

# Examples
```jldoctest
julia> using Semisimple

julia> sub_dynkin_ordering(TypeD{4}, [2, 3, 4])
3-element Vector{Int64}:
 3
 2
 4
```
"""
sub_dynkin_ordering(::Type{DT}, vertices) where {DT<:DynkinType} =
  last(sub_dynkin_type_with_ordering(DT, vertices))

sub_dynkin_ordering(dt::DynkinType, vertices) = sub_dynkin_ordering(typeof(dt), vertices)
