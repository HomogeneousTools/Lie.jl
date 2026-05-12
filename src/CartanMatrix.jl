# ═══════════════════════════════════════════════════════════════════════════════
#  Cartan matrices — compile-time specialized via @generated functions
#
#  Convention: (a_{ij}) = ⟨αᵢ∨, αⱼ⟩  (following Bourbaki / Oscar)
# ═══════════════════════════════════════════════════════════════════════════════

export cartan_matrix, cartan_symmetrizer, cartan_bilinear_form, cartan_matrix_inverse
export omega_bilinear_form_scaled, cartan_determinant

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

# Helper to get raw matrix data at code-generation time
function _cartan_matrix_data(::Type{TypeA{N}}) where {N}
  check_dynkin_type(TypeA{N})
  C = zeros(Int, N, N)
  for i in 1:N
    C[i, i] = 2
  end
  for i in 1:(N - 1)
    C[i, i + 1] = -1
    C[i + 1, i] = -1
  end
  return C
end

function _cartan_matrix_data(::Type{TypeB{N}}) where {N}
  check_dynkin_type(TypeB{N})
  C = _cartan_matrix_data(TypeA{N})
  C[N, N - 1] = -2
  return C
end

function _cartan_matrix_data(::Type{TypeC{N}}) where {N}
  check_dynkin_type(TypeC{N})
  C = _cartan_matrix_data(TypeA{N})
  C[N - 1, N] = -2
  return C
end

function _cartan_matrix_data(::Type{TypeD{N}}) where {N}
  check_dynkin_type(TypeD{N})
  C = zeros(Int, N, N)
  for i in 1:N
    C[i, i] = 2
  end
  for i in 1:(N - 3)
    C[i, i + 1] = -1
    C[i + 1, i] = -1
  end
  C[N - 2, N - 1] = -1
  C[N - 1, N - 2] = -1
  C[N - 2, N] = -1
  C[N, N - 2] = -1
  return C
end

function _cartan_matrix_data(::Type{TypeE{N}}) where {N}
  check_dynkin_type(TypeE{N})
  C8 = _E8_cartan()
  return C8[1:N, 1:N]
end

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
julia> using Semisimple, StaticArrays

julia> cartan_matrix(TypeA{2}) == SMatrix{2,2}(2, -1, -1, 2)
true

julia> cartan_matrix(TypeG2) == SMatrix{2,2}(2, -1, -3, 2)
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

function _cartan_symmetrizer_data(::Type{DT}) where {DT<:SimpleDynkinType}
  C = _cartan_matrix_data(DT)
  N = rank(DT)
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
julia> using Semisimple, StaticArrays

julia> cartan_symmetrizer(TypeB{3}) == SVector(2, 2, 1)
true

julia> cartan_symmetrizer(TypeG2) == SVector(1, 3)
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
