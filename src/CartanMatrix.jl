# ═══════════════════════════════════════════════════════════════════════════════
#  Cartan matrices — compile-time specialized via @generated functions
#
#  Convention: (a_{ij}) = ⟨αᵢ∨, αⱼ⟩  (following Bourbaki / Oscar)
# ═══════════════════════════════════════════════════════════════════════════════

export cartan_matrix, cartan_symmetrizer, cartan_bilinear_form, cartan_matrix_inverse

# ─── Type A ──────────────────────────────────────────────────────────────────
# A_n: tridiagonal, 2 on diagonal, -1 on super/sub-diagonal

@generated function cartan_matrix(::Type{TypeA{N}}) where {N}
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
    return :(SMatrix{$N,$N,Int,$(N*N)}($(Tuple(entries))))
end

# ─── Type B ──────────────────────────────────────────────────────────────────
# B_n: like A but C[n, n-1] = -2

@generated function cartan_matrix(::Type{TypeB{N}}) where {N}
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
    return :(SMatrix{$N,$N,Int,$(N*N)}($(Tuple(entries))))
end

# ─── Type C ──────────────────────────────────────────────────────────────────
# C_n: like A but C[n-1, n] = -2

@generated function cartan_matrix(::Type{TypeC{N}}) where {N}
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
    return :(SMatrix{$N,$N,Int,$(N*N)}($(Tuple(entries))))
end

# ─── Type D ──────────────────────────────────────────────────────────────────
# D_n: like A_{n-2} extended with branching at node n-2
#   Dynkin: 1 - 2 - ... - (n-2) < (n-1)
#                                  \ n
#   C[n-2,n-1] = C[n-1,n-2] = -1
#   C[n-2,n] = C[n,n-2] = -1
#   C[n-1,n] = C[n,n-1] = 0

@generated function cartan_matrix(::Type{TypeD{N}}) where {N}
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
    return :(SMatrix{$N,$N,Int,$(N*N)}($(Tuple(entries))))
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
    for i in 1:8; C[i, i] = 2; end
    # edges: 1-3, 3-4, 4-5, 5-6, 6-7, 7-8, 2-4
    edges = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (2, 4)]
    for (a, b) in edges
        C[a, b] = -1
        C[b, a] = -1
    end
    return C
end

@generated function cartan_matrix(::Type{TypeE{N}}) where {N}
    C8 = _E8_cartan()
    C = C8[1:N, 1:N]
    entries = Tuple(C[i, j] for j in 1:N for i in 1:N)
    return :(SMatrix{$N,$N,Int,$(N*N)}($entries))
end

# ─── Type F4 ─────────────────────────────────────────────────────────────────
# F4: 1 - 2 >=> 3 - 4

function cartan_matrix(::Type{TypeF4})
    #  F₄ Cartan matrix (Bourbaki): 1 - 2 =>= 3 - 4
    #  C = [2 -1 0 0; -1 2 -1 0; 0 -2 2 -1; 0 0 -1 2]
    #  Stored column-major:
    SMatrix{4,4,Int,16}((
        2, -1,  0,  0,   # column 1
       -1,  2, -2,  0,   # column 2
        0, -1,  2, -1,   # column 3
        0,  0, -1,  2    # column 4
    ))
end

# ─── Type G2 ─────────────────────────────────────────────────────────────────
# G2: 1 <<< 2   (C[1,2] = -3, C[2,1] = -1)

function cartan_matrix(::Type{TypeG2})
    #  G₂ Cartan matrix: C = [2 -3; -1 2]
    #  Column-major: col1=[2,-1], col2=[-3,2]
    SMatrix{2,2,Int,4}((
        2, -1,   # column 1
       -3,  2    # column 2
    ))
end

# ─── Product types ───────────────────────────────────────────────────────────

"""
    cartan_matrix(::Type{ProductDynkinType{Ts}})

Block-diagonal Cartan matrix for a product of simple types.

# Examples
```jldoctest
julia> using Lie, StaticArrays

julia> cartan_matrix(TypeA{2}) == SMatrix{2,2}(2, -1, -1, 2)
true

julia> cartan_matrix(TypeG2) == SMatrix{2,2}(2, -1, -3, 2)
true
```
"""
@generated function cartan_matrix(::Type{ProductDynkinType{Ts}}) where {Ts}
    types = Ts.parameters
    R = sum(rank(T) for T in types)
    C = zeros(Int, R, R)
    offset = 0
    for T in types
        r = rank(T)
        # Build the Cartan matrix for this component at compile time
        C_block = _cartan_matrix_data(T)
        C[offset+1:offset+r, offset+1:offset+r] .= C_block
        offset += r
    end
    entries = Tuple(C[i, j] for j in 1:R for i in 1:R)
    return :(SMatrix{$R,$R,Int,$(R*R)}($entries))
end

# Helper to get raw matrix data at code-generation time
function _cartan_matrix_data(::Type{TypeA{N}}) where {N}
    C = zeros(Int, N, N)
    for i in 1:N; C[i, i] = 2; end
    for i in 1:N-1; C[i, i+1] = -1; C[i+1, i] = -1; end
    return C
end

function _cartan_matrix_data(::Type{TypeB{N}}) where {N}
    C = _cartan_matrix_data(TypeA{N})
    C[N, N-1] = -2
    return C
end

function _cartan_matrix_data(::Type{TypeC{N}}) where {N}
    C = _cartan_matrix_data(TypeA{N})
    C[N-1, N] = -2
    return C
end

function _cartan_matrix_data(::Type{TypeD{N}}) where {N}
    C = zeros(Int, N, N)
    for i in 1:N; C[i, i] = 2; end
    for i in 1:N-3; C[i, i+1] = -1; C[i+1, i] = -1; end
    C[N-2, N-1] = -1; C[N-1, N-2] = -1
    C[N-2, N]   = -1; C[N, N-2]   = -1
    return C
end

function _cartan_matrix_data(::Type{TypeE{N}}) where {N}
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
    types = Ts.parameters
    R = sum(rank(T) for T in types)
    C = zeros(Int, R, R)
    offset = 0
    for T in types
        r = rank(T)
        C_block = _cartan_matrix_data(T)
        C[offset+1:offset+r, offset+1:offset+r] .= C_block
        offset += r
    end
    return C
end

# Instance dispatch
cartan_matrix(dt::DynkinType) = cartan_matrix(typeof(dt))

# ─── Cartan symmetrizer ─────────────────────────────────────────────────────
# d_i such that d_i * C_{ij} = d_j * C_{ji}  (symmetrizes the Cartan matrix)

"""
    cartan_symmetrizer(::Type{DT}) -> SVector

Return the symmetrizer `d` such that `diag(d) * C` is symmetric, where `C` is
the Cartan matrix of `DT`. Entries are positive integers with gcd 1.

# Examples
```jldoctest
julia> using Lie, StaticArrays

julia> cartan_symmetrizer(TypeB{3}) == SVector(2, 2, 1)
true

julia> cartan_symmetrizer(TypeG2) == SVector(1, 3)
true
```
"""
@generated function cartan_symmetrizer(::Type{DT}) where {DT<:SimpleDynkinType}
    d = _cartan_symmetrizer_data(DT)
    N = rank(DT)
    entries = Tuple(d)
    return :(SVector{$N,Int}($entries))
end

@generated function cartan_symmetrizer(::Type{ProductDynkinType{Ts}}) where {Ts}
    types = Ts.parameters
    R = sum(rank(T) for T in types)
    d = Int[]
    for T in types
        d_T = _cartan_symmetrizer_data(T)
        append!(d, d_T)
    end
    entries = Tuple(d)
    return :(SVector{$R,Int}($entries))
end

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
                d[j] = d[i] * C[i, j] // C[j, i]
                if d[j] < 0
                    d[j] = -d[j]
                end
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

cartan_symmetrizer(dt::DynkinType) = cartan_symmetrizer(typeof(dt))

# ─── Symmetric bilinear form ────────────────────────────────────────────────

"""
    cartan_bilinear_form(::Type{DT}) -> SMatrix

Return the symmetrized Cartan matrix `diag(d) * C`, which is a symmetric
positive-definite matrix defining the inner product on the root space.
"""
function cartan_bilinear_form(::Type{DT}) where {DT<:DynkinType}
    C = cartan_matrix(DT)
    d = cartan_symmetrizer(DT)
    R = rank(DT)
    # diag(d) * C
    return SMatrix{R,R}(Tuple(d[i] * C[i, j] for j in 1:R for i in 1:R))
end

cartan_bilinear_form(dt::DynkinType) = cartan_bilinear_form(typeof(dt))

# ─── Inverse Cartan matrix (rational) ───────────────────────────────────────

"""
    cartan_matrix_inverse(::Type{DT}) -> SMatrix{R,R,Rational{Int}}

Return the inverse of the Cartan matrix over the rationals.
"""
@generated function cartan_matrix_inverse(::Type{DT}) where {DT<:DynkinType}
    R = rank(DT)
    C = _cartan_matrix_data_full(DT)
    # Compute inverse over Rational
    Crat = Rational{Int}.(C)
    Cinv = inv(Crat)
    entries = Tuple(Cinv[i, j] for j in 1:R for i in 1:R)
    return :(SMatrix{$R,$R,Rational{Int},$(R*R)}($entries))
end

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
        C[offset+1:offset+r, offset+1:offset+r] .= C_block
        offset += r
    end
    return C
end

cartan_matrix_inverse(dt::DynkinType) = cartan_matrix_inverse(typeof(dt))
