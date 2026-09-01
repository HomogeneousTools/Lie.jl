# ═══════════════════════════════════════════════════════════════════════════════
#  Bourbaki tables — intrinsic versions of the finite root-system plates
#
#  Bourbaki's plates use family-specific Euclidean realizations. Semisimple.jl
#  instead has one uniform realization: roots and coroots are expressed in the
#  simple-root basis. This file assembles the corresponding intrinsic data from
#  the package's existing root-system, Cartan, weight-lattice, and Weyl-group API.
# ═══════════════════════════════════════════════════════════════════════════════

export BourbakiTable, bourbaki_table

"""
    BourbakiTable{DT,R}

An intrinsic version of the Bourbaki plate for the irreducible Dynkin type `DT`.

Roots and coroots are represented in the simple-root basis. The columns of
`fundamental_weight_coefficients` express the fundamental weights in that same
basis. The remaining fields collect the affine Cartan matrix, Coxeter data,
the root-lattice quotient, Weyl-group data, and the opposition involution.

Construct a table with [`bourbaki_table`](@ref).
"""
struct BourbakiTable{
  DT<:SimpleDynkinType,
  R,
  CM<:AbstractMatrix{Int},
  FWM<:AbstractMatrix{Rational{Int}},
}
  dynkin_type::Type{DT}
  dynkin_diagram::DynkinDiagram
  positive_roots::Vector{RootSpaceElem{DT,R}}
  positive_coroots::Vector{RootSpaceElem{DT,R}}
  highest_root::RootSpaceElem{DT,R}
  highest_short_root::RootSpaceElem{DT,R}
  highest_coroot::RootSpaceElem{DT,R}
  fundamental_weights::Vector{WeightLatticeElem{DT,R}}
  fundamental_weight_coefficients::FWM
  weyl_vector::WeightLatticeElem{DT,R}
  sum_positive_roots::RootSpaceElem{DT,R}
  cartan_matrix::CM
  affine_cartan_matrix::Matrix{Int}
  coxeter_coefficients::SVector{R,Int}
  dual_coxeter_coefficients::SVector{R,Int}
  coxeter_number::Int
  dual_coxeter_number::Int
  connection_index::Int
  root_lattice_quotient::Vector{Int}
  exponents::Vector{Int}
  degrees_fundamental_invariants::Vector{Int}
  weyl_order::BigInt
  longest_element::WeylGroupElem{DT,R}
  opposition_involution::SVector{R,Int}
end

# The affine simple root is α₀ = -θ. Its coroot is -θ∨, where θ∨ is the
# coroot paired with the highest root (not the highest coroot in general).
function _affine_cartan_matrix(
  C::AbstractMatrix{Int},
  highest_root_coefficients::AbstractVector{Int},
  highest_root_coroot_coefficients::AbstractVector{Int},
)
  R = size(C, 1)
  affine = zeros(Int, R + 1, R + 1)
  affine[1, 1] = 2
  affine[2:end, 2:end] .= C
  affine[2:end, 1] = collect(-(C * highest_root_coefficients))
  affine[1, 2:end] = collect(-(transpose(highest_root_coroot_coefficients) * C))
  return affine
end

# P(R)/Q(R) has order det(C). For an irreducible finite root system it is
# cyclic unless its order is four and the exponent, read from C⁻¹, is two
# (the even D case). Empty invariant factors denote the trivial group.
function _root_lattice_quotient_invariants(
  Cinv::AbstractMatrix{Rational{Int}}, connection_index::Int
)
  connection_index == 1 && return Int[]
  quotient_exponent = foldl(lcm, denominator.(Cinv); init=1)
  if quotient_exponent == connection_index
    return [connection_index]
  elseif connection_index == 4 && quotient_exponent == 2
    return [2, 2]
  end
  error(
    "unexpected root-lattice quotient of order $connection_index and exponent $quotient_exponent"
  )
end

function _opposition_involution(
  fundamental_weights::Vector{WeightLatticeElem{DT,R}},
  w0::WeylGroupElem{DT,R},
) where {DT,R}
  return SVector{R,Int}(
    ntuple(Val(R)) do i
      image = -(fundamental_weights[i] * w0)
      j = findfirst(==(1), image.vec)
      j === nothing && error("the longest element did not permute the fundamental weights")
      image.vec == fundamental_weights[j].vec ||
        error("the longest element did not permute the fundamental weights")
      return j
    end,
  )
end

"""
    bourbaki_table(::Type{DT}) -> BourbakiTable{DT}
    bourbaki_table(dt::SimpleDynkinType) -> BourbakiTable

Assemble an intrinsic Bourbaki table for an irreducible finite Dynkin type.

Unlike the family-specific ambient-coordinate realizations in Bourbaki's
plates, this table expresses every root and coroot uniformly in the simple-root
basis. All entries are computed from Semisimple.jl's existing root data.
Product types are rejected because the Bourbaki plates are indexed by
irreducible root systems.

# Examples
```jldoctest
julia> using Semisimple

julia> table = bourbaki_table(TypeA{2});

julia> table.exponents
2-element Vector{Int64}:
 1
 2

julia> table.root_lattice_quotient
1-element Vector{Int64}:
 3

julia> table.opposition_involution
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 2
 1
```
"""
function bourbaki_table(::Type{DT}) where {DT<:SimpleDynkinType}
  check_dynkin_type(DT)
  R = rank(DT)
  RS = RootSystem(DT)
  positive_roots_data = positive_roots(RS)
  positive_coroots_data = positive_coroots(RS)
  highest_root_data = highest_root(RS)
  highest_short_root_data = highest_short_root(RS)
  highest_coroot_data = highest_coroot(RS)
  fundamental_weights_data = fundamental_weights(DT)
  C = cartan_matrix(DT)
  Cinv = cartan_matrix_inverse(DT)
  connection_index = cartan_determinant(DT)
  degrees = sort!(collect(degrees_fundamental_invariants(DT)))
  exponents = degrees .- 1
  w0 = longest_element(weyl_group(DT))

  root_sum = zeros(Int, R)
  for root in positive_roots_data
    root_sum .+= root.vec
  end

  affine_C = _affine_cartan_matrix(
    C,
    highest_root_data.vec,
    RS.positive_coroots_list[end],
  )

  return BourbakiTable{
    DT,
    R,
    typeof(C),
    typeof(Cinv),
  }(
    DT,
    dynkin_diagram(DT),
    positive_roots_data,
    positive_coroots_data,
    highest_root_data,
    highest_short_root_data,
    highest_coroot_data,
    fundamental_weights_data,
    Cinv,
    weyl_vector(DT),
    RootSpaceElem(DT, root_sum),
    C,
    affine_C,
    coxeter_coefficients(DT),
    dual_coxeter_coefficients(DT),
    coxeter_number(DT),
    dual_coxeter_number(DT),
    connection_index,
    _root_lattice_quotient_invariants(Cinv, connection_index),
    exponents,
    degrees,
    weyl_order(DT),
    w0,
    _opposition_involution(fundamental_weights_data, w0),
  )
end

bourbaki_table(dt::SimpleDynkinType) = bourbaki_table(typeof(dt))

function bourbaki_table(::Type{DT}) where {DT<:ProductDynkinType}
  throw(ArgumentError("bourbaki_table requires an irreducible Dynkin type, got $DT"))
end

bourbaki_table(dt::ProductDynkinType) = bourbaki_table(typeof(dt))

const _SUBSCRIPT_DIGITS = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')

function _subscript(n::Integer)
  n >= 0 || throw(ArgumentError("subscripts must be non-negative, got $n"))
  return join(_SUBSCRIPT_DIGITS[Int(digit) + 1] for digit in digits(n; base=10)[end:-1:1])
end

function _unicode_type_name(::Type{DT}) where {DT<:SimpleDynkinType}
  name = _type_name(DT)
  return string(first(name), _subscript(parse(Int, name[2:end])))
end

_alpha_label(i::Integer) = "α" * _subscript(i)
_coroot_label(i::Integer) = _alpha_label(i) * "∨"
_omega_label(i::Integer) = "ω" * _subscript(i)

_number_string(n::Integer) = replace(string(n), '-' => '−')

function _number_string(q::Rational)
  denominator(q) == 1 && return _number_string(numerator(q))
  return _number_string(numerator(q)) * "⁄" * _number_string(denominator(q))
end

_coordinate_string(v) = "[" * join(_number_string.(v), ", ") * "]"

function _quotient_string(invariants::Vector{Int})
  isempty(invariants) && return "trivial"
  return join(("ℤ/$(n)ℤ" for n in invariants), " × ")
end

function _show_text_box(io::IO, title::AbstractString, contents::AbstractString)
  lines = split(contents, '\n')
  title_width = textwidth(title)
  inner_width = max(title_width + 2, maximum(textwidth, lines))
  println(io, "╭─ ", title, " ", repeat("─", inner_width - title_width - 1), "╮")
  for line in lines
    println(io, "│ ", line, repeat(" ", inner_width - textwidth(line)), " │")
  end
  println(io, "╰", repeat("─", inner_width + 2), "╯")
  return nothing
end

function _summary_rows(table::BourbakiTable{DT,R}) where {DT,R}
  opposition = join(
    (
      _alpha_label(i) * " ↦ " * _alpha_label(table.opposition_involution[i]) for i in 1:R
    ),
    ", ",
  )
  return Any[
    "rank" R
    "|Φ|" 2 * length(table.positive_roots)
    "|Φ⁺|" length(table.positive_roots)
    "θ" _coordinate_string(table.highest_root.vec)
    "θₛ" _coordinate_string(table.highest_short_root.vec)
    "θ∨" _coordinate_string(table.highest_coroot.vec)
    "2ρ" _coordinate_string(table.sum_positive_roots.vec)
    "marks mᵢ" _coordinate_string(table.coxeter_coefficients)
    "dual marks mᵢ∨" _coordinate_string(table.dual_coxeter_coefficients)
    "h" table.coxeter_number
    "h∨" table.dual_coxeter_number
    "P/Q" _quotient_string(table.root_lattice_quotient)
    "f = |P/Q|" table.connection_index
    "exponents" _coordinate_string(table.exponents)
    "degrees" _coordinate_string(table.degrees_fundamental_invariants)
    "|W|" table.weyl_order
    "ℓ(w₀)" length(table.longest_element)
    "−w₀" opposition
  ]
end

function _show_summary_table(io::IO, table::BourbakiTable)
  pretty_table(
    io,
    _summary_rows(table);
    header=["invariant", "value"],
    title="ROOT-SYSTEM INVARIANTS",
    title_alignment=:l,
    tf=tf_unicode_rounded,
    alignment=[:l, :l],
    crop=:none,
  )
  return nothing
end

function _show_coordinate_table(io::IO, entries; coroot::Bool=false)
  n = length(entries)
  R = length(first(entries).vec)
  data = Matrix{Any}(undef, n, R + 2)
  for (i, entry) in enumerate(entries)
    data[i, 1] = i
    data[i, 2] = height(entry)
    for j in 1:R
      data[i, j + 2] = _number_string(entry.vec[j])
    end
  end

  labels = coroot ? [_coroot_label(i) for i in 1:R] : [_alpha_label(i) for i in 1:R]
  pretty_table(
    io,
    data;
    header=["#", "ht", labels...],
    title=coroot ? "(Φ∨)⁺ · POSITIVE COROOTS" : "Φ⁺ · POSITIVE ROOTS",
    title_alignment=:l,
    tf=tf_unicode_rounded,
    alignment=fill(:c, R + 2),
    crop=:none,
  )
  return nothing
end

function _show_fundamental_weight_table(io::IO, Cinv::AbstractMatrix)
  R = size(Cinv, 1)
  data = Matrix{String}(undef, R, R)
  for i in 1:R, j in 1:R
    data[i, j] = _number_string(Cinv[j, i])
  end

  pretty_table(
    io,
    data;
    header=[_alpha_label(i) for i in 1:R],
    row_labels=[_omega_label(i) for i in 1:R],
    row_label_column_title="",
    title="FUNDAMENTAL WEIGHTS · SIMPLE-ROOT COORDINATES",
    title_alignment=:l,
    tf=tf_unicode_rounded,
    alignment=:c,
    crop=:none,
  )
  return nothing
end

function _show_cartan_table(io::IO, matrix::AbstractMatrix{Int}; affine::Bool=false)
  first_node = affine ? 0 : 1
  last_node = first_node + size(matrix, 1) - 1
  nodes = first_node:last_node
  data = _number_string.(matrix)

  pretty_table(
    io,
    data;
    header=[_alpha_label(i) for i in nodes],
    row_labels=[_coroot_label(i) for i in nodes],
    row_label_column_title="",
    title=if affine
      "AFFINE CARTAN MATRIX · NODES ₀…$(_subscript(last_node))"
    else
      "CARTAN MATRIX"
    end,
    title_alignment=:l,
    tf=tf_unicode_rounded,
    alignment=:c,
    crop=:none,
  )
  return nothing
end

function Base.show(io::IO, table::BourbakiTable{DT}) where {DT}
  print(io, "BourbakiTable(", _unicode_type_name(DT), ")")
end

function Base.show(
  io::IO, ::MIME"text/plain", table::BourbakiTable{DT,R}
) where {DT,R}
  _show_text_box(
    io,
    "BOURBAKI PLATE · " * _unicode_type_name(DT),
    string(table.dynkin_diagram),
  )
  println(io)
  _show_summary_table(io, table)
  println(io)
  _show_coordinate_table(io, table.positive_roots)
  println(io)
  _show_coordinate_table(io, table.positive_coroots; coroot=true)
  println(io)
  _show_fundamental_weight_table(io, table.fundamental_weight_coefficients)
  println(io)
  _show_cartan_table(io, table.cartan_matrix)
  println(io)
  _show_cartan_table(io, table.affine_cartan_matrix; affine=true)
  return nothing
end
