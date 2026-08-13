using Test
using Semisimple
using Aqua
import Semisimple: _borel_weil_bott  # no longer publicly exported; tested here via explicit import
using StaticArrays
using LinearAlgebra: det

# ═══════════════════════════════════════════════════════════════════════
#  Dynkin types
# ═══════════════════════════════════════════════════════════════════════
@testset "Dynkin types" begin
  @test rank(TypeA{3}) == 3
  @test rank(TypeB{4}) == 4
  @test rank(TypeC{5}) == 5
  @test rank(TypeD{3}) == 3
  @test rank(TypeD{6}) == 6
  @test rank(TypeE{6}) == 6
  @test rank(TypeE{7}) == 7
  @test rank(TypeE{8}) == 8
  @test rank(TypeF4) == 4
  @test rank(TypeG2) == 2
  @test TypeB{3}() isa TypeB{3}
  @test TypeC{3}() isa TypeC{3}
  @test TypeD{4}() isa TypeD{4}

  # Product types
  PT = ProductDynkinType{Tuple{TypeA{3},TypeD{5}}}
  @test rank(PT) == 8
  @test n_components(PT) == 2
  @test component_type(PT, 1) == TypeA{3}
  @test component_type(PT, 2) == TypeD{5}

  PT2 = ProductDynkinType{Tuple{TypeA{3},TypeD{5},TypeE{6}}}
  @test rank(PT2) == 14

  # Invalid types
  @test_throws ArgumentError TypeA{0}()
  @test_throws ArgumentError TypeB{1}()
  @test_throws ArgumentError TypeC{1}()
  @test_throws ArgumentError TypeD{2}()
  @test_throws ArgumentError TypeE{5}()
  @test_throws ArgumentError rank(TypeA{0})
  @test_throws ArgumentError rank(TypeB{1})
  @test_throws ArgumentError rank(TypeC{1})
  @test_throws ArgumentError rank(TypeD{2})
  @test_throws ArgumentError rank(TypeE{9})
  @test_throws ArgumentError n_positive_roots(TypeD{2})
  @test_throws ArgumentError dimension(TypeA{0})
  @test_throws ArgumentError cartan_matrix(TypeA{0})
  @test_throws ArgumentError RootSystem(TypeB{1})
  @test_throws ArgumentError WeightLatticeElem(TypeA{0}, Int[])
  @test_throws ArgumentError fundamental_weight(TypeC{1}, 1)
  @test_throws ArgumentError rank(ProductDynkinType{Tuple{TypeA{0},TypeG2}})
  @test_throws ArgumentError rank(ProductDynkinType{Tuple{TypeA{1},Int}})

  # Display
  @test sprint(show, TypeA(3)) == "A3"
  @test sprint(show, TypeC(3)) == "C3"
  @test sprint(show, TypeD(4)) == "D4"
  @test sprint(show, TypeE(6)) == "E6"
  @test sprint(show, TypeG2()) == "G2"
  @test sprint(show, ProductDynkinType(TypeA{2}(), TypeG2())) == "A2 × G2"
end

@testset "Coverage edge cases" begin
  @testset "Cache configuration wrappers" begin
    @test configure_caches!(
      budget=10_000,
      dominant_frac=0.1,
      tensor_frac=0.2,
      sym_power_frac=0.3,
      ext_power_frac=0.4,
    ) === nothing
    info = cache_info()
    @test info.dominant_character.maxsize == 1000
    @test info.tensor.maxsize == 2000
    @test info.symmetric_power.maxsize == 3000
    @test info.exterior_power.maxsize == 4000
  end

  @testset "Dynkin type wrappers" begin
    dt::DynkinType = TypeA{2}()
    PT = ProductDynkinType{Tuple{TypeA{2},TypeB{3}}}
    diag = dynkin_diagram(TypeA{2})

    @test TypeB(3) isa TypeB{3}
    @test Semisimple.is_valid_dynkin_type(TypeF4)
    @test Semisimple.is_valid_dynkin_type(TypeG2)
    @test Semisimple._invalid_dynkin_type_message(TypeF4) == "TypeF4 is valid"
    @test Semisimple._invalid_dynkin_type_message(TypeG2) == "TypeG2 is valid"
    @test Base.invokelatest(rank, TypeF4) == 4
    @test Base.invokelatest(rank, TypeG2) == 2
    @test Base.invokelatest(rank, dt) == 2
    @test Base.invokelatest(n_positive_roots, TypeE{6}) == 36
    @test Base.invokelatest(n_positive_roots, TypeE{7}) == 63
    @test Base.invokelatest(n_positive_roots, TypeE{8}) == 120
    @test Base.invokelatest(n_positive_roots, TypeF4) == 24
    @test Base.invokelatest(n_positive_roots, TypeG2) == 6
    @test Base.invokelatest(n_positive_roots, dt) == 3
    @test n_components(TypeA{2}) == 1
    @test Base.invokelatest(n_components, dt) == 1
    @test component_ranks(PT) == (2, 3)
    @test component_offsets(PT) == (0, 2)
    @test sprint(show, MIME"text/plain"(), diag) == "○───○\n1   2"
    @test diag == "○───○\n1   2"
    @test "○───○\n1   2" == diag
    @test hash(diag, UInt(0)) == hash("○───○\n1   2", UInt(0))
  end

  @testset "Cartan runtime fallbacks" begin
    DT = TypeA{17}
    C = cartan_matrix(DT)
    @test Semisimple._cartan_matrix_runtime(DT) == C
    @test Base.invokelatest(cartan_matrix, DT()) == C

    B = cartan_bilinear_form(DT)
    @test Semisimple._cartan_bilinear_form_runtime(DT) == B
    @test Base.invokelatest(cartan_bilinear_form, DT()) == B

    Cinv = cartan_matrix_inverse(DT)
    @test Semisimple._cartan_matrix_inverse_runtime(DT) == Cinv
    @test Base.invokelatest(cartan_matrix_inverse, DT()) == Cinv

    S, Bω = omega_bilinear_form_scaled(DT)
    @test Semisimple._omega_bilinear_form_scaled_runtime(DT) == (S, Bω)
  end

  @testset "Root system wrappers" begin
    dt::DynkinType = TypeA{2}()
    RS = Base.invokelatest(RootSystem, dt)
    α1 = simple_root(RS, 1)
    α2 = simple_root(RS, 2)

    @test Semisimple._make_root_system_runtime(TypeA{17}) == RootSystem(TypeA{17})
    @test sprint(show, RS) == "Root system of type A2, rank 2 with 3 positive roots"
    @test RootSpaceElem(TypeA{2}, (1, 1)) == α1 + α2
    @test α1 - α2 == RootSpaceElem(TypeA{2}, [1, -1])
    @test α1 * 2 == RootSpaceElem(TypeA{2}, [2, 0])
    @test hash(α1, UInt(0)) == hash(simple_root(RS, 1), UInt(0))
    @test iszero(zero(typeof(α1)))
    @test positive_roots(RS) == [positive_root(RS, i) for i in 1:3]
    @test negative_root(RS, 1) == -α1
    @test negative_roots(RS) == [negative_root(RS, i) for i in 1:3]
    @test roots(RS) == vcat(positive_roots(RS), negative_roots(RS))
    @test root(RS, 1) == positive_root(RS, 1)
    @test root(RS, 4) == negative_root(RS, 1)
    @test_throws BoundsError root(RS, 7)
    @test simple_coroots(RS) == simple_roots(RS)
    @test positive_coroots(RS)[1:2] == simple_coroots(RS)
    @test Base.invokelatest(coxeter_coefficients, TypeC{3}) == SVector(2, 2, 1)
    @test dual_coxeter_coefficients(ProductDynkinType{Tuple{TypeA{2},TypeB{2}}}) ==
      SVector(1, 1, 1, 1)
    @test Base.invokelatest(dual_coxeter_coefficients, dt) == SVector(1, 1)
    @test Base.invokelatest(coxeter_number, dt) == 3
    @test Base.invokelatest(dual_coxeter_number, dt) == 3
  end

  @testset "Weight lattice wrappers" begin
    RS = RootSystem(TypeA{2})
    ω1 = fundamental_weight(TypeA{2}, 1)

    @test ω1 * 2 == 2 * ω1
    @test zero(ω1) == WeightLatticeElem(TypeA{2}, [0, 0])
    @test fundamental_weights(TypeA{2}) == [fundamental_weight(TypeA{2}, i) for i in 1:2]
    @test dot(simple_root(RS, 1), ω1) == 1//1
  end

  @testset "Character convenience methods" begin
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    z = zero(ω1)
    V = WeylCharacter(ω1)
    W = WeylCharacter(ω2)
    mixed = WeylCharacter(2 * ω1) + 2 * W
    signed = WeylCharacter(2 * ω1) - 2 * W

    @test WeylCharacter(TypeA{2}, [1, 0]) == V
    @test isone(WeylCharacter(z))
    sum_pairs = sort!(collect(pairs(V + W)); by=p -> p.first.vec, rev=true)
    @test first.(sum_pairs) == [ω1, ω2]
    @test last.(sum_pairs) == [1, 1]
    @test sum_pairs == [ω1 => 1, ω2 => 1]
    @test sprint(show, WeylCharacter(TypeA{2})) == "0"
    @test sprint(show, V) == "A2(1, 0)"
    @test sprint(show, -V) == "-A2(1, 0)"
    @test sprint(show, 2 * V) == "2*A2(1, 0)"
    @test occursin(" + A2(0, 1)", sprint(show, WeylCharacter(2 * ω1) + W))
    @test occursin(" + 2*A2(0, 1)", sprint(show, mixed))
    @test occursin(" - A2(0, 1)", sprint(show, WeylCharacter(2 * ω1) - W))
    @test occursin(" - 2*A2(0, 1)", sprint(show, signed))
    @test V * 2 == 2 * V
    n = -1
    @test_throws ArgumentError V^n
    @test V^0 == WeylCharacter(z)
    @test V^1 == V
    @test V^2 == tensor_product(ω1, ω1)
    @test weight_multiplicity(ω1 + ω2, z) == 2
  end

  @testset "Weyl group convenience methods" begin
    dt::DynkinType = TypeA{2}()
    W = Base.invokelatest(weyl_group, dt)
    x = W([1, 2]; normalize=false)
    e = one(W)
    PT = ProductDynkinType{Tuple{TypeA{2},TypeB{2}}}()
    ω1 = fundamental_weight(TypeA{2}, 1)

    @test sprint(show, W) == "Weyl group of type A2"
    @test sprint(show, e) == "id"
    @test sprint(show, x) == "s1 * s2"
    @test hash(x, UInt(0)) == hash(W([1, 2]; normalize=false), UInt(0))
    @test isone(e)
    @test x^0 == e
    @test x^2 == x * x
    @test x^(-1) == inv(x)
    @test Base.invokelatest(weyl_order, TypeC{3}) == 48
    @test Base.invokelatest(weyl_order, TypeD{4}) == 192
    @test Base.invokelatest(weyl_order, TypeE{6}) == BigInt(51840)
    @test Base.invokelatest(weyl_order, TypeF4) == BigInt(1152)
    @test weyl_order(PT) == 48
    @test Semisimple._weyl_denominator(TypeA{2}) == BigInt(2)
    @test length(Semisimple._weyl_dim_scaled_roots(TypeA{2})) == 3
    @test Semisimple._degree_runtime(TypeA{17}, fundamental_weight(TypeA{17}, 1)) == 18
    @test degree(TypeA{2}, [1, 0]) == 3
    @test degree(TypeA{2}(), [1, 0]) == 3
    @test weyl_dimension(TypeA{2}, ω1) == 3
    @test weyl_dimension(TypeA{2}, [1, 0]) == 3
    @test weyl_dimension(TypeA{2}(), [1, 0]) == 3
  end

  @testset "Weylloop constants" begin
    @test Semisimple._weylloop_subtype(TypeE{6}) == :D
    @test Semisimple._weylloop_subtype(TypeE{7}) == :A
    @test Semisimple._weylloop_subtype(TypeE{8}) == :D
    @test Semisimple._weylloop_subtype(TypeF4) == :B
    @test Semisimple._weylloop_subtype(TypeG2) == :A
    @test Semisimple._weylloop_eps_dim(TypeE{6}) == 6
    @test Semisimple._weylloop_eps_dim(TypeE{7}) == 8
    @test Semisimple._weylloop_eps_dim(TypeE{8}) == 8
    @test Semisimple._weylloop_eps_dim(TypeF4) == 4
    @test Semisimple._weylloop_eps_dim(TypeG2) == 3
    @test Semisimple._weylloop_perm_size(TypeE{6}) == 5
    @test Semisimple._weylloop_perm_size(TypeE{7}) == 8
    @test Semisimple._weylloop_perm_size(TypeE{8}) == 8
    @test Semisimple._weylloop_perm_size(TypeF4) == 4
    @test Semisimple._weylloop_perm_size(TypeG2) == 3
    @test length(Semisimple._build_coset_reps(TypeE{7})) == 72
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Dynkin diagram layouts
# ═══════════════════════════════════════════════════════════════════════
@testset "Dynkin diagrams" begin
  # Simple A-type: linear chain
  @test dynkin_diagram(TypeA{1}) == "○\n1"
  @test dynkin_diagram(TypeA{3}) == "○───○───○\n1   2   3"
  @test dynkin_diagram(TypeA{5}) == "○───○───○───○───○\n1   2   3   4   5"

  # B-type: double bond with arrow pointing right (to short root)
  @test dynkin_diagram(TypeB{2}) == "○═>═○\n1   2"
  @test dynkin_diagram(TypeB{3}) == "○───○═>═○\n1   2   3"

  # C-type: double bond with arrow pointing left (to long root)
  @test dynkin_diagram(TypeC{2}) == "○═<═○\n1   2"
  @test dynkin_diagram(TypeC{4}) == "○───○───○═<═○\n1   2   3   4"

  # D-type: forked tail (Bourbaki orientation — top node is N, fork from N-2)
  @test dynkin_diagram(TypeD{3}) == "    ○ 3\n   /\n○───○\n1   2"
  @test dynkin_diagram(TypeD{4}) == "        ○ 4\n       /\n○───○───○\n1   2   3"
  @test dynkin_diagram(TypeD{5}) ==
    "            ○ 5\n           /\n○───○───○───○\n1   2   3   4"

  # E-type: node 2 branches off node 4 (Bourbaki)
  @test dynkin_diagram(TypeE{6}) ==
    "        ○ 2\n        |\n○───○───○───○───○\n1   3   4   5   6"
  @test dynkin_diagram(TypeE{7}) ==
    "        ○ 2\n        |\n○───○───○───○───○───○\n1   3   4   5   6   7"
  @test dynkin_diagram(TypeE{8}) ==
    "        ○ 2\n        |\n○───○───○───○───○───○───○\n1   3   4   5   6   7   8"

  # F4 and G2
  @test dynkin_diagram(TypeF4) == "○───○═>═○───○\n1   2   3   4"
  @test dynkin_diagram(TypeG2) == "○≡<≡○\n1   2"

  # Product type: labelled sections separated by blank lines
  PT = ProductDynkinType{Tuple{TypeA{1},TypeB{2}}}()
  @test dynkin_diagram(PT) == "A1:\n○\n1\n\nB2:\n○═>═○\n1   2"

  # Instance dispatch matches type dispatch
  @test dynkin_diagram(TypeA{3}()) == dynkin_diagram(TypeA{3})
  @test dynkin_diagram(TypeE{6}()) == dynkin_diagram(TypeE{6})

  # Return type is DynkinDiagram, not raw String
  @test dynkin_diagram(TypeA{3}) isa DynkinDiagram
  @test dynkin_diagram(TypeG2) isa DynkinDiagram
  @test dynkin_diagram(PT) isa DynkinDiagram

  # string() recovers the raw multi-line string
  @test string(dynkin_diagram(TypeA{3})) == "○───○───○\n1   2   3"
  @test string(dynkin_diagram(TypeF4)) == "○───○═>═○───○\n1   2   3   4"
end
@testset "Cartan matrices" begin
  # A2
  C_A2 = cartan_matrix(TypeA{2})
  @test C_A2 == [2 -1; -1 2]

  # B2: C[2,1] = -2
  C_B2 = cartan_matrix(TypeB{2})
  @test C_B2 == [2 -1; -2 2]

  # C2: C[1,2] = -2
  C_C2 = cartan_matrix(TypeC{2})
  @test C_C2 == [2 -2; -1 2]

  # B3
  C_B3 = cartan_matrix(TypeB{3})
  @test C_B3 == [2 -1 0; -1 2 -1; 0 -2 2]

  # G2
  C_G2 = cartan_matrix(TypeG2)
  @test C_G2 == [2 -3; -1 2]

  # D3 (Bourbaki: node 1 branches to 2 and 3)
  C_D3 = cartan_matrix(TypeD{3})
  @test C_D3 == [2 -1 -1; -1 2 0; -1 0 2]

  # F4  (Bourbaki: 1 - 2 >=> 3 - 4, so C[3,2] = -2)
  C_F4 = cartan_matrix(TypeF4)
  @test C_F4 == [2 -1 0 0; -1 2 -1 0; 0 -2 2 -1; 0 0 -1 2]

  # A1 (simplest case)
  C_A1 = cartan_matrix(TypeA{1})
  @test C_A1[1, 1] == 2

  # Symmetry check: C is NOT symmetric in general, but diag(d)*C IS
  for DT in [TypeA{3}, TypeB{3}, TypeC{3}, TypeD{4}, TypeE{6}, TypeF4, TypeG2]
    B = cartan_bilinear_form(DT)
    @test B == B'
  end

  # Product type: block-diagonal
  C_prod = cartan_matrix(ProductDynkinType{Tuple{TypeA{2},TypeG2}})
  @test size(C_prod) == (4, 4)
  @test C_prod[1:2, 1:2] == cartan_matrix(TypeA{2})
  @test C_prod[3:4, 3:4] == cartan_matrix(TypeG2)
  @test C_prod[1:2, 3:4] == zeros(Int, 2, 2)

  # Cartan determinant (connection index)
  # The determinant depends on the root system structure
  @test cartan_determinant(TypeA{1}) == 2
  @test cartan_determinant(TypeA{2}) == 3
  @test cartan_determinant(TypeA{3}) == 4
  @test cartan_determinant(TypeD{4}) == 4
  @test cartan_determinant(TypeE{6}) == 3

  # Multiply-laced types
  @test cartan_determinant(TypeB{2}) == 2
  @test cartan_determinant(TypeB{3}) == 2
  @test cartan_determinant(TypeC{2}) == 2
  @test cartan_determinant(TypeC{3}) == 2
  @test cartan_determinant(TypeF4) == 1
  @test cartan_determinant(TypeG2) == 1

  # Instance dispatch
  @test cartan_determinant(TypeA{2}()) == cartan_determinant(TypeA{2})

  # Runtime symmetrizer fallback for high-rank simple and product types
  @test cartan_symmetrizer(TypeA{17}) == SVector{17,Int}(ntuple(_ -> 1, Val(17)))
  PT_runtime = ProductDynkinType{Tuple{TypeA{9},TypeB{8}}}
  @test collect(cartan_symmetrizer(PT_runtime)) ==
    vcat(collect(cartan_symmetrizer(TypeA{9})), collect(cartan_symmetrizer(TypeB{8})))

  # Cross-check: cartan_determinant == det(cartan_matrix) for all simple types
  for DT in [TypeA{1}, TypeA{2}, TypeA{3}, TypeA{4}, TypeB{2}, TypeB{3},
    TypeC{2}, TypeC{3}, TypeD{4}, TypeD{5},
    TypeE{6}, TypeE{7}, TypeE{8}, TypeF4, TypeG2]
    @test cartan_determinant(DT) == round(Int, det(Matrix{Float64}(cartan_matrix(DT))))
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Root systems — number of positive roots
# ═══════════════════════════════════════════════════════════════════════
@testset "Root systems" begin
  # Verify the number of positive roots matches the formula
  for (DT, expected) in [
    (TypeA{1}, 1), (TypeA{2}, 3), (TypeA{3}, 6), (TypeA{6}, 21),
    (TypeB{2}, 4), (TypeB{3}, 9), (TypeB{6}, 36),
    (TypeC{2}, 4), (TypeC{3}, 9), (TypeC{6}, 36),
    (TypeD{4}, 12), (TypeD{6}, 30),
    (TypeE{6}, 36), (TypeE{7}, 63), (TypeE{8}, 120),
    (TypeF4, 24),
    (TypeG2, 6),
  ]
    RS = RootSystem(DT)
    @test n_positive_roots(RS) == expected
    @test n_roots(RS) == 2 * expected
    @test n_simple_roots(RS) == rank(DT)
  end

  RS_A10 = RootSystem(TypeA{10})
  RS_A17 = RootSystem(TypeA{17})
  @test n_positive_roots(RS_A10) == 55
  @test n_positive_roots(RS_A17) == 153
  @test coefficients(highest_root(RS_A10)) == SVector{10,Int}(ntuple(_ -> 1, Val(10)))
  @test coefficients(highest_root(RS_A17)) == SVector{17,Int}(ntuple(_ -> 1, Val(17)))
  @test RootSystem(TypeA{17}) === RS_A17

  # Simple roots are standard basis vectors
  RS_A3 = RootSystem(TypeA{3})
  @test coefficients(simple_root(RS_A3, 1)) == [1, 0, 0]
  @test coefficients(simple_root(RS_A3, 2)) == [0, 1, 0]
  @test coefficients(simple_root(RS_A3, 3)) == [0, 0, 1]

  # Highest root
  RS_A2 = RootSystem(TypeA{2})
  hr = highest_root(RS_A2)
  @test coefficients(hr) == [1, 1]  # α1 + α2

  RS_B2 = RootSystem(TypeB{2})
  hr_B2 = highest_root(RS_B2)
  @test height(hr_B2) == sum(coefficients(hr_B2))

  # Highest short root
  # For B2: short roots have length 1 (±eᵢ); highest is e1 = α1+α2
  hsr_B2 = highest_short_root(RS_B2)
  @test coefficients(hsr_B2) == [1, 1]  # B2 highest short root = α1+α2

  # For A2 (simply-laced): highest short root = highest root
  hsr_A2 = highest_short_root(RS_A2)
  @test coefficients(hsr_A2) == coefficients(hr)

  # For G2: short roots have length² = 2; highest is 2α1+α2
  RS_G2 = RootSystem(TypeG2)
  hsr_G2 = highest_short_root(RS_G2)
  @test coefficients(hsr_G2) == [2, 1]

  # Highest coroot and Coxeter coefficients
  hcr_A2 = highest_coroot(RS_A2)
  @test coefficients(hcr_A2) == [1, 1]  # Same coroot structure for A2

  c_A2 = coxeter_coefficients(TypeA{2})
  @test c_A2 == [1, 1]
  @test length(c_A2) == rank(TypeA{2})

  c_A3 = coxeter_coefficients(TypeA{3})
  @test c_A3 == [1, 1, 1]  # Simply-laced: all 1s

  c_B2 = coxeter_coefficients(TypeB{2})
  @test c_B2 == [1, 2]  # Type B: [1, 2, ..., 2]

  c_G2 = coxeter_coefficients(TypeG2)
  @test c_G2 == [3, 2]  # G2 Coxeter coefficients

  # Product type: must return SVector (not Vector), matching simple-type contract
  c_prod = coxeter_coefficients(ProductDynkinType{Tuple{TypeA{2},TypeB{3}}}())
  @test c_prod isa SVector
  @test c_prod == vcat(coxeter_coefficients(TypeA{2}), coxeter_coefficients(TypeB{3}))

  # Dual Coxeter coefficients: highest short root of the dual root system
  dc_B2 = dual_coxeter_coefficients(TypeB{2})
  @test dc_B2 == [1, 1]  # B2∨=C2; highest short root of C2 has coefficients [1,1]

  dc_C2 = dual_coxeter_coefficients(TypeC{2})
  @test dc_C2 == [1, 1]  # C2∨=B2; highest short root of B2 (=e1) has coefficients [1,1]

  dc_G2 = dual_coxeter_coefficients(TypeG2)
  @test dc_G2 == [1, 2]  # G2 self-dual; highest short root of G2∨ has coefficients [1,2]

  dc_F4 = dual_coxeter_coefficients(TypeF4)
  @test dc_F4 == [1, 2, 3, 2]  # F4 self-dual; highest short root coefficients [1,2,3,2]

  # Coxeter number: h = 1 + ∑ cᵢ
  h_A1 = coxeter_number(TypeA{1})
  @test h_A1 == 2  # 1 + 1

  h_A2 = coxeter_number(TypeA{2})
  @test h_A2 == 3  # 1 + 1 + 1

  h_A3 = coxeter_number(TypeA{3})
  @test h_A3 == 4  # 1 + 1 + 1 + 1

  h_B2 = coxeter_number(TypeB{2})
  @test h_B2 == 4  # 1 + 1 + 2

  h_G2 = coxeter_number(TypeG2)
  @test h_G2 == 6  # 1 + 3 + 2

  # Dual Coxeter number: h* = 1 + ∑ c*ᵢ
  h_star_A1 = dual_coxeter_number(TypeA{1})
  @test h_star_A1 == 2  # 1 + 1

  h_star_A2 = dual_coxeter_number(TypeA{2})
  @test h_star_A2 == 3  # 1 + 1 + 1

  h_star_A3 = dual_coxeter_number(TypeA{3})
  @test h_star_A3 == 4  # 1 + 1 + 1 + 1

  h_star_B2 = dual_coxeter_number(TypeB{2})
  @test h_star_B2 == 3  # 1 + 1 + 1 = 3 = 2·2-1

  h_star_C2 = dual_coxeter_number(TypeC{2})
  @test h_star_C2 == 3  # 1 + 1 + 1 = 3 = 2+1

  h_star_B3 = dual_coxeter_number(TypeB{3})
  @test h_star_B3 == 5  # 1 + 1+2+1 = 5 = 2·3-1

  h_star_C3 = dual_coxeter_number(TypeC{3})
  @test h_star_C3 == 4  # 1 + 1+1+1 = 4 = 3+1

  h_star_G2 = dual_coxeter_number(TypeG2)
  @test h_star_G2 == 4  # 1 + 1 + 2 = 4

  h_star_F4 = dual_coxeter_number(TypeF4)
  @test h_star_F4 == 9  # 1 + 1+2+3+2 = 9

  # Degrees of fundamental invariants
  deg_A2 = degrees_fundamental_invariants(TypeA{2})
  @test deg_A2 == [2, 3]
  @test degrees_fundamental_invariants(TypeA{4}) == [2, 3, 4, 5]  # 2..n+1

  deg_G2 = degrees_fundamental_invariants(TypeG2)
  @test deg_G2 == [2, 6]
  @test degrees_fundamental_invariants(TypeF4) == [2, 6, 8, 12]

  # B_n and C_n: 2, 4, 6, ..., 2n
  @test degrees_fundamental_invariants(TypeB{2}) == [2, 4]
  @test degrees_fundamental_invariants(TypeB{3}) == [2, 4, 6]
  @test degrees_fundamental_invariants(TypeB{4}) == [2, 4, 6, 8]
  @test degrees_fundamental_invariants(TypeC{2}) == [2, 4]
  @test degrees_fundamental_invariants(TypeC{3}) == [2, 4, 6]

  # D_n: 2, 4, ..., 2(n-1), n
  @test degrees_fundamental_invariants(TypeD{4}) == [2, 4, 6, 4] # n=4 even: 4 twice
  @test degrees_fundamental_invariants(TypeD{5}) == [2, 4, 6, 8, 5] # n=5 odd: once
  @test degrees_fundamental_invariants(TypeD{6}) == [2, 4, 6, 8, 10, 6] # n=6 even

  # E-series
  @test degrees_fundamental_invariants(TypeE{6}) == [2, 5, 6, 8, 9, 12]
  @test degrees_fundamental_invariants(TypeE{7}) == [2, 6, 8, 10, 12, 14, 18]
  @test degrees_fundamental_invariants(TypeE{8}) == [2, 8, 12, 14, 18, 20, 24, 30]

  # Root operations
  α1 = simple_root(RS_A2, 1)
  α2 = simple_root(RS_A2, 2)
  @test coefficients(α1 + α2) == [1, 1]
  @test coefficients(-α1) == [-1, 0]
  @test coefficients(2 * α1) == [2, 0]
  @test α1 == α1
  @test α1 != α2

  # is_positive_root
  @test is_positive_root(RS_A2, α1)
  @test is_positive_root(RS_A2, α1 + α2)
  @test !is_positive_root(RS_A2, -α1)

  # clear_caches! must not break is_positive_root on re-use
  RS_E6 = RootSystem(TypeE{6})
  α_E6 = simple_roots(RS_E6)[1]
  @test is_positive_root(RS_E6, α_E6)
  clear_caches!()
  # After clearing, is_positive_root must still work (cache repopulates on demand)
  @test is_positive_root(RS_E6, α_E6)
  @test !is_positive_root(RS_E6, -α_E6)
  # And a fresh root system should also work
  RS_B3 = RootSystem(TypeB{3})
  @test is_positive_root(RS_B3, simple_roots(RS_B3)[2])

  # Compact show for RootSpaceElem
  io = IOBuffer()
  show(IOContext(io, :compact => true), α1)
  @test String(take!(io)) == "A2[1,0]"
  show(IOContext(io, :compact => true), α1 + α2)
  @test String(take!(io)) == "A2[1,1]"
  show(IOContext(io, :compact => true), -α1)
  @test String(take!(io)) == "A2[-1,0]"
  show(io, α1)
  @test String(take!(io)) == "α1"
  show(io, α1 + α2)
  @test String(take!(io)) == "α1 + α2"
  show(io, 2 * α1)
  @test String(take!(io)) == "2α1"
  show(io, RootSpaceElem(TypeA{2}, [1, -1]))
  @test String(take!(io)) == "α1 - α2"
  show(io, zero(typeof(α1)))
  @test String(take!(io)) == "0"
end

# ═══════════════════════════════════════════════════════════════════════
#  Weight lattice
# ═══════════════════════════════════════════════════════════════════════
@testset "Weight lattice" begin
  DT = TypeA{2}
  ω1 = fundamental_weight(DT, 1)
  ω2 = fundamental_weight(DT, 2)

  @test coefficients(ω1) == [1, 0]
  @test coefficients(ω2) == [0, 1]

  ρ = weyl_vector(DT)
  @test coefficients(ρ) == [1, 1]
  @test ρ == ω1 + ω2

  # Dominance
  @test is_dominant(ω1)
  @test is_dominant(ω2)
  @test is_dominant(ρ)
  @test !is_dominant(-ω1)
  @test is_dominant(WeightLatticeElem(DT, [0, 0]))

  # Indexing
  @test ω1[1] == 1
  @test ω1[2] == 0

  # Weight-root conversion
  RS = RootSystem(DT)
  α1 = simple_root(RS, 1)
  w_α1 = WeightLatticeElem(α1)
  @test w_α1 == WeightLatticeElem(DT, [2, -1])  # α1 = 2ω1 - ω2
  @test RootSpaceElem(w_α1) == α1
  @test_throws ArgumentError RootSpaceElem(ω1)

  # Reflection
  w = WeightLatticeElem(DT, [2, 1])
  w_reflected = reflect(w, 1)
  @test w_reflected == WeightLatticeElem(DT, [-2, 3])
  # s1(2ω1 + ω2) = (2ω1 + ω2) - 2*(α1) = (2ω1 + ω2) - 2*(2ω1 - ω2) = -2ω1 + 3ω2
  @test reflect(ω1, highest_root(RS)) == -ω2
  @test_throws ArgumentError reflect(w, RootSpaceElem(DT, [2, 0]))

  # Conjugation to dominant chamber
  w_neg = WeightLatticeElem(DT, [-1, 2])
  w_dom = conjugate_dominant_weight(w_neg)
  @test is_dominant(w_dom)

  # Conjugate_dominant_weight_with_length agrees with _with_elem
  for coords in [[-1, 2], [-3, 5], [2, -4], [-2, -3], [0, 0]]
    wt = WeightLatticeElem(DT, coords)
    dom_e, word = conjugate_dominant_weight_with_elem(wt)
    dom_l, len = conjugate_dominant_weight_with_length(wt)
    @test dom_e == dom_l
    @test length(word) == len
  end

  # Compact show: explicit IOContext(:compact => true) → "DT[c1,c2,…]"
  io = IOBuffer()
  show(IOContext(io, :compact => true), ω1)
  @test String(take!(io)) == "A2[1,0]"

  show(IOContext(io, :compact => true), ω2)
  @test String(take!(io)) == "A2[0,1]"

  show(IOContext(io, :compact => true), ω1 + ω2)
  @test String(take!(io)) == "A2[1,1]"

  show(IOContext(io, :compact => true), WeightLatticeElem(DT, [0, 0]))
  @test String(take!(io)) == "A2[0,0]"

  show(IOContext(io, :compact => true), WeightLatticeElem(DT, [-3, 2]))
  @test String(take!(io)) == "A2[-3,2]"

  # Default show unchanged (no :compact key)
  show(io, ω1)
  @test String(take!(io)) == "ω1"
  show(io, ω1 + 2ω2)
  @test String(take!(io)) == "ω1 + 2ω2"
  show(io, WeightLatticeElem(DT, [0, -1]))
  @test String(take!(io)) == "-ω2"
  show(io, WeightLatticeElem(DT, [1, -1]))
  @test String(take!(io)) == "ω1 - ω2"
  show(io, WeightLatticeElem(DT, [0, 0]))
  @test String(take!(io)) == "0"

  # compact_display! global toggle
  compact_display!(true)
  try
    show(io, ω1)
    @test String(take!(io)) == "A2[1,0]"
    show(io, WeightLatticeElem(DT, [0, 0]))
    @test String(take!(io)) == "A2[0,0]"
    # IOContext(:compact => false) overrides the global toggle
    show(IOContext(io, :compact => false), ω1)
    @test String(take!(io)) == "ω1"
  finally
    compact_display!(false)
  end

  # Confirm restored
  show(io, ω1)
  @test String(take!(io)) == "ω1"

  # Vector constructor: zero-padding and warn-truncation
  @testset "WeightLatticeElem vector constructor" begin
    # Exact length
    @test WeightLatticeElem(TypeA{3}, [1, 2, 3]) == WeightLatticeElem(TypeA{3}, (1, 2, 3))
    # Shorter → pad with zeros
    @test WeightLatticeElem(TypeA{3}, [1, 2]) == WeightLatticeElem(TypeA{3}, (1, 2, 0))
    @test WeightLatticeElem(TypeA{3}, [5]) == WeightLatticeElem(TypeA{3}, (5, 0, 0))
    @test WeightLatticeElem(TypeA{3}, Int[]) == WeightLatticeElem(TypeA{3}, (0, 0, 0))
    # Longer → truncate (with warning)
    @test_warn r"truncating" WeightLatticeElem(TypeA{2}, [1, 2, 3]) ==
      WeightLatticeElem(TypeA{2}, (1, 2))
    @test_warn r"truncating" WeightLatticeElem(TypeA{2}, [7, 8, 9, 10]) ==
      WeightLatticeElem(TypeA{2}, (7, 8))
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Weyl group
# ═══════════════════════════════════════════════════════════════════════
@testset "Weyl group" begin
  @testset "A2" begin
    W = weyl_group(TypeA{2})
    RS = root_system(W)
    s = gens(W)
    @test word(W([2, 1]; normalize=false)) == UInt8[2, 1]

    # Simple reflections are involutions
    @test s[1] * s[1] == one(W)
    @test s[2] * s[2] == one(W)

    # Order of W(A2) = 6
    @test weyl_order(TypeA{2}) == 6

    # Longest element
    w0 = longest_element(W)
    @test length(w0) == n_positive_roots(RS)  # length = # positive roots = 3

    # w0² = 1
    @test w0 * w0 == one(W)

    # Action on weights: w0(ρ) = -ρ
    ρ = weyl_vector(TypeA{2})
    @test ρ * w0 == -ρ

    # Action on roots
    α1 = simple_root(RS, 1)
    α2 = simple_root(RS, 2)
    @test α1 * s[1] == -α1
    @test α1 * s[2] == α1 + α2  # s2(α1) = α1 + α2 in type A2
  end

  @testset "B2" begin
    W = weyl_group(TypeB{2})
    RS = root_system(W)
    w0 = longest_element(W)

    @test weyl_order(TypeB{2}) == 8
    @test length(w0) == n_positive_roots(RS)
    @test w0 * w0 == one(W)

    ρ = weyl_vector(TypeB{2})
    @test ρ * w0 == -ρ
  end

  @testset "G2" begin
    W = weyl_group(TypeG2)
    RS = root_system(W)
    w0 = longest_element(W)

    @test weyl_order(TypeG2) == 12
    @test length(w0) == n_positive_roots(RS)
    @test w0 * w0 == one(W)

    ρ = weyl_vector(TypeG2)
    @test ρ * w0 == -ρ
  end

  @testset "Reflections send root to -root" begin
    for DT in [TypeA{3}, TypeB{3}, TypeD{4}]
      W = weyl_group(DT)
      RS = root_system(W)
      s = gens(W)
      for i in 1:rank(DT)
        αi = simple_root(RS, i)
        @test αi * s[i] == -αi
      end
    end
  end

  @testset "Weyl orbit" begin
    ω1 = fundamental_weight(TypeA{2}, 1)
    orb = weyl_orbit(ω1)
    @test length(orb) == 3  # Orbit of ω1 in A2 has 3 elements

    ρ = weyl_vector(TypeA{2})
    orb_ρ = weyl_orbit(ρ)
    @test length(orb_ρ) == 6  # Regular weight, full orbit
  end

  @testset "Descent sets" begin
    W = weyl_group(TypeA{2})
    e = one(W)
    s1 = gen(W, 1)
    s2 = gen(W, 2)
    w0 = longest_element(W)

    @test right_descent_set(e) == Int[]
    @test left_descent_set(e) == Int[]
    @test right_descent_set(s1) == [1]
    @test left_descent_set(s2) == [2]
    @test sort(right_descent_set(w0)) == [1, 2]
    @test sort(left_descent_set(w0)) == [1, 2]
  end

  @testset "Bruhat order" begin
    W = weyl_group(TypeA{2})
    e = one(W)
    s1 = gen(W, 1)
    s2 = gen(W, 2)
    w0 = longest_element(W)

    @test bruhat_leq(e, s1)
    @test bruhat_leq(e, s2)
    @test bruhat_leq(e, w0)
    @test bruhat_leq(s1, w0)
    @test bruhat_leq(s2, w0)
    @test !bruhat_leq(s1, s2)
    @test !bruhat_leq(s2, s1)
    @test !bruhat_leq(w0, s1)

    # Reflexive and transitive sanity checks
    elems = [e, s1, s2, s1 * s2, s2 * s1, w0]
    for x in elems
      @test bruhat_leq(x, x)
    end
    for x in elems, y in elems, z in elems
      if bruhat_leq(x, y) && bruhat_leq(y, z)
        @test bruhat_leq(x, z)
      end
    end
  end

  @testset "Bruhat descendants" begin
    W = weyl_group(TypeA{2})
    w0 = longest_element(W)
    desc = bruhat_descendants(w0)
    @test length(desc) == 2
    @test sort(length.(desc)) == [2, 2]
    for v in desc
      @test bruhat_leq(v, w0)
      @test length(v) == length(w0) - 1
    end
  end

  @testset "Right-multiplication insertion path" begin
    W = weyl_group(TypeA{2})
    x = W([2, 1])
    b, pos, letter = Semisimple._explain_rmul(
      x, UInt8(2), root_system(W).refl, rank(TypeA{2})
    )
    @test (b, pos, Int(letter)) == (true, 1, 1)

    y = deepcopy(x)
    Semisimple.rmul!(y, UInt8(2))
    @test word(y) == UInt8[1, 2, 1]
  end

  @testset "Parabolic coset representatives" begin
    W = weyl_group(TypeA{2})

    # |W| = 6, |W_{\{1\}}| = 2, so |W/W_{\{1\}}| = 3
    reps_right = right_coset_reps(W, [1])
    @test length(reps_right) == 3
    for w in reps_right
      @test !(1 in right_descent_set(w))
    end

    reps_left = left_coset_reps(W, [2])
    @test length(reps_left) == 3
    for w in reps_left
      @test !(2 in left_descent_set(w))
    end
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Weyl dimension formula
# ═══════════════════════════════════════════════════════════════════════
@testset "Dimension formula" begin
  # A1: dim V(nomega1) = n+1
  for n in 1:10
    hw = WeightLatticeElem(TypeA{1}, [n])
    @test degree(hw) == n + 1
  end

  # A2: dim V(ω1) = 3 (standard rep)
  @test degree(fundamental_weight(TypeA{2}, 1)) == 3

  # A2: dim V(ω2) = 3 (dual standard rep)
  @test degree(fundamental_weight(TypeA{2}, 2)) == 3

  # A2: dim V(ω1 + ω2) = 8 (adjoint rep)
  @test degree(WeightLatticeElem(TypeA{2}, [1, 1])) == 8

  # A3: dim V(ω1) = 4
  @test degree(fundamental_weight(TypeA{3}, 1)) == 4

  # A3: dim V(ω2) = 6 (exterior square)
  @test degree(fundamental_weight(TypeA{3}, 2)) == 6

  # B2: dim V(ω1) = 5 (standard rep of SO(5))
  @test degree(fundamental_weight(TypeB{2}, 1)) == 5

  # B2: dim V(ω2) = 4 (spin rep)
  @test degree(fundamental_weight(TypeB{2}, 2)) == 4

  # B3: dim V(ω1) = 7 (standard rep of SO(7))
  @test degree(fundamental_weight(TypeB{3}, 1)) == 7

  # C2: dim V(ω1) = 4 (standard rep of Sp(4))
  @test degree(fundamental_weight(TypeC{2}, 1)) == 4

  # G2: dim V(ω1) = 7 (standard rep)
  @test degree(fundamental_weight(TypeG2, 1)) == 7

  # G2: dim V(ω2) = 14 (adjoint rep)
  @test degree(fundamental_weight(TypeG2, 2)) == 14

  # D4: dim V(ω1) = 8 (standard rep of SO(8))
  @test degree(fundamental_weight(TypeD{4}, 1)) == 8

  # Higher-rank cached Weyl-dimension data still uses the same public formula.
  @test degree(fundamental_weight(TypeA{10}, 1)) == 11
  @test degree(fundamental_weight(TypeA{17}, 1)) == 18

  # E8: all fundamental representations
  @testset "E8 fundamental representations" begin
    expected_dims = [3875, 147250, 6696000, 6899079264,
      146325270, 2450240, 30380, 248]
    for (i, expected) in enumerate(expected_dims)
      @test degree(fundamental_weight(TypeE{8}, i)) == expected
    end
  end

  # E8: high-dimensional representation 3ω3 + 5ω8
  @test degree(WeightLatticeElem(TypeE{8}, [0, 0, 3, 0, 0, 0, 0, 5])) ==
    big"18190674254761844256000000"

  # Synonyms
  @test weyl_dimension(fundamental_weight(TypeA{2}, 1)) == 3
end

# ═══════════════════════════════════════════════════════════════════════
#  Dominant weights
# ═══════════════════════════════════════════════════════════════════════
@testset "Dominant weights" begin
  # A2, hw = ω1 + ω2: adjoint rep has 2 dominant weights
  hw = WeightLatticeElem(TypeA{2}, [1, 1])
  dw = dominant_weights(hw)
  @test length(dw) == 2
  @test hw in dw
  @test WeightLatticeElem(TypeA{2}, [0, 0]) in dw

  # A1, hw = 3ω1: dominant weights are 3ω, ω
  hw1 = WeightLatticeElem(TypeA{1}, [3])
  dw1 = dominant_weights(hw1)
  @test length(dw1) == 2
  @test WeightLatticeElem(TypeA{1}, [3]) in dw1
  @test WeightLatticeElem(TypeA{1}, [1]) in dw1
end

# ═══════════════════════════════════════════════════════════════════════
#  Singularity and Borel–Weil–Bott
# ═══════════════════════════════════════════════════════════════════════
@testset "Singularity" begin
  # ρ is strictly dominant → regular
  @test !is_singular(weyl_vector(TypeA{2}))
  @test !is_singular(weyl_vector(TypeB{3}))
  @test !is_singular(weyl_vector(TypeG2))

  # The zero weight is singular
  @test is_singular(WeightLatticeElem(TypeA{2}, [0, 0]))

  # A weight with one zero coordinate
  @test is_singular(WeightLatticeElem(TypeA{2}, [1, 0]))
  @test is_singular(WeightLatticeElem(TypeA{2}, [0, 1]))

  # Strictly dominant → regular
  @test !is_singular(WeightLatticeElem(TypeA{2}, [1, 1]))
  @test !is_singular(WeightLatticeElem(TypeA{2}, [3, 5]))

  # Non-dominant but conjugate to a singular weight
  @test is_singular(WeightLatticeElem(TypeA{2}, [-1, 1]))  # → [1, 0]

  # Non-dominant but conjugate to a regular weight
  @test !is_singular(WeightLatticeElem(TypeA{2}, [-1, 3])) # → [1, 2] or similar
end

@testset "dot_reduce" begin
  zero_A2 = WeightLatticeElem(TypeA{2}, [0, 0])
  @test Semisimple.dot_reduce(fundamental_weight(TypeA{2}, 1)) ==
    (1, fundamental_weight(TypeA{2}, 1))
  @test Semisimple.dot_reduce(WeightLatticeElem(TypeA{2}, [-2, 1])) == (-1, zero_A2)
  @test Semisimple.dot_reduce(WeightLatticeElem(TypeA{2}, [-1, 0])) == (0, zero_A2)

  zero_B2 = WeightLatticeElem(TypeB{2}, [0, 0])
  @test Semisimple.dot_reduce(fundamental_weight(TypeB{2}, 1)) ==
    (1, fundamental_weight(TypeB{2}, 1))
  @test Semisimple.dot_reduce(WeightLatticeElem(TypeB{2}, [2, -2])) ==
    (-1, fundamental_weight(TypeB{2}, 1))
  @test Semisimple.dot_reduce(WeightLatticeElem(TypeB{2}, [0, -2])) == (0, zero_B2)

  zero_G2 = WeightLatticeElem(TypeG2, [0, 0])
  @test Semisimple.dot_reduce(fundamental_weight(TypeG2, 1)) ==
    (1, fundamental_weight(TypeG2, 1))
  @test Semisimple.dot_reduce(WeightLatticeElem(TypeG2, [-2, 2])) ==
    (-1, fundamental_weight(TypeG2, 2))
  @test Semisimple.dot_reduce(WeightLatticeElem(TypeG2, [0, -2])) == (0, zero_G2)
end

@testset "Borel–Weil–Bott" begin
  # ── A2 ──────────────────────────────────────────────────────────────
  # Dominant weight: degree 0, representation is itself
  @testset "A2" begin
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    ρ = weyl_vector(TypeA{2})

    # ω1 is dominant: H⁰ = V(ω1), dim = 3
    result = _borel_weil_bott(ω1)
    @test result !== nothing
    d, μ = result
    @test d == 0
    @test μ == ω1

    # ω2 is dominant: H⁰ = V(ω2), dim = 3
    result = _borel_weil_bott(ω2)
    @test result !== nothing
    d, μ = result
    @test d == 0
    @test μ == ω2

    # λ = -ρ: λ + ρ = 0, singular → nothing
    @test _borel_weil_bott(-ρ) === nothing

    # λ = [-2, 1]: λ+ρ = [-1, 2], s1 gives [1, 1], d=1, μ = [0, 0]
    λ = WeightLatticeElem(TypeA{2}, [-2, 1])
    result = _borel_weil_bott(λ)
    @test result !== nothing
    d, μ = result
    @test d == 1
    @test μ == WeightLatticeElem(TypeA{2}, [0, 0])  # trivial rep

    # λ = [-3, 3]: λ+ρ = [-2, 4], s1 gives [2, 2], d=1, μ = [1, 1]
    λ = WeightLatticeElem(TypeA{2}, [-3, 3])
    result = _borel_weil_bott(λ)
    @test result !== nothing
    d, μ = result
    @test d == 1
    @test μ == WeightLatticeElem(TypeA{2}, [1, 1])  # adjoint rep

    # λ = [-3, 1]: λ+ρ = [-2, 2], conjugates to singular weight [2, 0]
    @test _borel_weil_bott(WeightLatticeElem(TypeA{2}, [-3, 1])) === nothing
  end

  # ── A1 ──────────────────────────────────────────────────────────────
  @testset "A1" begin
    # nomega1 dominant: degree 0, result is nomega1
    for n in 0:5
      result = _borel_weil_bott(WeightLatticeElem(TypeA{1}, [n]))
      @test result !== nothing
      d, μ = result
      @test d == 0
      @test μ == WeightLatticeElem(TypeA{1}, [n])
    end

    # λ = -1: λ+ρ = 0, singular
    @test _borel_weil_bott(WeightLatticeElem(TypeA{1}, [-1])) === nothing

    # λ = -3: λ+ρ = -2, s1 → 2, dominant, d=1, μ = 2-1 = 1
    result = _borel_weil_bott(WeightLatticeElem(TypeA{1}, [-3]))
    @test result !== nothing
    d, μ = result
    @test d == 1
    @test μ == WeightLatticeElem(TypeA{1}, [1])
  end

  # ── B2 ──────────────────────────────────────────────────────────────
  @testset "B2" begin
    # Dominant weight: degree 0
    ω1 = fundamental_weight(TypeB{2}, 1)
    result = _borel_weil_bott(ω1)
    @test result !== nothing
    d, μ = result
    @test d == 0
    @test μ == ω1
  end

  # ── Consistency: degree 0 ⟺ dominant ────────────────────────────────
  @testset "Degree 0 iff dominant" begin
    for DT in [TypeA{2}, TypeB{2}, TypeG2]
      R = rank(DT)
      for i in 1:R
        ωi = fundamental_weight(DT, i)
        result = _borel_weil_bott(ωi)
        @test result !== nothing
        d, μ = result
        @test d == 0
        @test μ == ωi
      end
    end
  end
  # ── E8 example ──────────────────────────────────────────────────────────────────────────
  @testset "E8" begin
    λ = WeightLatticeElem(TypeE{8}, [-5, 3, -2, -3, 5, -8, 2, 1])
    # λ+ρ conjugates to a singular weight, so all cohomology vanishes
    @test _borel_weil_bott(λ) === nothing
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Folding and Borel–Weil–Bott inside a root subsystem
#
#  Reflecting only in a subset S of the simple roots folds a weight into the
#  dominant chamber of the Levi subgroup L_S.  The ground truth is the same
#  computation performed inside the sub-root-system itself.
# ═══════════════════════════════════════════════════════════════════════
@testset "Levi-restricted fold" begin
  CASES = [
    (TypeA{3}, (1, 2, 3)), (TypeA{3}, (2, 3)), (TypeA{3}, (1, 3)), (TypeA{3}, (2,)),
    (TypeA{4}, (1, 2, 4)), (TypeB{3}, (2, 3)), (TypeB{4}, (1, 3, 4)),
    (TypeC{3}, (1, 2)), (TypeD{4}, (2, 3, 4)), (TypeD{5}, (1, 2, 4)),
    (TypeG2, (2,)), (TypeF4, (2, 3, 4)), (TypeE{6}, (2, 4, 5)),
  ]
  COORDS = [-3, -1, 0, 1, 4]

  @testset "$DT restricted to $S" for (DT, S) in CASES
    R = rank(DT)
    LT, ord = sub_dynkin_type_with_ordering(DT, S)

    for seed in 1:6
      λ = WeightLatticeElem(DT, Int[COORDS[1 + (seed * i) % length(COORDS)] for i in 1:R])
      dom, len = conjugate_dominant_weight_with_length(λ, S)

      # Dominant for the subsystem, and only for it.
      @test all(coefficients(dom)[s] >= 0 for s in S)

      # Ground truth: the same fold carried out inside the sub-root-system.
      sub = WeightLatticeElem(LT, Int[coefficients(λ)[ord[k]] for k in 1:rank(LT)])
      sub_dom, sub_len = conjugate_dominant_weight_with_length(sub)
      @test Int[coefficients(dom)[ord[k]] for k in 1:rank(LT)] == coefficients(sub_dom)
      @test len == sub_len

      # The word from _with_elem has the same length and reproduces `dom`.
      dom_e, word = conjugate_dominant_weight_with_elem(λ, S)
      @test dom_e == dom
      @test length(word) == len
      @test issubset(word, S)
      @test foldl(reflect, word; init=λ) == dom

      # Folding is idempotent, and the coordinates outside S move only by roots
      # of the subsystem, so a node not touched by S keeps its coordinate.
      @test conjugate_dominant_weight(dom, S) == dom
      for i in 1:R
        if all(cartan_matrix(DT)[i, s] == 0 for s in S)
          @test coefficients(dom)[i] == coefficients(λ)[i]
        end
      end
    end

    # Passing every node is the absolute statement.
    λ = WeightLatticeElem(DT, Int[i % 3 == 0 ? -2 : 1 for i in 1:R])
    @test conjugate_dominant_weight(λ, 1:R) == conjugate_dominant_weight(λ)
    @test conjugate_dominant_weight_with_length(λ, Tuple(1:R)) ==
      conjugate_dominant_weight_with_length(λ)
    @test _borel_weil_bott(λ, 1:R) == _borel_weil_bott(λ)
  end
end

@testset "Levi-restricted fold: node argument" begin
  λ = WeightLatticeElem(TypeA{3}, [-1, 2, -1])

  # Any container of node indices works, since membership is all that is asked
  # of it.
  expected = conjugate_dominant_weight(λ, (2, 3))
  @test conjugate_dominant_weight(λ, [2, 3]) == expected
  @test conjugate_dominant_weight(λ, 2:3) == expected
  @test conjugate_dominant_weight(λ, Set([2, 3])) == expected
  @test conjugate_dominant_weight(λ, (3, 2)) == expected
  @test conjugate_dominant_weight(λ, (2, 3, 2)) == expected   # duplicates are harmless

  # A node outside the diagram is a mistake, not a reflection to skip.  Note the
  # test suite runs under --check-bounds=yes; this guard is what keeps an ordinary
  # build from indexing past the end of the coordinate vector.
  for nodes in [(0,), (4,), (17,), (1, 5), [-1]]
    @test_throws ArgumentError conjugate_dominant_weight(λ, nodes)
    @test_throws ArgumentError conjugate_dominant_weight_with_length(λ, nodes)
    @test_throws ArgumentError conjugate_dominant_weight_with_elem(λ, nodes)
    @test_throws ArgumentError _borel_weil_bott(λ, nodes)
    @test_throws ArgumentError is_singular(λ, nodes)
  end

  # A range is checked by its extremes rather than element by element, so it
  # needs its own out-of-range cases.
  @test_throws ArgumentError conjugate_dominant_weight(λ, 2:5)
  @test_throws ArgumentError conjugate_dominant_weight(λ, 0:2)
  @test_throws ArgumentError _borel_weil_bott(λ, 0:4)
  @test conjugate_dominant_weight(λ, 1:3) == conjugate_dominant_weight(λ)
  @test conjugate_dominant_weight(λ, Base.OneTo(3)) == conjugate_dominant_weight(λ)

  # No nodes at all: nothing moves.  An empty range is vacuously in range, even
  # when its endpoints are not.
  @test conjugate_dominant_weight(λ, ()) == λ
  @test conjugate_dominant_weight(λ, 3:2) == λ
  @test conjugate_dominant_weight(λ, 9:1) == λ
  @test conjugate_dominant_weight_with_length(λ, ()) == (λ, 0)
  @test !is_singular(λ, ())
end

@testset "Levi-restricted Borel–Weil–Bott" begin
  # Agreement with the absolute statement inside the sub-root-system: the degree
  # is the same, and the output weight restricts to the sub-diagram output.  Note
  # ρ_G is the correct shift on both sides, since ρ_G - ρ_S is W_S-invariant.
  @testset "$DT restricted to $S" for (DT, S) in
                                      [(TypeA{3}, (1, 3)), (TypeA{4}, (2, 3, 4)),
    (TypeB{3}, (1, 2)), (TypeC{3}, (2, 3)),
    (TypeD{4}, (1, 3, 4)), (TypeF4, (1, 2))]
    R = rank(DT)
    LT, ord = sub_dynkin_type_with_ordering(DT, S)
    ρ, ρ_S = weyl_vector(DT), weyl_vector(LT)

    for seed in 1:8
      λ = WeightLatticeElem(DT, Int[((seed * i) % 7) - 3 for i in 1:R])
      result = _borel_weil_bott(λ, S)

      sub_λ =
        WeightLatticeElem(
          LT, Int[coefficients(λ)[ord[k]] + coefficients(ρ)[ord[k]] for k in 1:rank(LT)]
        ) - ρ_S
      sub_result = _borel_weil_bott(sub_λ)

      @test (result === nothing) == (sub_result === nothing)
      result === nothing && continue
      d, μ = result
      sub_d, sub_μ = sub_result
      @test d == sub_d
      @test Int[coefficients(μ)[ord[k]] for k in 1:rank(LT)] == coefficients(sub_μ)
      # The output is dominant for the subsystem, as the theorem promises.
      @test all(coefficients(μ)[s] >= 0 for s in S)
    end
  end

  # is_singular is the vanishing criterion, so it must agree with _borel_weil_bott
  # on exactly when nothing survives.
  @testset "is_singular restricted: $DT / $S" for (DT, S) in
                                                  [(TypeA{3}, (1, 3)), (TypeB{3}, (2, 3)),
    (TypeC{3}, (1, 2)), (TypeD{4}, (2, 3, 4)),
    (TypeG2, (1,))]
    R = rank(DT)
    ρ = weyl_vector(DT)
    RS = RootSystem(DT)
    sub_positive = [
      α for α in positive_roots(RS) if
      all(coefficients(α)[i] == 0 for i in 1:R if !(i in S))
    ]

    for seed in 1:10
      λ = WeightLatticeElem(DT, Int[((seed * i) % 7) - 3 for i in 1:R])
      # Ground truth: pair λ + ρ against every positive root of the subsystem.
      expected = any(iszero(dot(α, λ + ρ)) for α in sub_positive)
      @test is_singular(λ + ρ, S) == expected
      @test (_borel_weil_bott(λ, S) === nothing) == expected

      # Singular for the subsystem implies singular for the whole system, since
      # the offending root is a root of both.
      @test !is_singular(λ + ρ, S) || is_singular(λ + ρ)
    end

    # Passing every node is the absolute statement.
    λ = WeightLatticeElem(DT, Int[i % 2 == 0 ? 0 : 2 for i in 1:R])
    @test is_singular(λ, 1:R) == is_singular(λ)
  end

  # A weight that is regular for the whole group but singular for a subsystem,
  # and one that is singular for the whole group but regular for a subsystem.
  @test _borel_weil_bott(WeightLatticeElem(TypeA{2}, [-2, 1])) ==
    (1, WeightLatticeElem(TypeA{2}))
  @test _borel_weil_bott(WeightLatticeElem(TypeA{2}, [-2, 1]), (2,)) ==
    (0, WeightLatticeElem(TypeA{2}, [-2, 1]))
  @test _borel_weil_bott(WeightLatticeElem(TypeA{2}, [0, -1])) === nothing
  @test _borel_weil_bott(WeightLatticeElem(TypeA{2}, [0, -1]), (1,)) ==
    (0, WeightLatticeElem(TypeA{2}, [0, -1]))

  # No nodes to reflect in: nothing can be singular and nothing moves.
  @test _borel_weil_bott(WeightLatticeElem(TypeA{2}, [-5, -5]), ()) ==
    (0, WeightLatticeElem(TypeA{2}, [-5, -5]))
end

# ═══════════════════════════════════════════════════════════════════════
#  StaticArrays: verify types are compile-time static
# ═══════════════════════════════════════════════════════════════════════
@testset "Static type system" begin
  # Cartan matrices are SMatrix
  C = cartan_matrix(TypeA{3})
  @test C isa SMatrix{3,3,Int}

  C2 = cartan_matrix(ProductDynkinType{Tuple{TypeA{2},TypeB{3}}})
  @test C2 isa SMatrix{5,5,Int}

  # Weights and roots use SVector
  w = fundamental_weight(TypeA{3}, 1)
  @test coefficients(w) isa SVector{3,Int}

  RS = RootSystem(TypeA{3})
  α = simple_root(RS, 1)
  @test coefficients(α) isa SVector{3,Int}
end

# ═══════════════════════════════════════════════════════════════════════
#  Product types
# ═══════════════════════════════════════════════════════════════════════
@testset "Product types" begin
  PT = ProductDynkinType{Tuple{TypeA{2},TypeB{3}}}

  @test rank(PT) == 5
  @test n_positive_roots(PT) == 3 + 9  # A2 has 3, B3 has 9

  RS = RootSystem(PT)
  @test n_positive_roots(RS) == 12
  @test n_simple_roots(RS) == 5

  # Cartan matrix is block diagonal
  C = cartan_matrix(PT)
  @test C[1:2, 1:2] == cartan_matrix(TypeA{2})
  @test C[3:5, 3:5] == cartan_matrix(TypeB{3})

  # Weyl group of product
  @test weyl_order(PT) == factorial(BigInt(3)) * factorial(BigInt(3)) * BigInt(2)^3
end

# ═══════════════════════════════════════════════════════════════════════
#  Characters — Freudenthal, tensor products, exterior / symmetric powers
# ═══════════════════════════════════════════════════════════════════════
@testset "Characters" begin

  # ─── WeylCharacter basics ─────────────────────────────────────
  @testset "WeylCharacter basics" begin
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    V1 = WeylCharacter(ω1)
    V2 = WeylCharacter(ω2)

    @test is_effective(V1)
    @test is_irreducible(V1)
    @test highest_weight(V1) == ω1
    @test !iszero(V1)
    @test iszero(WeylCharacter(TypeA{2}))

    # Arithmetic
    @test V1 + V2 == V2 + V1
    @test V1 - V1 == WeylCharacter(TypeA{2})
    @test 2 * V1 == V1 + V1
    @test is_effective(V1 + V2)
    @test !is_irreducible(V1 + V2)
  end

  # ─── add! and addmul! ────────────────────────────────────────
  @testset "add! and addmul!" begin
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)

    # add! is equivalent to +
    V = WeylCharacter(ω1)
    W = WeylCharacter(ω2)
    expected = V + W
    add!(V, W)
    @test V == expected

    # add! with self-cancellation
    V2 = WeylCharacter(ω1)
    add!(V2, -WeylCharacter(ω1))
    @test iszero(V2)

    # addmul! basic
    V3 = WeylCharacter(TypeA{2})
    addmul!(V3, WeylCharacter(ω1), 5)
    @test V3 == 5 * WeylCharacter(ω1)

    # addmul! with negative coefficient
    V4 = WeylCharacter(ω1) + WeylCharacter(ω2)
    addmul!(V4, WeylCharacter(ω1), -1)
    @test V4 == WeylCharacter(ω2)

    # addmul! with c=0 is identity
    V5 = WeylCharacter(ω1)
    addmul!(V5, WeylCharacter(ω2), 0)
    @test V5 == WeylCharacter(ω1)

    # add! returns the modified object
    V6 = WeylCharacter(ω1)
    @test add!(V6, WeylCharacter(ω2)) === V6

    # addmul! returns the modified object
    V7 = WeylCharacter(ω1)
    @test addmul!(V7, WeylCharacter(ω2), 2) === V7
  end

  # ─── Dominant character ─────────────────────────────────────────
  @testset "Dominant character" begin
    # A2 standard: V(ω1) has dim 3, only 1 dominant weight (ω1 itself)
    dc = dominant_character(fundamental_weight(TypeA{2}, 1))
    @test length(dc) == 1
    @test dc[SVector(1, 0)] == 1

    # A2 adjoint: V(ω1+ω2) has dim 8, dominant weights are ω1+ω2 and 0
    dc_adj = dominant_character(
      fundamental_weight(TypeA{2}, 1) + fundamental_weight(TypeA{2}, 2)
    )
    @test length(dc_adj) == 2
    @test dc_adj[SVector(1, 1)] == 1  # highest weight
    @test dc_adj[SVector(0, 0)] == 2  # zero weight mult 2

    # Consistency: sum over Weyl orbits = full character dimension
    for (DT, i) in [(TypeA{3}, 1), (TypeB{3}, 3), (TypeC{3}, 1),
      (TypeD{4}, 1), (TypeG2, 1), (TypeF4, 4)]
      λ = fundamental_weight(DT, i)
      dc = dominant_character(λ)
      full = freudenthal_formula(λ)
      # Every dominant weight in dc must appear in full with same multiplicity
      for (μ_vec, m) in dc
        @test haskey(full, μ_vec)
        @test full[μ_vec] == m
      end
      # Total dimension via orbit expansion must match degree
      @test sum(values(full)) == degree(λ)
    end

    # E8 adjoint: V(ω8) dim 248
    dc_e8 = dominant_character(fundamental_weight(TypeE{8}, 8))
    @test dc_e8[SVector(0, 0, 0, 0, 0, 0, 0, 1)] == 1  # highest weight
    @test haskey(dc_e8, SVector(0, 0, 0, 0, 0, 0, 0, 0))  # zero weight

    # Caching: repeated calls should reuse the cached value without relying
    # on Dict object identity.
    clear_all_caches!()
    info0 = cache_info()
    λ = fundamental_weight(TypeA{3}, 1)
    dc1 = dominant_character(λ)
    info1 = cache_info()
    dc2 = dominant_character(λ)
    info2 = cache_info()
    @test dc1 == dc2
    @test info1.dominant_character.length == info0.dominant_character.length + 1
    @test info2.dominant_character.length == info1.dominant_character.length

    # Higher weight: A3 V(ρ) dim 20
    ρ = weyl_vector(TypeA{3})
    dc_rho = dominant_character(ρ)
    full_rho = freudenthal_formula(ρ)
    for (μ_vec, m) in dc_rho
      @test full_rho[μ_vec] == m
    end

    # Regression: dominant_character(ρ) for E8 used to throw
    # `DomainError("non-integer multiplicity")` because the Freudenthal inner
    # sum overflowed Int64. The recursion now uses BigInt internally; intermediate
    # multiplicities here genuinely exceed typemax(Int64).
    ρ_e8 = weyl_vector(TypeE{8})
    dc_e8_rho = dominant_character(ρ_e8)
    @test eltype(values(dc_e8_rho)) === BigInt
    @test dc_e8_rho[SVector(1, 1, 1, 1, 1, 1, 1, 1)] == 1  # highest weight, mult 1
    @test dc_e8_rho[SVector(0, 1, 1, 1, 1, 0, 1, 1)] == big"19828749079454812"
    @test maximum(values(dc_e8_rho)) > typemax(Int64)
  end

  # ─── Freudenthal formula: simply-laced ───────────────────────────
  @testset "Freudenthal (simply-laced)" begin
    # A2 standard: dim 3
    m = freudenthal_formula(fundamental_weight(TypeA{2}, 1))
    @test sum(values(m)) == 3
    @test all(v == 1 for v in values(m))  # all multiplicities 1

    # A2 adjoint: dim 8, with zero weight multiplicity 2
    m_adj = freudenthal_formula(
      fundamental_weight(TypeA{2}, 1) + fundamental_weight(TypeA{2}, 2)
    )
    @test sum(values(m_adj)) == 8
    @test m_adj[SVector(0, 0)] == 2  # zero weight has multiplicity 2

    # D4 fundamental: dim 8
    m_d4 = freudenthal_formula(fundamental_weight(TypeD{4}, 1))
    @test sum(values(m_d4)) == 8

    # E6 fundamental ω1: dim 27
    m_e6 = freudenthal_formula(fundamental_weight(TypeE{6}, 1))
    @test sum(values(m_e6)) == 27

    # E8 fundamental ω8: dim 248
    m_e8 = freudenthal_formula(fundamental_weight(TypeE{8}, 8))
    @test sum(values(m_e8)) == 248
  end

  # ─── Freudenthal formula: non-simply-laced ───────────────────────
  @testset "Freudenthal (non-simply-laced)" begin
    # B2: std (dim 5), spin (dim 4)
    @test sum(values(freudenthal_formula(fundamental_weight(TypeB{2}, 1)))) == 5
    @test sum(values(freudenthal_formula(fundamental_weight(TypeB{2}, 2)))) == 4

    # B3: std (dim 7), spin (dim 8)
    @test sum(values(freudenthal_formula(fundamental_weight(TypeB{3}, 1)))) == 7
    @test sum(values(freudenthal_formula(fundamental_weight(TypeB{3}, 3)))) == 8

    # C3: std (dim 6)
    @test sum(values(freudenthal_formula(fundamental_weight(TypeC{3}, 1)))) == 6

    # G2: 7-dim and 14-dim
    @test sum(values(freudenthal_formula(fundamental_weight(TypeG2, 1)))) == 7
    @test sum(values(freudenthal_formula(fundamental_weight(TypeG2, 2)))) == 14

    # F4: 52-dim and 26-dim
    @test sum(values(freudenthal_formula(fundamental_weight(TypeF4, 1)))) == 52
    @test sum(values(freudenthal_formula(fundamental_weight(TypeF4, 4)))) == 26
  end

  # ─── Tensor products ─────────────────────────────────────────────
  @testset "Tensor products" begin
    # A2: V(ω1) ⊗ V(ω1) = V(2ω1) + V(ω2)
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    V1 = WeylCharacter(ω1)
    tp = V1 * V1
    @test tp == WeylCharacter(2 * ω1) + WeylCharacter(ω2)

    # A2: V(ω1) ⊗ V(ω2) = V(ω1+ω2) + V(0)
    tp2 = V1 * WeylCharacter(ω2)
    @test tp2 ==
      WeylCharacter(ω1 + ω2) +
          WeylCharacter(WeightLatticeElem(TypeA{2}, SVector(0, 0)))

    # B2: V(ω1) ⊗ V(ω1) = V(2ω1) + V(ω2) + V(0) (dims: 25 = 14+10+1)
    ω1_b = fundamental_weight(TypeB{2}, 1)
    ω2_b = fundamental_weight(TypeB{2}, 2)
    tp_b = WeylCharacter(ω1_b) * WeylCharacter(ω1_b)
    @test tp_b ==
      WeylCharacter(2 * ω1_b) +
          WeylCharacter(WeightLatticeElem(TypeB{2}, SVector(0, 2))) +
          WeylCharacter(WeightLatticeElem(TypeB{2}, SVector(0, 0)))

    # Dimension check: tensor product preserves dimension
    @test sum(degree(k) * v for (k, v) in tp.terms) == 9

    # Tensor product of virtual (non-effective) characters
    # V(ω1) - V(ω2) tensored with V(ω1):
    # = V(ω1) ⊗ V(ω1) - V(ω2) ⊗ V(ω1)
    # = [V(2ω1) + V(ω2)] - [V(ω1+ω2) + V(0)]
    virtual = WeylCharacter(ω1) - WeylCharacter(ω2)
    @test !is_effective(virtual)
    tp_virt = virtual * WeylCharacter(ω1)
    z = WeightLatticeElem(TypeA{2}, SVector(0, 0))
    expected_virt =
      WeylCharacter(2 * ω1) + WeylCharacter(ω2) - WeylCharacter(ω1 + ω2) -
      WeylCharacter(z)
    @test tp_virt == expected_virt

    # Character arithmetic prunes zero-multiplicity terms
    @test iszero(WeylCharacter(ω1) + (-WeylCharacter(ω1)))
    @test iszero(WeylCharacter(ω1) - WeylCharacter(ω1))
  end

  # ─── Littlewood–Richardson rule ──────────────────────────────────
  @testset "Littlewood–Richardson rule" begin
    # Verify LR matches Brauer–Klimyk for all Type A tests

    # Helper: compute tensor product via Brauer–Klimyk only
    function bk_tensor(λ, μ)
      if Semisimple.degree(λ) > Semisimple.degree(μ)
        Semisimple.brauer_klimyk(Semisimple.freudenthal_formula(μ), λ)
      else
        Semisimple.brauer_klimyk(Semisimple.freudenthal_formula(λ), μ)
      end
    end

    # A1: simplest case
    ω1_a1 = fundamental_weight(TypeA{1}, 1)
    @test lr_tensor_product(ω1_a1, ω1_a1) == bk_tensor(ω1_a1, ω1_a1)
    @test lr_tensor_product(2ω1_a1, ω1_a1) == bk_tensor(2ω1_a1, ω1_a1)
    @test lr_tensor_product(3ω1_a1, 2ω1_a1) == bk_tensor(3ω1_a1, 2ω1_a1)

    # A2: comprehensive tests
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    z = WeightLatticeElem(TypeA{2}, SVector(0, 0))

    @test lr_tensor_product(ω1, ω1) == WeylCharacter(2ω1) + WeylCharacter(ω2)
    @test lr_tensor_product(ω1, ω2) == WeylCharacter(ω1 + ω2) + WeylCharacter(z)
    @test lr_tensor_product(ω2, ω2) == WeylCharacter(2ω2) + WeylCharacter(ω1)
    @test lr_tensor_product(ω1 + ω2, ω1) == bk_tensor(ω1 + ω2, ω1)
    @test lr_tensor_product(ω1 + ω2, ω2) == bk_tensor(ω1 + ω2, ω2)
    @test lr_tensor_product(ω1 + ω2, ω1 + ω2) == bk_tensor(ω1 + ω2, ω1 + ω2)
    @test lr_tensor_product(2ω1, ω1) == bk_tensor(2ω1, ω1)
    @test lr_tensor_product(2ω1, ω2) == bk_tensor(2ω1, ω2)
    @test lr_tensor_product(2ω1, 2ω1) == bk_tensor(2ω1, 2ω1)
    @test lr_tensor_product(3ω1, ω2) == bk_tensor(3ω1, ω2)

    # Edge case: tensor with trivial
    @test lr_tensor_product(ω1, z) == WeylCharacter(ω1)
    @test lr_tensor_product(z, ω1) == WeylCharacter(ω1)
    @test lr_tensor_product(z, z) == WeylCharacter(z)

    # A3: tests
    ω = [fundamental_weight(TypeA{3}, i) for i in 1:3]
    @test lr_tensor_product(ω[1], ω[1]) == bk_tensor(ω[1], ω[1])
    @test lr_tensor_product(ω[1], ω[2]) == bk_tensor(ω[1], ω[2])
    @test lr_tensor_product(ω[1], ω[3]) == bk_tensor(ω[1], ω[3])
    @test lr_tensor_product(ω[2], ω[2]) == bk_tensor(ω[2], ω[2])
    @test lr_tensor_product(ω[2], ω[3]) == bk_tensor(ω[2], ω[3])
    @test lr_tensor_product(ω[1] + ω[3], ω[1]) == bk_tensor(ω[1] + ω[3], ω[1])
    @test lr_tensor_product(ω[1] + ω[3], ω[2]) == bk_tensor(ω[1] + ω[3], ω[2])
    @test lr_tensor_product(2ω[1], ω[2]) == bk_tensor(2ω[1], ω[2])
    @test lr_tensor_product(2ω[1], 2ω[1]) == bk_tensor(2ω[1], 2ω[1])

    # A4: tests
    ω4 = [fundamental_weight(TypeA{4}, i) for i in 1:4]
    @test lr_tensor_product(ω4[1], ω4[1]) == bk_tensor(ω4[1], ω4[1])
    @test lr_tensor_product(ω4[1], ω4[4]) == bk_tensor(ω4[1], ω4[4])
    @test lr_tensor_product(ω4[2], ω4[2]) == bk_tensor(ω4[2], ω4[2])
    @test lr_tensor_product(ω4[2], ω4[3]) == bk_tensor(ω4[2], ω4[3])

    # A5: tests
    ω5 = [fundamental_weight(TypeA{5}, i) for i in 1:5]
    @test lr_tensor_product(ω5[1], ω5[1]) == bk_tensor(ω5[1], ω5[1])
    @test lr_tensor_product(ω5[1], ω5[5]) == bk_tensor(ω5[1], ω5[5])
    @test lr_tensor_product(ω5[2], ω5[2]) == bk_tensor(ω5[2], ω5[2])
    @test lr_tensor_product(2ω5[1], ω5[1]) == bk_tensor(2ω5[1], ω5[1])
    @test lr_tensor_product(2ω5[1], 2ω5[1]) == bk_tensor(2ω5[1], 2ω5[1])

    # A7: higher rank
    ω7 = [fundamental_weight(TypeA{7}, i) for i in 1:7]
    @test lr_tensor_product(ω7[1], ω7[1]) == bk_tensor(ω7[1], ω7[1])
    @test lr_tensor_product(ω7[1], ω7[7]) == bk_tensor(ω7[1], ω7[7])
    @test lr_tensor_product(ω7[2], ω7[2]) == bk_tensor(ω7[2], ω7[2])

    # Dimension consistency: tensor product dimension = dim(V) * dim(W)
    for (λ, μ) in [(ω1, ω1), (ω1, ω2), (ω1 + ω2, ω1),
      (ω[1], ω[2]), (ω4[2], ω4[3])]
      result = lr_tensor_product(λ, μ)
      dim_sum = sum(Semisimple.degree(k) * v for (k, v) in result.terms)
      @test dim_sum == Semisimple.degree(λ) * Semisimple.degree(μ)
    end

    # Commutativity: LR(λ, μ) == LR(μ, λ)
    @test lr_tensor_product(ω1, ω2) == lr_tensor_product(ω2, ω1)
    @test lr_tensor_product(ω[1], ω[3]) == lr_tensor_product(ω[3], ω[1])
    @test lr_tensor_product(2ω1, ω2) == lr_tensor_product(ω2, 2ω1)

    # tensor_product dispatches to LR for TypeA
    empty!(Semisimple._tensor_cache)
    tp_dispatch = tensor_product(ω1, ω2)
    @test tp_dispatch == lr_tensor_product(ω1, ω2)

    normalize_lr(d) = Dict(Tuple(k) => v for (k, v) in d)
    @test normalize_lr(Semisimple._lr_coefficients([1], [1, 0], 3)) ==
      normalize_lr(Semisimple._lr_coefficients([1, 0, 0], [1], 3))
  end

  # ─── Dual ────────────────────────────────────────────────────────
  @testset "Dual" begin
    # A2: dual(ω1) = ω2 (A2 has non-trivial outer automorphism)
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    @test dual(ω1) == ω2
    @test dual(ω2) == ω1

    # B2: dual = identity (all reps self-dual)
    ω1_b = fundamental_weight(TypeB{2}, 1)
    @test dual(ω1_b) == ω1_b

    # Dual of virtual character
    V = WeylCharacter(ω1)
    @test highest_weight(dual(V)) == ω2
  end

  # ─── Exterior powers ─────────────────────────────────────────────
  @testset "Exterior powers" begin
    # A2: ⋀²V(ω1) = V(ω2)
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    @test ⋀(2, ω1) == WeylCharacter(ω2)
    @test ⋀(3, ω1) == WeylCharacter(WeightLatticeElem(TypeA{2}, SVector(0, 0)))

    # A3: ⋀²V(ω1) = V(ω2), ⋀³V(ω1) = V(ω3)
    ω1_a3 = fundamental_weight(TypeA{3}, 1)
    ω2_a3 = fundamental_weight(TypeA{3}, 2)
    ω3_a3 = fundamental_weight(TypeA{3}, 3)
    @test ⋀(2, ω1_a3) == WeylCharacter(ω2_a3)
    @test ⋀(3, ω1_a3) == WeylCharacter(ω3_a3)

    # A3: ⋀⁴V(ω1) = trivial (top exterior power of std rep)
    z_a3 = WeightLatticeElem(TypeA{3}, SVector(0, 0, 0))
    @test ⋀(4, ω1_a3) == WeylCharacter(z_a3)

    # A3: ⋀ᵏV(ω1) = 0 for k > dim = 4
    @test ⋀(5, ω1_a3) == WeylCharacter(TypeA{3})

    # E8: ⋀²V(ω1) has 4 irreducible components
    ω1_e8 = fundamental_weight(TypeE{8}, 1)
    r = ⋀(2, ω1_e8)
    @test length(r.terms) == 4
    @test is_effective(r)

    # ─── Dimension identity: dim ⋀ᵏV = C(dim V, k) ─────────────
    # A4: V(ω1) has dim 5, so ⋀ᵏ has dim C(5,k)
    ω1_a4 = fundamental_weight(TypeA{4}, 1)
    for k in 0:5
      r = ⋀(k, ω1_a4)
      @test sum(m * degree(μ) for (μ, m) in r.terms; init=0) == binomial(5, k)
    end

    # B3: V(ω3) is 8-dimensional spin rep
    ω3_b3 = fundamental_weight(TypeB{3}, 3)
    d = degree(ω3_b3)
    for k in 1:3
      r = ⋀(k, ω3_b3)
      @test is_effective(r)
      @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(d, k)
    end

    # ─── Newton identity: V ⊗ V = Sym²V + ⋀²V ──────────────────
    for λ in [
      fundamental_weight(TypeA{3}, 1),
      fundamental_weight(TypeA{3}, 2),
      fundamental_weight(TypeB{3}, 1),
      fundamental_weight(TypeC{3}, 1),
      fundamental_weight(TypeG2, 1),
    ]
      @test tensor_product(λ, λ) == Sym(2, λ) + ⋀(2, λ)
    end

    # ─── Larger exterior powers across types ─────────────────────
    # A5: ⋀³V(ω1) = V(ω3)  (fundamental rep)
    ω1_a5 = fundamental_weight(TypeA{5}, 1)
    ω3_a5 = fundamental_weight(TypeA{5}, 3)
    @test ⋀(3, ω1_a5) == WeylCharacter(ω3_a5)

    # A7: ⋀⁴V(ω1) = V(ω4)
    ω1_a7 = fundamental_weight(TypeA{7}, 1)
    ω4_a7 = fundamental_weight(TypeA{7}, 4)
    @test ⋀(4, ω1_a7) == WeylCharacter(ω4_a7)

    # D4: ⋀²V(ω1) has specific structure
    ω1_d4 = fundamental_weight(TypeD{4}, 1)
    r_d4 = ⋀(2, ω1_d4)
    @test is_effective(r_d4)
    @test sum(m * degree(μ) for (μ, m) in r_d4.terms) == binomial(8, 2)

    # G2: dim ⋀ᵏV(ω1) = C(7,k) (7-dim rep)
    ω1_g2 = fundamental_weight(TypeG2, 1)
    for k in 2:4
      r = ⋀(k, ω1_g2)
      @test is_effective(r)
      @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(7, k)
    end

    # ─── Non-minuscule exterior powers ───────────────────────────
    # A3: ⋀²V(ω1+ω3) — adjoint rep (15-dim)
    ω1_a3 = fundamental_weight(TypeA{3}, 1)
    ω3_a3 = fundamental_weight(TypeA{3}, 3)
    adj = ω1_a3 + ω3_a3
    r_adj = ⋀(2, adj)
    @test is_effective(r_adj)
    @test sum(m * degree(μ) for (μ, m) in r_adj.terms) == binomial(15, 2)

    # ─── WeylCharacter overloads ──────────────────────────────────
    # Delegation: ⋀(k, V) == ⋀(k, λ) when V is irreducible
    ω1_v = fundamental_weight(TypeA{3}, 1)
    V_ext = WeylCharacter(ω1_v)
    @test ⋀(2, V_ext) == ⋀(2, ω1_v)
    @test ⋀(3, V_ext) == ⋀(3, ω1_v)
    @test ⋀(4, V_ext) == ⋀(4, ω1_v)
    @test ⋀(5, V_ext) == ⋀(5, ω1_v)

    # Newton identity via WeylCharacter
    for λ in [
      fundamental_weight(TypeA{3}, 1),
      fundamental_weight(TypeB{3}, 1),
      fundamental_weight(TypeG2, 1),
    ]
      Vλ = WeylCharacter(λ)
      @test Vλ * Vλ == Sym(2, Vλ) + ⋀(2, Vλ)
    end

    # Reducible character: ⋀²(V1 ⊕ V2) = ⋀²V1 ⊕ (V1 ⊗ V2) ⊕ ⋀²V2
    ω2_v = fundamental_weight(TypeA{3}, 2)
    V1 = WeylCharacter(ω1_v)
    V2 = WeylCharacter(ω2_v)
    @test ⋀(2, V1 + V2) == ⋀(2, V1) + V1 * V2 + ⋀(2, V2)
  end

  # ─── Symmetric powers ───────────────────────────────────────────
  @testset "Symmetric powers" begin
    # A2: Sym²V(ω1) = V(2ω1)
    ω1 = fundamental_weight(TypeA{2}, 1)
    @test Sym(2, ω1) == WeylCharacter(2 * ω1)

    # A2: Sym³V(ω1) = V(3ω1)
    @test Sym(3, ω1) == WeylCharacter(3 * ω1)

    # Sym⁰ = trivial, Sym¹ = identity
    z = WeightLatticeElem(TypeA{2}, SVector(0, 0))
    @test Sym(0, ω1) == WeylCharacter(z)
    @test Sym(1, ω1) == WeylCharacter(ω1)

    # ─── Type A: SymᵏV(ω1) = V(komega1) (always irreducible) ──────
    for (DT, k_max) in [(TypeA{2}, 5), (TypeA{3}, 4), (TypeA{5}, 3)]
      ω1 = fundamental_weight(DT, 1)
      for k in 2:k_max
        @test Sym(k, ω1) == WeylCharacter(k * ω1)
      end
    end

    # ─── Dimension identity: dim Symᵏ(V) = C(dim V + k - 1, k) ─
    # A3: dim V(ω1) = 4, so dim Symᵏ = C(4+k-1, k)
    ω1_a3 = fundamental_weight(TypeA{3}, 1)
    for k in 2:5
      r = Sym(k, ω1_a3)
      @test is_effective(r)
      @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(4 + k - 1, k)
    end

    # B2: dim V(ω1) = 5, dim Symᵏ = C(5+k-1, k)
    ω1_b2 = fundamental_weight(TypeB{2}, 1)
    for k in 2:4
      r = Sym(k, ω1_b2)
      @test is_effective(r)
      @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(5 + k - 1, k)
    end

    # ─── Cross-type symmetric powers ────────────────────────────
    # G2: Sym²V(ω1) decomposes; verify effectiveness and dimension
    ω1_g2 = fundamental_weight(TypeG2, 1)
    r = Sym(2, ω1_g2)
    @test is_effective(r)
    @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(7 + 1, 2)

    # C3: Sym²V(ω1) decomposes; verify dimension
    ω1_c3 = fundamental_weight(TypeC{3}, 1)
    r = Sym(2, ω1_c3)
    @test is_effective(r)
    @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(6 + 1, 2)

    # ─── WeylCharacter overloads ──────────────────────────────────
    # Delegation: Sym(k, V) == Sym(k, λ) when V is irreducible
    ω1_v = fundamental_weight(TypeA{3}, 1)
    V_sym = WeylCharacter(ω1_v)
    @test Sym(2, V_sym) == Sym(2, ω1_v)
    @test Sym(3, V_sym) == Sym(3, ω1_v)

    # Boundary cases
    z3 = WeightLatticeElem(TypeA{3}, SVector(0, 0, 0))
    @test Sym(0, V_sym) == WeylCharacter(z3)
    @test Sym(1, V_sym) == V_sym

    # Reducible character: Sym²(2V) = 3·Sym²V + ⋀²V
    @test Sym(2, 2 * V_sym) == 3 * Sym(2, V_sym) + ⋀(2, V_sym)

    # Reducible: Sym²(V1 ⊕ V2) = Sym²V1 + V1⊗V2 + Sym²V2
    ω2_v = fundamental_weight(TypeA{3}, 2)
    V1 = WeylCharacter(ω1_v)
    V2 = WeylCharacter(ω2_v)
    @test Sym(2, V1 + V2) == Sym(2, V1) + V1 * V2 + Sym(2, V2)
  end

  # ─── Adams operators ─────────────────────────────────────────────
  @testset "Adams operators" begin
    ω1 = fundamental_weight(TypeA{2}, 1)

    # ψ¹ = the weight multiplicities of V(ω1)
    ψ1 = adams_operator(ω1, 1)
    @test ψ1 == freudenthal_formula(ω1)

    # Newton identity: ψ²(V) as a virtual character = Sym²(V) - ⋀²(V)
    ψ2_raw = adams_operator(ω1, 2)
    ψ2_char = character_from_weights(TypeA{2}, ψ2_raw)
    @test ψ2_char == Sym(2, ω1) - ⋀(2, ω1)
  end

  # ─── E8 exterior power cross-checks ─────────────────────────────
  @testset "E8 exterior powers" begin
    ω = [fundamental_weight(TypeE{8}, i) for i in 1:8]

    # ⋀²V(ω1): 4 irreducibles
    r1 = ⋀(2, ω[1])
    @test length(r1.terms) == 4
    @test haskey(r1.terms, WeightLatticeElem(TypeE{8}, SVector(0, 0, 0, 0, 0, 0, 1, 0)))
    @test haskey(r1.terms, WeightLatticeElem(TypeE{8}, SVector(1, 0, 0, 0, 0, 0, 0, 1)))
    @test haskey(r1.terms, WeightLatticeElem(TypeE{8}, SVector(0, 0, 0, 0, 0, 0, 0, 1)))
    @test haskey(r1.terms, WeightLatticeElem(TypeE{8}, SVector(0, 0, 1, 0, 0, 0, 0, 0)))

    # ⋀²V(ω2): 13 irreducibles
    r2 = ⋀(2, ω[2])
    @test length(r2.terms) == 13

    # ⋀⁵V(ω8): 12 irreducibles
    r3 = ⋀(5, ω[8])
    @test length(r3.terms) == 12

    # ⋀²V(2ω8): 7 irreducibles
    r4 = ⋀(2, 2 * ω[8])
    @test length(r4.terms) == 7
  end

  # ─── character_from_weights ──────────────────────────────────────
  @testset "character_from_weights" begin
    # Build the standard A2 rep from explicit weights
    m = Dict(SVector(1, 0) => 1, SVector(-1, 1) => 1, SVector(0, -1) => 1)
    V = character_from_weights(TypeA{2}, m)
    @test is_irreducible(V)
    @test highest_weight(V) == fundamental_weight(TypeA{2}, 1)

    # Adjoint A2: 8 = V(1,1) with zero weight mult 2
    m_adj = Dict(
      SVector(1, 1) => 1, SVector(2, -1) => 1, SVector(-1, 2) => 1,
      SVector(-2, 1) => 1, SVector(1, -2) => 1, SVector(-1, -1) => 1,
      SVector(0, 0) => 2,
    )
    V_adj = character_from_weights(TypeA{2}, m_adj)
    @test is_irreducible(V_adj)
    @test highest_weight(V_adj) ==
      fundamental_weight(TypeA{2}, 1) + fundamental_weight(TypeA{2}, 2)
  end

  # ─── Plethysm ─────────────────────────────────────────────────────
  @testset "Plethysm" begin
    # Symmetric power = plethysm with one-row partition
    ω1_A4 = fundamental_weight(TypeA{4}, 1)
    for k in 2:5
      @test plethysm(vcat([k]), ω1_A4) == Sym(k, ω1_A4)
    end

    # Exterior power = plethysm with one-column partition
    for k in 2:4
      @test plethysm(ones(Int, k), ω1_A4) == ⋀(k, ω1_A4)
    end

    # Mixed symmetry: S_{(2,1)} functor
    ω1_A3 = fundamental_weight(TypeA{3}, 1)
    p21 = plethysm([2, 1], ω1_A3)
    @test is_irreducible(p21)
    @test highest_weight(p21) ==
      fundamental_weight(TypeA{3}, 1) + fundamental_weight(TypeA{3}, 2)
    @test degree(p21) == 20

    # Plethysm on non-type-A: B3
    ω1_B3 = fundamental_weight(TypeB{3}, 1)
    @test plethysm([2], ω1_B3) == Sym(2, ω1_B3)
    @test plethysm([1, 1], ω1_B3) == ⋀(2, ω1_B3)

    # Plethysm on G2
    ω1_G2 = fundamental_weight(TypeG2, 1)
    @test plethysm([2], ω1_G2) == Sym(2, ω1_G2)
    @test plethysm([1, 1], ω1_G2) == ⋀(2, ω1_G2)

    # S_{(2,1)} on B3 ω1: dimension check
    # dim(V(ω1)) = 7 for B3, S_{(2,1)} has hook content dim = 7*6*5/3 = 70
    # but in general the formula is more complex
    p21_B3 = plethysm([2, 1], ω1_B3)
    @test degree(p21_B3) == 112  # known value

    # Trivial cases
    @test plethysm([1], ω1_A3) == WeylCharacter(ω1_A3)
    @test plethysm(Int[], ω1_A3) ==
      WeylCharacter(WeightLatticeElem{TypeA{3},3}(zero(SVector{3,Int})))
    @test plethysm([2, 1, 0], ω1_A3) == plethysm([2, 1], ω1_A3)
    @test Semisimple._mn_char_val([2, 1, 0], [2, 1, 0]) ==
      Semisimple._mn_char_val([2, 1], [2, 1])
  end

  # ─── ProductDynkinType characters ──────────────────────────────
  @testset "ProductDynkinType characters" begin
    PT = ProductDynkinType{Tuple{TypeA{2},TypeB{3}}}
    ω1 = fundamental_weight(PT, 1)  # A2 fundamental weight
    ω3 = fundamental_weight(PT, 3)  # B3 fundamental weight

    # Degree of product fundamental weights factors
    @test degree(ω1) == degree(fundamental_weight(TypeA{2}, 1))
    @test degree(ω3) == degree(fundamental_weight(TypeB{3}, 1))

    # Freudenthal formula
    m = freudenthal_formula(ω1)
    @test sum(values(m)) == degree(ω1)
    m3 = freudenthal_formula(ω3)
    @test sum(values(m3)) == degree(ω3)

    # Tensor product dimensions are multiplicative
    V1 = WeylCharacter(ω1)
    V3 = WeylCharacter(ω3)
    @test degree(V1 * V3) == degree(V1) * degree(V3)
    @test degree(V1 * V1) == 9  # 3 ⊗ 3 = 6 ⊕ 3̄ in A2

    # Symmetric and exterior powers
    @test degree(symmetric_power(ω1, 2)) == 6   # Sym²(3) in A2
    @test degree(exterior_power(ω1, 2)) == 3     # ∧²(3) in A2
  end

  # ─── Dual is involution ─────────────────────────────────────────
  @testset "Dual is involution" begin
    for (DT, R) in [
      (TypeA{2}, 2), (TypeA{4}, 4), (TypeB{3}, 3),
      (TypeC{3}, 3), (TypeD{4}, 4), (TypeG2, 2), (TypeF4, 4),
    ]
      for i in 1:R
        ω = fundamental_weight(DT, i)
        @test dual(dual(ω)) == ω
      end
    end
  end

  # ─── Tensor product dimension ────────────────────────────────────
  @testset "Tensor product dimension" begin
    for (DT, i, j) in [
      (TypeA{3}, 1, 2), (TypeA{3}, 1, 3),
      (TypeB{3}, 1, 3), (TypeC{3}, 1, 2),
      (TypeD{4}, 1, 3), (TypeG2, 1, 2),
    ]
      ω_i = fundamental_weight(DT, i)
      ω_j = fundamental_weight(DT, j)
      V = tensor_product(ω_i, ω_j)
      @test degree(V) == degree(ω_i) * degree(ω_j)
    end
  end

  # ─── Dimension consistency ───────────────────────────────────────
  @testset "Dimension consistency" begin
    # Freudenthal dimension matches Weyl dimension formula
    for (DT, idx) in [(TypeA{3}, 1), (TypeA{3}, 2), (TypeB{3}, 1),
      (TypeB{3}, 3), (TypeC{3}, 1), (TypeD{4}, 1),
      (TypeG2, 1), (TypeG2, 2), (TypeF4, 4)]
      λ = fundamental_weight(DT, idx)
      m = freudenthal_formula(λ)
      @test sum(values(m)) == degree(λ)
    end
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Representations with the same degree (Lübeck, arXiv:2601.18786)
# ═══════════════════════════════════════════════════════════════════════
@testset "Representations with the same degree (Lübeck)" begin

  # ─── Proposition 2: exceptional types, A2, and B2 ───────────────
  @testset "Proposition 2" begin
    # A2: V(ω1+2ω2) and V(4ω2) both have degree 15
    @test degree(WeightLatticeElem(TypeA{2}, [1, 2])) == 15
    @test degree(WeightLatticeElem(TypeA{2}, [0, 4])) == 15

    # B2: V(ω1+2ω2) and V(4ω2) both have degree 35
    @test degree(WeightLatticeElem(TypeB{2}, [1, 2])) == 35
    @test degree(WeightLatticeElem(TypeB{2}, [0, 4])) == 35

    # G2: V(3ω1) and V(2ω2) both have degree 77
    # The paper uses the opposite labeling to Bourbaki for G2 (ω1 ↔ ω2).
    @test degree(WeightLatticeElem(TypeG2, [3, 0])) == 77
    @test degree(WeightLatticeElem(TypeG2, [0, 2])) == 77

    # F4: V(ω1+ω4) and V(2ω1) both have degree 1053
    @test degree(WeightLatticeElem(TypeF4, [1, 0, 0, 1])) == 1053
    @test degree(WeightLatticeElem(TypeF4, [2, 0, 0, 0])) == 1053

    # E6: V(2ω1) and V(ω3) both have degree 351
    @test degree(WeightLatticeElem(TypeE{6}, [2, 0, 0, 0, 0, 0])) == 351
    @test degree(WeightLatticeElem(TypeE{6}, [0, 0, 1, 0, 0, 0])) == 351

    # E7: V(ω4+ω5) and V(2ω6+3ω7) both have degree 1903725824
    @test degree(WeightLatticeElem(TypeE{7}, [0, 0, 0, 1, 1, 0, 0])) ==
      1903725824
    @test degree(WeightLatticeElem(TypeE{7}, [0, 0, 0, 0, 0, 2, 3])) ==
      1903725824

    # E8: V(ω1+ω3) and V(ω1+ω7+ω8) both have degree 8634368000
    @test degree(WeightLatticeElem(TypeE{8}, [1, 0, 1, 0, 0, 0, 0, 0])) ==
      8634368000
    @test degree(WeightLatticeElem(TypeE{8}, [1, 0, 0, 0, 0, 0, 1, 1])) ==
      8634368000
  end

  # ─── Theorem 3(a): Type Aₗ ───────────────────────────────────────
  # V((l-1)ω2) and V(ω1+(l-2)ω2) have the same degree
  # = (2l-1) ∏_{k=l+1}^{2l-2} k² / (l-1)!²
  @testset "Theorem 3(a): Type A" begin
    for l in 2:15
      DT = TypeA{l}
      coords_λ = zeros(Int, l)
      coords_λ[2] = l - 1
      coords_μ = zeros(Int, l)
      coords_μ[1] = 1
      coords_μ[2] = l - 2
      λ = WeightLatticeElem(DT, coords_λ)
      μ = WeightLatticeElem(DT, coords_μ)
      expected =
        BigInt(2l - 1) *
        prod(BigInt(k)^2 for k in (l + 1):(2l - 2); init=BigInt(1)) ÷
        factorial(BigInt(l - 1))^2
      @test degree(λ) == degree(μ)
      @test degree(λ) == expected
    end
  end

  # ─── Theorem 3(b): Type Bₗ ───────────────────────────────────────
  # V((2l-2)ω2) and V(ω1+(2l-3)ω2) have the same degree
  # = 3·(4l-5)·(6l-5)·(6l-7) ∏_{k=2l}^{4l-6} k² / (2l-3)!²
  @testset "Theorem 3(b): Type B" begin
    for l in 3:10
      DT = TypeB{l}
      coords_λ = zeros(Int, l)
      coords_λ[2] = 2l - 2
      coords_μ = zeros(Int, l)
      coords_μ[1] = 1
      coords_μ[2] = 2l - 3
      λ = WeightLatticeElem(DT, coords_λ)
      μ = WeightLatticeElem(DT, coords_μ)
      expected =
        BigInt(3) * BigInt(4l - 5) * BigInt(6l - 5) *
        BigInt(6l - 7) *
        prod(BigInt(k)^2 for k in (2l):(4l - 6); init=BigInt(1)) ÷
        factorial(BigInt(2l - 3))^2
      @test degree(λ) == degree(μ)
      @test degree(λ) == expected
    end
  end

  # ─── Theorem 3(c): Type Dₗ ───────────────────────────────────────
  # V((2l-3)ω2) and V(ω1+(2l-4)ω2) have the same degree
  # = 3·(3l-4)·(3l-5)·(4l-7) ∏_{k=2l-1}^{4l-8} k² / ((l-2)²·(2l-5)!²)
  @testset "Theorem 3(c): Type D" begin
    for l in 4:10
      DT = TypeD{l}
      coords_λ = zeros(Int, l)
      coords_λ[2] = 2l - 3
      coords_μ = zeros(Int, l)
      coords_μ[1] = 1
      coords_μ[2] = 2l - 4
      λ = WeightLatticeElem(DT, coords_λ)
      μ = WeightLatticeElem(DT, coords_μ)
      expected =
        BigInt(3) * BigInt(3l - 4) * BigInt(3l - 5) *
        BigInt(4l - 7) *
        prod(BigInt(k)^2 for k in (2l - 1):(4l - 8); init=BigInt(1)) ÷
        (BigInt(l - 2)^2 * factorial(BigInt(2l - 5))^2)
      @test degree(λ) == degree(μ)
      @test degree(λ) == expected
    end
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Representation invariants (Phase 1)
# ═══════════════════════════════════════════════════════════════════════
@testset "Representation invariants" begin

  # ─── Dynkin index ──────────────────────────────────────────────────
  @testset "Dynkin index" begin
    # Fundamental reps of A_n have index 1/2
    for n in 1:6
      DT = TypeA{n}
      for i in 1:n
        @test dynkin_index(fundamental_weight(DT, i)) ==
          Rational{BigInt}(1, 2) * degree(fundamental_weight(DT, i)) *
              dot(
                fundamental_weight(DT, i), fundamental_weight(DT, i) + 2 * weyl_vector(DT)
              ) /
              dimension(DT)
      end
    end

    # Adjoint representation has index = dual Coxeter number
    for DT in (
      TypeA{2},
      TypeA{3},
      TypeB{3},
      TypeC{3},
      TypeD{4},
      TypeE{6},
      TypeE{7},
      TypeE{8},
      TypeF4,
      TypeG2,
    )
      RS = RootSystem(DT)
      R = rank(DT)
      θ = highest_root(RS)
      C = cartan_matrix(DT)
      θ_w = WeightLatticeElem{DT,R}(C * θ.vec)
      @test dynkin_index(θ_w) == dual_coxeter_number(DT)
    end

    # Zero weight has index 0
    @test dynkin_index(zero(WeightLatticeElem{TypeA{2},2})) == 0

    # Index additivity for tensor products: ℓ(V⊗W) = ℓ(V)dim(W) + dim(V)ℓ(W)
    ω1 = fundamental_weight(TypeA{3}, 1)
    ω2 = fundamental_weight(TypeA{3}, 2)
    V = tensor_product(ω1, ω2)
    idx_sum = Rational{BigInt}(0)
    for (λ, m) in V
      idx_sum += m * dynkin_index(λ)
    end
    @test idx_sum == dynkin_index(ω1) * degree(ω2) + degree(ω1) * dynkin_index(ω2)
  end

  # ─── Casimir eigenvalue ────────────────────────────────────────────
  @testset "Casimir eigenvalue" begin
    ρ = weyl_vector(TypeA{2})
    ω1 = fundamental_weight(TypeA{2}, 1)
    @test casimir_eigenvalue(ω1) == dot(ω1, ω1 + 2 * ρ)

    # In non-simply-laced types the result is normalized so long roots have length² 2
    ω3_b3 = fundamental_weight(TypeB{3}, 3)
    ρ_b3 = weyl_vector(TypeB{3})
    θ_b3 = highest_root(RootSystem(TypeB{3}))
    θ_b3_w = WeightLatticeElem{TypeB{3},3}(cartan_matrix(TypeB{3}) * θ_b3.vec)
    @test casimir_eigenvalue(ω3_b3) ==
      2 * dot(ω3_b3, ω3_b3 + 2 * ρ_b3) // dot(θ_b3_w, θ_b3_w)
    @test casimir_eigenvalue(ω3_b3) == 21//4

    # C2(0) = 0
    z = zero(WeightLatticeElem{TypeA{2},2})
    @test casimir_eigenvalue(z) == 0
  end

  # ─── Congruency class ──────────────────────────────────────────────
  @testset "Congruency class" begin
    # A2: Σ i·λᵢ mod 3
    @test congruency_class(fundamental_weight(TypeA{2}, 1)) == 1
    @test congruency_class(fundamental_weight(TypeA{2}, 2)) == 2
    @test congruency_class(WeightLatticeElem(TypeA{2}, [1, 1])) == 0  # adjoint

    # B3: λ3 mod 2
    @test congruency_class(fundamental_weight(TypeB{3}, 1)) == 0
    @test congruency_class(fundamental_weight(TypeB{3}, 3)) == 1

    # C3: Σ λᵢ for odd i mod 2
    @test congruency_class(fundamental_weight(TypeC{3}, 1)) == 1
    @test congruency_class(fundamental_weight(TypeC{3}, 2)) == 0
    @test congruency_class(fundamental_weight(TypeC{3}, 3)) == 1

    # D4: Z/2 × Z/2 center
    @test congruency_class(fundamental_weight(TypeD{4}, 1)) == (1, 1)
    @test congruency_class(fundamental_weight(TypeD{4}, 2)) == (0, 0)
    @test congruency_class(fundamental_weight(TypeD{4}, 3)) == (1, 0)
    @test congruency_class(fundamental_weight(TypeD{4}, 4)) == (0, 1)

    # D5: Z/4 center
    @test congruency_class(fundamental_weight(TypeD{5}, 1)) == 2
    @test congruency_class(fundamental_weight(TypeD{5}, 4)) == 1
    @test congruency_class(fundamental_weight(TypeD{5}, 5)) == 3

    # E6: λ1 - λ2 + λ4 - λ5 mod 3
    @test congruency_class(fundamental_weight(TypeE{6}, 1)) == 1
    @test congruency_class(fundamental_weight(TypeE{6}, 6)) == 2

    # E7: λ2 + λ5 + λ7 mod 2
    @test congruency_class(fundamental_weight(TypeE{7}, 1)) == 0
    @test congruency_class(fundamental_weight(TypeE{7}, 7)) == 1

    # Trivial center types always return 0
    @test congruency_class(fundamental_weight(TypeE{8}, 1)) == 0
    @test congruency_class(fundamental_weight(TypeF4, 1)) == 0
    @test congruency_class(fundamental_weight(TypeG2, 1)) == 0

    # Weights in same congruency class differ by root lattice element
    # A3: class = Σ i·λᵢ mod 4; [1,0,1] and [0,2,0] both have class 0
    @test congruency_class(WeightLatticeElem(TypeA{3}, [1, 0, 1])) ==
      congruency_class(WeightLatticeElem(TypeA{3}, [0, 2, 0]))
  end

  # ─── Self-dual and Frobenius–Schur ─────────────────────────────────
  @testset "Self-dual and Frobenius-Schur" begin
    # A_n: ωᵢ is self-dual iff 2i = n+1 (i.e., middle weight for odd n)
    @test !is_self_dual(fundamental_weight(TypeA{2}, 1))
    @test is_self_dual(WeightLatticeElem(TypeA{2}, [1, 1]))
    @test !is_self_dual(fundamental_weight(TypeA{3}, 1))
    @test is_self_dual(fundamental_weight(TypeA{3}, 2))  # A3: ω2 is self-dual (4×4 antisymmetric)

    # B, C, G2, F4, E7, E8: all fundamental reps are self-dual
    for DT in (TypeB{3}, TypeC{3}, TypeG2, TypeF4, TypeE{7}, TypeE{8})
      R = rank(DT)
      for i in 1:R
        @test is_self_dual(fundamental_weight(DT, i))
      end
    end

    # E6: ω1 and ω6 are NOT self-dual (conjugate to each other)
    @test !is_self_dual(fundamental_weight(TypeE{6}, 1))
    @test !is_self_dual(fundamental_weight(TypeE{6}, 6))
    @test is_self_dual(fundamental_weight(TypeE{6}, 2))

    # Frobenius-Schur indicator
    @test frobenius_schur_indicator(fundamental_weight(TypeA{2}, 1)) == 0
    @test frobenius_schur_indicator(WeightLatticeElem(TypeA{2}, [1, 1])) == 1

    # B3 vector is real, spinor of SO(7) is real (7 = 8k-1 pattern)
    @test frobenius_schur_indicator(fundamental_weight(TypeB{3}, 1)) == 1
    @test frobenius_schur_indicator(fundamental_weight(TypeB{3}, 3)) == 1

    # B2 spinor (SO(5)) is pseudoreal (5 = 8k+5 pattern)
    @test frobenius_schur_indicator(fundamental_weight(TypeB{2}, 2)) == -1

    # C2 standard rep is pseudoreal (symplectic form)
    @test frobenius_schur_indicator(fundamental_weight(TypeC{2}, 1)) == -1
    @test frobenius_schur_indicator(fundamental_weight(TypeC{2}, 2)) == 1

    # D4 spinors are real (8 = 8·1)
    @test frobenius_schur_indicator(fundamental_weight(TypeD{4}, 1)) == 1
    @test frobenius_schur_indicator(fundamental_weight(TypeD{4}, 3)) == 1
    @test frobenius_schur_indicator(fundamental_weight(TypeD{4}, 4)) == 1

    # G2 7-dim is real
    @test frobenius_schur_indicator(fundamental_weight(TypeG2, 1)) == 1
  end

  # ─── Adjoint representation ────────────────────────────────────────
  @testset "Adjoint representation" begin
    # Adjoint has dimension = dim(g)
    for DT in (
      TypeA{2},
      TypeA{3},
      TypeB{3},
      TypeC{3},
      TypeD{4},
      TypeE{6},
      TypeE{7},
      TypeE{8},
      TypeF4,
      TypeG2,
    )
      adj = adjoint_representation(DT)
      @test is_irreducible(adj)
      @test degree(adj) == dimension(DT)
    end

    # Instance dispatch
    @test degree(adjoint_representation(TypeA{3}())) == 15
  end

  @testset "adjoint_representation — product types" begin
    # A2×B2: dim(A2) = 8, dim(B2) = so(5) = 10, total = 18
    PDT1 = ProductDynkinType{Tuple{TypeA{2},TypeB{2}}}
    adj1 = adjoint_representation(PDT1)
    @test degree(adj1) == 18
    @test length(adj1) == 2

    # G2×F4: dim(G2) = 14, dim(F4) = 52, total = 66
    PDT2 = ProductDynkinType{Tuple{TypeG2,TypeF4}}
    adj2 = adjoint_representation(PDT2)
    @test degree(adj2) == 66

    # A3×A3: dim(A3) = 15 each, total = 30
    PDT3 = ProductDynkinType{Tuple{TypeA{3},TypeA{3}}}
    adj3 = adjoint_representation(PDT3)
    @test degree(adj3) == 30
    @test length(adj3) == 2

    # Instance dispatch works too
    @test degree(adjoint_representation(ProductDynkinType{Tuple{TypeA{2},TypeB{2}}}())) ==
      18
  end
end

@testset "Weylloop ε-basis roundtrip" begin
  # Test that _w2e! followed by _e2w! is the identity for all simple types.
  # Uses deterministic pseudo-random weight vectors to avoid a Random dependency.
  simple_types = [
    TypeA{4}, TypeB{3}, TypeC{3}, TypeD{5},
    TypeE{6}, TypeE{7}, TypeE{8}, TypeF4, TypeG2,
  ]
  for DT in simple_types
    R = rank(DT)
    ED = Semisimple._weylloop_eps_dim(DT)
    e = zeros(Int, ED)
    w2 = zeros(Int, R)
    for trial in 1:100
      w = [((trial * 7 + i * 13) % 11) - 5 for i in 1:R]
      Semisimple._w2e!(DT, e, w)
      Semisimple._e2w!(DT, w2, e)
      @test w2 == w
    end
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Aqua.jl: package quality checks
# ═══════════════════════════════════════════════════════════════════════
@testset "Aqua" begin
  # ambiguities=false: @generated functions can trigger false positives
  # stale_deps=false: Aqua flags itself as stale when run via include() rather than Pkg.test()
  Aqua.test_all(Semisimple; ambiguities=false, stale_deps=false)
end

# ═══════════════════════════════════════════════════════════════════════
#  Coset reps and Bruhat order — extended coverage (item 12)
# ═══════════════════════════════════════════════════════════════════════
@testset "Coset reps extended" begin
  # ─── B2: |W| = 8, parabolic by {1} has 4 cosets ──────────────────
  @testset "B2 right coset reps" begin
    W = weyl_group(TypeB{2})
    reps = right_coset_reps(W, [1])
    @test length(reps) == 4   # |W|/|W_{\{1\}}| = 8/2
    for w in reps
      @test !(1 in right_descent_set(w))
    end
    # Lengths: shortest coset rep has length 0,1,2,3
    lens = sort(length.(reps))
    @test lens == [0, 1, 2, 3]
  end

  @testset "B2 left coset reps" begin
    W = weyl_group(TypeB{2})
    reps = left_coset_reps(W, [2])
    @test length(reps) == 4
    for w in reps
      @test !(2 in left_descent_set(w))
    end
  end

  # ─── A3: |W| = 24, parabolic by {1,2} = S3, cosets = 4 ──────────
  @testset "A3 right coset reps by {1,2}" begin
    W = weyl_group(TypeA{3})
    reps = right_coset_reps(W, [1, 2])
    @test length(reps) == 4   # |S4|/|S3| = 4
    for w in reps
      @test !(1 in right_descent_set(w))
      @test !(2 in right_descent_set(w))
    end
  end

  @testset "A3 right coset reps by {1}" begin
    W = weyl_group(TypeA{3})
    reps = right_coset_reps(W, [1])
    @test length(reps) == 12   # 24/2
    for w in reps
      @test !(1 in right_descent_set(w))
    end
    # Lengths 0..3 each appear at least once (Grassmannian variety)
    lens = Set(length.(reps))
    @test 0 in lens
    @test 3 in lens
  end

  # ─── A3 Bruhat order ────────────────────────────────────────────
  @testset "A3 Bruhat order" begin
    W = weyl_group(TypeA{3})
    e = one(W)
    w0 = longest_element(W)
    s1 = gen(W, 1);
    s2 = gen(W, 2);
    s3 = gen(W, 3)

    # Identity ≤ everything, w0 ≥ everything
    elems = [e, s1, s2, s3, s1 * s2, s2 * s3, s1 * s2 * s3, w0]
    for x in elems
      @test bruhat_leq(e, x)
      @test bruhat_leq(x, w0)
      @test bruhat_leq(x, x)
    end

    # Simple reflections are incomparable
    @test !bruhat_leq(s1, s2)
    @test !bruhat_leq(s2, s1)
    @test !bruhat_leq(s1, s3)

    # Transitivity spot-check
    @test bruhat_leq(s1, s1 * s2)
    @test bruhat_leq(s1 * s2, w0)
    @test bruhat_leq(s1, w0)
  end

  # ─── E6 maximal parabolic (benchmark sanity) ───────────────────
  @testset "E6 right coset reps by {2,3,4,5,6}" begin
    W = weyl_group(TypeE{6})
    # Remove node 1: W/W_{2..6} ≅ E6/P1, |cosets| = dim(E6 27-rep) = 27
    reps = right_coset_reps(W, [2, 3, 4, 5, 6])
    @test length(reps) == 27
    for w in reps
      for i in 2:6
        @test !(i in right_descent_set(w))
      end
    end
  end

  @testset "E8 maximal parabolic by {1,2,3,4,5,6,7}" begin
    W = weyl_group(TypeE{8})
    # Remove node 8 (rightmost leaf): subdiagram is E7.
    # |W(E8)| / |W(E7)| = 696_729_600 / 2_903_040 = 240
    reps = right_coset_reps(W, [1, 2, 3, 4, 5, 6, 7])
    @test length(reps) == 240
    # Every representative has no right descent in I
    for w in reps
      for i in 1:7
        @test !(i in right_descent_set(w))
      end
    end
    # Left coset reps give the same count
    @test length(left_coset_reps(W, [1, 2, 3, 4, 5, 6, 7])) == 240
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Deterministic property tests for character identities (item 18)
# ═══════════════════════════════════════════════════════════════════════
@testset "Character property tests" begin

  # ─── dual∘dual = id ────────────────────────────────────────────────
  @testset "dual involution" begin
    for (DT, R, coords) in [
      (TypeA{2}, 2, [1, 0]),
      (TypeA{2}, 2, [2, 1]),
      (TypeA{3}, 3, [1, 0, 1]),
      (TypeB{3}, 3, [1, 0, 0]),
      (TypeB{3}, 3, [0, 1, 1]),
      (TypeC{3}, 3, [1, 0, 0]),
      (TypeD{4}, 4, [1, 0, 0, 0]),
      (TypeG2, 2, [1, 0]),
      (TypeG2, 2, [0, 1]),
      (TypeF4, 4, [1, 0, 0, 0]),
    ]
      λ = WeightLatticeElem(DT, coords)
      @test dual(dual(λ)) == λ
      V = WeylCharacter(λ)
      @test dual(dual(V)) == V
    end
  end

  # ─── degree(V) = sum(values(freudenthal_formula(λ))) ───────────────
  @testset "degree equals Freudenthal sum" begin
    for (DT, coords) in [
      (TypeA{1}, [5]),
      (TypeA{2}, [1, 0]),
      (TypeA{2}, [2, 1]),
      (TypeA{3}, [1, 0, 0]),
      (TypeA{3}, [1, 1, 0]),
      (TypeB{3}, [0, 0, 1]),
      (TypeC{3}, [1, 0, 0]),
      (TypeD{4}, [0, 1, 0, 0]),
      (TypeG2, [0, 1]),
      (TypeE{6}, [1, 0, 0, 0, 0, 0]),
    ]
      λ = WeightLatticeElem(DT, coords)
      @test degree(λ) == sum(values(freudenthal_formula(λ)))
    end
  end

  # ─── Tensor associativity (U⊗V)⊗W = U⊗(V⊗W) ─────────────────────
  @testset "tensor associativity" begin
    for (DT, a, b, c) in [
      (TypeA{2}, [1, 0], [0, 1], [1, 0]),
      (TypeA{2}, [1, 1], [1, 0], [0, 1]),
      (TypeA{3}, [1, 0, 0], [0, 1, 0], [0, 0, 1]),
      (TypeB{2}, [1, 0], [0, 1], [1, 0]),
      (TypeG2, [1, 0], [0, 1], [1, 0]),
    ]
      U = WeylCharacter(WeightLatticeElem(DT, a))
      V = WeylCharacter(WeightLatticeElem(DT, b))
      W = WeylCharacter(WeightLatticeElem(DT, c))
      @test (U * V) * W == U * (V * W)
    end
  end

  # ─── Sym^k ⊕ ⋀^k dimension identity ─────────────────────────────
  # For V of dim d: Σ_k degree(Sym^k(V)) x^k = 1/(1-x)^d
  # Check that degree(Sym^k(V)) + degree(⋀^k(V)) matches known formula
  @testset "Sym + exterior dimension identity" begin
    ω1_A2 = fundamental_weight(TypeA{2}, 1)  # dim 3
    V = WeylCharacter(ω1_A2)
    # Sym^2(3) = 6, ⋀^2(3) = 3
    @test degree(symmetric_power(V, 2)) == 6
    @test degree(exterior_power(V, 2)) == 3

    ω1_B3 = fundamental_weight(TypeB{3}, 1)  # dim 7
    V7 = WeylCharacter(ω1_B3)
    # Sym^2(7) = 28, ⋀^2(7) = 21
    @test degree(symmetric_power(V7, 2)) == 28
    @test degree(exterior_power(V7, 2)) == 21

    # Full exterior algebra of 7-dim: Σ_k C(7,k) = 128 = 2^7
    total = sum(degree(exterior_power(V7, k)) for k in 0:7)
    @test total == 2^7

    # Adams operator degree: degree(ψ^k(V)) via character = degree(V) (same)
    ψ2 = adams_operator(ω1_A2, 2)
    @test sum(values(ψ2)) == degree(ω1_A2)
  end

  # ─── degree(V::WeylCharacter) works for virtual characters ────────
  @testset "degree of virtual character" begin
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    V1 = WeylCharacter(ω1)
    V2 = WeylCharacter(ω2)
    @test degree(V1 - V2) == 0         # 3 - 3 = 0
    @test degree(V1 - 2 * V2) == -3   # 3 - 6 = -3
    @test degree(2 * V1 + 3 * V2) == 15  # 2*3 + 3*3 = 15

    # Zero character
    @test degree(V1 - V1) == 0

    # A2: V(ω1+ω2) = adjoint (8), V(2ω1) = Sym² (6)
    adj = WeylCharacter(WeightLatticeElem(TypeA{2}, [1, 1]))
    sym2 = WeylCharacter(WeightLatticeElem(TypeA{2}, [2, 0]))
    @test degree(adj - sym2) == 2           # 8 - 6 = 2
    @test degree(sym2 - adj) == -2          # 6 - 8 = -2
    @test degree(adj - sym2 + V1) == 5      # 2 + 3 = 5
    @test degree(3 * adj - 2 * sym2) == 12  # 24 - 12 = 12

    # B2: V(ω1)=5, V(ω2)=4, V(ω1+ω2)=16, V(2ω1)=14
    b2ω1 = fundamental_weight(TypeB{2}, 1)
    b2ω2 = fundamental_weight(TypeB{2}, 2)
    W1 = WeylCharacter(b2ω1)   # dim 5
    W2 = WeylCharacter(b2ω2)   # dim 4
    @test degree(W1 - W2) == 1             # 5 - 4 = 1
    @test degree(W2 - W1) == -1            # 4 - 5 = -1
    @test degree(2 * W1 - 3 * W2) == -2   # 10 - 12 = -2

    # G2: V(ω1)=7, V(ω2)=14
    g2ω1 = fundamental_weight(TypeG2, 1)
    g2ω2 = fundamental_weight(TypeG2, 2)
    G1 = WeylCharacter(g2ω1)   # dim 7
    G2char = WeylCharacter(g2ω2)  # dim 14
    @test degree(G1 - G2char) == -7        # 7 - 14 = -7
    @test degree(2 * G1 - G2char) == 0    # 14 - 14 = 0
    @test degree(3 * G1 - G2char) == 7    # 21 - 14 = 7

    # Newton identity: ψ2(V) = Sym²(V) - ⋀²(V) is a virtual character with
    # degree = dim Sym²(V) - dim ⋀²(V)
    ψ2_raw = adams_operator(ω1, 2)
    ψ2 = character_from_weights(TypeA{2}, ψ2_raw)
    @test degree(ψ2) == degree(Sym(2, ω1)) - degree(⋀(2, ω1))  # 6 - 3 = 3
    @test degree(ψ2) == 3

    # Additivity: degree(V + W) == degree(V) + degree(W)
    for (V, W) in [(V1, V2), (W1, W2), (G1, G2char)]
      @test degree(V + W) == degree(V) + degree(W)
    end

    # Multiplicativity: degree(V * W) == degree(V) * degree(W)
    for (V, W) in [(V1, V2), (W1, W2)]
      @test degree(V * W) == degree(V) * degree(W)
    end
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Product-type character tests (item 21)
# ═══════════════════════════════════════════════════════════════════════
@testset "Product-type characters extended" begin
  # ─── A2 × B2 (rank 4) ────────────────────────────────────────────
  @testset "A2 × B2" begin
    PT = ProductDynkinType{Tuple{TypeA{2},TypeB{2}}}
    ω = [fundamental_weight(PT, i) for i in 1:4]

    # Each fundamental weight lives in one factor
    @test degree(ω[1]) == 3    # A2 standard
    @test degree(ω[2]) == 3    # A2 dual standard
    @test degree(ω[3]) == 5    # B2 vector (= SO(5) standard, ω1 of B2)
    @test degree(ω[4]) == 4    # B2 spinor (= ω2 of B2)

    # Freudenthal formula consistency
    for i in 1:4
      @test degree(ω[i]) == sum(values(freudenthal_formula(ω[i])))
    end

    # tensor product: V(ω1) ⊗ V(ω3) has dim 3×5 = 15
    V1 = WeylCharacter(ω[1]);
    V3 = WeylCharacter(ω[3])
    @test degree(V1 * V3) == 15

    # V(ω1) ⊗ V(ω1) decomposes in A2 factor only
    @test degree(V1 * V1) == 9   # 3⊗3 in A2

    # Sym² and ⋀² of fundamental reps
    @test degree(symmetric_power(ω[1], 2)) == 6
    @test degree(exterior_power(ω[1], 2)) == 3
    @test degree(symmetric_power(ω[3], 2)) == 15  # Sym²(5) for SO(5) = 14-dim + trivial
    @test degree(exterior_power(ω[3], 2)) == 10   # adjoint of B2 = so(5) = 10-dim

    # dual involution
    for i in 1:4
      @test dual(dual(ω[i])) == ω[i]
    end

    # WeylCharacter equality on product type
    @test WeylCharacter(ω[1]) + WeylCharacter(ω[2]) ==
      WeylCharacter(ω[2]) + WeylCharacter(ω[1])

    # degree via WeylCharacter
    for i in 1:4
      @test degree(WeylCharacter(ω[i])) == degree(ω[i])
    end
  end

  # ─── A1 × A1 (simplest product) ─────────────────────────────────
  @testset "A1 × A1" begin
    PT = ProductDynkinType{Tuple{TypeA{1},TypeA{1}}}
    ω1 = fundamental_weight(PT, 1)
    ω2 = fundamental_weight(PT, 2)

    @test degree(ω1) == 2
    @test degree(ω2) == 2

    V1 = WeylCharacter(ω1);
    V2 = WeylCharacter(ω2)

    # (2⊗1) ⊗ (1⊗2) = 4-dim
    @test degree(V1 * V2) == 4

    # Sym²(2⊗1) in product type = Sym²(2) ⊗ 1 = 3⊗1 in product
    # In A1×A1: V(ω1)=2⊗1, Sym²(V(ω1)) = V(2ω1) = 3⊗1, dim=3
    @test degree(symmetric_power(ω1, 2)) == 3

    # Bruhat-like: dimension formula on product
    @test weyl_order(PT) == 4   # Z/2 × Z/2

    # freudenthal on product type
    m = freudenthal_formula(ω1)
    @test sum(values(m)) == 2
  end

  # ─── A2 × A2 × A2 (three factors) ───────────────────────────────
  @testset "A2 × A2 × A2" begin
    PT = ProductDynkinType{Tuple{TypeA{2},TypeA{2},TypeA{2}}}
    ω = [fundamental_weight(PT, i) for i in 1:6]

    # Each A2 has 2 fundamental weights with dim 3
    for i in 1:6
      @test degree(ω[i]) == 3
    end

    # Triple tensor product: V(ω1) ⊗ V(ω3) ⊗ V(ω5) has dim 27
    V1 = WeylCharacter(ω[1]);
    V3 = WeylCharacter(ω[3]);
    V5 = WeylCharacter(ω[5])
    @test degree(V1 * V3 * V5) == 27

    # Weyl order: (3!)^3 = 216
    @test weyl_order(PT) == 216

    # Cartan matrix is block diagonal 6×6
    C = cartan_matrix(PT)
    @test C[1:2, 1:2] == cartan_matrix(TypeA{2})
    @test C[3:4, 3:4] == cartan_matrix(TypeA{2})
    @test C[5:6, 5:6] == cartan_matrix(TypeA{2})
    @test C[1:2, 3:6] == zeros(Int, 2, 4)

    # rank and positive root count
    @test rank(PT) == 6
    @test n_positive_roots(PT) == 9
  end

  # ─── A3 × G2 ─────────────────────────────────────────────────────
  @testset "A3 × G2" begin
    PT = ProductDynkinType{Tuple{TypeA{3},TypeG2}}
    ω = [fundamental_weight(PT, i) for i in 1:5]

    @test degree(ω[1]) == 4    # A3 standard
    @test degree(ω[2]) == 6    # A3 antisym
    @test degree(ω[3]) == 4    # A3 det = dual standard (= 4 since self-dual A3 ω3)
    @test degree(ω[4]) == 7    # G2 standard
    @test degree(ω[5]) == 14   # G2 adjoint

    V4 = WeylCharacter(ω[4])
    V5 = WeylCharacter(ω[5])

    # G2 factor: 7⊗14 decomposes as known
    # In the product type, this is a G2 tensor product tensored with trivial A3
    @test degree(V4 * V5) == 98

    # Freudenthal consistency
    for i in 1:5
      @test degree(ω[i]) == sum(values(freudenthal_formula(ω[i])))
    end

    # dimension: rank + 2*n_pos
    @test dimension(PT) == dimension(TypeA{3}) + dimension(TypeG2)
  end

  # ─── character_from_weights round-trip ──────────────────────────
  @testset "character_from_weights round-trip" begin
    # For each irreducible, reconstruct from freudenthal_formula
    for (DT, coords) in [
      (TypeA{2}, [1, 0]),
      (TypeA{2}, [1, 1]),
      (TypeA{3}, [1, 0, 0]),
      (TypeB{3}, [0, 0, 1]),
      (TypeG2, [1, 0]),
      (TypeE{6}, [1, 0, 0, 0, 0, 0]),   # minuscule: ties in height exist
    ]
      λ = WeightLatticeElem(DT, coords)
      ff = freudenthal_formula(λ)
      V = character_from_weights(DT, ff)
      @test V == WeylCharacter(λ)
    end

    # Multi-irreducible: V(ω1) ⊕ V(ω2) in A2
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    ff1 = freudenthal_formula(ω1)
    ff2 = freudenthal_formula(ω2)
    combined = merge(+, ff1, ff2)
    result = character_from_weights(TypeA{2}, combined)
    @test result == WeylCharacter(ω1) + WeylCharacter(ω2)
  end
end

@testset "parse_dynkin_type" begin
  @test parse_dynkin_type("A3") === TypeA{3}
  @test parse_dynkin_type("B4") === TypeB{4}
  @test parse_dynkin_type("G2") === TypeG2
  @test parse_dynkin_type("F4") === TypeF4
  @test parse_dynkin_type("E6") === TypeE{6}
  @test parse_dynkin_type("A2xB3") === ProductDynkinType{Tuple{TypeA{2},TypeB{3}}}
  @test parse_dynkin_type("A2×B3") === ProductDynkinType{Tuple{TypeA{2},TypeB{3}}}
  @test parse_dynkin_type(" A3 ") === TypeA{3}
  @test parse_dynkin_type("a3") === TypeA{3}

  @testset "malformed input" begin
    @test_throws ArgumentError parse_dynkin_type("")
    @test_throws ArgumentError parse_dynkin_type("   ")
    @test_throws ArgumentError parse_dynkin_type("A")
    @test_throws ArgumentError parse_dynkin_type("3A")
    @test_throws ArgumentError parse_dynkin_type("Z9")
    @test_throws ArgumentError parse_dynkin_type("xx")
  end

  @testset "rank bounds agree with the type system" begin
    # the bounds are check_dynkin_type's, so the parser accepts exactly the
    # labels naming a type this package is willing to work with
    for label in ("A0", "B1", "C1", "D2", "E5", "E9", "F3", "G3")
      @test_throws ArgumentError parse_dynkin_type(label)
    end
    @test parse_dynkin_type("A1") === TypeA{1}
    @test parse_dynkin_type("D3") === TypeD{3}   # the same diagram as A3
  end
end
@testset "Sub-diagrams" begin
  @testset "sub_dynkin_type" begin
    # removing an end node of a path shortens it, a gap splits the diagram
    @test sub_dynkin_type(TypeA{5}, [2, 3, 4, 5]) === TypeA{4}
    @test sub_dynkin_type(TypeA{5}, 1:5) === TypeA{5}
    @test sub_dynkin_type(TypeA{5}, [1, 2, 4, 5]) ===
      ProductDynkinType{Tuple{TypeA{2},TypeA{2}}}
    @test sub_dynkin_type(TypeD{6}, [5, 6]) ===
      ProductDynkinType{Tuple{TypeA{1},TypeA{1}}}
    # the classical Levi factors of E8
    @test sub_dynkin_type(TypeE{8}, [1, 2, 3, 4, 5, 6, 7]) === TypeE{7}
    @test sub_dynkin_type(TypeE{8}, [2, 3, 4, 5, 6, 7, 8]) === TypeD{7}
    @test sub_dynkin_type(TypeE{8}, [1, 3, 4, 5, 6, 7, 8]) === TypeA{7}
    @test sub_dynkin_type(TypeE{6}, [1, 2, 3, 4, 5]) === TypeD{5}
    # B and C are told apart by the direction of the double bond
    @test sub_dynkin_type(TypeB{5}, [2, 3, 4, 5]) === TypeB{4}
    @test sub_dynkin_type(TypeC{5}, [2, 3, 4, 5]) === TypeC{4}
    @test sub_dynkin_type(TypeB{4}, [3, 4]) === TypeB{2}
    @test sub_dynkin_type(TypeC{4}, [3, 4]) === TypeC{2}
    # F4 keeps its double bond in the middle and loses it at the ends
    @test sub_dynkin_type(TypeF4, 1:4) === TypeF4
    @test sub_dynkin_type(TypeF4, [2, 3]) === TypeB{2}
    @test sub_dynkin_type(TypeF4, [1, 2]) === TypeA{2}
    @test sub_dynkin_type(TypeG2, [1, 2]) === TypeG2
    @test sub_dynkin_type(TypeG2, [1]) === TypeA{1}
    # a disconnected ambient diagram
    @test sub_dynkin_type(ProductDynkinType{Tuple{TypeA{2},TypeB{3}}}, [1, 3, 4, 5]) ===
      ProductDynkinType{Tuple{TypeA{1},TypeB{3}}}

    # an instance may stand in for the type
    @test sub_dynkin_type(TypeA{5}(), [2, 3, 4, 5]) === TypeA{4}
    @test sub_dynkin_ordering(TypeD{4}(), [2, 3, 4]) == [3, 2, 4]

    @test_throws ArgumentError sub_dynkin_type(TypeA{3}, Int[])
    @test_throws ArgumentError sub_dynkin_type(TypeA{3}, [1, 1])
    @test_throws ArgumentError sub_dynkin_type(TypeA{3}, [4])
    @test_throws ArgumentError sub_dynkin_type(TypeA{3}, [0])
  end

  @testset "sub_dynkin_ordering" begin
    # the ordering names the vertices in the sub-diagram's own Bourbaki order, so
    # permuting the ambient Cartan matrix by it gives the sub-diagram's Cartan matrix
    for (DT, vertices) in [
      (TypeA{5}, [2, 3, 4, 5]),
      (TypeA{5}, [1, 2, 4, 5]),
      (TypeE{8}, [1, 2, 3, 4, 5, 6, 7]),
      (TypeE{8}, [2, 3, 4, 5, 6, 7, 8]),
      (TypeE{6}, [1, 2, 3, 4, 5]),
      (TypeD{6}, 1:6),
      (TypeD{5}, [2, 3, 4, 5]),
      (TypeD{4}, [2, 3, 4]),
      (TypeF4, 1:4),
      (TypeB{5}, [2, 3, 4, 5]),
      (TypeC{5}, [2, 3, 4, 5]),
      (TypeG2, [1, 2]),
      (TypeD{6}, [5, 6]),
    ]
      sub, ordering = sub_dynkin_type_with_ordering(DT, vertices)
      @test sort(ordering) == sort(collect(vertices))
      ambient = cartan_matrix(DT)
      @test [ambient[a, b] for a in ordering, b in ordering] == cartan_matrix(sub)
    end
  end

  @testset "the whole diagram is a sub-diagram of itself" begin
    for DT in (TypeA{4}, TypeB{4}, TypeC{4}, TypeD{5}, TypeE{6}, TypeF4, TypeG2)
      @test sub_dynkin_type(DT, 1:rank(DT)) === DT
      @test sub_dynkin_ordering(DT, 1:rank(DT)) == collect(1:rank(DT))
    end
  end

  @testset "_classify_cartan_matrix on matrices no sub-diagram produces" begin
    # the classifier is reached only through sub_dynkin_type, so check the inputs
    # that cannot arise that way: a non-square matrix, B and C told apart in
    # isolation rather than as part of a bigger diagram, and a rank-2 component
    # whose bond runs the opposite way to the convention cartan_matrix uses
    @test_throws ArgumentError Semisimple._classify_cartan_matrix([2 -1 0; -1 2 -1])
    @test Semisimple._classify_cartan_matrix([2 -1; -2 2]) == ([(:B, 2)], [1, 2])
    @test Semisimple._classify_cartan_matrix([2 -2; -1 2]) == ([(:C, 2)], [1, 2])
    # G2 either way round; the ordering always names the short-root vertex first
    @test Semisimple._classify_cartan_matrix([2 -3; -1 2]) == ([(:G, 2)], [1, 2])
    @test Semisimple._classify_cartan_matrix([2 -1; -3 2]) == ([(:G, 2)], [2, 1])
  end
end
