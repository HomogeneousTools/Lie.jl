using Test
using Lie
using Aqua
import Lie: borel_weil_bott  # no longer publicly exported; tested here via explicit import
using StaticArrays

# ═══════════════════════════════════════════════════════════════════════
#  Dynkin types
# ═══════════════════════════════════════════════════════════════════════
@testset "Dynkin types" begin
  @test rank(TypeA{3}) == 3
  @test rank(TypeB{4}) == 4
  @test rank(TypeC{5}) == 5
  @test rank(TypeD{6}) == 6
  @test rank(TypeE{6}) == 6
  @test rank(TypeE{7}) == 7
  @test rank(TypeE{8}) == 8
  @test rank(TypeF4) == 4
  @test rank(TypeG2) == 2

  # Product types
  PT = ProductDynkinType{Tuple{TypeA{3},TypeD{5}}}
  @test rank(PT) == 8
  @test n_components(PT) == 2

  PT2 = ProductDynkinType{Tuple{TypeA{3},TypeD{5},TypeE{6}}}
  @test rank(PT2) == 14

  # Invalid types
  @test_throws ArgumentError TypeA{0}()
  @test_throws ArgumentError TypeB{1}()
  @test_throws ArgumentError TypeD{3}()
  @test_throws ArgumentError TypeE{5}()

  # Display
  @test sprint(show, TypeA(3)) == "A3"
  @test sprint(show, TypeG2()) == "G2"
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
  @test dynkin_diagram(TypeD{4}) == "        ○ 4\n       /\n○───○───○\n1   2   3"
  @test dynkin_diagram(TypeD{5}) == "            ○ 5\n           /\n○───○───○───○\n1   2   3   4"

  # E-type: node 2 branches off node 4 (Bourbaki)
  @test dynkin_diagram(TypeE{6}) == "        ○ 2\n        |\n○───○───○───○───○\n1   3   4   5   6"
  @test dynkin_diagram(TypeE{7}) == "        ○ 2\n        |\n○───○───○───○───○───○\n1   3   4   5   6   7"
  @test dynkin_diagram(TypeE{8}) == "        ○ 2\n        |\n○───○───○───○───○───○───○\n1   3   4   5   6   7   8"

  # F₄ and G₂
  @test dynkin_diagram(TypeF4) == "○───○═>═○───○\n1   2   3   4"
  @test dynkin_diagram(TypeG2) == "○≡≡≡○\n1   2"

  # Product type: labelled sections separated by blank lines
  PT = ProductDynkinType{Tuple{TypeA{1},TypeB{2}}}()
  @test dynkin_diagram(PT) == "A1:\n○\n1\n\nB2:\n○═>═○\n1   2"

  # Instance dispatch matches type dispatch
  @test dynkin_diagram(TypeA{3}()) == dynkin_diagram(TypeA{3})
  @test dynkin_diagram(TypeE{6}()) == dynkin_diagram(TypeE{6})
end
@testset "Cartan matrices" begin
  # A₂
  C_A2 = cartan_matrix(TypeA{2})
  @test C_A2 == [2 -1; -1 2]

  # B₂: C[2,1] = -2
  C_B2 = cartan_matrix(TypeB{2})
  @test C_B2 == [2 -1; -2 2]

  # C₂: C[1,2] = -2
  C_C2 = cartan_matrix(TypeC{2})
  @test C_C2 == [2 -2; -1 2]

  # B₃
  C_B3 = cartan_matrix(TypeB{3})
  @test C_B3 == [2 -1 0; -1 2 -1; 0 -2 2]

  # G₂
  C_G2 = cartan_matrix(TypeG2)
  @test C_G2 == [2 -3; -1 2]

  # F₄  (Bourbaki: 1 - 2 >=> 3 - 4, so C[3,2] = -2)
  C_F4 = cartan_matrix(TypeF4)
  @test C_F4 == [2 -1 0 0; -1 2 -1 0; 0 -2 2 -1; 0 0 -1 2]

  # A₁ (simplest case)
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
  @test coefficients(hr) == [1, 1]  # α₁ + α₂

  RS_B2 = RootSystem(TypeB{2})
  hr_B2 = highest_root(RS_B2)
  @test height(hr_B2) == sum(coefficients(hr_B2))

  # Highest short root
  # For B₂: short roots have length 1 (±eᵢ); highest is e₁ = α₁+α₂
  hsr_B2 = highest_short_root(RS_B2)
  @test coefficients(hsr_B2) == [1, 1]  # B₂ highest short root = α₁+α₂

  # For A₂ (simply-laced): highest short root = highest root
  hsr_A2 = highest_short_root(RS_A2)
  @test coefficients(hsr_A2) == coefficients(hr)

  # For G₂: short roots have length² = 2; highest is 2α₁+α₂
  RS_G2 = RootSystem(TypeG2)
  hsr_G2 = highest_short_root(RS_G2)
  @test coefficients(hsr_G2) == [2, 1]

  # Highest coroot and Coxeter coefficients
  hcr_A2 = highest_coroot(RS_A2)
  @test coefficients(hcr_A2) == [1, 1]  # Same coroot structure for A₂

  c_A2 = coxeter_coefficients(TypeA{2})
  @test c_A2 == [1, 1]
  @test length(c_A2) == rank(TypeA{2})

  c_A3 = coxeter_coefficients(TypeA{3})
  @test c_A3 == [1, 1, 1]  # Simply-laced: all 1s

  c_B2 = coxeter_coefficients(TypeB{2})
  @test c_B2 == [1, 2]  # Type B: [1, 2, ..., 2]

  c_G2 = coxeter_coefficients(TypeG2)
  @test c_G2 == [3, 2]  # G₂ Coxeter coefficients

  # Dual Coxeter coefficients: highest short root of the dual root system
  dc_B2 = dual_coxeter_coefficients(TypeB{2})
  @test dc_B2 == [1, 1]  # B₂∨=C₂; highest short root of C₂ has coefficients [1,1]

  dc_C2 = dual_coxeter_coefficients(TypeC{2})
  @test dc_C2 == [1, 1]  # C₂∨=B₂; highest short root of B₂ (=e₁) has coefficients [1,1]

  dc_G2 = dual_coxeter_coefficients(TypeG2)
  @test dc_G2 == [1, 2]  # G₂ self-dual; highest short root of G₂∨ has coefficients [1,2]

  dc_F4 = dual_coxeter_coefficients(TypeF4)
  @test dc_F4 == [1, 2, 3, 2]  # F₄ self-dual; highest short root coefficients [1,2,3,2]

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
  @test w_α1 == WeightLatticeElem(DT, [2, -1])  # α₁ = 2ω₁ - ω₂
  @test_throws ArgumentError RootSpaceElem(ω1)

  # Reflection
  w = WeightLatticeElem(DT, [2, 1])
  w_reflected = reflect(w, 1)
  @test w_reflected == WeightLatticeElem(DT, [-2, 3])
  # s₁(2ω₁ + ω₂) = (2ω₁ + ω₂) - 2*(α₁) = (2ω₁ + ω₂) - 2*(2ω₁ - ω₂) = -2ω₁ + 3ω₂
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

  # Compact show: explicit IOContext(:compact => true) → "DT[c₁,c₂,…]"
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
end

# ═══════════════════════════════════════════════════════════════════════
#  Weyl group
# ═══════════════════════════════════════════════════════════════════════
@testset "Weyl group" begin
  @testset "A₂" begin
    W = weyl_group(TypeA{2})
    RS = root_system(W)
    s = gens(W)

    # Simple reflections are involutions
    @test s[1] * s[1] == one(W)
    @test s[2] * s[2] == one(W)

    # Order of W(A₂) = 6
    @test weyl_order(TypeA{2}) == 6

    # Longest element
    w0 = longest_element(W)
    @test length(w0) == n_positive_roots(RS)  # length = # positive roots = 3

    # w₀² = 1
    @test w0 * w0 == one(W)

    # Action on weights: w₀(ρ) = -ρ
    ρ = weyl_vector(TypeA{2})
    @test ρ * w0 == -ρ

    # Action on roots
    α1 = simple_root(RS, 1)
    α2 = simple_root(RS, 2)
    @test α1 * s[1] == -α1
    @test α1 * s[2] == α1 + α2  # s₂(α₁) = α₁ + α₂ in type A₂
  end

  @testset "B₂" begin
    W = weyl_group(TypeB{2})
    RS = root_system(W)
    w0 = longest_element(W)

    @test weyl_order(TypeB{2}) == 8
    @test length(w0) == n_positive_roots(RS)
    @test w0 * w0 == one(W)

    ρ = weyl_vector(TypeB{2})
    @test ρ * w0 == -ρ
  end

  @testset "G₂" begin
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
    @test length(orb) == 3  # Orbit of ω₁ in A₂ has 3 elements

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
  # A₁: dim V(nω₁) = n+1
  for n in 1:10
    hw = WeightLatticeElem(TypeA{1}, [n])
    @test degree(hw) == n + 1
  end

  # A₂: dim V(ω₁) = 3 (standard rep)
  @test degree(fundamental_weight(TypeA{2}, 1)) == 3

  # A₂: dim V(ω₂) = 3 (dual standard rep)
  @test degree(fundamental_weight(TypeA{2}, 2)) == 3

  # A₂: dim V(ω₁ + ω₂) = 8 (adjoint rep)
  @test degree(WeightLatticeElem(TypeA{2}, [1, 1])) == 8

  # A₃: dim V(ω₁) = 4
  @test degree(fundamental_weight(TypeA{3}, 1)) == 4

  # A₃: dim V(ω₂) = 6 (exterior square)
  @test degree(fundamental_weight(TypeA{3}, 2)) == 6

  # B₂: dim V(ω₁) = 5 (standard rep of SO(5))
  @test degree(fundamental_weight(TypeB{2}, 1)) == 5

  # B₂: dim V(ω₂) = 4 (spin rep)
  @test degree(fundamental_weight(TypeB{2}, 2)) == 4

  # B₃: dim V(ω₁) = 7 (standard rep of SO(7))
  @test degree(fundamental_weight(TypeB{3}, 1)) == 7

  # C₂: dim V(ω₁) = 4 (standard rep of Sp(4))
  @test degree(fundamental_weight(TypeC{2}, 1)) == 4

  # G₂: dim V(ω₁) = 7 (standard rep)
  @test degree(fundamental_weight(TypeG2, 1)) == 7

  # G₂: dim V(ω₂) = 14 (adjoint rep)
  @test degree(fundamental_weight(TypeG2, 2)) == 14

  # D₄: dim V(ω₁) = 8 (standard rep of SO(8))
  @test degree(fundamental_weight(TypeD{4}, 1)) == 8

  # Higher-rank cached Weyl-dimension data still uses the same public formula.
  @test degree(fundamental_weight(TypeA{10}, 1)) == 11
  @test degree(fundamental_weight(TypeA{17}, 1)) == 18

  # E₈: all fundamental representations
  @testset "E₈ fundamental representations" begin
    expected_dims = [3875, 147250, 6696000, 6899079264,
      146325270, 2450240, 30380, 248]
    for (i, expected) in enumerate(expected_dims)
      @test degree(fundamental_weight(TypeE{8}, i)) == expected
    end
  end

  # E₈: high-dimensional representation 3ω₃ + 5ω₈
  @test degree(WeightLatticeElem(TypeE{8}, [0, 0, 3, 0, 0, 0, 0, 5])) ==
    big"18190674254761844256000000"

  # Synonyms
  @test weyl_dimension(fundamental_weight(TypeA{2}, 1)) == 3
end

# ═══════════════════════════════════════════════════════════════════════
#  Dominant weights
# ═══════════════════════════════════════════════════════════════════════
@testset "Dominant weights" begin
  # A₂, hw = ω₁ + ω₂: adjoint rep has 2 dominant weights
  hw = WeightLatticeElem(TypeA{2}, [1, 1])
  dw = dominant_weights(hw)
  @test length(dw) == 2
  @test hw in dw
  @test WeightLatticeElem(TypeA{2}, [0, 0]) in dw

  # A₁, hw = 3ω₁: dominant weights are 3ω, ω
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

@testset "Borel–Weil–Bott" begin
  # ── A₂ ──────────────────────────────────────────────────────────────
  # Dominant weight: degree 0, representation is itself
  @testset "A₂" begin
    ω1 = fundamental_weight(TypeA{2}, 1)
    ω2 = fundamental_weight(TypeA{2}, 2)
    ρ = weyl_vector(TypeA{2})

    # ω₁ is dominant: H⁰ = V(ω₁), dim = 3
    result = borel_weil_bott(ω1)
    @test result !== nothing
    d, μ = result
    @test d == 0
    @test μ == ω1

    # ω₂ is dominant: H⁰ = V(ω₂), dim = 3
    result = borel_weil_bott(ω2)
    @test result !== nothing
    d, μ = result
    @test d == 0
    @test μ == ω2

    # λ = -ρ: λ + ρ = 0, singular → nothing
    @test borel_weil_bott(-ρ) === nothing

    # λ = [-2, 1]: λ+ρ = [-1, 2], s₁ gives [1, 1], d=1, μ = [0, 0]
    λ = WeightLatticeElem(TypeA{2}, [-2, 1])
    result = borel_weil_bott(λ)
    @test result !== nothing
    d, μ = result
    @test d == 1
    @test μ == WeightLatticeElem(TypeA{2}, [0, 0])  # trivial rep

    # λ = [-3, 3]: λ+ρ = [-2, 4], s₁ gives [2, 2], d=1, μ = [1, 1]
    λ = WeightLatticeElem(TypeA{2}, [-3, 3])
    result = borel_weil_bott(λ)
    @test result !== nothing
    d, μ = result
    @test d == 1
    @test μ == WeightLatticeElem(TypeA{2}, [1, 1])  # adjoint rep

    # λ = [-3, 1]: λ+ρ = [-2, 2], conjugates to singular weight [2, 0]
    @test borel_weil_bott(WeightLatticeElem(TypeA{2}, [-3, 1])) === nothing
  end

  # ── A₁ ──────────────────────────────────────────────────────────────
  @testset "A₁" begin
    # nω₁ dominant: degree 0, result is nω₁
    for n in 0:5
      result = borel_weil_bott(WeightLatticeElem(TypeA{1}, [n]))
      @test result !== nothing
      d, μ = result
      @test d == 0
      @test μ == WeightLatticeElem(TypeA{1}, [n])
    end

    # λ = -1: λ+ρ = 0, singular
    @test borel_weil_bott(WeightLatticeElem(TypeA{1}, [-1])) === nothing

    # λ = -3: λ+ρ = -2, s₁ → 2, dominant, d=1, μ = 2-1 = 1
    result = borel_weil_bott(WeightLatticeElem(TypeA{1}, [-3]))
    @test result !== nothing
    d, μ = result
    @test d == 1
    @test μ == WeightLatticeElem(TypeA{1}, [1])
  end

  # ── B₂ ──────────────────────────────────────────────────────────────
  @testset "B₂" begin
    # Dominant weight: degree 0
    ω1 = fundamental_weight(TypeB{2}, 1)
    result = borel_weil_bott(ω1)
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
        result = borel_weil_bott(ωi)
        @test result !== nothing
        d, μ = result
        @test d == 0
        @test μ == ωi
      end
    end
  end
  # ── E₈ example ──────────────────────────────────────────────────────────────────────────
  @testset "E₈" begin
    λ = WeightLatticeElem(TypeE{8}, [-5, 3, -2, -3, 5, -8, 2, 1])
    # λ+ρ conjugates to a singular weight, so all cohomology vanishes
    @test borel_weil_bott(λ) === nothing
  end
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
  @test n_positive_roots(PT) == 3 + 9  # A₂ has 3, B₃ has 9

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
    ω₁ = fundamental_weight(TypeA{2}, 1)
    ω₂ = fundamental_weight(TypeA{2}, 2)
    V1 = WeylCharacter(ω₁)
    V2 = WeylCharacter(ω₂)

    @test is_effective(V1)
    @test is_irreducible(V1)
    @test highest_weight(V1) == ω₁
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
    ω₁ = fundamental_weight(TypeA{2}, 1)
    ω₂ = fundamental_weight(TypeA{2}, 2)

    # add! is equivalent to +
    V = WeylCharacter(ω₁)
    W = WeylCharacter(ω₂)
    expected = V + W
    add!(V, W)
    @test V == expected

    # add! with self-cancellation
    V2 = WeylCharacter(ω₁)
    add!(V2, -WeylCharacter(ω₁))
    @test iszero(V2)

    # addmul! basic
    V3 = WeylCharacter(TypeA{2})
    addmul!(V3, WeylCharacter(ω₁), 5)
    @test V3 == 5 * WeylCharacter(ω₁)

    # addmul! with negative coefficient
    V4 = WeylCharacter(ω₁) + WeylCharacter(ω₂)
    addmul!(V4, WeylCharacter(ω₁), -1)
    @test V4 == WeylCharacter(ω₂)

    # addmul! with c=0 is identity
    V5 = WeylCharacter(ω₁)
    addmul!(V5, WeylCharacter(ω₂), 0)
    @test V5 == WeylCharacter(ω₁)

    # add! returns the modified object
    V6 = WeylCharacter(ω₁)
    @test add!(V6, WeylCharacter(ω₂)) === V6

    # addmul! returns the modified object
    V7 = WeylCharacter(ω₁)
    @test addmul!(V7, WeylCharacter(ω₂), 2) === V7
  end

  # ─── Dominant character ─────────────────────────────────────────
  @testset "Dominant character" begin
    # A₂ standard: V(ω₁) has dim 3, only 1 dominant weight (ω₁ itself)
    dc = dominant_character(fundamental_weight(TypeA{2}, 1))
    @test length(dc) == 1
    @test dc[SVector(1, 0)] == 1

    # A₂ adjoint: V(ω₁+ω₂) has dim 8, dominant weights are ω₁+ω₂ and 0
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

    # E₈ adjoint: V(ω₈) dim 248
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

    # Higher weight: A₃ V(ρ) dim 20
    ρ = weyl_vector(TypeA{3})
    dc_rho = dominant_character(ρ)
    full_rho = freudenthal_formula(ρ)
    for (μ_vec, m) in dc_rho
      @test full_rho[μ_vec] == m
    end
  end

  # ─── Freudenthal formula: simply-laced ───────────────────────────
  @testset "Freudenthal (simply-laced)" begin
    # A₂ standard: dim 3
    m = freudenthal_formula(fundamental_weight(TypeA{2}, 1))
    @test sum(values(m)) == 3
    @test all(v == 1 for v in values(m))  # all multiplicities 1

    # A₂ adjoint: dim 8, with zero weight multiplicity 2
    m_adj = freudenthal_formula(
      fundamental_weight(TypeA{2}, 1) + fundamental_weight(TypeA{2}, 2)
    )
    @test sum(values(m_adj)) == 8
    @test m_adj[SVector(0, 0)] == 2  # zero weight has multiplicity 2

    # D₄ fundamental: dim 8
    m_d4 = freudenthal_formula(fundamental_weight(TypeD{4}, 1))
    @test sum(values(m_d4)) == 8

    # E₆ fundamental ω₁: dim 27
    m_e6 = freudenthal_formula(fundamental_weight(TypeE{6}, 1))
    @test sum(values(m_e6)) == 27

    # E₈ fundamental ω₈: dim 248
    m_e8 = freudenthal_formula(fundamental_weight(TypeE{8}, 8))
    @test sum(values(m_e8)) == 248
  end

  # ─── Freudenthal formula: non-simply-laced ───────────────────────
  @testset "Freudenthal (non-simply-laced)" begin
    # B₂: std (dim 5), spin (dim 4)
    @test sum(values(freudenthal_formula(fundamental_weight(TypeB{2}, 1)))) == 5
    @test sum(values(freudenthal_formula(fundamental_weight(TypeB{2}, 2)))) == 4

    # B₃: std (dim 7), spin (dim 8)
    @test sum(values(freudenthal_formula(fundamental_weight(TypeB{3}, 1)))) == 7
    @test sum(values(freudenthal_formula(fundamental_weight(TypeB{3}, 3)))) == 8

    # C₃: std (dim 6)
    @test sum(values(freudenthal_formula(fundamental_weight(TypeC{3}, 1)))) == 6

    # G₂: 7-dim and 14-dim
    @test sum(values(freudenthal_formula(fundamental_weight(TypeG2, 1)))) == 7
    @test sum(values(freudenthal_formula(fundamental_weight(TypeG2, 2)))) == 14

    # F₄: 52-dim and 26-dim
    @test sum(values(freudenthal_formula(fundamental_weight(TypeF4, 1)))) == 52
    @test sum(values(freudenthal_formula(fundamental_weight(TypeF4, 4)))) == 26
  end

  # ─── Tensor products ─────────────────────────────────────────────
  @testset "Tensor products" begin
    # A₂: V(ω₁) ⊗ V(ω₁) = V(2ω₁) + V(ω₂)
    ω₁ = fundamental_weight(TypeA{2}, 1)
    ω₂ = fundamental_weight(TypeA{2}, 2)
    V₁ = WeylCharacter(ω₁)
    tp = V₁ * V₁
    @test tp == WeylCharacter(2 * ω₁) + WeylCharacter(ω₂)

    # A₂: V(ω₁) ⊗ V(ω₂) = V(ω₁+ω₂) + V(0)
    tp2 = V₁ * WeylCharacter(ω₂)
    @test tp2 ==
      WeylCharacter(ω₁ + ω₂) +
          WeylCharacter(WeightLatticeElem(TypeA{2}, SVector(0, 0)))

    # B₂: V(ω₁) ⊗ V(ω₁) = V(2ω₁) + V(ω₂) + V(0) (dims: 25 = 14+10+1)
    ω₁_b = fundamental_weight(TypeB{2}, 1)
    ω₂_b = fundamental_weight(TypeB{2}, 2)
    tp_b = WeylCharacter(ω₁_b) * WeylCharacter(ω₁_b)
    @test tp_b ==
      WeylCharacter(2 * ω₁_b) +
          WeylCharacter(WeightLatticeElem(TypeB{2}, SVector(0, 2))) +
          WeylCharacter(WeightLatticeElem(TypeB{2}, SVector(0, 0)))

    # Dimension check: tensor product preserves dimension
    @test sum(degree(k) * v for (k, v) in tp.terms) == 9

    # Tensor product of virtual (non-effective) characters
    # V(ω₁) - V(ω₂) tensored with V(ω₁):
    # = V(ω₁) ⊗ V(ω₁) - V(ω₂) ⊗ V(ω₁)
    # = [V(2ω₁) + V(ω₂)] - [V(ω₁+ω₂) + V(0)]
    virtual = WeylCharacter(ω₁) - WeylCharacter(ω₂)
    @test !is_effective(virtual)
    tp_virt = virtual * WeylCharacter(ω₁)
    z = WeightLatticeElem(TypeA{2}, SVector(0, 0))
    expected_virt =
      WeylCharacter(2 * ω₁) + WeylCharacter(ω₂) - WeylCharacter(ω₁ + ω₂) -
      WeylCharacter(z)
    @test tp_virt == expected_virt
  end

  # ─── Littlewood–Richardson rule ──────────────────────────────────
  @testset "Littlewood–Richardson rule" begin
    # Verify LR matches Brauer–Klimyk for all Type A tests

    # Helper: compute tensor product via Brauer–Klimyk only
    function bk_tensor(λ, μ)
      if Lie.degree(λ) > Lie.degree(μ)
        Lie.brauer_klimyk(Lie.freudenthal_formula(μ), λ)
      else
        Lie.brauer_klimyk(Lie.freudenthal_formula(λ), μ)
      end
    end

    # A₁: simplest case
    ω₁_a1 = fundamental_weight(TypeA{1}, 1)
    @test lr_tensor_product(ω₁_a1, ω₁_a1) == bk_tensor(ω₁_a1, ω₁_a1)
    @test lr_tensor_product(2ω₁_a1, ω₁_a1) == bk_tensor(2ω₁_a1, ω₁_a1)
    @test lr_tensor_product(3ω₁_a1, 2ω₁_a1) == bk_tensor(3ω₁_a1, 2ω₁_a1)

    # A₂: comprehensive tests
    ω₁ = fundamental_weight(TypeA{2}, 1)
    ω₂ = fundamental_weight(TypeA{2}, 2)
    z = WeightLatticeElem(TypeA{2}, SVector(0, 0))

    @test lr_tensor_product(ω₁, ω₁) == WeylCharacter(2ω₁) + WeylCharacter(ω₂)
    @test lr_tensor_product(ω₁, ω₂) == WeylCharacter(ω₁ + ω₂) + WeylCharacter(z)
    @test lr_tensor_product(ω₂, ω₂) == WeylCharacter(2ω₂) + WeylCharacter(ω₁)
    @test lr_tensor_product(ω₁ + ω₂, ω₁) == bk_tensor(ω₁ + ω₂, ω₁)
    @test lr_tensor_product(ω₁ + ω₂, ω₂) == bk_tensor(ω₁ + ω₂, ω₂)
    @test lr_tensor_product(ω₁ + ω₂, ω₁ + ω₂) == bk_tensor(ω₁ + ω₂, ω₁ + ω₂)
    @test lr_tensor_product(2ω₁, ω₁) == bk_tensor(2ω₁, ω₁)
    @test lr_tensor_product(2ω₁, ω₂) == bk_tensor(2ω₁, ω₂)
    @test lr_tensor_product(2ω₁, 2ω₁) == bk_tensor(2ω₁, 2ω₁)
    @test lr_tensor_product(3ω₁, ω₂) == bk_tensor(3ω₁, ω₂)

    # Edge case: tensor with trivial
    @test lr_tensor_product(ω₁, z) == WeylCharacter(ω₁)
    @test lr_tensor_product(z, ω₁) == WeylCharacter(ω₁)
    @test lr_tensor_product(z, z) == WeylCharacter(z)

    # A₃: tests
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

    # A₄: tests
    ω4 = [fundamental_weight(TypeA{4}, i) for i in 1:4]
    @test lr_tensor_product(ω4[1], ω4[1]) == bk_tensor(ω4[1], ω4[1])
    @test lr_tensor_product(ω4[1], ω4[4]) == bk_tensor(ω4[1], ω4[4])
    @test lr_tensor_product(ω4[2], ω4[2]) == bk_tensor(ω4[2], ω4[2])
    @test lr_tensor_product(ω4[2], ω4[3]) == bk_tensor(ω4[2], ω4[3])

    # A₅: tests
    ω5 = [fundamental_weight(TypeA{5}, i) for i in 1:5]
    @test lr_tensor_product(ω5[1], ω5[1]) == bk_tensor(ω5[1], ω5[1])
    @test lr_tensor_product(ω5[1], ω5[5]) == bk_tensor(ω5[1], ω5[5])
    @test lr_tensor_product(ω5[2], ω5[2]) == bk_tensor(ω5[2], ω5[2])
    @test lr_tensor_product(2ω5[1], ω5[1]) == bk_tensor(2ω5[1], ω5[1])
    @test lr_tensor_product(2ω5[1], 2ω5[1]) == bk_tensor(2ω5[1], 2ω5[1])

    # A₇: higher rank
    ω7 = [fundamental_weight(TypeA{7}, i) for i in 1:7]
    @test lr_tensor_product(ω7[1], ω7[1]) == bk_tensor(ω7[1], ω7[1])
    @test lr_tensor_product(ω7[1], ω7[7]) == bk_tensor(ω7[1], ω7[7])
    @test lr_tensor_product(ω7[2], ω7[2]) == bk_tensor(ω7[2], ω7[2])

    # Dimension consistency: tensor product dimension = dim(V) * dim(W)
    for (λ, μ) in [(ω₁, ω₁), (ω₁, ω₂), (ω₁ + ω₂, ω₁),
      (ω[1], ω[2]), (ω4[2], ω4[3])]
      result = lr_tensor_product(λ, μ)
      dim_sum = sum(Lie.degree(k) * v for (k, v) in result.terms)
      @test dim_sum == Lie.degree(λ) * Lie.degree(μ)
    end

    # Commutativity: LR(λ, μ) == LR(μ, λ)
    @test lr_tensor_product(ω₁, ω₂) == lr_tensor_product(ω₂, ω₁)
    @test lr_tensor_product(ω[1], ω[3]) == lr_tensor_product(ω[3], ω[1])
    @test lr_tensor_product(2ω₁, ω₂) == lr_tensor_product(ω₂, 2ω₁)

    # tensor_product dispatches to LR for TypeA
    empty!(Lie._tensor_cache)
    tp_dispatch = tensor_product(ω₁, ω₂)
    @test tp_dispatch == lr_tensor_product(ω₁, ω₂)
  end

  # ─── Dual ────────────────────────────────────────────────────────
  @testset "Dual" begin
    # A₂: dual(ω₁) = ω₂ (A₂ has non-trivial outer automorphism)
    ω₁ = fundamental_weight(TypeA{2}, 1)
    ω₂ = fundamental_weight(TypeA{2}, 2)
    @test dual(ω₁) == ω₂
    @test dual(ω₂) == ω₁

    # B₂: dual = identity (all reps self-dual)
    ω₁_b = fundamental_weight(TypeB{2}, 1)
    @test dual(ω₁_b) == ω₁_b

    # Dual of virtual character
    V = WeylCharacter(ω₁)
    @test highest_weight(dual(V)) == ω₂
  end

  # ─── Exterior powers ─────────────────────────────────────────────
  @testset "Exterior powers" begin
    # A₂: ⋀²V(ω₁) = V(ω₂)
    ω₁ = fundamental_weight(TypeA{2}, 1)
    ω₂ = fundamental_weight(TypeA{2}, 2)
    @test ⋀(2, ω₁) == WeylCharacter(ω₂)
    @test ⋀(3, ω₁) == WeylCharacter(WeightLatticeElem(TypeA{2}, SVector(0, 0)))

    # A₃: ⋀²V(ω₁) = V(ω₂), ⋀³V(ω₁) = V(ω₃)
    ω₁_a3 = fundamental_weight(TypeA{3}, 1)
    ω₂_a3 = fundamental_weight(TypeA{3}, 2)
    ω₃_a3 = fundamental_weight(TypeA{3}, 3)
    @test ⋀(2, ω₁_a3) == WeylCharacter(ω₂_a3)
    @test ⋀(3, ω₁_a3) == WeylCharacter(ω₃_a3)

    # A₃: ⋀⁴V(ω₁) = trivial (top exterior power of std rep)
    z_a3 = WeightLatticeElem(TypeA{3}, SVector(0, 0, 0))
    @test ⋀(4, ω₁_a3) == WeylCharacter(z_a3)

    # A₃: ⋀ᵏV(ω₁) = 0 for k > dim = 4
    @test ⋀(5, ω₁_a3) == WeylCharacter(TypeA{3})

    # E₈: ⋀²V(ω₁) has 4 irreducible components
    ω₁_e8 = fundamental_weight(TypeE{8}, 1)
    r = ⋀(2, ω₁_e8)
    @test length(r.terms) == 4
    @test is_effective(r)

    # ─── Dimension identity: dim ⋀ᵏV = C(dim V, k) ─────────────
    # A₄: V(ω₁) has dim 5, so ⋀ᵏ has dim C(5,k)
    ω₁_a4 = fundamental_weight(TypeA{4}, 1)
    for k in 0:5
      r = ⋀(k, ω₁_a4)
      @test sum(m * degree(μ) for (μ, m) in r.terms; init=0) == binomial(5, k)
    end

    # B₃: V(ω₃) is 8-dimensional spin rep
    ω₃_b3 = fundamental_weight(TypeB{3}, 3)
    d = degree(ω₃_b3)
    for k in 1:3
      r = ⋀(k, ω₃_b3)
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
    # A₅: ⋀³V(ω₁) = V(ω₃)  (fundamental rep)
    ω₁_a5 = fundamental_weight(TypeA{5}, 1)
    ω₃_a5 = fundamental_weight(TypeA{5}, 3)
    @test ⋀(3, ω₁_a5) == WeylCharacter(ω₃_a5)

    # A₇: ⋀⁴V(ω₁) = V(ω₄)
    ω₁_a7 = fundamental_weight(TypeA{7}, 1)
    ω₄_a7 = fundamental_weight(TypeA{7}, 4)
    @test ⋀(4, ω₁_a7) == WeylCharacter(ω₄_a7)

    # D₄: ⋀²V(ω₁) has specific structure
    ω₁_d4 = fundamental_weight(TypeD{4}, 1)
    r_d4 = ⋀(2, ω₁_d4)
    @test is_effective(r_d4)
    @test sum(m * degree(μ) for (μ, m) in r_d4.terms) == binomial(8, 2)

    # G₂: dim ⋀ᵏV(ω₁) = C(7,k) (7-dim rep)
    ω₁_g2 = fundamental_weight(TypeG2, 1)
    for k in 2:4
      r = ⋀(k, ω₁_g2)
      @test is_effective(r)
      @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(7, k)
    end

    # ─── Non-minuscule exterior powers ───────────────────────────
    # A₃: ⋀²V(ω₁+ω₃) — adjoint rep (15-dim)
    ω₁_a3 = fundamental_weight(TypeA{3}, 1)
    ω₃_a3 = fundamental_weight(TypeA{3}, 3)
    adj = ω₁_a3 + ω₃_a3
    r_adj = ⋀(2, adj)
    @test is_effective(r_adj)
    @test sum(m * degree(μ) for (μ, m) in r_adj.terms) == binomial(15, 2)

    # ─── WeylCharacter overloads ──────────────────────────────────
    # Delegation: ⋀(k, V) == ⋀(k, λ) when V is irreducible
    ω₁_v = fundamental_weight(TypeA{3}, 1)
    V_ext = WeylCharacter(ω₁_v)
    @test ⋀(2, V_ext) == ⋀(2, ω₁_v)
    @test ⋀(3, V_ext) == ⋀(3, ω₁_v)
    @test ⋀(4, V_ext) == ⋀(4, ω₁_v)
    @test ⋀(5, V_ext) == ⋀(5, ω₁_v)

    # Newton identity via WeylCharacter
    for λ in [
      fundamental_weight(TypeA{3}, 1),
      fundamental_weight(TypeB{3}, 1),
      fundamental_weight(TypeG2, 1),
    ]
      Vλ = WeylCharacter(λ)
      @test Vλ * Vλ == Sym(2, Vλ) + ⋀(2, Vλ)
    end

    # Reducible character: ⋀²(V₁ ⊕ V₂) = ⋀²V₁ ⊕ (V₁ ⊗ V₂) ⊕ ⋀²V₂
    ω₂_v = fundamental_weight(TypeA{3}, 2)
    V1 = WeylCharacter(ω₁_v)
    V2 = WeylCharacter(ω₂_v)
    @test ⋀(2, V1 + V2) == ⋀(2, V1) + V1 * V2 + ⋀(2, V2)
  end

  # ─── Symmetric powers ───────────────────────────────────────────
  @testset "Symmetric powers" begin
    # A₂: Sym²V(ω₁) = V(2ω₁)
    ω₁ = fundamental_weight(TypeA{2}, 1)
    @test Sym(2, ω₁) == WeylCharacter(2 * ω₁)

    # A₂: Sym³V(ω₁) = V(3ω₁)
    @test Sym(3, ω₁) == WeylCharacter(3 * ω₁)

    # Sym⁰ = trivial, Sym¹ = identity
    z = WeightLatticeElem(TypeA{2}, SVector(0, 0))
    @test Sym(0, ω₁) == WeylCharacter(z)
    @test Sym(1, ω₁) == WeylCharacter(ω₁)

    # ─── Type A: SymᵏV(ω₁) = V(kω₁) (always irreducible) ──────
    for (DT, k_max) in [(TypeA{2}, 5), (TypeA{3}, 4), (TypeA{5}, 3)]
      ω₁ = fundamental_weight(DT, 1)
      for k in 2:k_max
        @test Sym(k, ω₁) == WeylCharacter(k * ω₁)
      end
    end

    # ─── Dimension identity: dim Symᵏ(V) = C(dim V + k - 1, k) ─
    # A₃: dim V(ω₁) = 4, so dim Symᵏ = C(4+k-1, k)
    ω₁_a3 = fundamental_weight(TypeA{3}, 1)
    for k in 2:5
      r = Sym(k, ω₁_a3)
      @test is_effective(r)
      @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(4 + k - 1, k)
    end

    # B₂: dim V(ω₁) = 5, dim Symᵏ = C(5+k-1, k)
    ω₁_b2 = fundamental_weight(TypeB{2}, 1)
    for k in 2:4
      r = Sym(k, ω₁_b2)
      @test is_effective(r)
      @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(5 + k - 1, k)
    end

    # ─── Cross-type symmetric powers ────────────────────────────
    # G₂: Sym²V(ω₁) decomposes; verify effectiveness and dimension
    ω₁_g2 = fundamental_weight(TypeG2, 1)
    r = Sym(2, ω₁_g2)
    @test is_effective(r)
    @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(7 + 1, 2)

    # C₃: Sym²V(ω₁) decomposes; verify dimension
    ω₁_c3 = fundamental_weight(TypeC{3}, 1)
    r = Sym(2, ω₁_c3)
    @test is_effective(r)
    @test sum(m * degree(μ) for (μ, m) in r.terms) == binomial(6 + 1, 2)

    # ─── WeylCharacter overloads ──────────────────────────────────
    # Delegation: Sym(k, V) == Sym(k, λ) when V is irreducible
    ω₁_v = fundamental_weight(TypeA{3}, 1)
    V_sym = WeylCharacter(ω₁_v)
    @test Sym(2, V_sym) == Sym(2, ω₁_v)
    @test Sym(3, V_sym) == Sym(3, ω₁_v)

    # Boundary cases
    z3 = WeightLatticeElem(TypeA{3}, SVector(0, 0, 0))
    @test Sym(0, V_sym) == WeylCharacter(z3)
    @test Sym(1, V_sym) == V_sym

    # Reducible character: Sym²(2V) = 3·Sym²V + ⋀²V
    @test Sym(2, 2 * V_sym) == 3 * Sym(2, V_sym) + ⋀(2, V_sym)

    # Reducible: Sym²(V₁ ⊕ V₂) = Sym²V₁ + V₁⊗V₂ + Sym²V₂
    ω₂_v = fundamental_weight(TypeA{3}, 2)
    V1 = WeylCharacter(ω₁_v)
    V2 = WeylCharacter(ω₂_v)
    @test Sym(2, V1 + V2) == Sym(2, V1) + V1 * V2 + Sym(2, V2)
  end

  # ─── Adams operators ─────────────────────────────────────────────
  @testset "Adams operators" begin
    ω₁ = fundamental_weight(TypeA{2}, 1)

    # ψ¹ = the weight multiplicities of V(ω₁)
    ψ1 = adams_operator(ω₁, 1)
    @test ψ1 == freudenthal_formula(ω₁)

    # Newton identity: ψ²(V) as a virtual character = Sym²(V) - ⋀²(V)
    ψ2_raw = adams_operator(ω₁, 2)
    ψ2_char = character_from_weights(TypeA{2}, ψ2_raw)
    @test ψ2_char == Sym(2, ω₁) - ⋀(2, ω₁)
  end

  # ─── E₈ exterior power cross-checks ─────────────────────────────
  @testset "E₈ exterior powers" begin
    ω = [fundamental_weight(TypeE{8}, i) for i in 1:8]

    # ⋀²V(ω₁): 4 irreducibles
    r1 = ⋀(2, ω[1])
    @test length(r1.terms) == 4
    @test haskey(r1.terms, WeightLatticeElem(TypeE{8}, SVector(0, 0, 0, 0, 0, 0, 1, 0)))
    @test haskey(r1.terms, WeightLatticeElem(TypeE{8}, SVector(1, 0, 0, 0, 0, 0, 0, 1)))
    @test haskey(r1.terms, WeightLatticeElem(TypeE{8}, SVector(0, 0, 0, 0, 0, 0, 0, 1)))
    @test haskey(r1.terms, WeightLatticeElem(TypeE{8}, SVector(0, 0, 1, 0, 0, 0, 0, 0)))

    # ⋀²V(ω₂): 13 irreducibles
    r2 = ⋀(2, ω[2])
    @test length(r2.terms) == 13

    # ⋀⁵V(ω₈): 12 irreducibles
    r3 = ⋀(5, ω[8])
    @test length(r3.terms) == 12

    # ⋀²V(2ω₈): 7 irreducibles
    r4 = ⋀(2, 2 * ω[8])
    @test length(r4.terms) == 7
  end

  # ─── character_from_weights ──────────────────────────────────────
  @testset "character_from_weights" begin
    # Build the standard A₂ rep from explicit weights
    m = Dict(SVector(1, 0) => 1, SVector(-1, 1) => 1, SVector(0, -1) => 1)
    V = character_from_weights(TypeA{2}, m)
    @test is_irreducible(V)
    @test highest_weight(V) == fundamental_weight(TypeA{2}, 1)

    # Adjoint A₂: 8 = V(1,1) with zero weight mult 2
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
    ω₁_A4 = fundamental_weight(TypeA{4}, 1)
    for k in 2:5
      @test plethysm(vcat([k]), ω₁_A4) == Sym(k, ω₁_A4)
    end

    # Exterior power = plethysm with one-column partition
    for k in 2:4
      @test plethysm(ones(Int, k), ω₁_A4) == ⋀(k, ω₁_A4)
    end

    # Mixed symmetry: S_{(2,1)} functor
    ω₁_A3 = fundamental_weight(TypeA{3}, 1)
    p21 = plethysm([2, 1], ω₁_A3)
    @test is_irreducible(p21)
    @test highest_weight(p21) ==
      fundamental_weight(TypeA{3}, 1) + fundamental_weight(TypeA{3}, 2)
    @test degree(p21) == 20

    # Plethysm on non-type-A: B₃
    ω₁_B3 = fundamental_weight(TypeB{3}, 1)
    @test plethysm([2], ω₁_B3) == Sym(2, ω₁_B3)
    @test plethysm([1, 1], ω₁_B3) == ⋀(2, ω₁_B3)

    # Plethysm on G₂
    ω₁_G2 = fundamental_weight(TypeG2, 1)
    @test plethysm([2], ω₁_G2) == Sym(2, ω₁_G2)
    @test plethysm([1, 1], ω₁_G2) == ⋀(2, ω₁_G2)

    # S_{(2,1)} on B₃ ω₁: dimension check
    # dim(V(ω₁)) = 7 for B₃, S_{(2,1)} has hook content dim = 7*6*5/3 = 70
    # but in general the formula is more complex
    p21_B3 = plethysm([2, 1], ω₁_B3)
    @test degree(p21_B3) == 112  # known value

    # Trivial cases
    @test plethysm([1], ω₁_A3) == WeylCharacter(ω₁_A3)
    @test plethysm(Int[], ω₁_A3) ==
      WeylCharacter(WeightLatticeElem{TypeA{3},3}(zero(SVector{3,Int})))
  end

  # ─── ProductDynkinType characters ──────────────────────────────
  @testset "ProductDynkinType characters" begin
    PT = ProductDynkinType{Tuple{TypeA{2},TypeB{3}}}
    ω₁ = fundamental_weight(PT, 1)  # A₂ fundamental weight
    ω₃ = fundamental_weight(PT, 3)  # B₃ fundamental weight

    # Degree of product fundamental weights factors
    @test degree(ω₁) == degree(fundamental_weight(TypeA{2}, 1))
    @test degree(ω₃) == degree(fundamental_weight(TypeB{3}, 1))

    # Freudenthal formula
    m = freudenthal_formula(ω₁)
    @test sum(values(m)) == degree(ω₁)
    m3 = freudenthal_formula(ω₃)
    @test sum(values(m3)) == degree(ω₃)

    # Tensor product dimensions are multiplicative
    V1 = WeylCharacter(ω₁)
    V3 = WeylCharacter(ω₃)
    @test degree(V1 * V3) == degree(V1) * degree(V3)
    @test degree(V1 * V1) == 9  # 3 ⊗ 3 = 6 ⊕ 3̄ in A₂

    # Symmetric and exterior powers
    @test degree(symmetric_power(ω₁, 2)) == 6   # Sym²(3) in A₂
    @test degree(exterior_power(ω₁, 2)) == 3     # ∧²(3) in A₂
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

  # ─── Proposition 2: exceptional types, A₂, and B₂ ───────────────
  @testset "Proposition 2" begin
    # A₂: V(ω₁+2ω₂) and V(4ω₂) both have degree 15
    @test degree(WeightLatticeElem(TypeA{2}, [1, 2])) == 15
    @test degree(WeightLatticeElem(TypeA{2}, [0, 4])) == 15

    # B₂: V(ω₁+2ω₂) and V(4ω₂) both have degree 35
    @test degree(WeightLatticeElem(TypeB{2}, [1, 2])) == 35
    @test degree(WeightLatticeElem(TypeB{2}, [0, 4])) == 35

    # G₂: V(3ω₁) and V(2ω₂) both have degree 77
    # The paper uses the opposite labeling to Bourbaki for G₂ (ω₁ ↔ ω₂).
    @test degree(WeightLatticeElem(TypeG2, [3, 0])) == 77
    @test degree(WeightLatticeElem(TypeG2, [0, 2])) == 77

    # F₄: V(ω₁+ω₄) and V(2ω₁) both have degree 1053
    @test degree(WeightLatticeElem(TypeF4, [1, 0, 0, 1])) == 1053
    @test degree(WeightLatticeElem(TypeF4, [2, 0, 0, 0])) == 1053

    # E₆: V(2ω₁) and V(ω₃) both have degree 351
    @test degree(WeightLatticeElem(TypeE{6}, [2, 0, 0, 0, 0, 0])) == 351
    @test degree(WeightLatticeElem(TypeE{6}, [0, 0, 1, 0, 0, 0])) == 351

    # E₇: V(ω₄+ω₅) and V(2ω₆+3ω₇) both have degree 1903725824
    @test degree(WeightLatticeElem(TypeE{7}, [0, 0, 0, 1, 1, 0, 0])) ==
      1903725824
    @test degree(WeightLatticeElem(TypeE{7}, [0, 0, 0, 0, 0, 2, 3])) ==
      1903725824

    # E₈: V(ω₁+ω₃) and V(ω₁+ω₇+ω₈) both have degree 8634368000
    @test degree(WeightLatticeElem(TypeE{8}, [1, 0, 1, 0, 0, 0, 0, 0])) ==
      8634368000
    @test degree(WeightLatticeElem(TypeE{8}, [1, 0, 0, 0, 0, 0, 1, 1])) ==
      8634368000
  end

  # ─── Theorem 3(a): Type Aₗ ───────────────────────────────────────
  # V((l-1)ω₂) and V(ω₁+(l-2)ω₂) have the same degree
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
  # V((2l-2)ω₂) and V(ω₁+(2l-3)ω₂) have the same degree
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
  # V((2l-3)ω₂) and V(ω₁+(2l-4)ω₂) have the same degree
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
    ω₁ = fundamental_weight(TypeA{3}, 1)
    ω₂ = fundamental_weight(TypeA{3}, 2)
    V = tensor_product(ω₁, ω₂)
    idx_sum = Rational{BigInt}(0)
    for (λ, m) in V
      idx_sum += m * dynkin_index(λ)
    end
    @test idx_sum == dynkin_index(ω₁) * degree(ω₂) + degree(ω₁) * dynkin_index(ω₂)
  end

  # ─── Casimir eigenvalue ────────────────────────────────────────────
  @testset "Casimir eigenvalue" begin
    ρ = weyl_vector(TypeA{2})
    ω₁ = fundamental_weight(TypeA{2}, 1)
    @test casimir_eigenvalue(ω₁) == dot(ω₁, ω₁ + 2 * ρ)

    # C₂(0) = 0
    z = zero(WeightLatticeElem{TypeA{2},2})
    @test casimir_eigenvalue(z) == 0
  end

  # ─── Congruency class ──────────────────────────────────────────────
  @testset "Congruency class" begin
    # A₂: Σ i·λᵢ mod 3
    @test congruency_class(fundamental_weight(TypeA{2}, 1)) == 1
    @test congruency_class(fundamental_weight(TypeA{2}, 2)) == 2
    @test congruency_class(WeightLatticeElem(TypeA{2}, [1, 1])) == 0  # adjoint

    # B₃: λ₃ mod 2
    @test congruency_class(fundamental_weight(TypeB{3}, 1)) == 0
    @test congruency_class(fundamental_weight(TypeB{3}, 3)) == 1

    # C₃: Σ λᵢ for odd i mod 2
    @test congruency_class(fundamental_weight(TypeC{3}, 1)) == 1
    @test congruency_class(fundamental_weight(TypeC{3}, 2)) == 0
    @test congruency_class(fundamental_weight(TypeC{3}, 3)) == 1

    # D₄: Z/2 × Z/2 center
    @test congruency_class(fundamental_weight(TypeD{4}, 1)) == (1, 1)
    @test congruency_class(fundamental_weight(TypeD{4}, 2)) == (0, 0)
    @test congruency_class(fundamental_weight(TypeD{4}, 3)) == (1, 0)
    @test congruency_class(fundamental_weight(TypeD{4}, 4)) == (0, 1)

    # E₆: λ₁ - λ₂ + λ₄ - λ₅ mod 3
    @test congruency_class(fundamental_weight(TypeE{6}, 1)) == 1
    @test congruency_class(fundamental_weight(TypeE{6}, 6)) == 2

    # E₇: λ₂ + λ₅ + λ₇ mod 2
    @test congruency_class(fundamental_weight(TypeE{7}, 1)) == 0
    @test congruency_class(fundamental_weight(TypeE{7}, 7)) == 1

    # Trivial center types always return 0
    @test congruency_class(fundamental_weight(TypeE{8}, 1)) == 0
    @test congruency_class(fundamental_weight(TypeF4, 1)) == 0
    @test congruency_class(fundamental_weight(TypeG2, 1)) == 0

    # Weights in same congruency class differ by root lattice element
    # A₃: class = Σ i·λᵢ mod 4; [1,0,1] and [0,2,0] both have class 0
    @test congruency_class(WeightLatticeElem(TypeA{3}, [1, 0, 1])) ==
      congruency_class(WeightLatticeElem(TypeA{3}, [0, 2, 0]))
  end

  # ─── Self-dual and Frobenius–Schur ─────────────────────────────────
  @testset "Self-dual and Frobenius-Schur" begin
    # A_n: ωᵢ is self-dual iff 2i = n+1 (i.e., middle weight for odd n)
    @test !is_self_dual(fundamental_weight(TypeA{2}, 1))
    @test is_self_dual(WeightLatticeElem(TypeA{2}, [1, 1]))
    @test !is_self_dual(fundamental_weight(TypeA{3}, 1))
    @test is_self_dual(fundamental_weight(TypeA{3}, 2))  # A₃: ω₂ is self-dual (4×4 antisymmetric)

    # B, C, G₂, F₄, E₇, E₈: all fundamental reps are self-dual
    for DT in (TypeB{3}, TypeC{3}, TypeG2, TypeF4, TypeE{7}, TypeE{8})
      R = rank(DT)
      for i in 1:R
        @test is_self_dual(fundamental_weight(DT, i))
      end
    end

    # E₆: ω₁ and ω₆ are NOT self-dual (conjugate to each other)
    @test !is_self_dual(fundamental_weight(TypeE{6}, 1))
    @test !is_self_dual(fundamental_weight(TypeE{6}, 6))
    @test is_self_dual(fundamental_weight(TypeE{6}, 2))

    # Frobenius-Schur indicator
    @test frobenius_schur_indicator(fundamental_weight(TypeA{2}, 1)) == 0
    @test frobenius_schur_indicator(WeightLatticeElem(TypeA{2}, [1, 1])) == 1

    # B₃ vector is real, spinor of SO(7) is real (7 = 8k-1 pattern)
    @test frobenius_schur_indicator(fundamental_weight(TypeB{3}, 1)) == 1
    @test frobenius_schur_indicator(fundamental_weight(TypeB{3}, 3)) == 1

    # B₂ spinor (SO(5)) is pseudoreal (5 = 8k+5 pattern)
    @test frobenius_schur_indicator(fundamental_weight(TypeB{2}, 2)) == -1

    # C₂ standard rep is pseudoreal (symplectic form)
    @test frobenius_schur_indicator(fundamental_weight(TypeC{2}, 1)) == -1
    @test frobenius_schur_indicator(fundamental_weight(TypeC{2}, 2)) == 1

    # D₄ spinors are real (8 = 8·1)
    @test frobenius_schur_indicator(fundamental_weight(TypeD{4}, 1)) == 1
    @test frobenius_schur_indicator(fundamental_weight(TypeD{4}, 3)) == 1
    @test frobenius_schur_indicator(fundamental_weight(TypeD{4}, 4)) == 1

    # G₂ 7-dim is real
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
end

# ─── Weylloop ε-basis roundtrip ──────────────────────────────────────────────

@testset "Weylloop ε-basis roundtrip" begin
  # Test that _w2e! followed by _e2w! is the identity for all simple types.
  # Uses deterministic pseudo-random weight vectors to avoid a Random dependency.
  simple_types = [
    TypeA{4}, TypeB{3}, TypeC{3}, TypeD{5},
    TypeE{6}, TypeE{7}, TypeE{8}, TypeF4, TypeG2,
  ]
  for DT in simple_types
    R = rank(DT)
    ED = Lie._weylloop_eps_dim(DT)
    e = zeros(Int, ED)
    w2 = zeros(Int, R)
    for trial in 1:100
      w = [((trial * 7 + i * 13) % 11) - 5 for i in 1:R]
      Lie._w2e!(DT, e, w)
      Lie._e2w!(DT, w2, e)
      @test w2 == w
    end
  end
end

# ═══════════════════════════════════════════════════════════════════════
#  Aqua.jl: package quality checks
# ═══════════════════════════════════════════════════════════════════════
@testset "Aqua" begin
  # ambiguities=false: @generated functions can trigger false positives
  Aqua.test_all(Lie; ambiguities=false)
end

# ═══════════════════════════════════════════════════════════════════════
#  Coset reps and Bruhat order — extended coverage (item 12)
# ═══════════════════════════════════════════════════════════════════════
@testset "Coset reps extended" begin
  # ─── B₂: |W| = 8, parabolic by {1} has 4 cosets ──────────────────
  @testset "B₂ right coset reps" begin
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

  @testset "B₂ left coset reps" begin
    W = weyl_group(TypeB{2})
    reps = left_coset_reps(W, [2])
    @test length(reps) == 4
    for w in reps
      @test !(2 in left_descent_set(w))
    end
  end

  # ─── A₃: |W| = 24, parabolic by {1,2} = S₃, cosets = 4 ──────────
  @testset "A₃ right coset reps by {1,2}" begin
    W = weyl_group(TypeA{3})
    reps = right_coset_reps(W, [1, 2])
    @test length(reps) == 4   # |S₄|/|S₃| = 4
    for w in reps
      @test !(1 in right_descent_set(w))
      @test !(2 in right_descent_set(w))
    end
  end

  @testset "A₃ right coset reps by {1}" begin
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

  # ─── A₃ Bruhat order ────────────────────────────────────────────
  @testset "A₃ Bruhat order" begin
    W = weyl_group(TypeA{3})
    e = one(W)
    w0 = longest_element(W)
    s1 = gen(W, 1); s2 = gen(W, 2); s3 = gen(W, 3)

    # Identity ≤ everything, w₀ ≥ everything
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

  # ─── E₆ maximal parabolic (benchmark sanity) ───────────────────
  @testset "E₆ right coset reps by {2,3,4,5,6}" begin
    W = weyl_group(TypeE{6})
    # Remove node 1: W/W_{2..6} ≅ E₆/P₁, |cosets| = dim(E₆ 27-rep) = 27
    reps = right_coset_reps(W, [2, 3, 4, 5, 6])
    @test length(reps) == 27
    for w in reps
      for i in 2:6
        @test !(i in right_descent_set(w))
      end
    end
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
    ω₁_A2 = fundamental_weight(TypeA{2}, 1)  # dim 3
    V = WeylCharacter(ω₁_A2)
    # Sym^2(3) = 6, ⋀^2(3) = 3
    @test degree(symmetric_power(V, 2)) == 6
    @test degree(exterior_power(V, 2)) == 3

    ω₁_B3 = fundamental_weight(TypeB{3}, 1)  # dim 7
    V7 = WeylCharacter(ω₁_B3)
    # Sym^2(7) = 28, ⋀^2(7) = 21
    @test degree(symmetric_power(V7, 2)) == 28
    @test degree(exterior_power(V7, 2)) == 21

    # Full exterior algebra of 7-dim: Σ_k C(7,k) = 128 = 2^7
    total = sum(degree(exterior_power(V7, k)) for k in 0:7)
    @test total == 2^7

    # Adams operator degree: degree(ψ^k(V)) via character = degree(V) (same)
    ψ2 = adams_operator(ω₁_A2, 2)
    @test sum(values(ψ2)) == degree(ω₁_A2)
  end

  # ─── degree(V::WeylCharacter) works for virtual characters ────────
  @testset "degree of virtual character" begin
    ω₁ = fundamental_weight(TypeA{2}, 1)
    ω₂ = fundamental_weight(TypeA{2}, 2)
    V1 = WeylCharacter(ω₁)
    V2 = WeylCharacter(ω₂)
    @test degree(V1 - V2) == 0         # 3 - 3 = 0
    @test degree(V1 - 2 * V2) == -3   # 3 - 6 = -3
    @test degree(2 * V1 + 3 * V2) == 15  # 2*3 + 3*3 = 15

    # Zero character
    @test degree(V1 - V1) == 0

    # A₂: V(ω₁+ω₂) = adjoint (8), V(2ω₁) = Sym² (6)
    adj = WeylCharacter(WeightLatticeElem(TypeA{2}, [1, 1]))
    sym2 = WeylCharacter(WeightLatticeElem(TypeA{2}, [2, 0]))
    @test degree(adj - sym2) == 2           # 8 - 6 = 2
    @test degree(sym2 - adj) == -2          # 6 - 8 = -2
    @test degree(adj - sym2 + V1) == 5      # 2 + 3 = 5
    @test degree(3 * adj - 2 * sym2) == 12  # 24 - 12 = 12

    # B₂: V(ω₁)=5, V(ω₂)=4, V(ω₁+ω₂)=16, V(2ω₁)=14
    b2ω₁ = fundamental_weight(TypeB{2}, 1)
    b2ω₂ = fundamental_weight(TypeB{2}, 2)
    W1 = WeylCharacter(b2ω₁)   # dim 5
    W2 = WeylCharacter(b2ω₂)   # dim 4
    @test degree(W1 - W2) == 1             # 5 - 4 = 1
    @test degree(W2 - W1) == -1            # 4 - 5 = -1
    @test degree(2 * W1 - 3 * W2) == -2   # 10 - 12 = -2

    # G₂: V(ω₁)=7, V(ω₂)=14
    g2ω₁ = fundamental_weight(TypeG2, 1)
    g2ω₂ = fundamental_weight(TypeG2, 2)
    G1 = WeylCharacter(g2ω₁)   # dim 7
    G2char = WeylCharacter(g2ω₂)  # dim 14
    @test degree(G1 - G2char) == -7        # 7 - 14 = -7
    @test degree(2 * G1 - G2char) == 0    # 14 - 14 = 0
    @test degree(3 * G1 - G2char) == 7    # 21 - 14 = 7

    # Newton identity: ψ₂(V) = Sym²(V) - ⋀²(V) is a virtual character with
    # degree = dim Sym²(V) - dim ⋀²(V)
    ψ2_raw = adams_operator(ω₁, 2)
    ψ2 = character_from_weights(TypeA{2}, ψ2_raw)
    @test degree(ψ2) == degree(Sym(2, ω₁)) - degree(⋀(2, ω₁))  # 6 - 3 = 3
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
  # ─── A₂ × B₂ (rank 4) ────────────────────────────────────────────
  @testset "A₂ × B₂" begin
    PT = ProductDynkinType{Tuple{TypeA{2},TypeB{2}}}
    ω = [fundamental_weight(PT, i) for i in 1:4]

    # Each fundamental weight lives in one factor
    @test degree(ω[1]) == 3    # A₂ standard
    @test degree(ω[2]) == 3    # A₂ dual standard
    @test degree(ω[3]) == 5    # B₂ vector (= SO(5) standard, ω₁ of B₂)
    @test degree(ω[4]) == 4    # B₂ spinor (= ω₂ of B₂)

    # Freudenthal formula consistency
    for i in 1:4
      @test degree(ω[i]) == sum(values(freudenthal_formula(ω[i])))
    end

    # tensor product: V(ω₁) ⊗ V(ω₃) has dim 3×5 = 15
    V1 = WeylCharacter(ω[1]); V3 = WeylCharacter(ω[3])
    @test degree(V1 * V3) == 15

    # V(ω₁) ⊗ V(ω₁) decomposes in A₂ factor only
    @test degree(V1 * V1) == 9   # 3⊗3 in A₂

    # Sym² and ⋀² of fundamental reps
    @test degree(symmetric_power(ω[1], 2)) == 6
    @test degree(exterior_power(ω[1], 2)) == 3
    @test degree(symmetric_power(ω[3], 2)) == 15  # Sym²(5) for SO(5) = 14-dim + trivial
    @test degree(exterior_power(ω[3], 2)) == 10   # adjoint of B₂ = so(5) = 10-dim

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

  # ─── A₁ × A₁ (simplest product) ─────────────────────────────────
  @testset "A₁ × A₁" begin
    PT = ProductDynkinType{Tuple{TypeA{1},TypeA{1}}}
    ω₁ = fundamental_weight(PT, 1)
    ω₂ = fundamental_weight(PT, 2)

    @test degree(ω₁) == 2
    @test degree(ω₂) == 2

    V1 = WeylCharacter(ω₁); V2 = WeylCharacter(ω₂)

    # (2⊗1) ⊗ (1⊗2) = 4-dim
    @test degree(V1 * V2) == 4

    # Sym²(2⊗1) in product type = Sym²(2) ⊗ 1 = 3⊗1 in product
    # In A₁×A₁: V(ω₁)=2⊗1, Sym²(V(ω₁)) = V(2ω₁) = 3⊗1, dim=3
    @test degree(symmetric_power(ω₁, 2)) == 3

    # Bruhat-like: dimension formula on product
    @test weyl_order(PT) == 4   # Z/2 × Z/2

    # freudenthal on product type
    m = freudenthal_formula(ω₁)
    @test sum(values(m)) == 2
  end

  # ─── A₂ × A₂ × A₂ (three factors) ───────────────────────────────
  @testset "A₂ × A₂ × A₂" begin
    PT = ProductDynkinType{Tuple{TypeA{2},TypeA{2},TypeA{2}}}
    ω = [fundamental_weight(PT, i) for i in 1:6]

    # Each A₂ has 2 fundamental weights with dim 3
    for i in 1:6
      @test degree(ω[i]) == 3
    end

    # Triple tensor product: V(ω₁) ⊗ V(ω₃) ⊗ V(ω₅) has dim 27
    V1 = WeylCharacter(ω[1]); V3 = WeylCharacter(ω[3]); V5 = WeylCharacter(ω[5])
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

  # ─── A₃ × G₂ ─────────────────────────────────────────────────────
  @testset "A₃ × G₂" begin
    PT = ProductDynkinType{Tuple{TypeA{3},TypeG2}}
    ω = [fundamental_weight(PT, i) for i in 1:5]

    @test degree(ω[1]) == 4    # A₃ standard
    @test degree(ω[2]) == 6    # A₃ antisym
    @test degree(ω[3]) == 4    # A₃ det = dual standard (= 4 since self-dual A₃ ω₃)
    @test degree(ω[4]) == 7    # G₂ standard
    @test degree(ω[5]) == 14   # G₂ adjoint

    V4 = WeylCharacter(ω[4])
    V5 = WeylCharacter(ω[5])

    # G₂ factor: 7⊗14 decomposes as known
    # In the product type, this is a G₂ tensor product tensored with trivial A₃
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

    # Multi-irreducible: V(ω₁) ⊕ V(ω₂) in A₂
    ω₁ = fundamental_weight(TypeA{2}, 1)
    ω₂ = fundamental_weight(TypeA{2}, 2)
    ff1 = freudenthal_formula(ω₁)
    ff2 = freudenthal_formula(ω₂)
    combined = merge(+, ff1, ff2)
    result = character_from_weights(TypeA{2}, combined)
    @test result == WeylCharacter(ω₁) + WeylCharacter(ω₂)
  end
end
