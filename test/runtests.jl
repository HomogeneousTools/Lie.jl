using Test
using RepresentationTheory
using StaticArrays

@testset "RepresentationTheory.jl" begin

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
        PT = ProductDynkinType{Tuple{TypeA{3}, TypeD{5}}}
        @test rank(PT) == 8
        @test n_components(PT) == 2

        PT2 = ProductDynkinType{Tuple{TypeA{3}, TypeD{5}, TypeE{6}}}
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
    #  Cartan matrices
    # ═══════════════════════════════════════════════════════════════════════
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
        C_prod = cartan_matrix(ProductDynkinType{Tuple{TypeA{2}, TypeG2}})
        @test size(C_prod) == (4, 4)
        @test C_prod[1:2, 1:2] == cartan_matrix(TypeA{2})
        @test C_prod[3:4, 3:4] == cartan_matrix(TypeG2)
        @test C_prod[1:2, 3:4] == zeros(Int, 2, 2)
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

        # Reflection
        w = WeightLatticeElem(DT, [2, 1])
        w_reflected = reflect(w, 1)
        @test w_reflected == WeightLatticeElem(DT, [-2, 3])
        # s₁(2ω₁ + ω₂) = (2ω₁ + ω₂) - 2*(α₁) = (2ω₁ + ω₂) - 2*(2ω₁ - ω₂) = -2ω₁ + 3ω₂

        # Conjugation to dominant chamber
        w_neg = WeightLatticeElem(DT, [-1, 2])
        w_dom = conjugate_dominant_weight(w_neg)
        @test is_dominant(w_dom)
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
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  Weyl dimension formula
    # ═══════════════════════════════════════════════════════════════════════
    @testset "Dimension formula" begin
        # A₁: dim V(nω₁) = n+1
        for n in 1:10
            hw = WeightLatticeElem(TypeA{1}, [n])
            @test dim_of_simple_module(hw) == n + 1
        end

        # A₂: dim V(ω₁) = 3 (standard rep)
        @test dim_of_simple_module(fundamental_weight(TypeA{2}, 1)) == 3

        # A₂: dim V(ω₂) = 3 (dual standard rep)
        @test dim_of_simple_module(fundamental_weight(TypeA{2}, 2)) == 3

        # A₂: dim V(ω₁ + ω₂) = 8 (adjoint rep)
        @test dim_of_simple_module(WeightLatticeElem(TypeA{2}, [1, 1])) == 8

        # A₃: dim V(ω₁) = 4
        @test dim_of_simple_module(fundamental_weight(TypeA{3}, 1)) == 4

        # A₃: dim V(ω₂) = 6 (exterior square)
        @test dim_of_simple_module(fundamental_weight(TypeA{3}, 2)) == 6

        # B₂: dim V(ω₁) = 5 (standard rep of SO(5))
        @test dim_of_simple_module(fundamental_weight(TypeB{2}, 1)) == 5

        # B₂: dim V(ω₂) = 4 (spin rep)
        @test dim_of_simple_module(fundamental_weight(TypeB{2}, 2)) == 4

        # B₃: dim V(ω₁) = 7 (standard rep of SO(7))
        @test dim_of_simple_module(fundamental_weight(TypeB{3}, 1)) == 7

        # C₂: dim V(ω₁) = 4 (standard rep of Sp(4))
        @test dim_of_simple_module(fundamental_weight(TypeC{2}, 1)) == 4

        # G₂: dim V(ω₁) = 7 (standard rep)
        @test dim_of_simple_module(fundamental_weight(TypeG2, 1)) == 7

        # G₂: dim V(ω₂) = 14 (adjoint rep)
        @test dim_of_simple_module(fundamental_weight(TypeG2, 2)) == 14

        # D₄: dim V(ω₁) = 8 (standard rep of SO(8))
        @test dim_of_simple_module(fundamental_weight(TypeD{4}, 1)) == 8

        # E₈: all fundamental representations
        @testset "E₈ fundamental representations" begin
            expected_dims = [3875, 147250, 6696000, 6899079264,
                             146325270, 2450240, 30380, 248]
            for (i, expected) in enumerate(expected_dims)
                @test dim_of_simple_module(fundamental_weight(TypeE{8}, i)) == expected
            end
        end

        # E₈: high-dimensional representation 3ω₃ + 5ω₈
        @test dim_of_simple_module(WeightLatticeElem(TypeE{8}, [0, 0, 3, 0, 0, 0, 0, 5])) ==
              big"18190674254761844256000000"
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

            # λ = [-3, 1]: λ+ρ = [-2, 2], s₁ gives [2, 0], which gives μ-ρ = [1, -1]
            result = borel_weil_bott(WeightLatticeElem(TypeA{2}, [-3, 1]))
            @test result !== nothing
            d, μ = result
            @test d == 1
            @test μ == WeightLatticeElem(TypeA{2}, [1, -1])
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
            result = borel_weil_bott(λ)
            @test result !== nothing
            d, μ = result
            @test d == 49
            @test μ == WeightLatticeElem(TypeE{8}, [0, -1, -1, -1, 0, -1, -1, 0])
        end    end

    # ═══════════════════════════════════════════════════════════════════════
    #  StaticArrays: verify types are compile-time static
    # ═══════════════════════════════════════════════════════════════════════
    @testset "Static type system" begin
        # Cartan matrices are SMatrix
        C = cartan_matrix(TypeA{3})
        @test C isa SMatrix{3,3,Int}

        C2 = cartan_matrix(ProductDynkinType{Tuple{TypeA{2}, TypeB{3}}})
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
        PT = ProductDynkinType{Tuple{TypeA{2}, TypeB{3}}}

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

end
