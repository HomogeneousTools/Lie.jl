module RepresentationTheory

using StaticArrays
using LinearAlgebra: dot as _dot, I as _I

# ─── Type-level Dynkin types ────────────────────────────────────────────────
include("DynkinTypes.jl")

# ─── Cartan matrices (compile-time specialized) ─────────────────────────────
include("CartanMatrix.jl")

# ─── Root systems ───────────────────────────────────────────────────────────
include("RootSystem.jl")

# ─── Weight lattice ─────────────────────────────────────────────────────────
include("WeightLattice.jl")

# ─── Weyl groups ────────────────────────────────────────────────────────────
include("WeylGroup.jl")

end # module
