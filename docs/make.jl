using Documenter
using Lie

makedocs(;
  sitename="Lie.jl",
  modules=[Lie],
  pages=[
    "Home" => "index.md",
    "Dynkin types and Cartan matrices" => "types.md",
    "Root systems" => "roots.md",
    "Weight lattice" => "weights.md",
    "Weyl groups" => "weyl.md",
    "Characters and representations" => "characters.md",
    "Implementation details" => "details.md",
  ],
  doctest=true,
  warnonly=[:missing_docs],
  format=Documenter.HTML(;
    canonical="https://HomogeneousTools.github.io/Lie.jl"
  ),
)

deploydocs(;
  repo="git@github.com:HomogeneousTools/Lie.jl.git",
  target="build",
  branch="gh-pages",
  devbranch="main",
  push_preview=true,
)
