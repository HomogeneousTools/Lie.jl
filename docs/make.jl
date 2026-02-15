using Documenter
using Lie

makedocs(;
  sitename="Lie.jl",
  modules=[Lie],
  pages=["Home" => "index.md"],
  doctest=true,
)
