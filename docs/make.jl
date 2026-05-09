using Documenter
using Lie

const SITE_NAME = "Lie.jl"
const AUTHORS = "Pieter Belmans"
const PDF_NAME = "Lie.jl.pdf"
const HTML_BUILD = joinpath(@__DIR__, "build")
const PAGES = [
  "Home" => "index.md",
  "Dynkin types and Cartan matrices" => "types.md",
  "Root systems" => "roots.md",
  "Weight lattice" => "weights.md",
  "Weyl groups" => "weyl.md",
  "Characters and representations" => "characters.md",
  "Implementation details" => "details.md",
]
const COMMON_DOCS_KWARGS = (
  sitename=SITE_NAME,
  authors=AUTHORS,
  modules=[Lie],
  pages=PAGES,
)

function build_html_docs()
  makedocs(;
    COMMON_DOCS_KWARGS...,
    build=HTML_BUILD,
    doctest=true,
    format=Documenter.HTML(;
      canonical="https://homogeneous.tools/Lie.jl/",
      assets=["assets/analytics.js"],
    ),
  )
end

function build_pdf_docs()
  mktempdir() do pdf_root
    build_dir = joinpath(pdf_root, "build")
    cp(joinpath(@__DIR__, "src"), joinpath(pdf_root, "src"); force=true)
    makedocs(;
      COMMON_DOCS_KWARGS...,
      root=pdf_root,
      source="src",
      build="build",
      doctest=false,
      remotes=nothing,
      format=Documenter.LaTeX(;
        platform="native",
      ),
    )
    pdf_files = filter(path -> endswith(path, ".pdf"), readdir(build_dir; join=true))
    length(pdf_files) == 1 ||
      error("Expected exactly one PDF output in $(build_dir), found $(length(pdf_files)).")
    cp(only(pdf_files), joinpath(HTML_BUILD, PDF_NAME); force=true)
  end
end

build_html_docs()
build_pdf_docs()

deploydocs(;
  repo="https://github.com/HomogeneousTools/Lie.jl.git",
  target="build",
  branch="gh-pages",
  devbranch="main",
  push_preview=true,
)
