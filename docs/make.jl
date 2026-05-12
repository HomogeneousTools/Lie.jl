using Documenter
using Documenter: RawHTMLHeadContent
using Semisimple

const SITE_NAME = "Semisimple.jl"
const AUTHORS = "Pieter Belmans"
const PDF_NAME = "Semisimple.jl.pdf"
const HTML_BUILD = joinpath(@__DIR__, "build")
const PLAUSIBLE_HEAD = RawHTMLHeadContent("""
<script async src="https://plausible.io/js/pa-XnO99azZJG-BAZutMei1M.js"></script>
<script>
  window.plausible=window.plausible||function(){(plausible.q=plausible.q||[]).push(arguments)},plausible.init=plausible.init||function(i){plausible.o=i||{}};
  plausible.init()
</script>
""")
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
  modules=[Semisimple],
  pages=PAGES,
)

function build_html_docs()
  makedocs(;
    COMMON_DOCS_KWARGS...,
    build=HTML_BUILD,
    doctest=true,
    format=Documenter.HTML(;
      canonical="https://homogeneous.tools/Semisimple.jl/",
      assets=[PLAUSIBLE_HEAD],
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
  repo="https://github.com/HomogeneousTools/Semisimple.jl.git",
  target="build",
  branch="gh-pages",
  devbranch="main",
  push_preview=true,
)
