using CellArrays
using Documenter
using DocExtensions
using DocExtensions.DocumenterExtensions

const DOCSRC      = joinpath(@__DIR__, "src")
const DOCASSETS   = joinpath(DOCSRC, "assets")
const EXAMPLEROOT = joinpath(@__DIR__, "..", "examples")

DocMeta.setdocmeta!(CellArrays, :DocTestSetup, :(using CellArrays); recursive=true)

@info "Copy examples folder to assets"
mkpath(DOCASSETS)
cp(EXAMPLEROOT, joinpath(DOCASSETS, "examples"); force=true)


@info "Preprocessing .MD-files..."
include("reflinks.jl")
MarkdownExtensions.expand_reflinks(reflinks; rootdir=DOCSRC)


@info "Building documentation website using Documenter.jl..."
makedocs(;
    modules  = [CellArrays],
    authors  = "Samuel Omlin",
    repo     = "https://github.com/omlins/CellArrays.jl/blob/{commit}{path}#{line}",
    sitename = "CellArrays.jl",
    format   = Documenter.HTML(;
        prettyurls       = true, #get(ENV, "CI", "false") == "true",
        canonical        = "https://omlins.github.io/CellArrays.jl",
        collapselevel    = 1,
        sidebar_sitename = true,
        edit_link        = "main",
        #assets           = [asset("https://img.shields.io/github/stars/omlins/CellArrays.jl.svg", class = :ico)],
        #warn_outdated    = true,
    ),
    pages   = [
        "Introduction"  => "index.md",
        "Usage"         => "usage.md",
        "Examples"      => [hide("..." => "examples.md"),
                            "examples/memcopyCellArray3D.md",
                            "examples/memcopyCellArray3D_ParallelStencil.md",
                           ],
        "API reference" => "api.md",
    ],
)


@info "Deploying docs..."
deploydocs(;
    repo         = "github.com/omlins/CellArrays.jl",
    push_preview = true,
    devbranch    = "main",
)
