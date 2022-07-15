using CellArrays
using Documenter

DocMeta.setdocmeta!(CellArrays, :DocTestSetup, :(using CellArrays); recursive=true)

makedocs(;
    modules  = [CellArrays],
    authors  = "Samuel Omlin",
    repo     = "https://github.com/omlins/CellArrays.jl/blob/{commit}{path}#{line}",
    sitename = "CellArrays.jl",
    format   = Documenter.HTML(;
        prettyurls       = get(ENV, "CI", "false") == "true",
        canonical        = "https://omlins.github.io/CellArrays.jl",
        collapselevel    = 1,
        sidebar_sitename = true,
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

deploydocs(;
    repo         = "github.com/omlins/CellArrays.jl",
    push_preview = true,
    devbranch    ="main",
)
