using CellArrays
using Documenter

DocMeta.setdocmeta!(CellArrays, :DocTestSetup, :(using CellArrays); recursive=true)

makedocs(;
    modules=[CellArrays],
    authors="Samuel Omlin",
    repo="https://github.com/omlins/CellArrays.jl/blob/{commit}{path}#{line}",
    sitename="CellArrays.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://omlins.github.io/CellArrays.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/omlins/CellArrays.jl",
    devbranch="main",
)
