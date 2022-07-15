# Examples

## Example 1: memory copy of [`CuCellArray`](@ref)s with 4x4 `SMatrix` cells
````@eval
using Markdown
Markdown.parse("""
```julia
$(read(joinpath("..", "..", "..", "examples", "memcopyCellArray3D.jl"), String))
```
""")
````

## Example 2: memory copy of [`CellArray`](@ref)s with 4x4 cells using [ParallelStencil.jl]

````@eval
using Markdown
Markdown.parse("""
```julia
$(read(joinpath("..", "..", "..", "examples", "memcopyCellArray3D_ParallelStencil.jl"), String))
```
""")
````
