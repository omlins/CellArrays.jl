# Examples

### Memory copy of `CuCellArray`s with 4x4 `SMatrix` cells.
````@eval
using Markdown
Markdown.parse("""
```julia
$(read(joinpath("..", "..", "..", "examples", "memcopyCellArray3D.jl"), String))
```
""")
````

### Memory copy of `CellArray`s with 4x4 cells using `ParallelStencil.jl`.

````@eval
using Markdown
Markdown.parse("""
```julia
$(read(joinpath("..", "..", "..", "examples", "memcopyCellArray3D_ParallelStencil.jl"), String))
```
""")
````
