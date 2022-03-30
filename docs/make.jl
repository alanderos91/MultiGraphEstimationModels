using Documenter, MultiGraphEstimationModels

makedocs(
    modules = [MultiGraphEstimationModels],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Alfonso Landeros",
    sitename = "MultiGraphEstimationModels.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/alanderos91/MultiGraphEstimationModels.jl.git",
    push_preview = true
)
