include("../src/HelperFunctions.jl")

using DataFrames
using Statistics

test_data = DataFrame(Dict(:a => ["a", "a", "a"], :b => [1,2,3]))

@testset "Helper functions" begin
	@test standardize(test_data) == DataFrame(Dict(:a => ["a", "a", "a"], :b => [-1.0,0.0,1.0]))
end
