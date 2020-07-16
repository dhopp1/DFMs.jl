include("../src/HelperFunctions.jl")

using DataFrames
using Statistics

@testset "Helper functions" begin

	test_data = DataFrame(Dict(:a => ["a", "a", "a"], :b => [1,2,3]))
	@test standardize(test_data) == DataFrame(Dict(:a => ["a", "a", "a"], :b => [-1.0,0.0,1.0]))

	@test isequal(spline_fill([1,2,missing,4]), [1.0,2.0,3.0,4.0])
	@test isequal(spline_fill([1,2,missing,5,missing]), [1.0,2.0,3.375,5.0,missing])
end
