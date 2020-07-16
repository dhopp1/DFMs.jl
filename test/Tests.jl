include("../src/HelperFunctions.jl")

using DataFrames
using Statistics

@testset "Helper functions" begin
	
	# testing standardize function
	test_data = DataFrame(Dict(:a => ["a", "a", "a"], :b => [1,2,3]))
	@test standardize(test_data) == DataFrame(Dict(:a => ["a", "a", "a"], :b => [-1.0,0.0,1.0]))

	#testing spline fill function
	@test isequal(spline_fill([1,2,missing,4]), [1.0,2.0,3.0,4.0])
	@test isequal(spline_fill([1,2,missing,5,missing]), [1.0,2.0,3.375,5.0,missing])

	# testing digital filter function
	@test (sum((digital_filter([missing, 1,2,5,3,missing]) .- [2.16, 1.0, 2.0, 5.0, 3.0, 2.33]))) < 0.01
	
	# testing fill na function
	@test ((DataFrame(Dict(:x=>[missing, 1, 2, missing, 3.5], :y=>[24, 42, 76, 89.0, missing])) |> fill_na) .- DataFrame(Dict(:x=>[2.07725, 1.0, 2.0, 2.8125, 3.5], :y=>[24.0, 42.0, 76.0, 89.0, 61.9689])) |> x-> sum(x.x) + sum(x.y)) < 0.001
end
