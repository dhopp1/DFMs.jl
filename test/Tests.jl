include("../src/HelperFunctions.jl")
include("../src/InitialConditions.jl")

using DataFrames
using Statistics
using Dates
using LinearAlgebra
using SparseArrays

@testset "Helper functions" begin

	# testing standardize function
	test_data = DataFrame(Dict(:a => ["a", "a", "a"], :b => [1,2,3]))
	@test standardize(test_data) == DataFrame(Dict(:a => ["a", "a", "a"], :b => [-1.0,0.0,1.0]))

	#testing spline_fill function
	@test isequal(spline_fill([1,2,missing,4]), [1.0,2.0,3.0,4.0])
	@test isequal(spline_fill([1,2,missing,5,missing]), [1.0,2.0,3.375,5.0,missing])

	# testing digital_filter function
	@test (sum((digital_filter([missing, 1,2,5,3,missing]) .- [2.16, 1.0, 2.0, 5.0, 3.0, 2.33]))) < 0.01

	# testing fill_na function
	@test ((DataFrame(Dict(:x=>[missing, 1, 2, missing, 3.5], :y=>[24, 42, 76, 89.0, missing])) |> fill_na)[:output] .- DataFrame(Dict(:x=>[2.07725, 1.0, 2.0, 2.8125, 3.5], :y=>[24.0, 42.0, 76.0, 89.0, 61.9689])) |> x-> sum(x.x) + sum(x.y)) < 0.001
	@test (DataFrame(Dict(:x=>[missing, 1, 2, missing, 3.5], :y=>[24, 42, 76, 89.0, missing])) |> fill_na)[:na_indices] == DataFrame(Dict(:x=>[1,0,0,1,0], :y=>[0,0,0,0,1]))

	# testing gen_monthly_quarterly_matrix function
	test_dates = [Dates.Date(2020,i,1) for i in 1:12]
	test_data = DataFrame(
		Dict(
			:a => [0 for i in 1:12],
			:b => [missing,missing,0,missing,missing,0,missing,missing,0,missing,missing,0]
		)
	)
	@test gen_monthly_quarterly(test_dates, test_data) == [0,1]
end


@testset "Initial conditions" begin
	Y = DataFrame(Dict(:a=>[1,2,4,3,2], :b=>[2,1,3,1,5]))
	dates = [Dates.Date(2020,1,1), Dates.Date(2020,2,1), Dates.Date(2020,3,1), Dates.Date(2020,4,1), Dates.Date(2020,5,1)]
	blocks = DataFrame(Dict(:a=>[1.0, 1.0], :b=>[1.0, 1.0]))
	R_mat = [2 -1 0 0 0; 3 0 -1 0 0; 2 0 0 -1 0; 1 0 0 0 -1]
	tmp = initialize_conditions(Y; dates=dates, p=1, blocks=blocks, R_mat=R_mat)

	@test sum(tmp[:A]) - 2.438 < 0.001
	@test sum(tmp[:C]) - 4.2649 < 0.001
	@test isequal(sum(tmp[:Q]), NaN)
	@test tmp[:R] == [0.0001 0.0; 0.0 0.0001]
	@test tmp[:Z0] == zeros(12)
	@test isequal(sum(tmp[:V0]), NaN)
end
