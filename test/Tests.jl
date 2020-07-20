include("../src/HelperFunctions.jl")
include("../src/InitialConditions.jl")
include("../src/KalmanFilter.jl")

using Suppressor

using DataFrames
using Statistics
using Dates
using LinearAlgebra
using SparseArrays
using CSV

sample_data = CSV.read("test_data.csv") |> DataFrame

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

	# testing date_col_name function
	test_data = DataFrame(a=[Dates.Date(2020,1,1), Dates.Date(2020,2,1)], b=[1,2])
	@test date_col_name(test_data) == :a
	error_out = @capture_out date_col_name(test_data[!, Not(:a)])
	@test error_out == "No column of type Dates.Date\n"
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

@testset "Kalman filter" begin
	q = zeros(4)
	y_tmp = sample_data[!, Not(:date)]
	dates = sample_data[!, :date]
	monthly_quarterly_array = gen_monthly_quarterly(dates, y_tmp)
	R_mat = [2 -1 0 0 0; 3 0 -1 0 0; 2 0 0 -1 0; 1 0 0 0 -1]
	blocks = DataFrame(a=ones(size(y_tmp)[2]-1), b=ones(size(y_tmp)[2]-1))
	init_conds = initialize_conditions(y_tmp; dates=dates, p=1, blocks=blocks, R_mat=R_mat)
    A = init_conds[:A]; C = init_conds[:C]; Q = init_conds[:Q]; R = init_conds[:R]; Z0 = init_conds[:Z0]; V0 = init_conds[:V0]
	y_est = y_tmp[[sum(.!ismissing.(Array(x))) > 0 for x in eachrow(y_tmp)], :] |> Array |> transpose
	output = kalman_filter(y_est, A, C, Q, R, Z0, V0)

	@test sum(output[:Zsmooth]) ≈ 16.9734395062919
	@test sum(output[:Vsmooth]) ≈ 872.1521836796258
	@test sum(output[:VVsmooth]) ≈ 283.92273510769024
	@test sum(output[:loglik]) ≈ 509.2892978152599
end
