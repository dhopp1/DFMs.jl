include("../src/HelperFunctions.jl")
include("../src/InitialConditions.jl")
include("../src/KalmanFilter.jl")
include("../src/EM.jl")
include("../src/EstimateDFM.jl")
include("../src/Update.jl")

using Suppressor

using DataFrames
using Statistics
using Dates
using LinearAlgebra
using SparseArrays
using CSV


# data set up
sample_data = CSV.read("test_data.csv") |> DataFrame
y_tmp = sample_data[!, Not(:date)]
dates = sample_data[!, :date]
q = zeros(4)
p = 1
monthly_quarterly_array = gen_monthly_quarterly(dates, y_tmp)
R_mat = [2 -1 0 0 0; 3 0 -1 0 0; 2 0 0 -1 0; 1 0 0 0 -1]
blocks = DataFrame(a=ones(size(y_tmp)[2]-1), b=ones(size(y_tmp)[2]-1))
init_conds = initialize_conditions(y_tmp; dates=dates, p=1, blocks=blocks, R_mat=R_mat)
A = init_conds[:A]; C = init_conds[:C]; Q = init_conds[:Q]; R = init_conds[:R]; Z0 = init_conds[:Z0]; V0 = init_conds[:V0]
y_est = y_tmp[[sum(.!ismissing.(Array(x))) > 0 for x in eachrow(y_tmp)], :] |> Array |> transpose
nM = sum(.!monthly_quarterly_array)

# creating an artificial lagged dataset
data_lag = create_lag(sample_data, 1)
old_y = copy(data_lag)
new_y = copy(sample_data)
target_variable = Symbol("x_world.sa")
target_period = Dates.Date(2020,6,1)


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
	tmp = initialize_conditions(Y; dates=dates, p=p, blocks=blocks, R_mat=R_mat)

	@test sum(tmp[:A]) - 2.438 < 0.001
	@test sum(tmp[:C]) - 4.2649 < 0.001
	@test isequal(sum(tmp[:Q]), NaN)
	@test tmp[:R] == [0.0001 0.0; 0.0 0.0001]
	@test tmp[:Z0] == zeros(12)
	@test isequal(sum(tmp[:V0]), NaN)
end

@testset "Kalman filter" begin
	output = kalman_filter(y_est, A, C, Q, R, Z0, V0)

	@test sum(output[:Zsmooth]) ≈ 16.9734395062919
	@test sum(output[:Vsmooth]) ≈ 872.1521836796258
	@test sum(output[:VVsmooth]) ≈ 283.92273510769024
	@test sum(output[:loglik]) ≈ 509.2892978152599

	output_dfm = estimate_dfm(sample_data; blocks=blocks, p=p, max_iter=10, threshold=1e-5)
	constparams = kalman_filter_constparams(y_tmp; output_dfm=output_dfm, lag=0)

	@test sum(constparams[:Plag][1]) ≈ 97.58891907171578
	@test sum(constparams[:X_smooth]) ≈ 14.04281014538804
	@test sum(constparams[:Vsmooth]) ≈ 98.00373560282938
	@test sum(constparams[:F]) ≈ 114.31015137657359

	constparams = kalman_filter_constparams(y_tmp; output_dfm=output_dfm, lag=2)
	@test sum(skipmissing(constparams[:Plag][2])) ≈ 31.4925876069849
	@test sum(skipmissing(constparams[:Plag][3])) ≈ 29.398312508379988
end

@testset "EM" begin
	em_output = EM_step(y_est; A=A, C=C, Q=Q, R=R, Z0=Z0, V0=V0, p=p, blocks=blocks, R_mat=R_mat, q=q, nM=nM, monthly_quarterly_array=monthly_quarterly_array)

	@test sum(em_output[:A_new]) ≈ 11.6799558151333
	@test sum(em_output[:C_new]) ≈ 22.349648436221734
	@test sum(em_output[:Q_new]) ≈ 1.6498771441810458
	@test sum(em_output[:Z0]) ≈ 0.031060939148336537
	@test sum(em_output[:V0]) ≈ 49.20972973859572
	@test sum(em_output[:loglik]) ≈ 509.2892978152599

	loglik = 10; prev_loglik = 1e-6;
	converged, decrease = values(EM_convergence(loglik, prev_loglik, 1e-5))
	@test converged == 0
	@test decrease == 0

	loglik = 1e-8; prev_loglik = 5;
	converged, decrease = values(EM_convergence(loglik, prev_loglik, 1e-1))
	@test converged == 0
	@test decrease == 1

	loglik = 2; prev_loglik = 5;
	converged, decrease = values(EM_convergence(loglik, prev_loglik, 1))
	@test converged == 1
	@test decrease == 1
end

@testset "EstimateDFM" begin
	output = estimate_dfm(sample_data; blocks=blocks, p=p, max_iter=10, threshold=1e-5)

	@test sum(output[:Xsmooth_std]) ≈ 10.511990707459224
	@test sum(output[:C]) ≈ 21.979826198523067
	@test sum(output[:R]) ≈ 0.0011
	@test sum(output[:A]) ≈ 12.768581356259535
	@test sum(output[:Q]) ≈ 0.15526910701420402
	@test sum(output[:means]) ≈ 0.060732284509809845
	@test sum(output[:stds]) ≈ 0.5829554656784255
	@test sum(output[:Z0]) ≈ 0.7554301411482052
	@test sum(output[:V0]) ≈ 49.20972973859572
	@test sum(output[:loglik]) ≈ 6414.119209820474
	@test sum(output[:LL]) ≈ 53946.7817953308
end

@testset "Update" begin
	output_dfm = estimate_dfm(sample_data; blocks=blocks, p=p, max_iter=10, threshold=1e-5)
	news = news_dfm(;old_y=data_lag, new_y=sample_data, output_dfm=output_dfm, target_variable=Symbol("x_world.sa"), target_period=Dates.Date(2020,6,1))

	@test sum(news[:actual]) ≈ -0.6080187052009832
	@test sum(news[:forecast]) ≈ -0.7934991526023685
	@test sum(news[:weight]) ≈ 0.3377434672020498
	@test sum(news[:innov]) ≈ 15.119737715315845
	@test sum(news[:row_miss]) ≈ 2427
	@test sum(news[:col_miss]) ≈ 66
	@test sum(news[:singlenews]) ≈ -0.013155651969646982
	@test sum(news[:y_old]) ≈ 0.010865864832867203
	@test sum(news[:y_new]) ≈ -0.01335101140300618

	updated_nowcast = update_nowcast(;old_y=create_lag(new_y, 1), new_y=new_y, output_dfm=output_dfm, target_variable=Symbol("x_world.sa"), target_period=Dates.Date(2020,6,1))

	@test sum(skipmissing(updated_nowcast[!, :forecast])) ≈ -79.34991526023686
	@test sum(skipmissing(updated_nowcast[!, :actual])) ≈ -60.80187052009832
	@test sum(skipmissing(updated_nowcast[!, :weight])) ≈ 0.3377434672020498
	@test sum(skipmissing(updated_nowcast[!, :impact_releases])) ≈ -1.3155651969646984
	@test sum(skipmissing(updated_nowcast[!, :impact_total])) ≈ -1.3155651969646984
	@test sum(skipmissing(updated_nowcast[!, :data_release])) ≈ 2
end
