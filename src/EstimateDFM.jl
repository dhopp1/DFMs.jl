include("HelperFunctions.jl")
include("InitialConditions.jl")
include("EM.jl")

using DataFrames
using Dates
using Statistics
using CSV


"""
    estimate a dynamic factor model.
    parameters:
        Y : DataFrame | Array{Number, 2}
            matrix of variables, size (n_obs, n_variables). Must include a column of type Dates.Date.
        blocks : DataFrame, size (n_variables, n_blocks). Note don't include date column in n_variables
            matrix of 1s or 0s for block loadings, i.e. included in block.
            ex for Y 2 x 3, 2 blocks:
                [1, 0]
                [1, 1]
                [0, 1]
        p : Int
            number of lags in transition equation
        max_iter : Int
            number of iterations for EM
        threshold : Float
            EM loop threshold
    returns: Dict
      :X_smooth => Kalman-smoothed data where missing values are replaced by their expectation
      :Z => smoothed states, rows give time, and columns are organized according to matrix C
      :C => measurement matrix, rows correspond to each series, and the columns are organized as
        - columns 1-20 give the factor loadings. For example, 1-5 give loadings for the first
        and are organized in reverse-chronological order (f^G_t, f^G_t-1, f^G_t-2, f^G_t-3, f^G_t-4)
        - Columns 6-10, 11-15, and 16-20 give loadings for the second, third, and fourth blocks respectively.
      :R => covariance for measurement matrix residuals
      :A => transition matrix, a square matrix that follows the same organization scheme as matrix C's columns.
        Identity matrices are used to account for matching terms on the left and righthand side.
        For example, we place an I4 matrix to account for matching (f_t-1; f_t-2; f_t-3; f_t-4) terms
      :Q => covariance for transition equation residuals
      :means =>  series mean
      :stds => series standard deviation
      :Z0 => initial value of state
      :V0 => initial value of covariance matrix
      :p => number of lags in transition equation
      :num_iter => number of iterations required for convergence
      :convergence => bool if algorithm converged successfully (given max_iter)
      :loglik => log likelihood of last iteration
      :LL => sequence of log likelihoods per iteration
"""
function estimate_dfm(Y; blocks, p, max_iter=5000, threshold=1e-5)
    R_mat = [2 -1 0 0 0; 3 0 -1 0 0; 2 0 0 -1 0; 1 0 0 0 -1] # R*λ = q; constraints on loadings of quarterly variables
    q = zeros(4)
    dates = Y[!, date_col_name(Y)]
    y_tmp = Y[!, Not(date_col_name(Y))]
    monthly_quarterly_array = gen_monthly_quarterly(dates, y_tmp)
    # put all quarterly variables to the end
    y_tmp = [y_tmp[!, .!monthly_quarterly_array]  y_tmp[!, monthly_quarterly_array]]
    monthly_quarterly_array = gen_monthly_quarterly(dates, y_tmp)
    nM = sum(.!monthly_quarterly_array)

    # calculate initial variables
    init_conds = initialize_conditions(y_tmp; dates=dates, p=p, blocks=blocks, R_mat=R_mat)
    A = init_conds[:A]; C = init_conds[:C]; Q = init_conds[:Q]; R = init_conds[:R]; Z0 = init_conds[:Z0]; V0 = init_conds[:V0]

    # initialize EM loop values
    #= EM model
    y = C*Z + e
    z = A*Z(-1) + v
    where y is n_obs x n_variables, Z is (pr) x n_variables, etc.
    =#

    # remove leading and ending rows where all missing, transpose
    y_est = y_tmp[[sum(.!ismissing.(Array(x))) > 0 for x in eachrow(y_tmp)], :] |> Array |> transpose
    n_obs = size(y_est)[2]
    prev_loglik = -1e6
    num_iter = 0
    LL = [-1e6]
    converged = false
    y = transpose(y_tmp |> Array) # y for the estimation is WITH missing data

    # EM loop
    gain = []
    while !converged & (num_iter <= max_iter)
        em_output = EM_step(y_est; A=A, C=C, Q=Q, R=R, Z0=Z0, V0=V0, p=p, blocks=blocks, R_mat=R_mat, q=q, nM=nM, monthly_quarterly_array=monthly_quarterly_array)
        A = em_output[:A_new]; C = em_output[:C_new]; Q = em_output[:Q_new]; R = em_output[:R_new]; Z0 = em_output[:Z0]; V0 = em_output[:V0]; global loglik = em_output[:loglik];

        em_conv = EM_convergence(loglik, prev_loglik, threshold)
        converged = em_conv[:converged] == 1

        if ((mod(num_iter, 10) == 0) & (num_iter > 0))
            println("Now running iteration number $num_iter out of a maximum of $max_iter")
            println("Loglik: $(round(loglik, digits=4)); % change: $(round(100 * (loglik - prev_loglik)/prev_loglik, digits=10))%")
        end

        # break loop if marginal gains
        push!(gain, (loglik - prev_loglik)/prev_loglik)
        if num_iter > 50
            if mean(gain[num_iter-20:num_iter]) < 1e-6
                break
            end
        end

        push!(LL, loglik)
        prev_loglik = loglik
        num_iter += 1
    end

    # Final run of the Kalman filter
    kf_output = kalman_filter(y_est, A, C, Q, R, Z0, V0)
    Zsmooth = transpose(kf_output[:Zsmooth])
    Xsmooth_std = Zsmooth[2:end,:] * transpose(C)
    Xsmooth = [std(skipmissing(row)) for row in eachrow(y_est)] |> x->
        reshape(x, 1, size(x)[1]) |> x->
        repeat(x, n_obs) |> x->
        x .* Xsmooth_std + ([mean(skipmissing(row)) for row in eachrow(y_est)] |> x-> reshape(x, 1, size(x)[1]) |> x-> repeat(x, n_obs))

    # final results
    nowcast = Dict(
        :Xsmooth_std => Xsmooth_std,
        :Xsmooth => Xsmooth,
        :Z => Zsmooth[2:end,:],
        :C => C,
        :R => R,
        :A => A,
        :Q => Q,
        :means => [mean(skipmissing(row)) for row in eachrow(y_est)] |> x-> reshape(x, 1, size(x)[1]),
        :stds => [std(skipmissing(row)) for row in eachrow(y_est)] |> x-> reshape(x, 1, size(x)[1]),
        :Z0 => Z0,
        :V0 => V0,
        :p => p,
        :num_iter => num_iter,
        :convergence => converged,
        :loglik => loglik,
        :LL => LL[2:end]
    )
    return nowcast
end

"outputs an output_dfm dict to disk (path = foldername)"
function export_dfm(;output_dfm::Dict, out_path::String)
    DataFrame(output_dfm[:Xsmooth]) |> x-> CSV.write("$out_path/X_smooth.csv", x) # :X_smooth
    DataFrame(output_dfm[:Z]) |> x-> CSV.write("$out_path/Z.csv", x) # :Z
    DataFrame(output_dfm[:C]) |> x-> CSV.write("$out_path/C.csv", x) # :C
    DataFrame(output_dfm[:R]) |> x-> CSV.write("$out_path/R.csv", x) # :R
    DataFrame(output_dfm[:A]) |> x-> CSV.write("$out_path/A.csv", x) # :A
    DataFrame(output_dfm[:Q]) |> x-> CSV.write("$out_path/Q.csv", x) # :Q
    DataFrame(output_dfm[:means]) |> x-> CSV.write("$out_path/means.csv", x) # :means
    DataFrame(output_dfm[:stds]) |> x-> CSV.write("$out_path/stds.csv", x) # :stds
    DataFrame(Z0=output_dfm[:Z0]) |> x-> CSV.write("$out_path/Z0.csv", x) # :Z0
    DataFrame(output_dfm[:V0]) |> x-> CSV.write("$out_path/V0.csv", x) # :V0
    DataFrame(p=output_dfm[:p]) |> x-> CSV.write("$out_path/p.csv", x) # :p
    DataFrame(num_iter=output_dfm[:num_iter]) |> x-> CSV.write("$out_path/num_iter.csv", x) # :num_iter
    DataFrame(convergence=output_dfm[:convergence]) |> x-> CSV.write("$out_path/convergence.csv", x) # :convergence
    DataFrame(loglik=output_dfm[:loglik]) |> x-> CSV.write("$out_path/loglik.csv", x) # :loglik
    DataFrame(LL=output_dfm[:LL]) |> x-> CSV.write("$out_path/LL.csv", x) # :LL
end


"read an output_dfm directory to disk (folder name)"
function import_dfm(;path::String)::Dict
    tmp = Dict()
    tmp[:Xsmooth] = CSV.read("$path/X_smooth.csv") |> Array # :X_smooth
    tmp[:Z] = CSV.read("$path/Z.csv") |> Array # :Z
    tmp[:C] = CSV.read("$path/C.csv") |> Array # :C
    tmp[:R] = CSV.read("$path/R.csv") |> Array # :R
    tmp[:A] = CSV.read("$path/A.csv") |> Array # :A
    tmp[:Q] = CSV.read("$path/Q.csv") |> Array # :Q
    tmp[:means] = CSV.read("$path/means.csv") |> Array # :means
    tmp[:stds] = CSV.read("$path/stds.csv") |> Array # :stds
    tmp[:Z0] = CSV.read("$path/Z0.csv") |> Array # :Z0
    tmp[:V0] = CSV.read("$path/V0.csv") |> Array # :V0
    tmp[:p] = CSV.read("$path/p.csv") |> x-> x[1,1] # :p
    tmp[:num_iter] = CSV.read("$path/num_iter.csv") |> x-> x[1,1]  # :num_iter
    tmp[:convergence] = CSV.read("$path/convergence.csv") |> x-> x[1,1] # :convergence
    tmp[:loglik] = CSV.read("$path/loglik.csv") |> x-> x[1,1] # :loglik
    tmp[:LL] = CSV.read("$path/LL.csv") |> x-> x[1,1] # :LL

    return tmp
end


"""
    get predictions from an already estimated DFM model for all series, all periods.
    parameters:
        data : DataFrame
            dataframe to generate predictions. Must include a date column, must contain the same columns the output_dfm was trained on.
        output_dfm : Dict
            output dictionary of the estimate_dfm function
        months_ahead : Int
            number of months ahead to forecast
        lag : Int
            number of lags for the kalman filter
    returns: DataFrame
      dataframe with all missing values filled + predictions
"""
function predict_dfm(data::DataFrame; output_dfm, months_ahead::Int, lag=0)
    date_col = date_col_name(data)
    Y = copy(data[!, Not(date_col)])
    dates = copy(data[!, date_col])

    # add months
    last_date = data[end, date_col]
    row = DataFrame(eachrow(Y)[1:months_ahead])
    row .= missing
    Y = vcat(Y, row)
    for i in 1:months_ahead
        push!(dates, last_date + Month(i))
    end

    predictions = kalman_filter_constparams(Y; output_dfm=output_dfm, lag=lag)[:X_smooth] |> DataFrame
    predictions[!, :date] = dates
    predictions =  predictions[!, [:date; names(predictions)[1:end-1]]]
    rename!(predictions, names(data))
    return predictions
end
