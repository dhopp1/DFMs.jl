include("HelperFunctions.jl")
include("InitialConditions.jl")
include("EM.jl")

using DataFrames
using Statistics


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
      :sdevs => series standard deviation
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
