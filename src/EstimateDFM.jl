cd("/home/danhopp/dhopp1/DFMs.jl/src")
include("HelperFunctions.jl")
include("InitialConditions.jl")

using DataFrames
using Statistics


### temporary, setting up sample data
using jdplyr

data = _read_csv("../data/sample_data.csv"; missingstring="NA")
target = Symbol("x_world.sa")
target_data = data[!, target]
# random sample
cols = [1, 78,134,57,95,102,93,73,18,5,62]
data = data[!, cols]
data[!, target] = target_data
# creating an artificial lagged dataset
function create_lag(x, lag)
    new = copy(x)
    for i in 2:ncol(x)
        last_data_index = findall(!ismissing, x[!, i])[end]
        new[(last_data_index-lag+1):last_data_index, i] = missing
    end
    return new
end
data_lag = create_lag(data, 1)
###

#=
# Factor Model
(1), (2), (3) form state space model where common factors R and idiosyncracies E are unobserved states, (2) + (3) describe dynamics of the system
Y = L*R + E | (1) measurement equation
    Y = observed variables y_1,t...n,t
    R = latent common factors f_1,t...r,t
    L = coefficients relating variables to factors λ_1,t...r,t
    E = idiosyncratic errors e_1,t...n,t | cross-sectionally orthagonal

factors and idiosycracies modeled as Gaussian AR processes
f_j,t = a_j * f_j,t-1 + u_j,t, u_j,t ~ N(0, σ^2_uj) for j = 1...r | (2) transition equation
e_i,t = p_i * e_i,t-1 + ε_j,t, u_j,t ~ N(0, σ^2_εi) for i = i...n | (3) transition equation

# Expectation maximization (EM) algorithm + Kalman smoother to estimate
1) first step, initialize by computing principal components, model params estimated with OLS, treating PCs as if they were the common factors R
2) second step, given estimated parameters, updated estimate of common factors R is obtained via Kalman smoother. Stopping here = two-step estimation of common factors.
3) third step, MLE obtained by iterating 1 and 2 until convergence, taking into account at each step uncertainty related to fact that factors are estimated

# Estimation of impact of new data (news)
a = model prediction for a series
b = actual value of the series
c = weight given in the model to the "surprise"/news (b - a). This is how important that variable was in estimating the factor (produced from Kalman filter?s)
impact = c(b-a), how this surprise changes the nowcast

=#

# temporary
Y = copy(data)
blocks = DataFrame(a=ones(size(Y)[2]-1), b=ones(size(Y)[2]-1))
p = 1
threshold = 1e-5
max_iter = 5000
# end temporary

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
    returns: ?
        ?
"""
function estimate_dfm(Y; blocks, r, p, max_iter=5000, threshold=1e-5)
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
    prev_loglik = -1e6
    num_iter = 0
    LL = -1e6
    converged = 0
    y = transpose(y_tmp |> Array) # y for the estimation is WITH missing data

    # EM loop
    while !converged & num_iter <= max_iter
        em_output = EM_step(y_est, A, C, Q, R, Z0, V0, p, blocks, R_mat, q, nM, nQ, monthly_quarterly_array)
    end


end
