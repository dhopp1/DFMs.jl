using DataFrames
using Statistics
include("HelperFunctions.jl")

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

print(standardize(data))
