include("KalmanFilter.jl")

"""
    This function reestimates parameters based on the Estimation Maximization (EM) algorithm. This is a two-step procedure:
    (1) E-step: the expectation of the log-likelihood is calculated using previous parameter estimates.
    (2) M-step: Parameters are re-estimated through the maximisation of the log-likelihood (maximise result from (1)).
    See "Maximum likelihood estimation of factor models on data sets with arbitrary pattern of missing data" for details about parameter derivation (Banbura & Modugno, 2010).

    parameters:
        y: Array
            data (standardized and ready to be used)
        A: Array
            transition matrix
        C: Array
            measurement matrix
        Q: Array
            covariance for transition equation residuals
        R: Array
            covariance for measurement equation residuals
        Z0: Array
            initial values of state
        V0: Array
            initial value of factor covariance matrix
        p: Int
            number of lags in transition equation
        blocks: Array
            block structure for each series
        R_mat: Array
            estimation structure for quarterly variables (i.e. "tent")
        q: Array
            constraints on loadings
        nM: Int
            number of monthly variables
        nQ: Int
            number of quarterly variables
        monthly_quarterly_array: Array
            indices for monthly variables
    returns: Dict
        A_new => updated transition matrix
        C_new => updated measurement matrix
        Q_new => updated covariance matrix for residuals for transition matrix
        R_new => updated covariance matrix for residuals of measurement equation
        Z0 => initial value of state
        V0 => initial value of factor covariance matrix
        loglik: log likelihood
"""
function EM_step(y_est, A, C, Q, R, Z0, V0, p, blocks, R_mat, q, nM, nQ, monthly_quarterly_array)
    n = size(y_est)[1]
    n_obs = size(y_est)[2]
    pC = size(R_mat)[2]
    ppC = maximum([p, pC])
    n_blocks = size(blocks)[2]

    output = kalman_filter(y_est, A, C, Q, R, Z0, V0)
end
