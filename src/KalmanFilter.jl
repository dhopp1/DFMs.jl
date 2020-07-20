"""
    This function applies a Kalman filter and fixed-interval smoother. The script uses the following model:
        y_t = C_t * Z_t + e_t, for e_t ~ N(0, R)
        Z_t = A * Z_{t-1} + mu_t, for mu_t ~ N(0, Q)
    It then applies a fixed-interval smoother using the results from the KF.
    Throughout this file,
        'm' denotes the number of elements in the state vector Z_t.
        'k' denotes the number of elements (observed variables) in y_t.
        'nobs' denotes the number of time periods for which data are observed.

     parameters:
        y_est : Array
            k x nobs matrix of input data
        A : Array
            m x m transition matrix
        C: Array
            k x m measurement matrix
        Q: Array
            m x m covariance matrix for transition equation residuals (mu_t)
        R: Array
            k x k covariance for measurement matrix residuals (e_t)
        Z0: Array
            1 x m vector, initial value of state
        V0 : Array
            m x m matrix, initial value of factor covariance matrix
    returns: Dict
       Zsmooth => k-by-(nobs+1) matrix, smoothed factor estimates (i.e. Zsmooth[, t + 1] = Z_t|T)
       Vsmooth => k-by-k-by-(nobs+1) array, smoothed factor covariance matrices (i.e. Vsmooth[, , t + 1) = Cov(Z_t|T))
       VVsmooth => k-by-k-by-nobs array, lag 1 factor covariance matrices (i.e. Cov(Z_t, Z_t-1|T))
       loglik => scalar, log-likelihood
"""
function kalman_filter(y_est, A, C, Q, R, Z0, V0)
    
end
