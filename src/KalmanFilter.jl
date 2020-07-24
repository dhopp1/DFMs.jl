include("HelperFunctions.jl")

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
       VmU => Filtered factor posterior covariance
       VVsmooth => k-by-k-by-nobs array, lag 1 factor covariance matrices (i.e. Cov(Z_t, Z_t-1|T))
       loglik => scalar, log-likelihood
"""
function kalman_filter(y_est, A, C, Q, R, Z0, V0)
    # initialize
    m = size(C)[2]
    n_obs = size(y_est)[2]
    Zm = Array{Union{Missing, Float64},2}(undef, m, n_obs) # Z_t | t-1 (prior)
    Vm = Array{Union{Missing, Float64},3}(undef, m, m, n_obs) # V_t | t-1 (prior)
    ZmU = Array{Union{Missing, Float64},2}(undef, m, n_obs+1) # Z_t | t (posterior/updated)
    VmU = Array{Union{Missing, Float64},3}(undef, m, m, n_obs+1) # V_t | t (posterior/updated)
    ZmT = [0.0 for i in 1:(m * (n_obs + 1))] |> x-> reshape(x, m, n_obs + 1) # Z_t | T (smoothed states)
    VmT = [0.0 for i in 1:(m * m * (n_obs + 1))] |> x-> reshape(x, m, m, n_obs + 1) # V_t | T = Cov(Z_t|T) (smoothed factor covariance)
    VmT_lag = Array{Union{Missing, Float64},3}(undef, m, m, n_obs) # Cov(Z_t, Z_t-1|T) (smoothed lag 1 factor covariance)
    loglik = 0

    # initial values
    Zu = Z0 # Z_0|0 (In loop, Zu gives Z_t | t)
    Vu = V0 # V_0|0 (In loop, Vu gives V_t | t)
    ZmU[:, 1] = Zu
    VmU[:,:, 1] = Vu

    # kalman filter
    for t in 1:n_obs
        ### Calculate prior distribution
        Z = A * Zu # Use transition equation to create prior estimate for factor, i.e. Z = Z_t|t-1
        V = A * Vu * transpose(A) + Q # Prior covariance matrix of Z (i.e. V = V_t|t-1)
        V = 0.5 * (V + transpose(V))  # Trick to make symmetric

        ### Calculate posterior distribution
        non_missing = findall(!ismissing, y_est[:,t]) # Remove missing series: These are removed from Y, C, and R
        global yt = y_est[:,t][non_missing]
        global Ct = C[non_missing,:]
        Rt = R[non_missing,non_missing]

        # Check if yt contains no data; if this is the case, replace Zu and Vu with prior
        if length(yt) == 0
            Zu = Z
            Vu = V
        else
            # Steps for variance and population regression coefficients:
            VC = V * transpose(Ct)
            iF = inv(Ct * VC + Rt)
            global VCF = VC * iF # Matrix of population regression coefficients (QuantEcon eqn #4)
            innov = yt - Ct * Z # Gives difference between actual and predicted measurement matrix values
            Zu  = Z  + VCF * innov # Update estimate of factor values (posterior)
            Vu = V - VCF * transpose(VC) # Update covariance matrix (posterior) for time t
            Vu = 0.5 * (Vu + transpose(Vu)) # Trick to make symmetric
            loglik = loglik + 0.5 * (log(det(iF)) - transpose(innov) * iF * innov) # Update log likelihood
        end

        ### Store output
        # Store covariance and observation values for t-1 (priors)
        Zm[:,t] = Z
        Vm[:,:,t] = V
        # Store covariance and state values for t (posteriors), i.e. Zu = Z_t|t & Vu = V_t|t
        ZmU[:,t + 1] = Zu
        VmU[:,:,t + 1] = Vu
    end

    # Store Kalman gain k_t
    if length(yt) == 0
        k_t = zeros(m, m)
    else
        k_t = VCF * Ct
    end

    ### Apply fixed interval smoother
    # Fill the final period of ZmT & VmT with posterior values from KF
    ZmT[:,n_obs + 1] = ZmU[:,n_obs + 1]
    VmT[:,:,n_obs + 1] = VmU[:,:,n_obs + 1]
    VmT_lag[:,:,n_obs] = (I(m) - k_t) * A * VmU[:,:,n_obs] # Initialize VmT_1 lag 1 covariance matrix for final period
    J_2 = VmU[:,:, n_obs] * transpose(A) * pinv(Vm[:,:,n_obs]) # Used for recursion process, see companion file for details

    ### Run smoothing algorithm
    # Loop through time reverse-chronologically (starting at final period nobs)
    for t in n_obs:-1:1
        # Store posterior and prior factor covariance values
        VmUt = VmU[:,:,t]
        Vmt = Vm[:,:,t]
        # Store previous period smoothed factor covariance and lag-1 covariance
        Vt = VmT[:,:,t+1]
        Vt_lag = VmT_lag[:,:,t]
        J_1 = copy(J_2)
        ZmT[:,t] = ZmU[:,t] + J_1 * (ZmT[:,t+1] - A * ZmU[:,t])  # Update smoothed factor estimate
        VmT[:,:,t] = VmUt + J_1 * (Vt - Vmt) * transpose(J_1) # Update smoothed factor covariance matrix
        if t > 1
            J_2 = VmU[:,:,t-1] * transpose(A) * pinv(Vm[:,:,t-1]) # Update weight
            VmT_lag[:,:,t-1] = VmUt * transpose(J_2) + J_1 * (Vt_lag - A * VmUt) * transpose(J_2) # Update lag 1 factor covariance matrix
        end
    end

    ### output
    return Dict(
        :Zsmooth => ZmT,
        :Vsmooth => VmT,
        :VmU => VmU,
        :VVsmooth => VmT_lag,
        :loglik => loglik
    )
end


"""
    This function applies a Kalman filter for news calculation step, when model parameters are already estimated. This procedure only smoothes and fills missing data for a given data matrix
     parameters:
        data : Array
            input data matrix
        output_dfm : Dict
            output of estimate_dfm function
        lag: Int
            number of lags
    returns: Dict
        Plag => Smoothed factor covariance for transition matrix
        Vsmooth => Smoothed factor covariance matrix
        X_smooth => Smoothed data matrix
        F => Smoothed factors
"""
function kalman_filter_constparams(data; output_dfm, lag)
    # initialization
    Z0 = output_dfm[:Z0]; V0 = output_dfm[:V0]; A = output_dfm[:A]; C = output_dfm[:C]; Q = output_dfm[:Q]; R = output_dfm[:R]; means = output_dfm[:means]; stds = output_dfm[:stds];
    stds = output_dfm[:stds]
    means = output_dfm[:means]
    y = transpose(Array(standardize(data)))
    # y = standardize(data) |> j-> j[[sum(.!ismissing.(Array(x))) > 0 for x in eachrow(j)], :] |> Array |> transpose # temporary, should I removing missing before here?
    kalman_output = kalman_filter(y, A, C, Q, R, Z0, V0)

    Vs = kalman_output[:Vsmooth][:,:,1:end-1] # Smoothed factor covariance for transition matrix
    Vf = kalman_output[:VmU][:,:,1:end-1] # Filtered factor posterior covariance
    Zsmooth = kalman_output[:Zsmooth] # Smoothed factors
    Vsmooth = kalman_output[:Vsmooth] # Smoothed covariance value
    Plag = []
    push!(Plag, Vs)

    if lag > 0
        for jk in 1:lag
            tmp_plag = Array{Union{Missing, Float64},3}(undef, size(C)[2], size(C)[2], size(y)[2])
            for jt in size(Plag[1])[3]:-1:(lag + 1)
                As = Vf[:,:, jt - jk] * transpose(A) * pinv(A * Vf[:,:,jt-jk] * transpose(A) + Q)
                tmp_plag[:,:,jt] = As * Plag[jk][:,:,jt]
            end
            push!(Plag, tmp_plag)
        end
    end
    Zsmooth = transpose(Zsmooth)
    x_sm = Zsmooth[1:end-1,:] * transpose(C) # Factors to series representation
    X_smooth = repeat(stds, size(y)[2]) .* x_sm .+ repeat(means, size(y)[2]) # standardized to unstandardized

    return Dict(
        :Plag => Plag,
        :Vsmooth => Vsmooth,
        :X_smooth => X_smooth,
        :F => Zsmooth[1:end-1,:]
    )
end
