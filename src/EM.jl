include("KalmanFilter.jl")

export EM_step
export EM_convergence

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
        monthly_quarterly_array: Array
            indices for quarterly variables
    returns: Dict
        A_new => updated transition matrix
        C_new => updated measurement matrix
        Q_new => updated covariance matrix for residuals for transition matrix
        R_new => updated covariance matrix for residuals of measurement equation
        Z0 => initial value of state
        V0 => initial value of factor covariance matrix
        loglik => log likelihood
"""
function EM_step(y_est; A, C, Q, R, Z0, V0, p, blocks, R_mat, q, nM, monthly_quarterly_array)
    n = size(y_est)[1]
    n_obs = size(y_est)[2]
    pC = size(R_mat)[2]
    ppC = maximum([p, pC])
    n_blocks = size(blocks)[2]
    nM = sum(.!monthly_quarterly_array)
    y = copy(y_est)

    output = kalman_filter(y_est, A, C, Q, R, Z0, V0)
    Zsmooth = output[:Zsmooth]; Vsmooth = output[:Vsmooth]; VVsmooth = output[:VVsmooth]; loglik = output[:loglik]

    ### MAXIMIZATION STEP (TRANSITION EQUATION). See Banbura & Modugno (2010) for details.
    A_new = copy(A)
    Q_new = copy(Q)
    V0_new = copy(V0)

    # 2A. Update factor parameters individually
    for i in 1:n_blocks # Loop for each block: factors are uncorrelated
        # Setup indexing
        p1 = (i - 1) * ppC
        b_subset = (p1 + 1):(p1 + p) # Subset blocks: Helps for subsetting Zsmooth, Vsmooth
        t_start = p1 + 1 # Transition matrix factor idx start
        t_end = p1 + ppC # Transition matrix factor idx end

        # Estimate factor portion of Q, A. Note: EZZ, EZZ_BB, EZZ_FB are parts of equations 6 and 8 in BM 2010
        # E[f_t * f_t' | Omega_T]
        EZZ = (Zsmooth[b_subset, 2:end] * transpose(Zsmooth[b_subset, 2:end])) |> x->
            x + reshape(sum(Vsmooth[b_subset, b_subset, 2:end], dims=3), size(x)...)
        # E[f_{t-1} * f_{t-1}' | Omega_T]
        EZZ_BB = (Zsmooth[b_subset, 1:(end-1)] * transpose(Zsmooth[b_subset, 1:(end-1)])) |> x->
            x + reshape(sum(Vsmooth[b_subset, b_subset, 1:(end-1)], dims=3), size(x)...)
        # E[f_t * f_{t-1}' | Omega_T]
        EZZ_FB = (Zsmooth[b_subset, 2:end] * transpose(Zsmooth[b_subset, 1:(end-1)])) |> x->
            x  + reshape(sum(VVsmooth[b_subset,b_subset,:], dims=3), size(x)...)
        # Select transition matrix/covariance matrix for block i
        Ai = A[t_start:t_end, t_start:t_end]
        Qi = Q[t_start:t_end, t_start:t_end]
        Ai[1,1:p] = EZZ_FB[1,1:p] * inv(EZZ_BB[1:p, 1:p]) # Equation 6: Estimate VAR(p) for factor
        Qi[1,1] = ((EZZ[1,1] .- Ai[1, 1:p] * transpose(EZZ_FB[1, 1:p])) / n_obs)[1] # Equation 8: Covariance matrix of residuals of VAR
        # Place updated results in output matrix
        A_new[t_start:t_end,t_start:t_end] = copy(Ai)
        Q_new[t_start:t_end,t_start:t_end] = copy(Qi)
        V0_new[t_start:t_end,t_start:t_end] = Vsmooth[t_start:t_end, t_start:t_end,1]
    end

    # 2B. Update parameters for idiosyncratic component
    rp1 = n_blocks * ppC # Col size of factor portion
    t_start = rp1 + 1 # Start of idiosyncratic component index
    i_subset = t_start:(rp1 + nM) # Gives indices for monthly idiosyncratic component values

    #% The three equations below estimate the idiosyncratic component (for eqns 6, 8 BM 2010)
    # E[f_t * f_t' | \Omega_T]
    EZZ = (diag(Zsmooth[t_start:end, 2:end] * transpose(Zsmooth[t_start:end, 2:end])) +
        diag(sum(Vsmooth[t_start:end, t_start:end, 2:end], dims = 3) |> x-> reshape(x, size(x)[1:2]...))) |>
        diagm
    # E[f_{t-1} * f_{t-1}' | \Omega_T]
    EZZ_BB = (diag(Zsmooth[t_start:end, 1:(end-1)] * transpose(Zsmooth[t_start:end, 1:(end-1)])) +
        diag(sum(Vsmooth[t_start:end, t_start:end, 1:(end-1)], dims = 3) |> x-> reshape(x, size(x)[1:2]...))) |>
        diagm
    # E[f_t * f_{t-1}' | \Omega_T]
    EZZ_FB = (diag(Zsmooth[t_start:end, 2:end] * transpose(Zsmooth[t_start:end, 1:(end-1)])) +
        diag(sum(VVsmooth[t_start:end, t_start:end, :], dims = 3) |> x-> reshape(x, size(x)[1:2]...))) |>
        diagm

    Ai = EZZ_FB * diagm(transpose(1 / diag(EZZ_BB))) # Equation 6
    Qi = (EZZ - Ai * transpose(EZZ_FB)) / n_obs # Equation 8

    # Place updated results in output matrix
    A_new[i_subset, i_subset] = Ai[1:nM, 1:nM]
    Q_new[i_subset, i_subset] = Qi[1:nM, 1:nM]
    V0_new[i_subset, i_subset] = diagm(diag(Vsmooth[i_subset, i_subset, 1]))

    # 3. Maximization step (measurement equation)
    Z0 = Zsmooth[:,1]
    # Set missing data series values to 0
    y[ismissing.(y)] .= 0
    C_new = copy(C) # loadings
    bl = unique(blocks) # Gives unique loadings
    n_bl = size(bl)[1] # Number of unique loadings
    # Initialize indices: these later help with subsetting
    bl_idxM = [] # Indicator for monthly factor loadings
    bl_idxQ = [] # Indicator for quarterly factor loadings
    R_con = [] # Block diagonal matrix giving monthly-quarterly aggreg scheme
    q_con = []
    # Loop through each block
    for i in 1:n_blocks
        push!(bl_idxQ, repeat(bl[:,i], 1, ppC))
        push!(bl_idxM, hcat(repeat(bl[:,i], 1, 1), zeros(n_bl, ppC - 1)))
        push!(R_con, R_mat)
        push!(q_con, zeros(size(R_mat)[1], 1))
    end
    bl_idxQ = hcat(bl_idxQ...) .== 1.0
    bl_idxM = hcat(bl_idxM...) .== 1.0
    R_con = blockdiag(sparse.(R_con)...) |> collect
    q_con = vcat(q_con...)

    # Indicator for monthly/quarterly blocks in measurement matrix
    index_freq_M = .!monthly_quarterly_array[1:nM] # Gives 1 for monthly series
    n_index_M = length(index_freq_M) # Number of monthly series
    c_index_freq = cumsum(.!monthly_quarterly_array) # Cumulative number of monthly series

    for i in 1:n_bl # Loop through unique loadings
        bl_i = bl[i,:]
        rs = sum(bl_i) # Total num of blocks loaded
        idx_i = findall(x->x == true, [x == bl_i for x in eachrow(blocks)]) # Indices for bl_i (which variables in block)
        idx_iM = idx_i[idx_i .<= nM] # Only monthly series
        n_i = length(idx_iM) # Number of monthly series
        # Initialize sums in equation 13 of BGR 2010
        denom = zeros(Int(n_i * rs), Int(n_i * rs))
        nom = zeros(Int(n_i), Int(rs))

        # Stores monthly indicies. These are done for input robustness
        index_freq_i = index_freq_M[idx_iM]
        index_freq_ii = c_index_freq[idx_iM]
        index_freq_ii = index_freq_ii[index_freq_i]

        # Update monthly variables: loop through each period
        for t in 1:n_obs
            Wt = diagm(.!ismissing.(y_est)[idx_iM,t]) * 1 # Gives selection matrix (TRUE for nonmissing values)
            global bl_idxM_ext = vcat(bl_idxM[i,:], repeat([false], size(Zsmooth)[1] - length(bl_idxM)))
            # E[f_t * t_t' | Omega_T]
            denom = denom + kron(Zsmooth[bl_idxM_ext, t + 1] * transpose(Zsmooth[bl_idxM_ext, t + 1]) + Vsmooth[bl_idxM_ext, bl_idxM_ext, t + 1], Wt)
            # E[y_t * f_t' | Omega_T]
            nom = nom + y[idx_iM, t] *
                transpose(Zsmooth[bl_idxM_ext, t + 1]) -
                Wt[:,index_freq_i] * (Zsmooth[rp1 .+ index_freq_ii, t + 1] *
                transpose(Zsmooth[bl_idxM_ext, t + 1]) +
                Vsmooth[rp1 .+ index_freq_ii, bl_idxM_ext, t + 1])
        end
        vec_C = inv(denom) * reshape(nom, prod(collect(size(nom)), dims=1)[1]) # Eqn 13 BGR 2010
        C_new[idx_iM, bl_idxM_ext] = reshape(vec_C, Int(n_i), Int(rs)) # Place updated monthly results in output matrix
        idx_iQ = idx_i[idx_i .> nM] # Index for quarterly series
        rps = rs * ppC
        # Monthly-quarterly aggregation scheme
        R_con_i = R_con[:,bl_idxQ[i,:]]
        q_con_i = copy(q_con)
        no_c = [!all(x->Int(x)==0,row) for row in eachrow(R_con_i)] # any non-zero values in row
        R_con_i = R_con_i[no_c,:]
        q_con_i = q_con_i[no_c,:]

        # Loop through quarterly series in loading, this parallels monthly code
        for j in idx_iQ
            # Initialization
            denom = zeros(Int(rps), Int(rps))
            nom = zeros(1, Int(rps))
            idx_jQ = j - nM # Ordinal position of quarterly variable
            # Location of factor structure corresponding to quarterly variable residuals
            index_freq_jQ = (rp1 + n_index_M + 5 * (idx_jQ - 1) + 1):(rp1 + n_index_M + 5 * idx_jQ)
            # Place quarterly values in output matrix
            V0_new[index_freq_jQ, index_freq_jQ] = Vsmooth[index_freq_jQ, index_freq_jQ, 1]
            A_new[index_freq_jQ[1], index_freq_jQ[1]] = Ai[index_freq_jQ[1] - rp1, index_freq_jQ[1] - rp1]
            Q_new[index_freq_jQ[1], index_freq_jQ[1]] = Qi[index_freq_jQ[1] - rp1, index_freq_jQ[1] - rp1]
            # Update quarterly variables: loop through each period
            for t in 1:n_obs
                Wt = diagm(.!ismissing.(y_est)[idx_iQ,t]) .* 1.0 # Selection matrix for quarterly values
                # Intermediate steps in BGR equation 13
                bl_idxQ_ext = vcat(bl_idxQ[i,:], repeat([false], size(Zsmooth)[1] - size(bl_idxQ)[2]))
                denom = denom + kron(Zsmooth[bl_idxQ_ext, t + 1] * transpose(Zsmooth[bl_idxQ_ext, t + 1]) +  Vsmooth[bl_idxQ_ext, bl_idxQ_ext, t + 1], Wt)
                nom = nom + y[j,t] * transpose(Zsmooth[bl_idxQ_ext, t + 1])
                nom = nom - Wt * ([1 2 3 2 1] * Zsmooth[index_freq_jQ, t + 1] * transpose(Zsmooth[bl_idxQ_ext, t + 1]) + [1 2 3 2 1] * Vsmooth[index_freq_jQ, bl_idxQ_ext, t + 1])
            end
            C_i = inv(denom) * transpose(nom)
            # BGR equation 13
            C_i_constr = C_i - inv(denom) * transpose(R_con_i) * inv(R_con_i * inv(denom) * transpose(R_con_i)) * (R_con_i * C_i - q_con_i)
            C_new[j, bl_idxQ_ext] = C_i_constr # Place updated values in output structure
        end
    end

    # 3B. Update covariance of residuales for measurement equation
    # Initialize covariance of residuals of observation equation
    R_new = zeros(Int(n), Int(n))
    for t in 1:n_obs
        Wt = diagm(.!ismissing.(y_est)[:,t]) * 1.0 # Selection matrix
        # BGR equation 15
        R_new = R_new + (y[:,t] - Wt * C_new * Zsmooth[:,t + 1]) * transpose(y[:,t] - Wt * C_new * Zsmooth[:,t + 1]) +
            Wt * C_new * Vsmooth[:,:,t + 1] * transpose(C_new) * Wt + (collect(I(n)) - Wt) * R * (collect(I(n)) - Wt)
    end
    RR = diag(R_new)
    RR[.!monthly_quarterly_array] .= 1e-4 # Ensure non-zero measurement error, see Doz, Giannone, Reichlin (2012) for reference
    RR[(nM + 1):end] .= 1e-4
    R_new = diagm(RR)

    # final return
    return Dict(
        :A_new => A_new,
        :C_new => C_new,
        :Q_new => Q_new,
        :R_new => R_new,
        :Z0 => Z0,
        :V0 => V0,
        :loglik => loglik
        )
end


"""
  This function checks whether EM has converged. Convergence occurs if the slope of the log-likelihood function falls below 'threshold' (i.e. f(t) - f(t-1)| / avg < threshold) where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log likelihood at iteration t.
      This stopping criterion is from Numerical Recipes in C (pg. 423). With MAP estimation (using priors), the likelihood can decrease even if the mode of the posterior increases.
  (1) E-step: the expectation of the log-likelihood is calculated using previous parameter estimates.
  (2) M-step: Parameters are re-estimated through the maximisation of the log-likelihood (maximise result from (1)).
  See "Maximum likelihood estimation of factor models on data sets with arbitrary pattern of missing data" for details about parameter derivation (Banbura & Modugno, 2010).

  parameters:
      loglik: Float
          log-likelihood from current EM iteration
      prev_loglik: Float
          log-likelihood from previous EM iteration
      threshold: Float
          convergence threshhold
  returns: Dict
      converged => 1 if convergence criteria satistifed, 0 otherwise
      decrease => 1 if log likelihood decreased, 0 otherwise
"""
function EM_convergence(loglik, prev_loglik, threshold)::Dict{Symbol, Int64}
    # initialization
    converged = 0
    decrease = 0

    if (loglik - prev_loglik) < -1e-3 # allows for imprecision
        decrease = 1
    end
    # check convergence criteria
    delta = abs(loglik - prev_loglik)
    avg_loglik = (abs(loglik) + abs(prev_loglik) + eps()) / 2

    if ((delta / avg_loglik) < threshold)
        converged = 1
    end

    return Dict(
        :converged => converged,
        :decrease => decrease
    )
end
