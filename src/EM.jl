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
            indices for quarterly variables
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
            println(sum(Wt))
            bl_idxM_ext = vcat(bl_idxM[i,:], repeat([false], size(Zsmooth)[1] - length(bl_idxM)))
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

    end
end
