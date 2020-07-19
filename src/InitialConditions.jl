include("HelperFunctions.jl")

using LinearAlgebra
using SparseArrays

"""
    calculates initial conditions for parameter estimation.
    parameters:
        Y : DataFrame | Array{Number, 2}
            matrix of variables, size (n, z). Requires standardized matrix, pass standardize(df) if not already standardized. Requires no missings, pass fill_na(df) if not the case.
        blocks : DataFrame, size (n_variables, n_blocks)
            matrix of 1s or 0s for block loadings, i.e. included in block.
            ex for Y 2 x 3, 2 blocks:
                [1, 0]
                [1, 1]
                [0, 1]
        R_mat : Array{Int64, 2}
            quarterly-monthly aggregation scheme matrix
    returns: ?
        ?
"""
# temporary
blocks = DataFrame(ones(ncol(data[!, Not(:date)]), 2))
function initialize_conditions(Y; dates, p, blocks, R_mat)
    y_tmp = copy(Y) |> standardize
    q = zeros(4, 1)
    monthly_quarterly_array = gen_monthly_quarterly(dates, y_tmp)
    nM = (monthly_quarterly_array .== false) |> sum # number of monthly variables
    nQ = sum(monthly_quarterly_array) # number of quarterly variables
    nBlocks = size(blocks)[2]
    # put all quarterly variables to the end
    y_tmp = [y_tmp[!, .!monthly_quarterly_array]  y_tmp[!, monthly_quarterly_array]]

    tmp = fill_na(y_tmp)
    y_filled = tmp[:output]
    na_indices = tmp[:na_indices]
    n_rows = nrow(y_filled)
    n_cols = ncol(y_filled)

    res = copy(y_filled)
    res_tmp = copy(res)
    allowmissing!(res_tmp)
    for i in 1:ncol(res_tmp)
        res_tmp[na_indices[!, i], i] .= missing
    end

    pC = 5
    ppC = maximum([p, pC])
    C = missing
    A = missing
    Q = missing
    V0 = missing
    na_indices[1:(pC-1), :] .= true

    for i in 1:nBlocks
        C_i = zeros(n_cols, ppC) # Initialize state variable matrix helper
        idx_i = findall(x-> x==1.0, blocks[!, i]) # Indices of series loading in block i
        idx_iM = idx_i[idx_i .<= nM] # Monthly series indices that load in block i
        idx_iQ = idx_i[idx_i .> nM] # Quarterly series indices that loaded in block i
        cov_matrix = cov(res[!, idx_iM] |> Array) # Eigenvector of cov matrix of monthly data and largest eigenvalue
        d = eigvals(cov_matrix)[end] # Largest eigenvalue...
        v = eigvecs(cov_matrix)[:, end] |> x-> # ...and the associated eigenvector
            sum(x) < 0 ? -x : x # Flip sign for clearer output (not required, but facilitates reading)
        C_i[idx_iM, 1] = v
        f = Array(res[!, idx_iM]) * v # Data projection for eigenvector direction
        for j in 0:(maximum([p+1, pC]) - 1)
            global F = j == 0 ?  f[(pC-j):(length(f)-j)] : hcat(F, f[(pC-j):(length(f)-j)]) # Lag matrix
        end

        ff = F[:, 1:pC] # Projected data with lag structure, so pC-1 fewer entries
        for j in idx_iQ
            xx_j = res_tmp[pC:nrow(res_tmp), j] # For series j, values are dropped to accommodate lag structure
            if sum(ismissing.(xx_j)) < size(ff, 2) + 2
                xx_j = res[pC:nrow(res), j] # Replaces xx_j with spline if too many NaNs
            end
            ff_j = ff[.!ismissing.(xx_j),:]
            xx_j = xx_j[.!ismissing.(xx_j)]
            iff_j = inv(transpose(ff_j) * ff_j)
            Cc = iff_j * transpose(ff_j) * xx_j   # OLS
            Cc = Cc - iff_j * transpose(R_mat) * inv(R_mat * iff_j * transpose(R_mat)) * (R_mat * Cc - q) # Spline data monthly to quarterly conversion
            C_i[j, 1:pC] = transpose(Cc)   # Place in output matrix
        end

        ff = vcat(zeros(pC-1, pC), ff)   # Zeros in first pC-1 entries (replacing dropped from lag)
        global res = res .- ff * transpose(C_i)   # Residual calculations
        global res_tmp = copy(res)
        allowmissing!(res_tmp)
        for i in 1:ncol(res_tmp)
            res_tmp[na_indices[!, i], i] .= missing
        end
        global C = ismissing(C) ? C_i : hcat(C, C_i) # Combine past loadings together

        # transition equation
        z = F[:, 1] # Projected data (no lag)
        Z = F[:, 2:(p+1)] # Data with lag 1
        A_i = zeros(ppC, ppC) # Initialize transition matrix
        A_tmp = inv(transpose(Z) * Z) * transpose(Z) * z # OLS: gives coefficient value AR(p) process
        A_i[1, 1:p] = transpose(A_tmp)
        A_i[2:size(A_i)[1], 1:(ppC-1)] = I(ppC-1)

        Q_i = zeros(ppC, ppC)
        e = z - Z * A_tmp # VAR residuals
        Q_i[1, 1] = cov(e) # VAR covariance matrix
        initV_i = reshape(inv(I(ppC ^ 2) - kron(A_i, A_i)) * reshape(Q_i, (prod(size(Q_i)),1)), (ppC, ppC))
        # Gives top left block for the transition matrix
        global A = ismissing(A) ? A_i : blockdiag(sparse(A), sparse(A_i)) |> Array
        global Q = ismissing(Q) ? Q_i : blockdiag(sparse(Q), sparse(Q_i)) |> Array
        global V0 = ismissing(V0) ? initV_i : blockdiag(sparse(V0), sparse(initV_i)) |> Array
    end

end
