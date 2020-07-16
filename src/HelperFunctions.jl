using DataFrames
using Statistics
using CubicSplines

export standardize
export fill_na

"return boolean vector of whether df columns are numeric"
numeric_cols(df::DataFrame) = [(Float64 <: eltype(df[!, i])) | (Int64 <: eltype(df[!, i])) for i in 1:ncol(df)]

"""
    standardize all numeric columns in a dataframe. _newxᵢ = (xᵢ - μ) / σ²_
    parameters:
        df : DataFrame
            dataframe to standardize
    returns: DataFrame
        dataframe with standardized columns
"""
function standardize(df::DataFrame)
    tmp = copy(df)
    is_numeric = numeric_cols(tmp)
    for i in 1:ncol(tmp)
        if is_numeric[i]
            tmp[!, i] = (tmp[!, i] .- mean(skipmissing(tmp[!, i]))) / std(skipmissing(tmp[!, i]))
        end
    end
    return tmp
end


"""
    replace intermediate missings in a series with cubic spline
    parameters:
        series : Array
            series to fill missings
    returns: Array{Float64}
        series with intermediate missings filled in
"""
function spline_fill(series::Array)
    tmp = copy(series) |> x->
        [ismissing(a) ? missing : Float64(a) for a in x]
    xs = findall(.!ismissing.(tmp))
    ys = tmp[xs]
    spline = CubicSpline(xs, ys)
    tmp[xs[1]:xs[end]] = spline[xs[1]:xs[end]]
    return tmp
end


"""
    remove missing/NAs in a dataframe
    parameters:
        df : DataFrame
            dataframe to fill NAs
    returns: DataFrame
        dataframe with filled NAs
"""
function fill_na(df::DataFrame)
    tmp = copy(df)
    # if more than 80% of columns are missing in the beginning, drop the row
    threshold = 0.8
    mask = ([(x |> Array .|> ismissing |> sum) / ncol(tmp) for x in eachrow(tmp)] .<= threshold)
    mask = (mask .== 1) .| ([mask[1:end-1]; 1] .== 1)
    tmp = tmp[mask, :]

    # fill missing values between with cubic spline
    is_numeric = numeric_cols(tmp)
    for i in 1:ncol(tmp)
        if is_numeric[i]

        end
    end

    return tmp
end
