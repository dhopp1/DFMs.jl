using DataFrames
using Statistics
using CubicSplines
using DSP

export standardize
export spline_fill
export digital_filter
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
    fill in missing tails of series using 1-D digital filter
    parameters:
        series : Array
            series to fill leading/trailing missings
        k : Int
            rational transfer function argument's numerator
    returns: Array
        series with filled missings
"""
function digital_filter(series::Array)
    tmp = copy(series) |> x->
        [ismissing(a) ? missing : Float64(a) for a in x]
    missing_indices = findall(ismissing.(tmp))
    tmp[missing_indices] .= median(skipmissing(tmp))
    tmp = tmp .|> Float64

    responsetype = Lowpass(0.9)
    designmethod = Butterworth(1)
    ma = filt(digitalfilter(responsetype, designmethod), tmp)

    tmp[missing_indices] = ma[missing_indices]
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
    is_numeric = numeric_cols(tmp)

    # if more than 80% of columns are missing in the beginning, drop the row
    threshold = 0.8
    mask = ([(x |> Array .|> ismissing |> sum) / ncol(tmp) for x in eachrow(tmp)] .<= threshold)
    mask = (mask .== 1) .| ([mask[1:end-1]; 1] .== 1)
    tmp = tmp[mask, :]

    # if all entries are missing at the end, drop the row
    mask = [!all(ismissing, Array(x)) for x in eachrow(tmp[!, is_numeric])]
    tmp = tmp[mask, :]

    # fill missing values between with cubic spline, fill head/tail with digital filter
    for i in 1:ncol(tmp)
        if is_numeric[i]
            tmp[!, i] = spline_fill(tmp[!, i])
            tmp[!, i] = digital_filter(tmp[!, i])
        end
    end

    return tmp
end
