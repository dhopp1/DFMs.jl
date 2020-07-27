using DataFrames
using Statistics
using CubicSplines
using DSP
using Dates

export standardize
export spline_fill
export digital_filter
export fill_na
export date_col_name

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
    returns: Dict
        Dict(:output => df with NAs filled, :na_indices => df of Bool is na)
"""
function fill_na(df::DataFrame)
    tmp = copy(df)
    # adding a false at end for the index column
    is_numeric = [numeric_cols(tmp); false]
    na_indices = ismissing.(tmp)
    tmp[!, :index] = 1:nrow(tmp)

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

    return Dict(:output => tmp[!, Not(:index)], :na_indices => na_indices[tmp.index,:])
end


"""
    generate matrix of monthly and quarterly variables
    parameters:
        dates : Array{Dates.Date}
            array of corresponding dates for the matrix
        df : DataFrame
            dataframe to get monthly and quarterly variables from
    returns: Array{Bool, 1}
        array of 1's (is quarterly) and 0's (is not quarterly, i.e. monthly)
"""
function gen_monthly_quarterly(dates::Array{Dates.Date}, df::DataFrame)::Array{Bool, 1}
    is_quarterly = []
    for i in 1:ncol(df)
        quarterly = ([Month(x).value for x in dates[.!ismissing.(df[!, i])]] |> Set) |> x->
            isequal(x, Set([3,6,9,12]))
        push!(is_quarterly, quarterly)
    end
    return is_quarterly
end


"""
    return name of first date column of a dataframe
    parameters:
        df : DataFrame
            dataframe containing a date column
    returns: Symbol
        name of the first date column
"""
function date_col_name(df::DataFrame)
    try
        return findall(x->x==Dates.Date, eltype.(eachcol(df)))[1] |> i->
            Symbol(names(df)[i])
    catch e
        println("No column of type Dates.Date")
    end
end


"""
    given a dataframe, create a lagged dataframe where lag months back will be set to missing
    parameters:
        x : DataFrame
            dataframe to be artificially lagged
        lag : Int
            number of months to lag dataframe by
    returns: DataFrame
        lagged dataframe
"""
function create_lag(x, lag)
    new = copy(x)
    for i in 2:ncol(x)
        last_data_index = findall(!ismissing, x[!, i])[end]
        new[(last_data_index-lag+1):last_data_index, i] .= missing
    end
    return new
end
