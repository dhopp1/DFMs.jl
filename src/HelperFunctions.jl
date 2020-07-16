using DataFrames
using Statistics

export standardize

"""
    standardize all numeric columns in a dataframe. _newxᵢ = (xᵢ - μ) / σ²_
    parameters:
        df : DataFrame
            dataframe to standardize
    returns: DataFrame
        dataframe with standardized columns
"""
function standardize(df)
    tmp = copy(df)
    for i in 1:ncol(tmp)
        if (Float64 <: eltype(tmp[!, i])) | (Int64 <: eltype(tmp[!, i]))
            tmp[!, i] = (tmp[!, i] .- mean(skipmissing(tmp[!, i]))) / std(skipmissing(tmp[!, i]))
        end
    end
    return tmp
end

