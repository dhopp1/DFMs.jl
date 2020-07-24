include("HelperFunctions.jl")

"""
    This function calculates changes in news by using a given DFM results  structure. It inputs two datasets, DFM parameters, target time index, and target variable index. The function then produces Nowcast updates and decomposes the changes into news.
     parameters:
        data_old : DataFrame
            old/vintage dataframe, including a date column. Must have same columns as data_new
        data_new : DataFrame
            new data dataframe, including a date column
        output_dfm : Dict
            output of the estimate_dfm function
        target_variable : Symbol
            target variable column name
        target_period : Date
            target period dates
    returns: Dict
       y_old => old nowcast
       y_new => new nowcast
       singlenews => news for each data series
       actual => observed series release values
       forecast => forecasted series values
       weight => news weight
       t_miss => time index for data releases
       v_miss => series index for data releases
       innov => difference between observed and predicted series values ("innovation")
"""
function news_dfm(;old_y, new_y, output_dfm, target_variable, target_period)
    # initializing data
    old_date_col = date_col_name(old_y)
    new_date_col = date_col_name(new_y)
    old_dates = old_y[!, old_date_col]
    new_dates = new_y[!, new_date_col]
    data_old = old_y[!, Not(old_date_col)]
    data_new = new_y[!, Not(new_date_col)]
    # making sure old data has same number of rows/dates as new data
    old_y = join(DataFrame(date=new_dates), old_y, on=Pair(:date, old_date_col), kind=:left)
    data_old = old_y[!, Not(:date)]
    target_index = findall(x-> x == target_variable, names(data_new))
    target_old = target_new = zeros(1,length(target_index))
    target_period_index = findall(x-> x==target_period, new_dates)[1]

    r = size(output_dfm[:C])[2]
    N = size(data_new)[2]
    singlenews = repeat([0], N) |> x-> reshape(x, 1, length(x))

    # forecasting
    if !ismissing(data_new[target_period_index, target_index][1]) # no forecast if value for target already exists at target date
        results_old = kalman_filter_constparams(data_old, output_dfm, 0)
    else # yes forecast case, steps A and B
    end

    tmp = join(DataFrame(date=new_dates), old_y, on=Pair(:date, old_date_col), kind=:left)


end


"""
    tmp
     parameters:
        tmp : Array
            tmp
    returns: Dict
       tmp
"""
function update_nowcast()
end
