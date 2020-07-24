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
    singlenews = repeat([0.0], N) |> x-> reshape(x, 1, length(x))

    y_old = y_new = repeat([0.0],1,length(target_index))

    # forecasting
    if !ismissing(data_new[target_period_index, target_index][1]) # no forecast if value for target already exists at target date
        results_old = kalman_filter_constparams(data_old; output_dfm=output_dfm, lag=0)
        # Loop for each target variable
        for i in 1:length(target_index)
            # (Observed value) - (predicted value)
            singlenews[:,target_index[i]] .= data_new[target_period_index, target_index[i]] - results_old[:X_smooth][target_period_index, target_index[i]]
            # Set predicted and observed y values
            y_old[1,i] = results_old[:X_smooth][target_period_index, target_index[i]]
            y_new[1,i] = data_new[target_period_index, target_index[i]]
        end
        # Forecast-related output set to empty
        actual = missing; forecast = missing; weight = missing; t_miss = missing; v_miss = missing; innov = missing;
    else # yes forecast case, steps A and B
        # Initialize series mean/standard deviation
        means = output_dfm[:means]
        stds = output_dfm[:stds]
        # Calculate indicators for missing values (1 if missing, 0 otherwise)
        miss_old = ismissing.(data_old)
        miss_new = ismissing.(data_new)
        # Indicator for missing--combine above information to single matrix where:
        # (i) -1: Value is in the old data, but missing in new data
        # (ii) 1: Value is in the new data, but missing in old data
        # (iii) 0: Values are either both missing or both available in the datasets
        i_miss = miss_old .- miss_new
        row_miss = [i[1] for i in findall(x->x==1, Array(i_miss))] # Time/variable indices where case (ii) is true
        col_miss = [i[2] for i in findall(x->x==1, Array(i_miss))]

        # Forecast subcase (A): No new information
        if length(col_miss) == 0
            # Fill in missing variables using a Kalman filter
            results_old = kalman_filter_constparams(data_old; output_dfm=output_dfm, lag=0)
            results_new = kalman_filter_constparams(data_new; output_dfm=output_dfm, lag=0)
            # Set predicted and observed y values. New y value is set to old
            y_old = results_old[:X_smooth][target_period_index, target_index]
            y_new = y_old
            # No news, so nothing returned for news-related output
            groupnews = missing; singlenews = missing; gain = missing; gainSer = missing;
            actual = missing; forecast = missing; weight = missing; t_miss = missing; v_miss = missing; innov = missing
        # Forecast subcase (b): new informatoin
        else
            # Difference between forecast time and new data time
            lag = target_period_index .- row_miss
            # Gives biggest time interval between forecast and new data
            k = maximum([abs.(lag); (maximum(lag) - minimum(lag))])
            C =- output_dfm[:C] # Measurement matrix
            R = transpose(output_dfm[:R]) # Covariance for measurement matrix residuals
            # Number of new events
            n_news = length(lag)
            # Smooth old dataset
            results_old = kalman_filter_constparams(data_old; output_dfm=output_dfm, lag=k)
            Plag = results_old[:Plag]
            # Smooth new dataset
            results_new = kalman_filter_constparams(data_new; output_dfm=output_dfm, lag=0)
            # Subset for target variable and forecast time
            y_old = results_old[:X_smooth][target_period_index, target_index]
            y_new = results_new[:X_smooth][target_period_index, target_index]
            Vs = results_old[:Vsmooth][:,:,end-1]
            P1 = missing  # Initialize projection onto updates

            # Cycle through total number of updates
            for i in 1:n_news
                h = abs(target_period_index - row_miss[i])
                m = maximum([row_miss[i], target_period_index])
                # If location of update is later than the forecasting date
                if row_miss[i] > target_period_index
                    Pp = Plag[h + 1][:,:,m]  # P[1:r, h*r+1:h*r+r, m]'
                else
                    Pp = transpose(Plag[h + 1][:,:,m])  # P[1:r, h*r+1:h*r+r, m]
                end
                P1 = !ismissing(P1) ? hcat(P1, Pp * C[col_miss[i], 1:r]) : Pp * C[col_miss[i], 1:r]
            end

            innov = Array{Union{Missing, Float64},2}(undef, length(row_miss), 1)
            for i in 1:length(row_miss)
                # Standardize predicted and observed values
                X_new_norm = (data_new[row_miss[i], col_miss[i]] - means[col_miss[i]]) / stds[col_miss[i]]
                X_sm_norm = (results_old[:X_smooth][row_miss[i], col_miss[i]] - means[col_miss[i]]) / stds[col_miss[i]]
                # Innovation: gives [observed] data - [predicted data]
                innov[i] = X_new_norm - X_sm_norm
            end
            ins = size(innov)[1]
            P2 = missing
            p2 = missing
            WW = repeat([0.0], N, N)

            # Gives non-standardized series weights
            for i in 1:length(lag)
                for j in 1:length(lag)
                    h = abs(lag[i] - lag[j])
                    m = maximum([row_miss[i], row_miss[j]])
                    if row_miss[j] > row_miss[i]
                        Pp = Plag[h + 1][:,:,m]
                    else
                        Pp = transpose(Plag[h + 1][:,:,m])
                    end
                    if (col_miss[i] == col_miss[j]) & (row_miss[i] != row_miss[j])
                        WW[col_miss[i], col_miss[j]] = 0
                    else
                        WW[col_miss[i], col_miss[j]] = R[col_miss[i], col_miss[j]]
                    end
                    p2 = !ismissing(p2) ? hcat(p2, transpose(C[col_miss[i], 1:r]) * Pp * C[col_miss[j], 1:r] + WW[col_miss[i], col_miss[j]]) : transpose(C[col_miss[i], 1:r]) * Pp * C[col_miss[j], 1:r] + WW[col_miss[i], col_miss[j]]
                end
                P2 = vcat(P2, p2)
                p2 = missing
            end
            P2 = P2[.![sum(ismissing.(Array(row))) == length(row) for row in eachrow(P2)],:] # drop all missing rows

            totnews = repeat([0.0], 1, length(target_index))
            temp = Array{Union{Missing, Float64},3}(undef, 1, n_news, length(target_index))
            gain = Array{Union{Missing, Float64},3}(undef, 1, n_news, length(target_index))

            for i in 1:length(target_index) # loop on v_news
                # Convert to real units (unstadardized data)
                totnews[1,i] = (stds[target_index[i]] .* transpose(C[target_index[i], 1:r]) * P1 * inv(P2) * innov)[1]
                temp[1,:,i] = stds[target_index[i]] * transpose(C[target_index[i], 1:r]) * P1 * inv(P2) .* transpose(innov)
                gain[:,:,i] = stds[target_index[i]] * transpose(C[target_index[i], 1:r]) * P1 * inv(P2)
            end

            # Initialize output objects
            singlenews = Array{Union{Missing, Float64},3}(undef, maximum(row_miss) - minimum(row_miss) + 1, N, length(target_index))
            actual = Array{Union{Missing, Float64},2}(undef, N, 1) # Actual forecasted values
            forecast =  Array{Union{Missing, Float64},2}(undef, N, 1) # Forecasted values
            weight = Array{Union{Missing, Float64},3}(undef, N, 1, length(target_index))

            # Fill in output values
            for i in 1:length(innov)
                actual[col_miss[i],1] = data_new[row_miss[i], col_miss[i]]
                forecast[col_miss[i],1] = results_old[:X_smooth][row_miss[i], col_miss[i]]
                for j in 1:length(target_index)
                    singlenews[row_miss[i] - minimum(row_miss) + 1, col_miss[i], j] = temp[1,i,j]
                    weight[col_miss[i],1,j] = (gain[:,i,j] / stds[col_miss[i]])[1]
                end
            end
            singlenews = [sum(skipmissing(col)) for col in eachcol(singlenews[:,:,1])]
            col_miss = unique(col_miss)
        end
    end
    return Dict(
        :y_old => y_old,
        :y_new => y_new,
        :singlenews => singlenews,
        :actual => actual,
        :forecast => forecast,
        :weight => weight,
        :row_miss => row_miss,
        :col_miss => col_miss,
        :innov => innov
    )
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
