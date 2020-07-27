include("HelperFunctions.jl")

export gen_news

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
    old_y = leftjoin(DataFrame(date=new_y[!, date_col_name(new_y)]), old_y, on=:date=>date_col_name(old_y))
    data_old = old_y[!, Not(:date)]
    target_index = findall(x-> x == target_variable, Symbol.(names(data_new)))
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
        actual = missing; forecast = missing; weight = missing; t_miss = missing; v_miss = missing; innov = missing; row_miss = missing; col_miss = missing;
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
        # Forecast subcase (b): new information
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
    This function calculates the DFM nowcast estimates at a current data vintage and compare it with the results from a previous data vintage (pass the same dataset for old_y and new_y to generate an artifical 1-month lagged dataset). The changes are classified as data revisions and data releases, and their impact on the nowcast estimate is presented separately.
     parameters:
        old_y : DataFrame
            DataFrame of older data, including date column
        new_y : DataFrame of newer data
            DataFrame of newest data, including date column
        output_dfm : Dict
            Output/parameters of the estimate_dfm function. No model estimation happens here, this compares new data revisions with pre-estimated parameters.
        target_variable : Symbol
            the target variable column name
        target_period : Dates.Date
            the date for which a forecast is desired
    returns: Dict
        :news_table : DataFrame
           series => the name of the variable
           forecast => the forecast value for the desired time period
           actual => the most recent actually observed value of the variable
           weight => weight of each data release
           impact_releases => impact of data releases on nowcast
           impact_total => impact of data_release + data_revisions on nowcast
           data_release => was this series released between the old and newer datasets
       :y_old : Float
            previous prediction for y
       :y_hat : Float
            prediction for y from new data
"""
function gen_news(;old_y::DataFrame, new_y::DataFrame, output_dfm::Dict, target_variable::Symbol, target_period::Dates.Date)
    # artificial lagged dataset if no old data given
    if isequal(old_y, new_y)
        old_y = create_lag(new_y, 1)
    end
    # making sure old data has same number of rows/dates as new data
    old_y = leftjoin(DataFrame(date=new_y[!, date_col_name(new_y)]), old_y, on=:date=>date_col_name(old_y))

    # add 12 months to each dataset to allow for forecasting
    months_ahead = 12
    last_date = new_y[end, :date]
    row = DataFrame(eachrow(new_y)[1:months_ahead])
    row[!, Not(date_col_name(new_y))] .= missing
    old_y = vcat(old_y, row)
    new_y = vcat(new_y, row)
    old_y[end-months_ahead+1:end, :date] = new_y[end-months_ahead+1:end, :date] = [last_date + Month(i) for i in 1:months_ahead]

    # Update nowcast for target variable 'series' (i) at horizon 'target' (t)
    # Relate nowcast update into news from data releases:
    #   a. Compute the impact from data revisions
    #   b. Compute the impact from new data releases
    data_rev = copy(new_y)
    old_missing = ismissing.(old_y)
    for i in 1:ncol(old_y)
        if eltype(data_rev[!, i]) != Dates.Date
            data_rev[old_missing[!, i], i] .= missing
        end
    end

    # Compute impact from data revisions
    results_old = news_dfm(;old_y=old_y, new_y=data_rev, output_dfm=output_dfm, target_variable=target_variable, target_period=target_period)
    y_old = results_old[:y_old]; old_forecast = results_old[:forecast]

    # Compute impact from data releases
    results_new = news_dfm(;old_y=data_rev, new_y=new_y, output_dfm=output_dfm, target_variable=target_variable, target_period=target_period)
    y_rev = results_new[:y_old]; y_new = results_new[:y_new];
    actual = results_new[:actual]; forecast = results_new[:forecast]; weight = results_new[:weight]

    impact_revisions = y_rev - y_old # Impact from revisions
    news = actual .- forecast # News from releases
    impact_releases = weight[:,:,1] .* news # Impact of releases

    news_table = DataFrame(
        forecast=forecast[:,1],
        actual=actual[:,1],
        weight=weight[:,1],
        impact_releases=impact_releases[:,1],
        impact_total=impact_releases[:,1] .+ impact_revisions
    )
    news_table = DataFrame(Array(news_table) * diagm([100,100,1,100, 100])) |> x-> rename!(x, names(news_table))
    news_table[!, :series] = names(new_y) |> x-> x[.!occursin.(string.(x), String(date_col_name(new_y)))] .|> string
    news_table = news_table[!, [:series; Symbol.(names(news_table)[1:end-1])]]
    data_released = ismissing.(old_y[old_y[!, date_col_name(old_y)] .== target_period, Not(date_col_name(old_y))]) .& # data newly released between two datasets
        .!ismissing.(new_y[new_y[!, date_col_name(new_y)] .== target_period, Not(date_col_name(new_y))]) |> Array |> x-> reshape(x, size(x)[2])
    news_table[!, :data_release] = data_released

    return Dict(
        :news_table => news_table,
        :y_old => y_old,
        :y_new => y_new
    )
end
