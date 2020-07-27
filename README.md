# DFMs.jl
Dynamic factor models for Julia. Adapted from [Bok et al. 2017](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr830.pdf), [code](https://github.com/FRBNY-TimeSeriesAnalysis/Nowcasting).

## Installation
```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/dhopp1/DFMs.jl"))
using DFMs
```

## Usage

### Estimate a new model
1) To estimate a new model, two things are necessary:
- Dataframe to input to the model. Must include one column of type `Dates.Date`. N observations and R variables/columns (doesn't include date column).
- Block dataframe of dimension R x B, where B is number of blocks. Values are 1 or 0 for whether or not that variable is to be included in the block.

2) run `output_dfm = estimate_dfm(df; blocks=blocks_df, p=1, max_iter=5000, threshold=1e-5)`, where:
- `p` is number of lags for the AR dimension of the model
- `max_iter` is the max number of times to run the EM
- `threshold` is the threshold for convergence in the EM

3) save the estimated model parameters with `export_dfm(output_dfm=output_dfm, out_path="out_folder")`. The model can now be used for predictions on new datasets where columns remain the same as the dataset it was estimated on.

### Predict on an estimated model
1) either estimate a new model following the instructions above, or load a previously estimated one saved via `export_dfm` as follows:
- `output_dfm = import_dfm(path="out_folder")`

2) obtain predictions which outputs a dataframe with predicted values for all series:
- `predictions = predict_dfm(new_df; output_dfm=output_dfm, months_ahead=3, lag=0)`, where:
	- `new_df` is the new dataset with same columns as that the model was estimated on
	- `output_dfm` is the `estimate_dfm` output either newly estimated or loaded
	- `months_ahead` is the number of months forward to forecast
	- `lag` is the number of lags for the kalman filter (default 0)

### Calculate news/weights from a new data revision
1) `news = gen_news(old_y=old_df, new_y=new_df, output_dfm=output_dfm, target_variable=:target_col, target_period=Dates.Date(2020,1,1))`, where:
- `old_y/new_y` are the old and new datasets (i.e. same data, `new_y` has more recent values filled in)
- `output_dfm` is the output of a previously estimated model
- `target_variable` is which column you would like to forecast
- `target_period` is which date you would like to forecast for

2) Outputs a dictionary with `:y_old, :y_hat, :news_table`, run `?gen_news` for more info.
