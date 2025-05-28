# Suggested Further Improvements

Below are code sections where additional work may be beneficial.

## `preprocess.py`
```python
# around line 360
elif not isinstance(df_after_outliers.index, pd.DatetimeIndex) and df_after_outliers.index.name == time_col_name_common:
    df_after_outliers.index = pd.to_datetime(df_after_outliers.index, utc=True)
```
Consider adding error handling if index conversion fails due to invalid values.

## `fetch_energy.py`
Ensure that the dataset and column names configured for Energi Data Service are
correct. Currently they are marked with comments like `# <<< YOU MUST VERIFY THIS DATASET NAME`.

## Tests
No automated tests exist for these scripts. Creating unit tests for the
processing utilities would help catch regressions.
