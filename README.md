# [**Predict Future Sales**](https://www.kaggle.com/c/competitive-data-science-predict-future-sale)

**Objective and Overview**
- In this Notebook, I will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company.
- I will predict total sales for every product and store in the next month using Datasets given.

**Fastai Library or API**
- [Fast.ai](https://www.fast.ai/about/) is the first deep learning library to provide a single consistent interface to all the most commonly used deep learning applications for vision, text, tabular data, time series, and collaborative filtering.
- [Fast.ai](https://www.fast.ai/about/) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches.

**Preparing the Model**
- I have used [Fastai](https://www.fast.ai/about/) API to train the Model. It seems quite challenging to understand the code if you have never encountered with Fast.ai API before.
One important note for anyone who has never used Fastai API before is to go through [Fastai Documentation](https://docs.fast.ai/). And if you are using Fastai in Jupyter Notebook then you can use doc(function_name) to get the documentation instantly.

**Data**
- I had prepared the Data for this Project from [Kaggle](https://www.kaggle.com/c/competitive-data-science-predict-future-sale/data)

**XGBRegressor**
- XGBoost is an open-source software library which provides a gradient boosting framework for C++, Java, Python, R, Julia, Perl, and Scala. It works on Linux, Windows, and macOS. From the project description, it aims to provide a "Scalable, Portable and Distributed Gradient Boosting Library".

**Processing with Fastai API**
- While working on Timeseries Data then, adding following function or API will enhance the accuracy of the Model a lot.

```javascript
def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    if isinstance(fldnames, str):
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64
            
        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub("[Dd]ate$", '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elasped'] = fld.astype(np.int64) // 10**9
        if drop: df.drop(fldname, axis=1, inplace=True)
```
**Snapshot of using XGBRegressor**

![Image](https://res.cloudinary.com/dge89aqpc/image/upload/v1596688419/Regre_psefkn.png)

