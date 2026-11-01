# %% [markdown]
# The dates in the test data are for the 15 days after the last date in the training data.
# 
# https://www.kaggle.com/competitions/store-sales-time-series-forecasting

# %%
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error

# %%
# my db connection details stay hidden from public view in .env (which is not included in the repo)
load_dotenv()

DB_HOST=os.getenv("DB_HOST")
DB_PORT=os.getenv("DB_PORT")
DB_NAME=os.getenv("DB_NAME")
DB_USER=os.getenv("DB_USER")
DB_PASSWORD=os.getenv("DB_PASSWORD")

# %%
# using sqlalchemy to connect to my local maria db
connection_string = f"mariadb+mariadbconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

# %% [markdown]
# # DATA CHECK

# %%
# how many records per day? always 1782
q = text("""
select distinct count(*) n_records_daily
from train t
group by t.date
;
""")
pd.read_sql_query(q, con=engine)

# %%
# how many records per year? 648648 in a regular year, 650430 in leap year (ie 2016)
q = text("""
select year(`date`) `year`, count(*) n_records
from train
group by `year`
;
""")
pd.read_sql_query(q, con=engine)

# %%
# feb 29 2016 adds 1782 extra records
print(650430-648648)

# %%
# the math doesnt check out when multiplying by n days of a year, so there may be a missing day in each year's dataset
print(1782*365) # 650430, but my yearly record count for a non-leap year (eg 2015) says 648648
print(1782*366) # 652212, but my yearly record count for a leap year (eg 2016) says 650430

# %%
# how many days are present in each year of the dataset? 364 for regular years and 365 for a leap year, so a date is missing
q = text("""
    select year(`date`) `year`, count(*) n_days
    from (select distinct `date` from train) _
    group by year(`date`)
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# first and last days of training dataset
q=text("""
select min(t.date) first_date, max(t.date) last_date
from train t
""")
pd.read_sql_query(q, con=engine)

# %%
# to create a full calendar with no missing days between first and last dates, set a bigger recursion depth limit
with engine.begin() as con:
    con.execute(text("set session max_recursive_iterations = 2000;"))

# %%
# comparing the dates in the training set to the calendar, we can see it's only christmas that's constantly missing
q=text("""
    with recursive 
    cte_cal as (
        select '2013-01-01' d
        union all
        select d + interval 1 day from cte_cal
        where d < '2017-08-15'
    )
    select d from cte_cal where d not in (select distinct `date` from train)
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# there are always 33 distinct family values per day
q=text("""
    select distinct count(family) n_family
    from (select distinct `date`, family from train) _
    group by `date`
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# there are always 54 distinct store values per day
q = text("""
    select distinct count(store_nbr) n_stores
    from (select distinct `date`, store_nbr from train) _
    group by `date`
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# 54 stores x 33 families = 1782, ie the daily record count
# this suggests that date, store_nbr and family together form a sequence, so these must be aligned when lagging for my forecaster
print(54*33)

# 54 x 33 x n days
print(54*33*364) # 648648, which is correct for a non-leap year (eg 2015)
print(54*33*365) # 650430, which is correct for a leap year (eg 2016)

# %%
# onpromotion is an attribute that occurs every day
# it doesn't contribute to the sequence or else there'd be more than 1782 records per day
q=text("select distinct onpromotion from train;")
pd.read_sql_query(q, con=engine)

# %%
# there are no id gaps in the data for the range i need for the forecaster
q=text("""
    with sq as (
        select
            id, 
            lag(id, 1) over(order by id) id_lag 
        from train
        where `date` >= '2016-01-01'
    )
    select distinct (id-id_lag) delta from sq where id_lag is not null
""")
pd.read_sql_query(q, con=engine)

# %%
# but it appears there may be inconsistent ordering in the id field when i impose explicit ordering by the key columns
# this means i cant rely on the id field in order to perform the lag
q = text("""
    with sq as (
        select
            id, 
            lag(id, 1) over(order by `date`, store_nbr, family) id_lag 
        from train
        where `date` >= '2016-01-01'
    )
    select id, id_lag, id-id_lag delta from sq
    having delta <> 1
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# what indices do i have on train
pd.read_sql_query(text("show index from train;"), con=engine)

# %%
# without an index the joining used to perform the lag will take forever, so add one if it doesn't exist
df_keys = pd.read_sql_query(text("show index from train;"), con=engine)['Key_name']
if("idx_train_lookup" not in df_keys.values):
    with engine.begin() as con:
        con.execute(text("create index idx_train_lookup on train (date, store_nbr, family);"))

# %% [markdown]
# # NAIVE FORECASTER
# This compares a 15-day validation window at the end of my training set against the day immediately preceding that window. Use of a fixed cutoff like this simulates how a multi-step (ie multiple period) forecast can accumulate error the further it reaches out.

# %%
# validation window boundaries
pd.read_sql_query(text("""
select 
    max(`date`)-interval 15 day valid_start, 
    max(`date`) valid_end
from train
;
"""), con=engine)

# %%
# reference and validation sales columns
q=text("""
select
    t_ref.date date_ref,
    t_valid.date date_valid,
    t_ref.sales sales_ref,
    t_valid.sales sales_valid    
from train t_ref
join train t_valid
    on t_ref.date='2017-07-30'
    and t_ref.store_nbr=t_valid.store_nbr
    and t_ref.family=t_valid.family
where t_valid.date >= '2017-07-31' and t_valid.date <= '2017-08-15'
;
""")
df_naive = pd.read_sql_query(q, con=engine)
df_naive

# %%
rmse = root_mean_squared_error(df_naive['sales_valid'], df_naive['sales_ref'])
print(f"RMSE: {round(rmse,2)}")

# %% [markdown]
# # SEASONAL NAIVE FORECASTER
# This performs a day-aligned year-over-year comparison using a 15-day validation window at the end of my training set.

# %%
# lagging by 364 days aligns the days a year apart, not the dates, which matters because there are weekly cyclical patterns
q=text("""
    select
        t_ref.date date_ref,
        dayname(t_ref.date) day_ref,
        t_valid.date date_valid,
        dayname(t_valid.date) day_valid,    
        t_ref.sales sales_ref,
        t_valid.sales sales_valid
    from train t_ref
    join train t_valid
        on t_ref.date = t_valid.date - interval 364 day
        and t_ref.store_nbr = t_valid.store_nbr
        and t_ref.family = t_valid.family
    where t_valid.date >= '2017-07-31' and t_valid.date <= '2017-08-15'
""")
df_naive = pd.read_sql_query(q, con=engine)
df_naive

# %%
rmse = root_mean_squared_error(df_naive['sales_valid'], df_naive['sales_ref'])
print(f"RMSE: {round(rmse,2)}")

# %% [markdown]
# ## Comparing results vs statsforecast SeasonalNaive

# %%
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.preprocessing import fill_gaps

# %%
# validation set
q=text("""
with sq as (
    select `date`, store_nbr, family, sales
    from train t
    where t.date >= '2017-07-31' and t.date <= '2017-08-15'
)
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    sales y
from sq
""")
df_valid = pd.read_sql_query(q, con=engine).convert_dtypes()
df_valid['ds']=pd.to_datetime(df_valid['ds'])
df_valid

# %%
# training set
q=text("""
with sq as (
    select `date`, store_nbr, family, sales
    from train t
    where t.date < '2017-07-31'
)
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    sales y
from sq
""")
df_train = pd.read_sql_query(q, con=engine).convert_dtypes()
df_train['ds']=pd.to_datetime(df_train['ds'])
df_train

# %%
# is dec 25th missing from df_train? yes
df_train.loc[(df_train['ds'].dt.month==12) & (df_train['ds'].dt.day==25)]

# %%
# add new records for any missing dates
df_train = fill_gaps(df_train, freq='D', id_col='unique_id', time_col='ds')
df_train['y'] = df_train['y'].fillna(0)

# %%
# fit the model
fcast_naive = StatsForecast(
    models=[SeasonalNaive(season_length=364)],
    freq='D',
    n_jobs=-1
)
fcast_naive.fit(df_train)

# %%
# predict the validation window
df_naive = fcast_naive.predict(h=16, level=[90])
df_naive

# %%
# compare y_preds vs y_actuals, RMSE is the same as doing it manually
df_naive = pd.merge(df_naive, df_valid, on=['unique_id', 'ds'], how="inner")
rmse = root_mean_squared_error(df_naive['y'], df_naive['SeasonalNaive'])
print(f"RMSE: {round(rmse,2)}")

# %% [markdown]
# # ARIMA
# Use a 7-day season length to capture the dominant weekly cyclical pattern, bring in exogenous variables to inform the model of longer term trends, and train on a year's worth of data preceding the validation window.

# %%
from statsforecast.models import AutoARIMA

# %%
# validation set
q=text("""
with sq as (
    select `date`, store_nbr, family, sales
    from train t
    where t.date >= '2017-07-31' and t.date <= '2017-08-15'
)
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    sales y
from sq
""")
df_valid = pd.read_sql_query(q, con=engine).convert_dtypes()
df_valid['ds']=pd.to_datetime(df_valid['ds'])
df_valid

# %%
# I want to bring in holidays as an exogeneous variable, mapping training dates to holiday dates based on store properties
q=text("""
select distinct s.state, s.city, he.locale_name, he.locale
from stores s left join holidays_events he on s.city=he.locale_name or s.state=he.locale_name
order by state, city, locale desc
;
""")
pd.read_sql_query(q, con=engine)

# %%
# training set
q=text("""
with
sq_holiday as (select * from holidays_events where transferred=false and `type` in ('Holiday', 'Transfer','Additional')),
sq_train as (
select
	t.*,
	case
		when exists (select 1 from sq_holiday he where t.date=he.date and he.locale='National') then 1
        when exists (select 1 from sq_holiday he where t.date=he.date and he.locale='Regional' and s.state=he.locale_name) then 1
		when exists (select 1 from sq_holiday he where t.date=he.date and he.locale='Local' and s.city=he.locale_name) then 1
		else 0
	end as holiday,
    o.dcoilwtico
from train t 
    left join stores s on s.store_nbr=t.store_nbr
    left join oil o on t.date=o.date
where t.date >= ('2017-07-30'-interval 1 year) and t.date <= '2017-07-30'
)
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    sales y,
    onpromotion,
    holiday,
    dcoilwtico
from sq_train
;
""")
df_train = pd.read_sql_query(q, con=engine).convert_dtypes()
df_train['ds']=pd.to_datetime(df_train['ds'])
df_train

# %%
# checking for correlation, I only really get good correlation with onpromotion
print("sales-onpromotion:", df_train['y'].corr(df_train['onpromotion']))
print("sales-holiday:", df_train['y'].corr(df_train['holiday']))
print("sale-oil:", df_train['y'].corr(df_train['dcoilwtico']))

# %%
# going to use just the onpromotion field
df_train = df_train[['unique_id','ds','y','onpromotion']]
df_train

# %%
# on days with sales = 0, what is the onpromotion value? by far the most common value is also 0
q=text("""
    select 
        onpromotion,
        count(onpromotion) n
    from train
    where sales=0
    group by onpromotion
;
""")
pd.read_sql_query(q, con=engine)

# %%
# are those no-sales days with a promotion anywhere near christmas, the missing day? only once, so generally no
q=text("""
select distinct `date`, onpromotion
from train
where sales=0 and onpromotion <> 0
""")
pd.read_sql_query(q, con=engine)

# %%
# add new records for missing dates (in this case just christmas)
df_train = fill_gaps(df_train, freq='D', id_col='unique_id', time_col='ds')
df_train['y'] = df_train['y'].fillna(0)
df_train['onpromotion'] = df_train['onpromotion'].fillna(0)

# %%
# I need future exogenous values covering my future/validation window
q=text("""
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    onpromotion
from train t
where t.date >= '2017-07-31' and t.date <= '2017-08-15'
;
""")
df_Xvalid = pd.read_sql_query(q, con=engine).convert_dtypes()
df_Xvalid['ds']=pd.to_datetime(df_Xvalid['ds'])
df_Xvalid

# %%
# instantiate and fit
fcast_arima = StatsForecast(
    models=[AutoARIMA(season_length=7)],
    freq='D',
    n_jobs=-1
)
fcast_arima.fit(df_train)

# %%
# predict
df_arima = fcast_arima.predict(h=16, level=[90], X_df=df_Xvalid)
df_arima = pd.merge(df_arima, df_valid, on=['unique_id', 'ds'], how="inner")
df_arima

# %%
root_mean_squared_error(df_arima['y'], df_arima['AutoARIMA'])

# %%
root_mean_squared_log_error(df_arima['y'], abs(df_arima['AutoARIMA']))

# %% [markdown]
# ### ARIMA results
# - Arima, onpromotion only: 283.95
# - Arima, onpromotion and holiday: 290.23
# - Arima, onpromotion, holiday (new) and oil: 285.82

# %% [markdown]
# # LightGBM

# %%
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.linear_model import LinearRegression

# %%
# training set
q=text("""
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    sales y,
    onpromotion
from train t
where t.date < '2017-07-31'
;
""")
df_train = pd.read_sql_query(q, con=engine).convert_dtypes()
df_train['ds']=pd.to_datetime(df_train['ds'])
df_train

# %%
# add new records for missing dates (in this case just christmas)
df_train = fill_gaps(df_train, freq='D', id_col='unique_id', time_col='ds')
df_train['y'] = df_train['y'].fillna(0)
df_train['onpromotion'] = df_train['onpromotion'].fillna(0)

# %%
fcst = MLForecast(
    models=LinearRegression(),
    freq='D',
    lags=[1,2,3,7,14,21,28,364,365,366],
    target_transforms=[Differences([1])]
)
fcst.fit(df_train, static_features=[])

# %%
df_Xvalid

# %%
df_lgbm = fcst.predict(h=16, level=[90], X_df=df_Xvalid)
df_lgbm = pd.merge(df_lgbm, df_valid, on=['unique_id', 'ds'], how="inner")
df_lgbm

# %%
root_mean_squared_error(df_lgbm['y'], df_lgbm['LinearRegression'])

# %%
# training set
q=text("""
with
sq_holiday as (select * from holidays_events where transferred=false and `type` in ('Holiday', 'Transfer','Additional')),
sq_train as (
select
	t.*,
	case
		when exists (select 1 from sq_holiday he where t.date=he.date and he.locale='National') then 1
        when exists (select 1 from sq_holiday he where t.date=he.date and he.locale='Regional' and s.state=he.locale_name) then 1
		when exists (select 1 from sq_holiday he where t.date=he.date and he.locale='Local' and s.city=he.locale_name) then 1
		else 0
	end as holiday,
    o.dcoilwtico
from train t 
    left join stores s on s.store_nbr=t.store_nbr
    left join oil o on t.date=o.date
where t.date < '2017-07-31'
)
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    sales y,
    onpromotion,
    holiday,
    dcoilwtico
from sq_train
;
""")
df_train = pd.read_sql_query(q, con=engine).convert_dtypes()
df_train['ds']=pd.to_datetime(df_train['ds'])
df_train

# %%
# add new records for missing dates (in this case just christmas)
df_train = fill_gaps(df_train, freq='D', id_col='unique_id', time_col='ds')
df_train['y'] = df_train['y'].fillna(0)
df_train['onpromotion'] = df_train['onpromotion'].fillna(0)
df_train['holiday'] = df_train['holiday'].fillna(1)
df_train['dcoilwtico'] = df_train['dcoilwtico'].ffill().bfill()

# %%
# I need future exogenous values covering my future/validation window
q=text("""
with
sq_holiday as (select * from holidays_events where transferred=false and `type` in ('Holiday', 'Transfer','Additional')),
sq_train as (
select
	t.*,
	case
		when exists (select 1 from sq_holiday he where t.date=he.date and he.locale='National') then 1
        when exists (select 1 from sq_holiday he where t.date=he.date and he.locale='Regional' and s.state=he.locale_name) then 1
		when exists (select 1 from sq_holiday he where t.date=he.date and he.locale='Local' and s.city=he.locale_name) then 1
		else 0
	end as holiday,
    o.dcoilwtico
from train t 
    left join stores s on s.store_nbr=t.store_nbr
    left join oil o on t.date=o.date
where t.date >= '2017-07-31' and t.date <= '2017-08-15'
)
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    onpromotion,
    holiday,
    dcoilwtico
from sq_train
;
""")
df_Xvalid = pd.read_sql_query(q, con=engine).convert_dtypes()
df_Xvalid['ds']=pd.to_datetime(df_Xvalid['ds'])
df_Xvalid

# %%
# look for missing values
df_Xvalid['dcoilwtico'] = df_Xvalid['dcoilwtico'].ffill().bfill()
df_Xvalid.loc[df_Xvalid['dcoilwtico'].isnull()]


# %%
fcst = MLForecast(
    models=LinearRegression(),
    freq='D',
    lags=[1,2,3,7,14,21,28,364,365,366],
    target_transforms=[Differences([1])]
)
fcst.fit(df_train, static_features=[])

# %%
df_lgbm = fcst.predict(h=16, level=[90], X_df=df_Xvalid)
df_lgbm = pd.merge(df_lgbm, df_valid, on=['unique_id', 'ds'], how="inner")
df_lgbm

# %%
root_mean_squared_error(df_lgbm['y'], df_lgbm['LinearRegression']) # only marginally better

# %% [markdown]
# # Test

# %%



