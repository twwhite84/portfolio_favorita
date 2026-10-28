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
# how many records each year? 648648 in regular year, 650430 in leap year (eg 2016)
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
# number of records per day is always 1782
q = text("""
    select `date`, count(*) n
    from train
    group by `date`
    having n <> 1782
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# the math doesnt check out when multiplying by n days of a year, so there may be a missing day in each year's dataset
print(1782*365) # 650430, but my yearly record count for a non-leap year (eg 2015) says 648648
print(1782*366) # 652212, but my yearly record count for a leap year (eg 2016) says 650430

# %%
# how many days are present in each year of the dataset? 364 for regular years, 365 for leap year
q = text("""
    select year(`date`) `year`, count(*) n_days
    from (select distinct `date` from train) _
    group by year(`date`)
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# which day is missing? using 2015 as a candidate year, we find its missing day is christmas day, december 25
q=text("""
    with recursive 
    cal_2015 as (
        select '2015-01-01' d
        union all
        select d + interval 1 day from cal_2015
        where d < '2016-01-01' - interval 1 day
    ),
    train_2015 as (
        select distinct `date` d
        from train t
        where t.date >= '2015-01-01' and t.date < '2016-01-01'
    )
    select d from cal_2015 where d not in (select d from train_2015)
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# christmas day is missing every year in train. there is always only 1 day missing, so it's always going to be dec 25th each year
q=text("""
    select *
    from train t
    where date_format(t.date, '%m-%d')='12-25'
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# there are always 33 distinct family values per day
q=text("""
    with sq as (
        select distinct
            `date`,
            family
        from train t
    )
    select
        `date`,
        count(family) as n_family
    from sq
    group by `date`
    having count(*) <> 33
    ;
""")
pd.read_sql_query(q, con=engine)

# %%
# there are always 54 distinct store values per day
q = text("""
    with sq as (
        select distinct
            `date`,
            store_nbr
        from train t
    )
    select
        `date`,
        count(store_nbr) as n_stores
    from sq
    group by `date`
    having count(*) <> 54
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
    select id, id_lag, id-id_lag delta from sq
    having delta <> 1
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
# # SARIMAX
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
# training set spanning one year with exogenous variable (temp set to 6 month for testing purposes)
q=text("""
with sq as (
    select `date`, store_nbr, family, sales, onpromotion
    from train t
    where t.date >= ('2017-07-30'-interval 1 year) and t.date <= '2017-07-30'
)
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    sales y,
    onpromotion
from sq
""")
df_train = pd.read_sql_query(q, con=engine).convert_dtypes()
df_train['ds']=pd.to_datetime(df_train['ds'])
df_train

# %%
# is dec 25th missing from df_train? yes
df_train.loc[(df_train['ds'].dt.month==12) & (df_train['ds'].dt.day==25)]

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
# add new records for any missing dates
df_train = fill_gaps(df_train, freq='D', id_col='unique_id', time_col='ds')
df_train['y'] = df_train['y'].fillna(0)
df_train['onpromotion'] = df_train['onpromotion'].fillna(0)
df_train.loc[df_train['ds']=='2016-12-25']

# %%
# I need future exogenous values covering my future/validation window
q=text("""
with sq as (
    select `date`, store_nbr, family, onpromotion
    from train t
    where t.date >= '2017-07-31' and t.date <= '2017-08-15'
)
select
    concat(store_nbr,'_',family) unique_id,
    `date` ds,
    onpromotion
from sq
""")
df_Xvalid = pd.read_sql_query(q, con=engine).convert_dtypes()
df_Xvalid['ds']=pd.to_datetime(df_Xvalid['ds'])
df_Xvalid

# %%
# instantiate and fit
fcast_sarima = StatsForecast(
    models=[AutoARIMA(season_length=7)],
    freq='D',
    n_jobs=-1
)
fcast_sarima.fit(df_train)

# %%
# predict
df_sarima = fcast_sarima.predict(h=16, level=[90], X_df=df_Xvalid)
df_sarima = pd.merge(df_sarima, df_valid, on=['unique_id', 'ds'], how="inner")
df_sarima

# %%
# compared against actuals, RMSE is better than a seasonal naive
root_mean_squared_error(df_sarima['y'], df_sarima['AutoARIMA'])


