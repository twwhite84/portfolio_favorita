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
# # SEASONAL NAIVE FORECASTER
# This performs a year-over-year comparison to establish a baseline RMSE performance that I can then gauge my other models by. 2017 is my current YTD up to August 15.
# I am using a window on 2016 of the same range to serve as the predicted values. The project goal is to forecast the target 'sales' 15 days past YTD.

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
# without an index the joining used to perform the lag will take forever, so add one
with engine.begin() as con:
    con.execute(text("create index idx_train_lookup on train (date, store_nbr, family);"))

# %%
# the rmse is 644.47, rmsle is 0.86
q=text("""
    select
        t2017.date d,
        t2017.sales,
        t2016.sales sales_lag
    from train t2017
    join train t2016
        on t2016.date = t2017.date - interval 1 year
        and t2016.store_nbr = t2017.store_nbr
        and t2016.family = t2017.family
    where t2017.date >= '2017-01-01' and t2017.date <= '2017-08-15'
""")
df_naive = pd.read_sql_query(q, con=engine)
rmse = root_mean_squared_error(df_naive['sales'], df_naive['sales_lag'])
rmsle = root_mean_squared_log_error(df_naive['sales'], df_naive['sales_lag'])
print(f"RMSE: {rmse}, RMSLE: {rmsle}")


