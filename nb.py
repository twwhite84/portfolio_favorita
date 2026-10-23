# %%
import pandas as pd

# %%
# my db connection details stay hidden from public view in .env (which is not included in the repo)
import os
from dotenv import load_dotenv
load_dotenv()

DB_HOST=os.getenv("DB_HOST")
DB_PORT=os.getenv("DB_PORT")
DB_NAME=os.getenv("DB_NAME")
DB_USER=os.getenv("DB_USER")
DB_PASSWORD=os.getenv("DB_PASSWORD")

# %%
# using sqlalchemy to connect to my local maria db
from sqlalchemy import create_engine
connection_string = f"mariadb+mariadbconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

# %%
# SEASONAL NAIVE FORECASTER
# using this to establish a year-over-year baseline performance
# 2017 is my current incomplete year, project goal is to predict 'sales' 15 days past YTD

# %%
# how many records each year? 648648 in regular year, 650430 in leap year (eg 2016)
# 2016 is a leap year, it has a feb 29th in it
q = """
    select
        year(`date`) y,
        count(*) n
    from train
    group by y
    ;
"""
pd.read_sql(q, con=engine)

# %%
# feb 29 adds 1782 extra records
print(650430-648648)

# %%
# number of records per day is always 1782
q = """
    select
        `date` d,
        count(*) n
    from train
    group by d
    having n <> 1782
    ;
"""
pd.read_sql(q, con=engine)

# %%
# the math doesnt check out, so there may be a missing day in each year's dataset
print(1782*365) # 650430, but my yearly record count for a non-leap year (eg 2015) says 648648
print(1782*366) # 652212, but my yearly record count for a leap year (eg 2016) says 650430

# %%
# how many days are in each year of the dataset? 364 for regular years, 365 for leap year
q = """
    with sq as (
        select distinct `date` d
        from train
    )
    select count(*) from sq
    group by year(d)
    ;
"""
pd.read_sql(q, con=engine)

# %%
# which day is missing? christmas day, december 25
q="""
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
"""
pd.read_sql(q, con=engine)

# %%
# christmas day is missing every year in train
q = """
    select *
    from train t
    where date_format(t.date, '%m-%d')='12-25'
    ;
"""
pd.read_sql(q, con=engine)

# %%
# there are always 33 distinct family values per day
q="""
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
"""
pd.read_sql(q, con=engine)

# %%
# there are always 54 distinct store values per day
q = """
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
"""
pd.read_sql(q, con=engine)

# %%
# 54 stores x 33 families = 1782, the constant daily record count
# this suggests that store_nbr and family are key columns and that the data is already tidy 
print(54*33)

# 54 x 33 x n days
print(54*33*364) # 648648, which is correct for a non-leap year (eg 2015)
print(54*33*365) # 650430, which is correct for a leap year (eg 2016)

# %%
# there are no id gaps in the data for the range i need for the forecaster
q="""
    with sq as (
        select
            id, 
            lag(id, 1) over(order by id) id_lag 
        from train
        where `date` >= '2016-01-01'
    )
    select id, id_lag, id-id_lag delta from sq
    having delta <> 1
"""
pd.read_sql(q, con=engine)

# %%
# it appears there may be inconsistent ordering in the id field when i impose ordering by date and key columns store_nbr and family
q = """
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
"""
pd.read_sql(q, con=engine)

# %%
# this same inconsistency appears after dropping the remaining column
q="""
    with sq_nopromo as (
        select
            id,
            `date`,
            store_nbr,
            family,
            sales
        from train
    ),
    sq as (
        select
            id, 
            lag(id, 1) over(order by `date`, store_nbr, family) id_lag 
        from sq_nopromo
        where `date` >= '2016-01-01'
    )
    select id, id_lag, id-id_lag delta from sq
    having delta <> 1
    ;
"""
pd.read_sql(q, con=engine)

# %%
# to lag i'll use a window function to compute a row number by id at daily grain for 2016 and 2017, then join on both rn and on year
# use of an inner join will automatically discard feb 29 from 2016 when pairing with 2017
q = """
with 
sq_2016 as ( 
	select
		`date` d,
		sales,
		row_number() over(partition by `date` order by `date`, store_nbr, family) rn
	from vw_train_nopromo
	where year(`date`)=2016
),
sq_2017 as (
	select
		`date` d,
		sales,
		row_number() over(partition by `date` order by `date`, store_nbr, family) rn
	from vw_train_nopromo
	where year(`date`)=2017
)
select
	sq_2017.d,
	sq_2017.sales,
	sq_2016.sales sales_lag
from sq_2017 join sq_2016
	on sq_2016.d=sq_2017.d - interval 1 year
	and sq_2016.rn=sq_2017.rn
;
"""
pd.read_sql(q, con=engine)


