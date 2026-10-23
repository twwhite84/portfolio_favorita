-- PORTFOLIO: CORPORACION FAVORITA DB
-- https://www.kaggle.com/competitions/store-sales-time-series-forecasting

-- full unicode to handle some non-ascii characters, and case insensitive querying
create database portfolio_favorita
character set utf8mb4
collate utf8mb4_unicode_ci
;

use portfolio_favorita;

/*
 * TRAIN
 * The training data, comprising time series of features store_nbr, family, and onpromotion as well as the target sales
 * store_nbr:	identifies the store at which the products are sold.
 * family:		identifies the type of product sold.
 * sales:		gives the total sales for a product family at a particular store at a given date. products can be sold in fractional units (eg 1.5 kg of cheese)
 * onpromotion:	gives the total number of items in a product family that were being promoted at a store at a given date.
 */

create table train (
	id int primary key not null,
	`date` date not null,
	store_nbr int not null,
	family varchar(30) not null,
	sales decimal(14,7) not null, -- target
	onpromotion int not null
)
;

describe train;

load data local infile '/LEARNING/PORTFOLIO/corporacion_favorita/train.csv'
into table train
fields terminated by ','
optionally enclosed by '"'
lines terminated by '\n'
ignore 1 rows
(
	@id,
	@date,
	@store_nbr,
	@family,
	@sales,
	@onpromotion
)
set
id = nullif(@id, ''),
`date` = nullif(@date, ''),
store_nbr = nullif(@store_nbr, ''),
family = nullif(@family, ''),
sales = nullif(@sales, ''),
onpromotion = nullif(@onpromotion, '')
;

select * from train limit 100;


/*
 * STORES
 * store metadata, including city, state, type, and cluster
 * cluster is a grouping of similar stores
 */

create table stores (
	store_nbr int primary key not null,
	city varchar(20) not null,
	state varchar(30) not null,
	`type` char not null,
	cluster int not null
)
;

describe stores;

load data local infile '/LEARNING/PORTFOLIO/corporacion_favorita/stores.csv'
into table stores
fields terminated by ','
optionally enclosed by '"'
lines terminated by '\n'
ignore 1 rows
(
	@store_nbr,
	@city,
	@state,
	@type,
	@cluster
)
set
store_nbr = nullif(@store_nbr, ''),
city = nullif(@city, ''),
state = nullif(@state, ''),
`type` = nullif(@type, ''),
cluster = nullif(@cluster, '')
;

-- foreign key relation in train to this dimension table 
alter table train
add constraint fk_train_stores
foreign key (store_nbr) references stores(store_nbr)
;

select * from stores
;


/*
 * OIL
 * Daily oil price. Includes values during both the train and test data timeframes.
 * (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)
 */

create table oil (
	`date` date not null,
	dcoilwtico decimal(5,2)
)
;

describe oil;

load data local infile '/LEARNING/PORTFOLIO/corporacion_favorita/oil.csv'
into table oil
fields terminated by ','
optionally enclosed by '"'
lines terminated by '\n'
ignore 1 rows
(
	@date,
	@dcoilwtico
)
set
`date` = nullif(@date, ''),
dcoilwtico = nullif(@dcoilwtico, '')
;

select * from oil limit 100
;


/*
 * HOLIDAY EVENTS
 * Holidays and Events, with metadata
 * Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government.
 * A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer.
 * For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12.
 * Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend).
 * These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
 * Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).
 * Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
 * A magnitude 7.8 earthquake struck Ecuador on April 16, 2016.
 */

create table holidays_events (
	`date` date not null,
	`type` varchar(10) not null,
	locale varchar(20) not null,
	locale_name varchar(30) not null,
	description varchar(50) not null,
	transferred boolean not null
)
;

describe holidays_events
;

load data local infile '/LEARNING/PORTFOLIO/corporacion_favorita/holidays_events.csv'
into table holidays_events
fields terminated by ','
optionally enclosed by '"'
lines terminated by '\n'
ignore 1 rows
(
	@date,
	@type,
	@locale,
	@locale_name,
	@description,
	@transferred
)
set
`date` = nullif(@date, ''),
`type` = nullif(@type, ''),
locale = nullif(@locale, ''),
locale_name = nullif(@locale_name, ''),
description = nullif(@description, ''),
transferred = case
	when lower(@transferred)='true' then true
	when lower(@transferred)='false' then false
	else null
end
;

select * from holidays_events limit 100;

-- theres a lot of overlap between holidays_events.locale_name and stores.city
-- i could probably split out into a new locales table, maybe later as it's not really focus of this exercise

 -- 19 records in common
select count(*) from (
	select distinct s.city
	from stores s
		join holidays_events hs on s.city=hs.locale_name
) _
;

-- 8 not in common
select distinct s.city 
from stores s
	left join holidays_events hs
	on s.city=hs.locale_name
where hs.locale_name is null
union
select distinct hs.locale_name 
from holidays_events hs
	left join stores s
	on s.city=hs.locale_name
where s.city is null
;


/*
 * TRANSACTIONS
 */

create table transactions (
	`date` date not null,
 	store_nbr int not null,
 	transactions int not null
)
;

describe transactions;

load data local infile '/LEARNING/PORTFOLIO/corporacion_favorita/transactions.csv'
into table transactions
fields terminated by ','
optionally enclosed by '"'
lines terminated by '\n'
ignore 1 rows
(
	@date,
	@store_nbr,
	@transactions
)
set
`date` = nullif(@date, ''),
store_nbr = nullif(@store_nbr, ''),
transactions = nullif(@transactions, '')
;

select * from transactions order by `date`, store_nbr;

select count(*) from transactions;