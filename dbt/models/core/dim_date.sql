-- Daily calendar over the range of the transactions, without gaps.
with bounds as (
    select
        min(txn_date) as start_date,
        max(txn_date) as end_date
    from {{ ref('stg_transactions') }}
),

days as (
    select cast(unnest(generate_series(start_date, end_date, interval 1 day)) as date) as date_day
    from bounds
)

select
    date_day,
    year(date_day) as year,
    quarter(date_day) as quarter,
    month(date_day) as month,
    cast(date_trunc('month', date_day) as date) as month_start,
    day(date_day) as day_of_month,
    isodow(date_day) as iso_weekday,
    isodow(date_day) >= 6 as is_weekend
from days
