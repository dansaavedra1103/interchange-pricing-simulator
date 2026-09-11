-- Conservación del P&L: lo que reciben emisor, red y adquirente debe igualar el MDR que pagan
-- los comercios en cada fila, en cada mart y en los totales. Devuelve las filas que violan la
-- identidad; el test pasa cuando no devuelve ninguna.
with fct as (
    select * from {{ ref('fct_transactions') }}
),

row_level as (
    select
        'fct_transactions' as check_name,
        cast(txn_id as varchar) as grain,
        interchange_cop + scheme_fee_cop + acquirer_net_cop as received,
        mdr_cop as paid
    from fct
),

pnl_segments as (
    select
        'mart_pnl_by_actor' as check_name,
        concat_ws('|', txn_month, product, mcc_group) as grain,
        p.revenue_cop as received,
        f.mdr_cop as paid
    from (
        select txn_month, product, mcc_group, sum(mdr_cop) as mdr_cop
        from fct
        group by all
    ) as f
    full outer join (
        select txn_month, product, mcc_group, sum(revenue_cop) as revenue_cop
        from {{ ref('mart_pnl_by_actor') }}
        group by all
    ) as p
        using (txn_month, product, mcc_group)
),

interchange_segments as (
    select
        'mart_effective_interchange' as check_name,
        concat_ws('|', txn_month, product, mcc_group, channel, region) as grain,
        interchange_cop + scheme_fee_cop + acquirer_net_cop as received,
        mdr_cop as paid
    from {{ ref('mart_effective_interchange') }}
),

totals as (
    select
        'total mdr: mart_effective_interchange' as check_name,
        'all' as grain,
        (select sum(mdr_cop) from {{ ref('mart_effective_interchange') }}) as received,
        (select sum(mdr_cop) from fct) as paid
    union all
    select
        'total mdr: mart_merchant_relative_burden',
        'all',
        (select sum(mdr_cop) from {{ ref('mart_merchant_relative_burden') }}),
        (select sum(mdr_cop) from fct)
    union all
    select
        'purchases: mart_graph_edges',
        'all',
        (select sum(n_txns) from {{ ref('mart_graph_edges') }}),
        (select count(*) from fct)
    union all
    -- Un join que duplique filas inflaría el MDR sin romper la identidad fila a fila.
    select
        'rows: fct_transactions vs stg_transactions',
        'all',
        (select count(*) from fct),
        (select count(*) from {{ ref('stg_transactions') }})
),

checks as (
    select * from row_level
    union all
    select * from pnl_segments
    union all
    select * from interchange_segments
    union all
    select * from totals
)

select *
from checks
where
    received is null
    or paid is null
    or abs(received - paid) > 1e-6 * greatest(abs(paid), 1)
