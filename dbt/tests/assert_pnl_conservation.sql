-- Conservación del P&L: lo que se quedan emisor, red y adquirente debe igualar el MDR que
-- pagan los comercios en cada fila, en cada mart y en los totales; y el ingreso de la red debe
-- ser exactamente la suma de sus conceptos. Devuelve las filas que violan alguna identidad; el
-- test pasa cuando no devuelve ninguna.
with fct as (
    select * from {{ ref('fct_transactions') }}
),

row_level as (
    select
        'fct_transactions' as check_name,
        cast(txn_id as varchar) as grain,
        issuer_gross_cop + network_revenue_cop + acquirer_gross_cop as received,
        mdr_cop as paid
    from fct
),

network_drivers as (
    select
        'fct network drivers' as check_name,
        cast(txn_id as varchar) as grain,
        network_assessment_cop + network_authorization_cop + network_cross_border_cop
            as received,
        network_revenue_cop as paid
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
        issuer_gross_cop + network_revenue_cop + acquirer_gross_cop as received,
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
    select * from network_drivers
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
