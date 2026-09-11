-- P&L de cada actor que recibe una parte del MDR, por mes, producto y grupo de MCC:
-- el emisor cobra el interchange, la red el scheme fee y el adquirente el residuo.
with fct as (
    select * from {{ ref('fct_transactions') }}
),

receivers as (
    select
        txn_month,
        'issuer' as actor_type,
        issuer_id as actor_id,
        product,
        mcc_group,
        amount_cop,
        interchange_cop as revenue_cop
    from fct
    union all
    select txn_month, 'acquirer', acquirer_id, product, mcc_group, amount_cop, acquirer_net_cop
    from fct
    union all
    select txn_month, 'network', 'NETWORK', product, mcc_group, amount_cop, scheme_fee_cop
    from fct
)

select
    txn_month,
    actor_type,
    actor_id,
    product,
    mcc_group,
    count(*) as n_txns,
    sum(amount_cop) as gdv_cop,
    sum(revenue_cop) as revenue_cop
from receivers
group by all
