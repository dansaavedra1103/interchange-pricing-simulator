-- Ingreso bruto de cada actor que recibe una parte del MDR, por mes, producto y grupo de
-- MCC: el emisor se queda el interchange neto de sus fees de red, la red cobra los fees de
-- ambos lados y el adquirente se queda el residuo.
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
        issuer_gross_cop as revenue_cop
    from fct
    union all
    select txn_month, 'acquirer', acquirer_id, product, mcc_group, amount_cop, acquirer_gross_cop
    from fct
    union all
    select txn_month, 'network', 'NETWORK', product, mcc_group, amount_cop, network_revenue_cop
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
